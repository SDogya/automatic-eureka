"""Flat train/eval CLI; torchrun adds synchronous prompt data parallelism."""

import argparse
import itertools
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import yaml

from .checkpoint import restore_rng, save_checkpoint, write_json
from .data import epoch_batches, prepared_data
from .evaluation import evaluate
from .model import LogZ
from .objective import learning_rate
from .training import update_batch


def main():
    """Run one policy on one GPU or one identical replica per torchrun rank.

    Each rank owns local prompts, G rollouts, a frozen reference, independent
    rollout/mask RNG and a JSONL record. Group centering stays within each prompt.
    update_batch averages the accumulated trainable gradients in one collective
    before clipping and the joint optimizer step. This avoids reducer hooks on
    variable-length generation, detached prompt features and streamed backwards.
    Rank zero writes shared metadata/TensorBoard and the authoritative checkpoint;
    small per-rank runtime states preserve exact sharding/RNG on same-world resume.
    Evaluation shards prompts too, using globally weighted metrics.
    """
    parser = argparse.ArgumentParser(description="Online TraFL, Algorithm 1 / Eq.(4)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=("train", "eval", "smoke"), default="train")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", help="Full training checkpoint directory")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    resume = args.resume or cfg.get("resume_checkpoint")
    resume_state = None
    if resume:
        import torch
        resume_state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
        assert resume_state["config"].get("training_protocol") == cfg["training_protocol"], "Resume sampler protocol differs"
        assert resume_state["config"].get("distributed_world_size", 1) == world, "Exact resume requires the same world size"
        hardware = {key: cfg[key] for key in ("gpu", "local_files_only", "output_dir")}
        cfg = resume_state["config"] | hardware
        cfg["resume_checkpoint"] = None
        if world > 1:
            resume_state = resume_state | resume_state["rank_states"][rank]
    # torchrun honors the caller's CUDA_VISIBLE_DEVICES; plain python uses YAML gpu.
    if "LOCAL_RANK" not in os.environ:
        gpu_uuid = subprocess.check_output([
            "nvidia-smi", "-i", str(cfg["gpu"]), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(cfg["compile_threads"])

    import numpy as np
    import torch
    import torch.distributed as dist
    import dllm  # Register pinned A2D Qwen before AutoModel loading.
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from . import scoring as equations

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(backend=cfg["distributed_backend"], device_id=device)
    dtype = getattr(torch, cfg["dtype"])
    if dtype == torch.bfloat16:
        assert torch.cuda.is_bf16_supported(including_emulation=False), "GPU lacks native BF16 support"
    assert cfg["attention_implementation"] == "sdpa"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(cfg["sdpa_backend"] == "efficient")
    torch.backends.cuda.enable_math_sdp(cfg["sdpa_backend"] == "math")
    if args.mode == "smoke" and resume_state is None:
        cfg.update(updates=cfg["smoke_updates"], eval_limit=cfg["smoke_eval_limit"],
                   eval_max_new_tokens=cfg["smoke_eval_max_new_tokens"], log_every=cfg["smoke_log_every"])
    torch.manual_seed(cfg["seed"])  # Common adapter/head initialization on all ranks.
    model_path = snapshot_download(cfg["model_id"], revision=cfg["model_revision"], local_files_only=cfg["local_files_only"])
    load = dict(trust_remote_code=cfg["trust_remote_code"], local_files_only=True,
                dtype=dtype, attn_implementation=cfg["attention_implementation"])
    policy = AutoModelForMaskedLM.from_pretrained(model_path, **load).to(device)
    reference = AutoModelForMaskedLM.from_pretrained(model_path, **load).to(device).eval()
    reference.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_state = resume_state
    if args.mode == "eval" and model_state is None:
        state_path = Path(args.checkpoint) / "training_state.pt"
        if state_path.exists():
            model_state = torch.load(state_path, map_location="cpu", weights_only=False)
    if model_state is not None:
        saved = model_state["config"]
        assert (saved["model_id"], saved["model_revision"]) == (cfg["model_id"], cfg["model_revision"]), "Checkpoint base model differs"
    lora = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], target_modules=cfg["lora_targets"],
                      lora_dropout=cfg["lora_dropout"], bias=cfg["lora_bias"], init_lora_weights=cfg["lora_init"])
    policy = (get_peft_model(policy, lora) if args.mode != "eval" or model_state is not None
              else PeftModel.from_pretrained(policy, args.checkpoint))
    head = LogZ(policy.config.hidden_size, cfg["head_hidden"], cfg["head_activation"],
                cfg["head_bias"], cfg["head_zero_output"]).to(device)
    step = 0
    if model_state is not None:
        set_peft_model_state_dict(policy, model_state["policy"])
        head.load_state_dict(model_state["head"])
        step = model_state["step"]
    elif args.mode == "eval":
        exported = torch.load(Path(args.checkpoint) / "head.pt", map_location=device, weights_only=True)
        head.load_state_dict(exported["head"])
        step = exported["step"]
    parameters = [p for p in itertools.chain(policy.parameters(), head.parameters()) if p.requires_grad]
    if world > 1 and args.mode != "eval":
        # One initial broadcast, before graphs or optimizer exist.
        weights = torch.nn.utils.parameters_to_vector(parameters).detach()
        dist.broadcast(weights, src=0)
        torch.nn.utils.vector_to_parameters(weights, parameters)
    stride = cfg.get("rank_seed_stride", 1000003)
    rank_seed = cfg["seed"] + rank * stride
    random.seed(rank_seed)
    np.random.seed(rank_seed % 2**32)
    torch.manual_seed(rank_seed)
    if cfg["compile"]:
        for name in ("masked_token_mean", "batched_token_scores"):
            setattr(equations, name, torch.compile(getattr(equations, name),
                    backend=cfg["compile_backend"], dynamic=cfg["compile_dynamic"]))
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + args.mode
    run_path = str(Path(resume).resolve().parent if resume_state is not None else Path(cfg["output_dir"]) / name)
    if world > 1:
        path = [run_path]
        dist.broadcast_object_list(path, src=0)
        run_path = path[0]
    out = Path(run_path)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=resume_state is not None)
        if resume_state is not None:
            for pattern in ("answers-step*-*.json", "metrics-step*.json"):
                for path in out.glob(pattern):
                    recorded_step = int(path.name.split("step", 1)[1].split("-", 1)[0].split(".", 1)[0])
                    if recorded_step > step:
                        path.unlink()
        requested = f"requested-resume-step{step}.yaml" if resume_state is not None else "requested_config.yaml"
        (out / requested).write_text(Path(args.config).read_text())
    selected = resume_state["selected_examples"] if resume_state is not None else None
    if rank == 0:
        train, validation = prepared_data(cfg, tokenizer, out, selected)
    if world > 1:
        revision = [cfg["dataset_revision"]]
        dist.broadcast_object_list(revision, src=0)
        cfg["dataset_revision"] = revision[0]
        if rank != 0:
            train, validation = prepared_data(cfg, tokenizer, out, selected, write_output=False)
    cfg.update(resolved_model_path=model_path, model_hidden_size=policy.config.hidden_size,
               mask_token_id=tokenizer.mask_token_id, eos_token_id=tokenizer.eos_token_id,
               run_mode=args.mode, distributed_world_size=world,
               global_prompt_batch_size=world*cfg["prompt_batch_size"],
               resolved_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    writer = None
    if rank == 0:
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        write_json(out / "versions.json", {name: version(name) for name in
                   ("torch", "transformers", "peft", "datasets", "tensorboard", "pyyaml")})
        writer = SummaryWriter(str(out / "tensorboard"), purge_step=step+1 if resume_state is not None else None)
        print(json.dumps(dict(run_dir=str(out), model=cfg["model_id"], train_examples=len(train),
              evaluation_examples=len(validation), mode=args.mode, world_size=world,
              global_prompt_batch_size=cfg["global_prompt_batch_size"])), flush=True)

    def eval_current():
        return evaluate(policy, policy, reference, head, validation, cfg, tokenizer,
                        dtype, device, writer, out, step, rank=rank, world_size=world)

    if args.mode != "eval":
        if resume_state is None:
            eval_current()
        optimizer = getattr(torch.optim, cfg["optimizer"])(parameters, lr=cfg["lr_peak"],
                    betas=tuple(cfg["adam_betas"]), eps=cfg["adam_eps"],
                    weight_decay=cfg["weight_decay"], fused=cfg["optimizer_fused"])
        rng = torch.Generator(device=device).manual_seed(cfg["rollout_seed"] + rank*stride)
        mask_rng = torch.Generator().manual_seed(cfg["mask_seed"] + rank*stride)
        running, accumulated = torch.zeros(5, device=device), 0
        loader_state = dict(epoch=-1, order=None, next_batch=0,
            shuffle_rng=torch.Generator().manual_seed(cfg["seed"]).get_state(),
            worker_rng=torch.Generator().manual_seed(rank_seed).get_state())
        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
            loader_state = resume_state["loader"]
            running, accumulated = resume_state["running"].to(device), resume_state["accumulated"]
            step = resume_state["scheduler"]["next_update"]
        if step < cfg["updates"]:
            batches, batch_count = epoch_batches(train, cfg, loader_state, rank, world)
        if resume_state is not None:
            restore_rng(resume_state["rng"], rng, mask_rng)
        record_name = f"train_answers-rank{rank}.jsonl" if world > 1 else "train_answers.jsonl"
        with (out / record_name).open("r+" if resume_state is not None else "w", encoding="utf-8") as record:
            def checkpoint():
                save_checkpoint(policy, head, cfg, out, step, optimizer, rng, mask_rng,
                                loader_state, running, accumulated, record)
            if resume_state is not None:
                record.seek(resume_state["record_position"])
                record.truncate()
            else:
                checkpoint()
            for update in range(step, cfg["updates"]):
                if loader_state["next_batch"] == batch_count:
                    batches, batch_count = epoch_batches(train, cfg, loader_state, rank, world)
                lr = learning_rate(update, cfg["updates"], cfg["lr_peak"], cfg["lr_min"], cfg["warmup_updates"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                running += update_batch(policy, policy, reference, head, next(batches), cfg,
                           tokenizer, rng, mask_rng, dtype, device, optimizer, record)
                accumulated += 1
                step = update + 1
                loader_state["next_batch"] += 1
                if step % cfg["log_every"] == 0 or step == cfg["updates"]:
                    torch.cuda.synchronize(device)  # Timing/logging boundary only.
                    values = running / accumulated
                    perf = torch.tensor([time.perf_counter()-started, torch.cuda.max_memory_allocated(device)/2**30], device=device)
                    if world > 1:
                        dist.all_reduce(values)
                        values /= world
                        dist.all_reduce(perf, op=dist.ReduceOp.MAX)
                    if rank == 0:
                        tags = ("loss/train", "reward/train", "reward/mixed_groups", "score/difference", "log_z/mean")
                        seconds, peak = perf.cpu().tolist()
                        metrics = dict(zip(tags, values.cpu().tolist())) | dict(lr=lr)
                        metrics.update({"perf/update_seconds": seconds, "perf/peak_allocated_gib": peak,
                                        "perf/prompts_per_second": cfg["global_prompt_batch_size"]/seconds})
                        for tag, value in metrics.items():
                            writer.add_scalar(tag, value, step)
                        writer.flush()
                        print(json.dumps({"step": step, **metrics}), flush=True)
                    running.zero_()
                    accumulated = 0
                if cfg["eval_every"] and step % cfg["eval_every"] == 0:
                    eval_current()
                if cfg["save_every"] and step % cfg["save_every"] == 0 and step != cfg["updates"]:
                    checkpoint()
            checkpoint()
    eval_current()
    if rank == 0:
        writer.close()
        print(f"Completed: {out}", flush=True)
    if world > 1:
        dist.destroy_process_group()
