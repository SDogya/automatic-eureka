"""Исходная CLI orchestration train/eval/smoke и resume без изменения порядка вычислений."""

import argparse
import itertools
import json
import os
import subprocess
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
    r"""Запустить train, eval или smoke через argparse и один flat YAML.
    
    CLI:
        --config PATH: Requested YAML; default config.yaml.
        --mode train|eval|smoke: Train defaults, checkpoint evaluation или smoke overrides.
        --checkpoint DIR: Model checkpoint для evaluation.
        --resume DIR: Full training_state.pt для продолжения исходного run;
            также можно использовать YAML resume_checkpoint.
    
    Порядок:
        Разрешить saved config/resume, выбрать physical GPU через UUID до CUDA init,
        загрузить policy/reference/tokenizer фиксированной revision, настроить LoRA
        и head, затем заменить scoring.masked_token_mean compiled kernel.
        Подготовить данные, metadata и TensorBoard. Новый train сначала делает
        baseline evaluation; resume восстанавливает optimizer, batches, metrics/RNG,
        JSONL и удаляет future evaluation artifacts после checkpoint.
        Loop вычисляет LR от исходного next update и horizon; updates/scoring,
        logging, evaluation и checkpoint cadence сохраняют исходный порядок.
        Финальная evaluation выполняется после train или самостоятельно в eval mode.
    
    Resume применяет сохранённые hyperparameters/data/schedule; из requested YAML
    разрешены location/hardware keys gpu, local_files_only, output_dir.
    Old checkpoints без training_state.pt доступны для evaluation.
    Для BF16 нет GradScaler. Reference frozen; policy/head обновляются одним optimizer.
    Все model loads, device operations и дорогие вычисления остаются внутри main."""
    parser = argparse.ArgumentParser(description="Online TraFL, Algorithm 1 / Eq.(4)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=("train", "eval", "smoke"), default="train")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--resume", help="Training checkpoint directory containing training_state.pt"
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    resume = args.resume or cfg.get("resume_checkpoint")
    resume_state = None
    if resume:
        import torch

        resume_state = torch.load(
            Path(resume) / "training_state.pt", map_location="cpu", weights_only=False
        )
        # Continue the original horizon/data/algorithm; only location and GPU can change.
        hardware = {key: cfg[key] for key in ("gpu", "local_files_only", "output_dir")}
        cfg = resume_state["config"] | hardware
        cfg["resume_checkpoint"] = None
    gpu_uuid = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(cfg["gpu"]),
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(cfg["compile_threads"])

    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from . import scoring as equations

    if args.mode == "smoke" and resume_state is None:
        cfg.update(
            updates=cfg["smoke_updates"],
            eval_limit=cfg["smoke_eval_limit"],
            eval_max_new_tokens=cfg["smoke_eval_max_new_tokens"],
            log_every=cfg["smoke_log_every"],
        )
    torch.manual_seed(cfg["seed"])
    device, dtype = torch.device("cuda:0"), getattr(torch, cfg["dtype"])
    model_path = snapshot_download(
        cfg["model_id"],
        revision=cfg["model_revision"],
        local_files_only=cfg["local_files_only"],
    )
    load = dict(
        trust_remote_code=cfg["trust_remote_code"],
        local_files_only=True,
        dtype=dtype,
        attn_implementation=cfg["attention_implementation"],
    )
    policy = AutoModelForMaskedLM.from_pretrained(model_path, **load).to(device)
    reference = (
        AutoModelForMaskedLM.from_pretrained(model_path, **load).to(device).eval()
    )
    reference.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    lora = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_targets"],
        lora_dropout=cfg["lora_dropout"],
        bias=cfg["lora_bias"],
        init_lora_weights=cfg["lora_init"],
    )
    model_state = resume_state
    if args.mode == "eval" and model_state is None:
        state_path = Path(args.checkpoint) / "training_state.pt"
        if state_path.exists():
            model_state = torch.load(state_path, map_location="cpu", weights_only=False)
    policy = (
        get_peft_model(policy, lora)
        if args.mode != "eval" or model_state is not None
        else PeftModel.from_pretrained(policy, args.checkpoint)
    )
    head = LogZ(
        policy.config.hidden_size,
        cfg["head_hidden"],
        cfg["head_activation"],
        cfg["head_bias"],
        cfg["head_zero_output"],
    ).to(device)
    step = 0
    if model_state is not None:
        set_peft_model_state_dict(policy, model_state["policy"])
        head.load_state_dict(model_state["head"])
        step = model_state["step"]
    elif args.mode == "eval":
        exported = torch.load(
            Path(args.checkpoint) / "head.pt", map_location=device, weights_only=True
        )
        head.load_state_dict(exported["head"])
        step = exported["step"]
    if cfg["compile"]:
        equations.masked_token_mean = torch.compile(
            equations.masked_token_mean,
            backend=cfg["compile_backend"],
            dynamic=cfg["compile_dynamic"],
        )
    forward, ref_forward = policy, reference
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + args.mode
    out = (
        Path(resume).resolve().parent
        if resume_state is not None
        else Path(cfg["output_dir"]) / name
    )
    out.mkdir(parents=True, exist_ok=resume_state is not None)
    if resume_state is not None:
        for pattern in ("answers-step*-*.json", "metrics-step*.json"):
            for path in out.glob(pattern):
                recorded_step = int(
                    path.name.split("step", 1)[1].split("-", 1)[0].split(".", 1)[0]
                )
                if recorded_step > step:
                    path.unlink()
    requested_name = (
        f"requested-resume-step{step}.yaml"
        if resume_state is not None
        else "requested_config.yaml"
    )
    (out / requested_name).write_text(Path(args.config).read_text())
    selected_ids = (
        resume_state["selected_examples"] if resume_state is not None else None
    )
    train, validation = prepared_data(cfg, tokenizer, out, selected_ids)
    cfg.update(
        resolved_model_path=model_path,
        model_hidden_size=policy.config.hidden_size,
        mask_token_id=tokenizer.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        run_mode=args.mode,
        resolved_gpu_uuid=gpu_uuid,
    )
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    )
    write_json(
        out / "versions.json",
        {
            name: version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "datasets",
                "tensorboard",
                "pyyaml",
            )
        },
    )
    writer = SummaryWriter(
        str(out / "tensorboard"),
        purge_step=step + 1 if resume_state is not None else None,
    )
    print(
        json.dumps(
            {
                "run_dir": str(out),
                "model": cfg["model_id"],
                "train_examples": len(train),
                "evaluation_examples": len(validation),
                "mode": args.mode,
            }
        ),
        flush=True,
    )
    if args.mode != "eval":
        # Same evaluation generator/settings before and after, current/reference are initially identical.
        if resume_state is None:
            evaluate(
                policy,
                forward,
                ref_forward,
                head,
                validation,
                cfg,
                tokenizer,
                dtype,
                device,
                writer,
                out,
                0,
            )
        parameters = [
            parameter
            for parameter in itertools.chain(policy.parameters(), head.parameters())
            if parameter.requires_grad
        ]
        optimizer = getattr(torch.optim, cfg["optimizer"])(
            parameters,
            lr=cfg["lr_peak"],
            betas=tuple(cfg["adam_betas"]),
            eps=cfg["adam_eps"],
            weight_decay=cfg["weight_decay"],
            fused=cfg["optimizer_fused"],
        )
        rng = torch.Generator(device=device).manual_seed(cfg["rollout_seed"])
        mask_rng = torch.Generator().manual_seed(cfg["mask_seed"])
        running = torch.zeros(5, device=device)
        accumulated = 0
        loader_state = {
            "epoch": -1,
            "order": None,
            "next_batch": 0,
            "shuffle_rng": torch.Generator().manual_seed(cfg["seed"]).get_state(),
            "worker_rng": torch.Generator().manual_seed(cfg["seed"]).get_state(),
        }
        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
            loader_state = resume_state["loader"]
            running = resume_state["running"].to(device)
            accumulated = resume_state["accumulated"]
            step = resume_state["scheduler"]["next_update"]
        if step < cfg["updates"]:
            batches, batches_per_epoch = epoch_batches(train, cfg, loader_state)
        if resume_state is not None:
            # Initialization, tokenization and worker construction must not consume saved RNG.
            restore_rng(resume_state["rng"], rng, mask_rng)
        record_mode = "r+" if resume_state is not None else "w"
        with (out / "train_answers.jsonl").open(
            record_mode, encoding="utf-8"
        ) as record:
            if resume_state is not None:
                record.seek(resume_state["record_position"])
                record.truncate()
            else:
                save_checkpoint(
                    policy,
                    head,
                    cfg,
                    out,
                    step,
                    optimizer,
                    rng,
                    mask_rng,
                    loader_state,
                    running,
                    accumulated,
                    record,
                )
            for update in range(step, cfg["updates"]):
                if loader_state["next_batch"] == batches_per_epoch:
                    batches, batches_per_epoch = epoch_batches(train, cfg, loader_state)
                lr = learning_rate(
                    update,
                    cfg["updates"],
                    cfg["lr_peak"],
                    cfg["lr_min"],
                    cfg["warmup_updates"],
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                running += update_batch(
                    policy,
                    forward,
                    ref_forward,
                    head,
                    next(batches),
                    cfg,
                    tokenizer,
                    rng,
                    mask_rng,
                    dtype,
                    device,
                    optimizer,
                    record,
                )
                accumulated += 1
                step = update + 1
                loader_state["next_batch"] += 1
                if step % cfg["log_every"] == 0 or step == cfg["updates"]:
                    values = (running / accumulated).cpu().tolist()
                    tags = (
                        "loss/train",
                        "reward/train",
                        "reward/mixed_groups",
                        "score/difference",
                        "log_z/mean",
                    )
                    metrics = dict(zip(tags, values)) | {"lr": lr}
                    for tag, value in metrics.items():
                        writer.add_scalar(tag, value, step)
                    writer.flush()
                    print(json.dumps({"step": step, **metrics}), flush=True)
                    running.zero_()
                    accumulated = 0
                if cfg["eval_every"] and step % cfg["eval_every"] == 0:
                    evaluate(
                        policy,
                        forward,
                        ref_forward,
                        head,
                        validation,
                        cfg,
                        tokenizer,
                        dtype,
                        device,
                        writer,
                        out,
                        step,
                    )
                if (
                    cfg["save_every"]
                    and step % cfg["save_every"] == 0
                    and step != cfg["updates"]
                ):
                    save_checkpoint(
                        policy,
                        head,
                        cfg,
                        out,
                        step,
                        optimizer,
                        rng,
                        mask_rng,
                        loader_state,
                        running,
                        accumulated,
                        record,
                    )
            save_checkpoint(
                policy,
                head,
                cfg,
                out,
                step,
                optimizer,
                rng,
                mask_rng,
                loader_state,
                running,
                accumulated,
                record,
            )
    evaluate(
        policy,
        forward,
        ref_forward,
        head,
        validation,
        cfg,
        tokenizer,
        dtype,
        device,
        writer,
        out,
        step,
    )
    writer.close()
    print(f"Completed: {out}", flush=True)

