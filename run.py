"""Run: ../.venv/bin/python run.py --config config.yaml --mode smoke (or train/eval)."""

import argparse
import itertools
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import yaml

from trafl import (
    LogZ,
    antithetic_masks,
    balance_residual,
    boxed_reward,
    generate,
    learning_rate,
    masked_score,
    pass_at_k,
    prompt_features,
    trim_completion,
)


def collate(rows):
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepared_data(cfg, tokenizer, out, selected_ids=None):
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = cfg["dataset_revision"] or HfApi().dataset_info(cfg["dataset_id"]).sha
    cfg["dataset_revision"] = revision
    dataset = load_dataset(cfg["dataset_id"], cfg["dataset_config"], revision=revision)
    result, selected = [], {}
    for split_key, limit_key in (
        ("train_split", "train_limit"),
        ("eval_split", "eval_limit"),
    ):
        split = dataset[cfg[split_key]]
        if selected_ids is None:
            indices = torch.randperm(
                len(split), generator=torch.Generator().manual_seed(cfg["seed"])
            )
            if cfg[limit_key] is not None:
                indices = indices[: cfg[limit_key]]
        else:
            indices = torch.tensor(selected_ids[cfg[split_key]])
        rows = []
        for index in indices.tolist():
            example = split[index]
            text = cfg["prompt_template"].replace("{question}", example["question"])
            # Replace only the task placeholder; literal boxed braces stay unchanged.
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=cfg["enable_thinking"],
            )
            rows.append(
                {
                    "input_ids": torch.tensor(ids),
                    "answer": example["answer"],
                    "question": example["question"],
                    "example_id": index,
                }
            )
        result.append(rows)
        selected[cfg[split_key]] = indices.tolist()
    write_json(out / "selected_examples.json", selected)
    return result


def sample_group(model, prompt, count, length, prefix, cfg, tokenizer, rng, dtype):
    return generate(
        model,
        prompt,
        count,
        length,
        cfg[f"{prefix}_block_size"],
        cfg[f"{prefix}_temperature"],
        cfg[f"{prefix}_top_p"],
        cfg[f"{prefix}_confidence_threshold"],
        tokenizer.mask_token_id,
        cfg["generation_suppress_mask"],
        rng,
        dtype,
    )


def scored_completions(generated, tokenizer, answer, cfg):
    # One host transfer at the generation boundary: reward extraction requires text.
    token_lists = [
        trim_completion(ids, tokenizer.eos_token_id) for ids in generated.cpu().tolist()
    ]
    texts = tokenizer.batch_decode(token_lists, skip_special_tokens=True)
    rewards = [boxed_reward(text, answer, cfg["invalid_reward"]) for text in texts]
    return token_lists, texts, rewards


def update_batch(
    policy,
    forward,
    reference,
    head,
    rows,
    cfg,
    tokenizer,
    rng,
    mask_rng,
    dtype,
    device,
    optimizer,
    record,
):
    import torch

    optimizer.zero_grad(set_to_none=True)
    stats = torch.zeros(5, device=device)
    denominator = len(rows) * cfg["rollouts"]
    for row in rows:
        prompt = row["input_ids"].to(device, non_blocking=True)
        policy.eval()
        generated = sample_group(
            forward,
            prompt,
            cfg["rollouts"],
            cfg["train_max_new_tokens"],
            "train",
            cfg,
            tokenizer,
            rng,
            dtype,
        )
        token_lists, texts, reward_list = scored_completions(
            generated, tokenizer, row["answer"], cfg
        )
        rewards = torch.tensor(reward_list, device=device)
        centered = rewards - rewards.mean()
        features = prompt_features(policy, prompt, dtype)
        policy.train()
        stats[1] += rewards.sum() / denominator
        stats[2] += (rewards.max() != rewards.min()).float() / len(rows)
        for g, ids in enumerate(token_lists):
            completion = generated[g, : len(ids)]
            eligible = torch.tensor([token != tokenizer.eos_token_id for token in ids])
            masks = antithetic_masks(
                eligible, cfg["mask_replicates"], mask_rng, cfg["mask_endpoint"]
            )
            score = masked_score(
                forward,
                prompt,
                completion,
                masks,
                tokenizer.mask_token_id,
                dtype,
                cfg["empty_score"],
            )
            with torch.no_grad():
                ref_score = masked_score(
                    reference,
                    prompt,
                    completion,
                    masks,
                    tokenizer.mask_token_id,
                    dtype,
                    cfg["empty_score"],
                )
            log_z = head(features).squeeze(0)
            residual = balance_residual(
                score, ref_score, centered[g], log_z, cfg["beta"]
            )
            loss = residual.square() / denominator
            loss.backward()
            stats[0] += loss.detach()
            stats[3] += (score - ref_score).detach() / denominator
            stats[4] += log_z.detach() / denominator
        record.write(
            json.dumps(
                {
                    "example_id": row["example_id"],
                    "question": row["question"],
                    "completions": texts,
                    "rewards": reward_list,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    if cfg["grad_clip"] is not None:
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(policy.parameters(), head.parameters()), cfg["grad_clip"]
        )
    optimizer.step()  # Exactly ONE update after the complete prompt/rollout batch.
    return stats


def evaluate(
    policy,
    forward,
    reference,
    head,
    rows,
    cfg,
    tokenizer,
    dtype,
    device,
    writer,
    out,
    step,
):
    import torch

    policy.eval()
    results = {}
    for length in cfg["eval_max_new_tokens"]:
        rng = torch.Generator(device=device).manual_seed(cfg["eval_seed"] + length)
        mask_rng = torch.Generator().manual_seed(cfg["mask_seed"])
        totals, answers = {}, []
        validation_loss = torch.zeros((), device=device)
        for row in rows:
            prompt = row["input_ids"].to(device, non_blocking=True)
            generated = sample_group(
                forward,
                prompt,
                cfg["eval_samples"],
                length,
                "eval",
                cfg,
                tokenizer,
                rng,
                dtype,
            )
            ids, texts, reward_list = scored_completions(
                generated, tokenizer, row["answer"], cfg
            )
            correct = int(sum(reward_list))
            for k in cfg["eval_pass_k"]:
                name = f"pass@{k}"
                totals[name] = totals.get(name, 0.0) + pass_at_k(
                    correct, cfg["eval_samples"], k
                )
            # Validation loss uses its own G=5 group (first five independent evaluation draws).
            rewards = torch.tensor(reward_list[: cfg["rollouts"]], device=device)
            centered = rewards - rewards.mean()
            with torch.no_grad():
                log_z = head(prompt_features(policy, prompt, dtype)).squeeze(0)
                losses = []
                for g, tokens in enumerate(ids[: cfg["rollouts"]]):
                    eligible = torch.tensor(
                        [token != tokenizer.eos_token_id for token in tokens]
                    )
                    masks = antithetic_masks(
                        eligible, cfg["mask_replicates"], mask_rng, cfg["mask_endpoint"]
                    )
                    completion = generated[g, : len(tokens)]
                    score = masked_score(
                        forward,
                        prompt,
                        completion,
                        masks,
                        tokenizer.mask_token_id,
                        dtype,
                        cfg["empty_score"],
                    )
                    ref_score = masked_score(
                        reference,
                        prompt,
                        completion,
                        masks,
                        tokenizer.mask_token_id,
                        dtype,
                        cfg["empty_score"],
                    )
                    losses.append(
                        balance_residual(
                            score, ref_score, centered[g], log_z, cfg["beta"]
                        ).square()
                    )
                validation_loss += torch.stack(losses).mean()
            answers.append(
                {
                    "example_id": row["example_id"],
                    "question": row["question"],
                    "completions": texts,
                    "rewards": reward_list,
                }
            )
        totals["loss/val"] = validation_loss.cpu().item()
        metrics = {name: value / len(rows) for name, value in totals.items()}
        for name, value in metrics.items():
            tag = f"{name}/{length}" if name == "loss/val" else f"eval/{length}/{name}"
            writer.add_scalar(tag, value, step)
        write_json(out / f"answers-step{step}-{length}.json", answers)
        results[str(length)] = metrics
    writer.add_scalar(
        "loss/val", sum(m["loss/val"] for m in results.values()) / len(results), step
    )
    writer.add_scalar(
        "reward/val", sum(m["pass@1"] for m in results.values()) / len(results), step
    )
    writer.flush()
    write_json(out / f"metrics-step{step}.json", results)
    print(json.dumps({"step": step, "evaluation": results}), flush=True)
    policy.train()
    return results


def epoch_batches(train, cfg, state):
    import torch
    from torch.utils.data import DataLoader

    size = cfg["prompt_batch_size"]
    count = (len(train) + size - 1) // size
    if state["order"] is None or state["next_batch"] == count:
        shuffle_rng = torch.Generator().set_state(state["shuffle_rng"])
        state["order"] = torch.randperm(len(train), generator=shuffle_rng).tolist()
        state["shuffle_rng"] = shuffle_rng.get_state()
        state["epoch"] += 1
        state["next_batch"] = 0
        state["epoch_worker_rng"] = state["worker_rng"]
    # Prefetch never advances next_batch: only a completed optimizer update does.
    order = state["order"]
    batches = [order[i:i + size] for i in range(state["next_batch"] * size, len(order), size)]
    worker_rng = torch.Generator().set_state(state["epoch_worker_rng"])
    loader = DataLoader(train, batch_sampler=batches, num_workers=cfg["num_workers"],
                        pin_memory=cfg["pin_memory"], collate_fn=collate,
                        generator=worker_rng)
    iterator = iter(loader)
    state["worker_rng"] = worker_rng.get_state()
    return iterator, count


def restore_rng(state, rng, mask_rng):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    rng.set_state(state["rollout"])
    mask_rng.set_state(state["mask"])


def save_checkpoint(policy, head, cfg, out, step, optimizer, rng, mask_rng,
                    loader_state, running, accumulated, record):
    import numpy as np
    import torch
    from peft import get_peft_model_state_dict

    checkpoint = out / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    record.flush()
    os.fsync(record.fileno())
    state = {
        "policy": get_peft_model_state_dict(policy), "head": head.state_dict(),
        "optimizer": optimizer.state_dict(), "step": step, "config": dict(cfg),
        "scheduler": {"next_update": step, "horizon": cfg["updates"]},
        "loader": loader_state, "running": running.detach().cpu(),
        "accumulated": accumulated, "record_position": record.tell(),
        "selected_examples": json.loads((out / "selected_examples.json").read_text()),
        "rng": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
            "rollout": rng.get_state(), "mask": mask_rng.get_state(),
        },
    }
    # Self-contained authoritative resume state; old state survives interrupted writes.
    temporary = checkpoint / "training_state.pt.tmp"
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, checkpoint / "training_state.pt")
    directory = os.open(checkpoint, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(directory)
    os.close(directory)
    # Also keep the familiar standalone evaluation export.
    policy.save_pretrained(checkpoint)
    torch.save({"head": head.state_dict(), "step": step}, checkpoint / "head.pt")
    (checkpoint / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    )


def main():
    parser = argparse.ArgumentParser(description="Online TraFL, Algorithm 1 / Eq.(4)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=("train", "eval", "smoke"), default="train")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", help="Training checkpoint directory containing training_state.pt")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    resume = args.resume or cfg.get("resume_checkpoint")
    resume_state = None
    if resume:
        import torch

        resume_state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
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
    import trafl as equations

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
    policy = (get_peft_model(policy, lora) if args.mode != "eval" or model_state is not None
              else PeftModel.from_pretrained(policy, args.checkpoint))
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
    out = Path(resume).resolve().parent if resume_state is not None else Path(cfg["output_dir"]) / name
    out.mkdir(parents=True, exist_ok=resume_state is not None)
    if resume_state is not None:
        for pattern in ("answers-step*-*.json", "metrics-step*.json"):
            for path in out.glob(pattern):
                recorded_step = int(path.name.split("step", 1)[1].split("-", 1)[0].split(".", 1)[0])
                if recorded_step > step:
                    path.unlink()
    requested_name = f"requested-resume-step{step}.yaml" if resume_state is not None else "requested_config.yaml"
    (out / requested_name).write_text(Path(args.config).read_text())
    selected_ids = resume_state["selected_examples"] if resume_state is not None else None
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
    writer = SummaryWriter(str(out / "tensorboard"),
                           purge_step=step + 1 if resume_state is not None else None)
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
            evaluate(policy, forward, ref_forward, head, validation, cfg, tokenizer,
                     dtype, device, writer, out, 0)
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
            "epoch": -1, "order": None, "next_batch": 0,
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
        with (out / "train_answers.jsonl").open(record_mode, encoding="utf-8") as record:
            if resume_state is not None:
                record.seek(resume_state["record_position"])
                record.truncate()
            else:
                save_checkpoint(policy, head, cfg, out, step, optimizer, rng, mask_rng,
                                loader_state, running, accumulated, record)
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
                if cfg["save_every"] and step % cfg["save_every"] == 0 and step != cfg["updates"]:
                    save_checkpoint(policy, head, cfg, out, step, optimizer, rng, mask_rng,
                                    loader_state, running, accumulated, record)
            save_checkpoint(policy, head, cfg, out, step, optimizer, rng, mask_rng,
                            loader_state, running, accumulated, record)
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


if __name__ == "__main__":
    main()
