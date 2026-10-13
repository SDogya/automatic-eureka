"""Benchmark frozen pretrained BD3LM; no training-package imports or adapters.

Run from the project root with the existing remote environment::

    ../.venv/bin/python benchmarks/pretrain/run.py --config benchmarks/pretrain/config.yaml
    ../.venv/bin/python benchmarks/pretrain/run.py --resume runs/pretrain_gsm8k/RUN

The native dllm sampler owns generation, attention, cache and EOS handling.
An observer records revealed-token confidence after the original diffusion step
without drawing random numbers. Each four-draw chunk has a stable independent
seed and an atomic completion marker, so interrupted chunks can be regenerated.
"""

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import yaml

from artifacts import (
    aggregate, bootstrap, parse_answer, reconstruct_trace, sha256,
    trace_arrays, write_json, write_npz,
)


def chunk_seed(base, example_id, start):
    """Stable per-chunk seed, independent of interruption and Python hash salt."""
    digest = hashlib.sha256(f"{base}:{example_id}:{start}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def collate(rows):
    """Preserve variable-length prompt tensors for DataLoader's pin-memory walk."""
    return rows[0]


def load_chunks(out):
    """Read only committed chunks and verify their answer/trace hashes."""
    records, completed, chunks = [], set(), []
    for marker in sorted((out / "chunks").glob("*.complete.json")):
        entry = json.loads(marker.read_text())
        for name, digest in entry["sha256"].items():
            assert sha256(out / "chunks" / name) == digest, f"Artifact hash mismatch: {name}"
        payload = json.loads((out / "chunks" / entry["answers"]).read_text())
        assert len(payload["samples"]) == entry["count"]
        records.extend(payload["samples"])
        completed.add((entry["example_id"], entry["sample_start"]))
        chunks.append(payload["performance"])
    return records, completed, chunks


def log_progress(writer, out, cfg, records, chunks, started):
    """Log completed-task metrics, saved-draw diagnostics and measured throughput."""
    stats, _, _ = aggregate(records, cfg["samples"], cfg["pass_k"])
    step = stats["completed_problems"]
    for k, value in stats["pass_at_k"].items():
        writer.add_scalar(f"eval/pass@{k}", value, step)
    for name in ("parse_failure_rate", "mean_answer_tokens", "has_eos_rate", "has_mask_rate", "has_pad_rate"):
        writer.add_scalar("eval/" + name, stats[name], step)
    generation_seconds = sum(c["generation_seconds"] for c in chunks)
    generated_tokens = sum(c["generated_tokens"] for c in chunks)
    reveals = sum(c["revealed_tokens"] for c in chunks)
    mean_confidence = sum(c["confidence_sum"] for c in chunks) / max(1, reveals)
    mean_log_confidence = sum(c["log_confidence_sum"] for c in chunks) / max(1, reveals)
    stats.update(
        generation_seconds=generation_seconds,
        measured_session_seconds=time.monotonic() - started,
        samples_per_generation_second=len(records) / max(generation_seconds, 1e-9),
        generated_tokens_per_second=generated_tokens / max(generation_seconds, 1e-9),
        mean_confidence=mean_confidence,
        mean_log_confidence=mean_log_confidence,
        model_forwards=sum(c["model_forwards"] for c in chunks),
        projected_generation_seconds=generation_seconds / max(1, len(records)) * cfg["expected_problems"] * cfg["samples"],
        complete=step == cfg["expected_problems"],
    )
    writer.add_scalar("progress/problems", step, step)
    writer.add_scalar("progress/samples", len(records), step)
    writer.add_scalar("sampling/confidence", mean_confidence, step)
    writer.add_scalar("sampling/log_confidence", mean_log_confidence, step)
    writer.add_scalar("performance/samples_per_second", stats["samples_per_generation_second"], step)
    writer.add_scalar("performance/tokens_per_second", stats["generated_tokens_per_second"], step)
    writer.add_scalar("performance/model_forwards", stats["model_forwards"], step)
    if chunks:
        writer.add_scalar("performance/chunk_seconds", chunks[-1]["generation_seconds"], step)
        writer.add_scalar("performance/gpu_peak_allocated_mb", chunks[-1]["peak_allocated_bytes"] / 2**20, step)
        writer.add_scalar("performance/gpu_peak_reserved_mb", chunks[-1]["peak_reserved_bytes"] / 2**20, step)
    writer.flush()
    write_json(out / "progress.json", stats)
    print(json.dumps(stats), flush=True)
    return stats


def main():
    """Load raw weights once, generate complete test split and bootstrap task scores."""
    parser = argparse.ArgumentParser(description="Pure pretrained BD3LM GSM8K benchmark")
    parser.add_argument("--config")
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg_path = Path(args.config) if args.config else (
        Path(args.resume) / "config.yaml" if args.resume else Path(__file__).with_name("config.yaml")
    )
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["compile"] is False, "Native sampler benchmark uses eager inference"
    assert cfg["reward_mode"] == "last_boxed_exact_number"
    gpu_uuid = subprocess.check_output([
        "nvidia-smi", "-i", str(cfg["gpu"]), "--query-gpu=uuid", "--format=csv,noheader",
    ], text=True).strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    import dllm
    import dllm.core.samplers.bd3lm as native
    from dllm.utils.configs import ModelArguments

    started = time.monotonic()
    device, dtype = torch.device("cuda:0"), getattr(torch, cfg["dtype"])
    snapshot = Path(snapshot_download(cfg["model_id"], revision=cfg["model_revision"], local_files_only=True))
    dataset_path = Path(cfg["dataset_path"]).resolve()
    sampler_config = native.BD3LMSamplerConfig(**{
        name: cfg[name] for name in (
            "max_new_tokens", "steps", "steps_per_block", "block_size", "temperature",
            "remasking", "stochastic_transfer", "cfg_scale", "cfg_keep_tokens", "right_shift_logits",
        )
    }, return_dict=True)
    # Hash all executable dllm sources: changes to sampling/model/tokenization
    # cannot silently alter a resumed protocol, even with the same model ID.
    vendor = Path(inspect.getfile(dllm)).parent
    identity = {
        "model_files": {str(p.relative_to(snapshot)): sha256(p) for p in sorted(snapshot.rglob("*")) if p.is_file()},
        "dataset_sha256": sha256(dataset_path),
        "native_sources": {str(p.relative_to(vendor)): sha256(p) for p in sorted(vendor.rglob("*.py"))},
        "benchmark_sources": {name: sha256(Path(__file__).with_name(name)) for name in ("run.py", "artifacts.py")},
        "versions": {name: version(name) for name in ("torch", "transformers", "dllm", "numpy", "pyarrow", "tensorboard", "pyyaml", "huggingface-hub")},
        "gpu_uuid": gpu_uuid,
    }
    out = Path(args.resume) if args.resume else Path(cfg["output_dir"]) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if args.resume:
        previous = json.loads((out / "manifest.json").read_text())
        assert previous["config"] == cfg, "Resume generation/config mismatch"
        assert previous["identity"] == identity, "Resume model/data/source/environment mismatch"
    else:
        (out / "chunks").mkdir(parents=True)
        (out / "source").mkdir()
        for name in ("run.py", "artifacts.py"):
            shutil.copyfile(Path(__file__).with_name(name), out / "source" / name)
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        write_json(out / "manifest.json", {
            "config": cfg, "identity": identity, "resolved_model_path": str(snapshot),
            "resolved_dataset_path": str(dataset_path), "sampler_config": dataclasses.asdict(sampler_config),
            "dataset_download_manifest": json.loads(Path(cfg["dataset_manifest"]).read_text()),
            "protocol": "pure_pretrained_stochastic_boxed; differs from official greedy gsm8k_cot eval",
            "confidence_definition": "native-dtype softmax of raw logits at revealed token; FP32 log with tiny clamp",
            "seed_definition": "SHA256(base:example_id:sample_start), first 8 bytes little-endian modulo 2**63-1",
            "gpu": subprocess.check_output(["nvidia-smi", "-i", str(cfg["gpu"]), "-q"], text=True),
        })
    print(json.dumps({"run_dir": str(out.resolve()), "model": cfg["model_id"], "pure_pretrained": True}), flush=True)

    model_args = ModelArguments(model_name_or_path=str(snapshot), dtype=cfg["dtype"], lora=False, attn_implementation=cfg["attention_implementation"])
    model = dllm.utils.get_model(model_args).eval().requires_grad_(False)
    assert not hasattr(model, "peft_config"), "Benchmark must not load adapters"
    assert not any("lora_" in name for name, _ in model.named_parameters())
    assert not any(p.requires_grad for p in model.parameters())
    tokenizer = dllm.utils.get_tokenizer(model_args)
    write_json(out / "model_load.json", {
        "class": type(model).__name__, "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": 0, "peft": False, "dtype": str(next(model.parameters()).dtype),
        "attention_implementation": model.config._attn_implementation,
        "special_ids": {name: getattr(tokenizer, name + "_token_id") for name in ("mask", "eos", "pad", "bos")},
    })
    data = pq.read_table(dataset_path).to_pylist()
    assert len(data) == cfg["expected_problems"]
    rows, selected = [], []
    for index, row in enumerate(data):
        prompt = cfg["prompt_template"].replace("{question}", row["question"])
        ids = tokenizer.apply_chat_template([
            {"role": "user", "content": prompt},
        ], tokenize=True, add_generation_prompt=True, enable_thinking=cfg["enable_thinking"])
        padded_length = math_ceil_multiple(len(ids), cfg["block_size"])
        selected.append(row | {
            "example_id": index, "prompt": prompt, "prompt_ids": ids,
            "padded_prompt_ids": [tokenizer.pad_token_id] * (padded_length - len(ids)) + ids,
        })
        rows.append(row | {"example_id": index, "input_ids": torch.tensor(ids, dtype=torch.long)})
    if args.resume:
        assert json.loads((out / "selected_examples.json").read_text()) == selected
    else:
        write_json(out / "selected_examples.json", selected)
    records, completed, chunks = load_chunks(out)
    writer = SummaryWriter(str(out / "tensorboard"))
    log_progress(writer, out, cfg, records, chunks, started)
    loader = DataLoader(rows, batch_size=1, shuffle=False, num_workers=cfg["num_workers"],
                        pin_memory=cfg["pin_memory"], collate_fn=collate,
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    sampler = native.BD3LMSampler(model=model, tokenizer=tokenizer)
    observed, forward_count = [], 0
    original_step = native._diffusion_step_block

    def observe_step(**kwargs):
        """Call native transition first; record confidence without extra RNG draws."""
        updated = original_step(**kwargs)
        revealed = kwargs["mask_block"] & (updated != tokenizer.mask_token_id)
        probability = F.softmax(kwargs["logits"], dim=-1).gather(-1, updated.unsqueeze(-1)).squeeze(-1)
        observed.append((revealed.clone(), probability.float(), kwargs["num_transfer_step"].clone()))
        return updated

    def count_forward(module, inputs, output):
        """Count sampler model calls; do not touch logits or native cache objects."""
        nonlocal forward_count
        forward_count += 1

    native._diffusion_step_block = observe_step
    hook = model.register_forward_hook(count_forward)
    for row in loader:
        prompt = row["input_ids"].to(device, non_blocking=True)
        example_id = row["example_id"]
        for start in range(0, cfg["samples"], cfg["microbatch"]):
            if (example_id, start) in completed:
                continue
            count = min(cfg["microbatch"], cfg["samples"] - start)
            seed = chunk_seed(cfg["seed"], example_id, start)
            torch.manual_seed(seed)
            observed.clear()
            forward_count = 0
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            tick = time.monotonic()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
                output = sampler.sample(inputs=[prompt] * count, config=sampler_config)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - tick
            sequences = output.sequences.cpu().numpy().astype(np.int32)
            arrays = trace_arrays(output.histories, observed, tokenizer.mask_token_id, cfg["block_size"])
            assert np.array_equal(reconstruct_trace(arrays), sequences), "Native trace reconstruction mismatch"
            prefix_length = int(arrays["history_lengths"][0])
            batch_records = []
            for sample, sequence in enumerate(sequences):
                raw = sequence[prefix_length:].tolist()
                eos = raw.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in raw else None
                trimmed = raw[:eos + 1] if eos is not None else raw
                text = tokenizer.decode(trimmed, skip_special_tokens=True)
                sample_number = start + sample
                batch_records.append({
                    "example_id": example_id, "sample_id": sample_number, "seed": seed,
                    "sample_start": start, "row_in_chunk": sample,
                    "question": row["question"], "gold_answer": row["answer"],
                    "prompt_ids": row["input_ids"].tolist(), "padded_prompt_length": prefix_length,
                    "full_sequence_ids": sequence.tolist(), "raw_completion_ids": raw,
                    "answer_ids": trimmed, "answer_text": text,
                    "raw_text": tokenizer.decode(raw, skip_special_tokens=False),
                    "raw_clean_text": tokenizer.decode(raw, skip_special_tokens=True),
                    "eos_position": eos, "generated_length": len(raw), "answer_length": len(trimmed),
                    "has_eos": eos is not None, "has_mask": tokenizer.mask_token_id in raw,
                    "has_pad": tokenizer.pad_token_id in raw,
                } | parse_answer(text, row["answer"]))
            performance = {
                "generation_seconds": elapsed, "model_forwards": forward_count,
                "generated_tokens": int(sequences[:, prefix_length:].size),
                "revealed_tokens": len(arrays["confidence"]),
                "confidence_sum": float(arrays["confidence"].astype(np.float64).sum()),
                "log_confidence_sum": float(arrays["log_confidence"].astype(np.float64).sum()),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            name = f"problem-{example_id:04d}-samples-{start:02d}"
            answers, trace = name + ".json", name + ".npz"
            write_npz(out / "chunks" / trace, **arrays)
            write_json(out / "chunks" / answers, {"samples": batch_records, "performance": performance})
            write_json(out / "chunks" / (name + ".complete.json"), {
                "example_id": example_id, "sample_start": start, "count": count,
                "answers": answers, "trace": trace,
                "sha256": {file: sha256(out / "chunks" / file) for file in (answers, trace)},
            })
            records.extend(batch_records)
            completed.add((example_id, start))
            chunks.append(performance)
            del output, arrays
            observed.clear()
        completed_problems = example_id + 1
        if completed_problems <= cfg["initial_check_problems"] or completed_problems % cfg["log_every"] == 0:
            log_progress(writer, out, cfg, records, chunks, started)
    hook.remove()
    native._diffusion_step_block = original_step
    stats = log_progress(writer, out, cfg, records, chunks, started)
    assert len(records) == cfg["expected_problems"] * cfg["samples"]
    assert len({(r["example_id"], r["sample_id"]) for r in records}) == len(records)
    _, scores, problem_ids = aggregate(records, cfg["samples"], cfg["pass_k"])
    assert len(scores) == cfg["expected_problems"]
    draws, bounds = bootstrap(scores, cfg["bootstrap_repetitions"], cfg["bootstrap_seed"], cfg["confidence_level"])
    write_npz(out / "bootstrap.npz", distributions=draws, problem_ids=problem_ids, task_scores=scores,
              pass_k=np.asarray(cfg["pass_k"]), seed=np.asarray(cfg["bootstrap_seed"]))
    stats["confidence_intervals"] = {
        str(k): {"lower": float(bounds[0, i]), "upper": float(bounds[1, i]),
                 "degenerate": bool(bounds[0, i] == bounds[1, i])}
        for i, k in enumerate(cfg["pass_k"])
    }
    stats["bootstrap_method"] = "task-level percentile; individual intervals conditional on collected draws"
    for k, interval in stats["confidence_intervals"].items():
        for bound in ("lower", "upper"):
            writer.add_scalar(f"ci/pass@{k}/{bound}", interval[bound], len(scores))
    write_json(out / "metrics.json", stats)
    with (out / "answers.jsonl.tmp").open("w") as stream:
        for record in sorted(records, key=lambda r: (r["example_id"], r["sample_id"])):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out / "answers.jsonl.tmp").replace(out / "answers.jsonl")
    write_json(out / "complete.json", {"samples": len(records), "problems": len(scores), "metrics_sha256": sha256(out / "metrics.json")})
    writer.close()
    print(json.dumps({"completed": str(out.resolve()), "metrics": stats}), flush=True)


def math_ceil_multiple(length, block_size):
    """Align the prefix exactly as native BD3LMSampler does."""
    return ((length + block_size - 1) // block_size) * block_size


if __name__ == "__main__":
    main()
