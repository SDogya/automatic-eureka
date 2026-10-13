"""Install-time preparation, four native single-answer workers and dynamic scheduling.

run.sh selects --mode prepare, then --mode run; the controller starts --mode worker.
All RNG streams reset per attempt. The original BD3LMSampler owns transitions.
"""

import argparse
import inspect
import json
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
import pyarrow.parquet as pq
import yaml
from huggingface_hub import snapshot_download
import kagglehub
from artifacts import sha256, write_json
import dataclasses
import os
import random
import time
import numpy as np
from artifacts import parse_answer, reconstruct_trace, sha256, trace_arrays, write_json, write_npz
import fcntl
import signal
import sys
from collections import deque
from artifacts import write_json
from artifacts import attempt_seed, committed, archive_run, export_results, pending_tasks, progress

def prepare_main():
    """Prepare a fresh run or verify the exact saved config/data/source on resume."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config")
    parser.add_argument("--attempts", type=int)
    parser.add_argument("--gpus")
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root, run = Path(args.root).resolve(), Path(args.run).resolve()
    package = Path(__file__).resolve().parent
    cfg = yaml.safe_load((run / "config.yaml" if args.resume else Path(args.config or package / "config.yaml")).read_text())
    if not args.resume:
        if args.smoke:
            cfg.update(problem_limit=cfg["smoke_problems"], attempts=cfg["smoke_attempts"], smoke=True)
        else:
            cfg.update(problem_limit=cfg["expected_problems"], smoke=False)
        if args.attempts is not None:
            cfg["attempts"] = args.attempts
        if args.gpus:
            cfg["gpus"] = [int(g) for g in args.gpus.split(",")]
        if args.seed_start is not None:
            cfg["seed_start"] = args.seed_start
        cfg["pass_k"] = [k for k in cfg["pass_k"] if k <= cfg["attempts"]]
    assert len(cfg["gpus"]) == len(set(cfg["gpus"])) == 4, "Choose four distinct A100 GPU indices"
    assert 1 <= cfg["attempts"] < 2**32
    assert cfg["dtype"] == "bfloat16" and cfg["attention_implementation"] == "sdpa"
    assert cfg["sdpa_backend"] == "math" and not cfg["compile"]
    assert cfg["remasking"] == "low_confidence" and not cfg["stochastic_transfer"]
    assert cfg["cfg_scale"] == 0 and not cfg["right_shift_logits"]
    assert 1 <= cfg["problem_limit"] <= cfg["expected_problems"]
    attempt_seed(cfg["seed_start"], cfg["expected_problems"] - 1, cfg["attempts"] - 1)
    hardware = {}
    for gpu in cfg["gpus"]:
        values = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
            "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader,nounits"], text=True).strip().split(",")
        uuid, name, driver, memory = (v.strip() for v in values)
        assert "A100" in name, f"GPU {gpu} is {name}; this package requires four full A100 devices"
        hardware[str(gpu)] = {"uuid": uuid, "name": name, "driver": driver, "memory_mib": int(memory)}
    model = root / "models" / cfg["model_revision"]
    dataset = root / "datasets" / "gsm8k-v3"
    snapshot_download(cfg["model_id"], revision=cfg["model_revision"], local_dir=model, token=False)
    kagglehub.dataset_download(cfg["dataset_handle"], output_dir=str(dataset))
    test = dataset / cfg["dataset_file"]
    assert sha256(test) == cfg["dataset_sha256"], "Kaggle test split SHA256 mismatch"
    data = pq.read_table(test).to_pylist()
    assert len(data) == cfg["expected_problems"]
    selected = [row | {"example_id": i} for i, row in enumerate(data[:cfg["problem_limit"]])]
    # Native top-level import registers A2D Qwen classes with transformers.
    import dllm
    vendor = Path(inspect.getfile(dllm)).parent
    revision = subprocess.check_output(["git", "-C", str(root / "vendor/dllm"), "rev-parse", "HEAD"], text=True).strip()
    assert revision == cfg["dllm_revision"]
    identity = {
        "model_files": {str(p.relative_to(model)): sha256(p) for p in sorted(model.rglob("*"))
                        if p.is_file() and ".cache" not in p.relative_to(model).parts},
        "dataset_sha256": sha256(test),
        "native_sources": {str(p.relative_to(vendor)): sha256(p) for p in sorted(vendor.rglob("*.py"))},
        "package_sources": {p.name: sha256(p) for p in sorted(package.iterdir())
                            if p.is_file() and p.suffix in {".py", ".sh", ".yaml", ".lock", ".in"}},
        "versions": {n: version(n) for n in ("torch", "transformers", "dllm", "numpy", "pyarrow",
                     "tensorboard", "pyyaml", "huggingface-hub", "kagglehub", "accelerate")},
    }
    assert identity["versions"]["torch"] == "2.8.0+cu126", "Unexpected Torch/CUDA version"
    if args.resume:
        previous = json.loads((run / "manifest.json").read_text())
        assert previous["config"] == cfg and previous["identity"] == identity, "Resume identity mismatch"
        assert json.loads((run / "selected_examples.json").read_text()) == selected
        committed(run, cfg, selected)
    else:
        for name in ("attempts", "workers", "tensorboard"):
            (run / name).mkdir(exist_ok=True)
        (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        write_json(run / "selected_examples.json", selected)
        native_copy = run / "source/native_dllm"
        for p in vendor.rglob("*.py"):
            target = native_copy / p.relative_to(vendor)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        write_json(run / "manifest.json", {"config": cfg, "identity": identity, "hardware": hardware,
            "dllm_revision": revision, "model_path": str(model), "dataset_path": str(test),
            "seed_definition": "seed_start + (original_example_id << 32) + zero_based_attempt_index",
            "protocol": "pure_pretrained_stochastic_boxed; differs from official greedy gsm8k_cot",
            "reproducibility": "fixed A100 hardware, software and config; not guaranteed across platforms",
            "dataset": {"handle": cfg["dataset_handle"], "rows": len(data), "test_sha256": sha256(test)},
            "git_commit": subprocess.check_output(["git", "-C", str(package.parent), "rev-parse", "HEAD"],
                                                  text=True, stderr=subprocess.DEVNULL).strip()
                          if (package.parent / ".git").exists() else "source snapshot; see launch provenance"})
    write_json(run / "hardware-current.json", hardware)
    print(json.dumps({"prepared": str(run), "problems": len(selected), "attempts": cfg["attempts"],
                      "pure_pretrained": True}), flush=True)

def collate(rows):
    """Keep variable prompt tensors for DataLoader's recursive pin-memory walk."""
    return rows[0]

def reset_rng(seed, torch):
    """Reset every used RNG immediately before an independent native sample."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()
    run, gpu = Path(args.run), args.gpu
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    hardware = json.loads((run / "hardware-current.json").read_text())[str(gpu)]
    directory = run / "workers" / f"gpu{gpu}"
    directory.mkdir(exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=hardware["uuid"], CUBLAS_WORKSPACE_CONFIG=":4096:8",
        TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.data import DataLoader
    import dllm
    import dllm.core.samplers.bd3lm as native
    from dllm.utils.configs import ModelArguments

    torch.set_num_threads(cfg["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available(), "CUDA runtime is unavailable; check the NVIDIA driver"
    assert torch.cuda.get_device_capability() == (8, 0), "Expected A100 compute capability 8.0"
    assert torch.cuda.is_bf16_supported(including_emulation=False), "Native BF16 is unavailable"
    assert torch.version.cuda == "12.6", f"Expected CUDA 12.6 wheel, got {torch.version.cuda}"
    dtype, device = torch.bfloat16, torch.device("cuda:0")
    model_args = ModelArguments(model_name_or_path=manifest["model_path"], dtype=cfg["dtype"],
                                lora=False, attn_implementation=cfg["attention_implementation"])
    model = dllm.utils.get_model(model_args).eval().requires_grad_(False)
    assert not hasattr(model, "peft_config") and not any("lora_" in n for n, _ in model.named_parameters())
    assert all(p.dtype == dtype for p in model.parameters() if p.is_floating_point())
    tokenizer = dllm.utils.get_tokenizer(model_args)
    selected = json.loads((run / "selected_examples.json").read_text())
    rows = []
    for row in selected:
        prompt = cfg["prompt_template"].replace("{question}", row["question"])
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=cfg["enable_thinking"])
        rows.append(row | {"input_ids": torch.tensor(ids, dtype=torch.long), "prompt": prompt})
    loader = DataLoader(rows, batch_size=1, num_workers=cfg["num_workers"],
                        pin_memory=cfg["pin_memory"], collate_fn=collate,
                        multiprocessing_context="spawn" if cfg["num_workers"] else None,
                        generator=torch.Generator().manual_seed(cfg["assignment_seed"]))
    rows = list(loader)  # Pin the small prompt cache once, not once per task.
    config = native.BD3LMSamplerConfig(**{n: cfg[n] for n in ("max_new_tokens", "steps", "steps_per_block",
        "block_size", "temperature", "remasking", "stochastic_transfer", "cfg_scale", "cfg_keep_tokens",
        "right_shift_logits")}, return_dict=True)
    sampler = native.BD3LMSampler(model=model, tokenizer=tokenizer)
    observed, forward_count = [], 0
    original_step = native._diffusion_step_block

    def observe_step(**kwargs):
        """Execute the unmodified native step; collect confidence without RNG draws."""
        updated = original_step(**kwargs)
        revealed = kwargs["mask_block"] & (updated != tokenizer.mask_token_id)
        probability = F.softmax(kwargs["logits"], dim=-1).gather(-1, updated.unsqueeze(-1)).squeeze(-1)
        observed.append((revealed.clone(), probability.float(), kwargs["num_transfer_step"].clone()))
        return updated

    def count_forward(module, inputs, output):
        nonlocal forward_count
        forward_count += 1

    native._diffusion_step_block = observe_step
    hook = model.register_forward_hook(count_forward)

    def sample(prompt, seed, sampling_config):
        """Record only the native sample's GPU duration; transfer after completion."""
        nonlocal forward_count
        reset_rng(seed, torch)
        observed.clear()
        forward_count = 0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype), sdpa_kernel(SDPBackend.MATH):
            output = sampler.sample(inputs=[prompt], config=sampling_config)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - tick
        sequences = output.sequences.cpu().numpy().astype(np.int32)
        arrays = trace_arrays(output.histories, observed, tokenizer.mask_token_id, cfg["block_size"])
        assert np.array_equal(reconstruct_trace(arrays), sequences), "Trace reconstruction mismatch"
        result = sequences, arrays, elapsed, forward_count
        del output
        observed.clear()
        return result

    # Each device exercises the real model before the controller schedules tasks.
    prompt = rows[0]["input_ids"].to(device)
    masked = torch.cat((prompt, torch.full((cfg["block_size"],), tokenizer.mask_token_id, device=device)))
    mask, positions = native._prepare_for_sampling(masked.unsqueeze(0), cfg["block_size"], tokenizer.pad_token_id)
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype), sdpa_kernel(SDPBackend.MATH):
        logits = model(masked.unsqueeze(0), attention_mask=mask, position_ids=positions, use_cache=False).logits
    assert torch.isfinite(logits).all(), "BF16 real-model forward produced nonfinite logits"
    del logits
    tiny = dataclasses.replace(config, max_new_tokens=cfg["preflight_new_tokens"],
                               steps=cfg["preflight_steps"], steps_per_block=None)
    first, arrays, _, _ = sample(prompt, cfg["seed_start"], tiny)
    second, _, _, _ = sample(prompt, cfg["seed_start"], tiny)
    assert np.array_equal(first, second), "Same-seed native generation is not reproducible"
    write_json(directory / "preflight.json", {"passed": True, "native_bf16": True,
        "same_seed_same_tokens": True, "trace_reconstruction": True, "torch_cuda": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability()), "hardware": hardware,
        "parameter_dtype": str(next(model.parameters()).dtype), "gumbel_dtype": "torch.float64",
        "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": 0,
        "sampler_config": dataclasses.asdict(config)})
    del first, second, arrays, masked, mask, positions
    _, _, done = committed(run, cfg, selected)
    write_json(directory / "state.json", {"state": "ready"})
    while True:
        job = directory / "job.json"
        if not job.exists():
            time.sleep(0.25)
            continue
        example_id = json.loads(job.read_text())
        job.unlink()
        if example_id is None:
            break
        write_json(directory / "state.json", {"state": "running", "example_id": example_id})
        row = rows[example_id]
        prompt = row["input_ids"].to(device, non_blocking=True)
        for attempt in range(cfg["attempts"]):
            if (example_id, attempt) in done:
                continue
            seed = attempt_seed(cfg["seed_start"], example_id, attempt)
            sequence_batch, arrays, seconds, calls = sample(prompt, seed, config)
            sequence = sequence_batch[0].tolist()
            prefix = int(arrays["history_lengths"][0])
            raw = sequence[prefix:]
            eos = raw.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in raw else None
            trimmed = raw[:eos + 1] if eos is not None else raw
            text = tokenizer.decode(trimmed, skip_special_tokens=True)
            sample_record = {"example_id": example_id, "sample_id": attempt, "seed": seed,
                "question": row["question"], "gold_answer": row["answer"], "prompt": row["prompt"],
                "prompt_ids": row["input_ids"].tolist(), "padded_prompt_length": prefix,
                "full_sequence_ids": sequence, "raw_completion_ids": raw, "answer_ids": trimmed,
                "answer_text": text, "raw_text": tokenizer.decode(raw, skip_special_tokens=False),
                "raw_clean_text": tokenizer.decode(raw, skip_special_tokens=True), "eos_position": eos,
                "generated_length": len(raw), "answer_length": len(trimmed), "has_eos": eos is not None,
                "has_mask": tokenizer.mask_token_id in raw, "has_pad": tokenizer.pad_token_id in raw,
                "gpu": gpu, "gpu_uuid": hardware["uuid"]} | parse_answer(text, row["answer"])
            performance = {"gpu": gpu, "generation_seconds": seconds, "model_forwards": calls,
                "generated_tokens": len(raw), "revealed_tokens": len(arrays["confidence"]),
                "confidence_sum": float(arrays["confidence"].astype(np.float64).sum()),
                "log_confidence_sum": float(arrays["log_confidence"].astype(np.float64).sum()),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            name = f"problem-{example_id:04d}-attempt-{attempt:04d}"
            answers, trace = name + ".json", name + ".npz"
            sample_record["trace"] = "attempts/" + trace
            write_npz(run / "attempts" / trace, **arrays)
            write_json(run / "attempts" / answers, {"sample": sample_record, "performance": performance})
            write_json(run / "attempts" / (name + ".complete.json"), {"example_id": example_id,
                "sample_id": attempt, "answers": answers, "trace": trace,
                "sha256": {n: sha256(run / "attempts" / n) for n in (answers, trace)}})
            done.add((example_id, attempt))
            del sequence_batch, arrays
        write_json(directory / "state.json", {"state": "done", "example_id": example_id})
    hook.remove()
    native._diffusion_step_block = original_step

def stop_workers(workers):
    """Stop only processes launched by this controller before exporting artifacts."""
    for item in workers.values():
        process = item["process"]
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(w["process"].poll() is None for w in workers.values()):
        time.sleep(0.1)
    for item in workers.values():
        process = item["process"]
        if process.poll() is None:
            process.kill()
        process.wait()

def spawn_worker(run, gpu, failures):
    """Launch with an isolated log and clear obsolete assignment/state files."""
    directory = run / "workers" / f"gpu{gpu}"
    directory.mkdir(exist_ok=True)
    for name in ("job.json", "state.json"):
        (directory / name).unlink(missing_ok=True)
    log = directory / f"launch-{failures}.log"
    with log.open("a") as stream:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__)),
            "--mode", "worker", "--run", str(run), "--gpu", str(gpu)], stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT)
    write_json(directory / "launch.json", {"pid": process.pid, "log": str(log), "failures": failures})
    return {"process": process, "active": None, "failures": failures, "ready": False}

def interrupted(signum, frame):
    """Convert terminal/service termination into normal finally-based cleanup."""
    raise SystemExit(128 + signum)

def controller_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    run, root = Path(args.run).resolve(), Path(args.root).resolve()
    lock = (run / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    selected = json.loads((run / "selected_examples.json").read_text())
    _, _, pairs = committed(run, cfg, selected)
    queue = deque(pending_tasks(cfg, selected, pairs))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run / "tensorboard"))
    workers, states = {}, {}
    success = False
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    write_json(run / "status.json", {"state": "running", "controller_pid": __import__("os").getpid()})
    try:
        for gpu in cfg["gpus"]:
            workers[gpu] = spawn_worker(run, gpu, 0)
        last_log = 0.0
        while True:
            for gpu, item in workers.items():
                process = item["process"]
                directory = run / "workers" / f"gpu{gpu}"
                if process.poll() is not None:
                    if item["active"] is not None:
                        queue.appendleft(item["active"])
                    failures = item["failures"] + 1
                    states[str(gpu)] = {"state": "failed", "failures": failures, "exit_code": process.returncode}
                    if failures >= cfg["max_worker_failures"]:
                        write_json(run / "failure.json", {"gpu": gpu, "failures": failures,
                            "exit_code": process.returncode, "log": str(directory / f"launch-{item['failures']}.log")})
                        raise RuntimeError(f"GPU {gpu} worker failed {failures} times; see its log")
                    workers[gpu] = spawn_worker(run, gpu, failures)
                    continue
                state_file = directory / "state.json"
                state = json.loads(state_file.read_text()) if state_file.exists() else {"state": "preflight"}
                states[str(gpu)] = state | {"pid": process.pid, "failures": item["failures"]}
                if state["state"] == "ready":
                    item["ready"] = True
                if state["state"] == "done" and state["example_id"] == item["active"]:
                    _, _, pairs = committed(run, cfg, selected)
                    assert all((item["active"], a) in pairs for a in range(cfg["attempts"])), "Worker reported incomplete task"
                    item["active"] = None
            all_ready = all(item["ready"] for item in workers.values())
            if all_ready:
                for gpu, item in workers.items():
                    if item["active"] is None and queue:
                        item["active"] = queue.popleft()
                        write_json(run / "workers" / f"gpu{gpu}" / "job.json", item["active"])
                if not queue and all(item["active"] is None for item in workers.values()):
                    success = True
                    break
            if time.monotonic() - last_log >= cfg["progress_interval"]:
                stats, _, _, _ = progress(run, cfg, selected, writer, states)
                print(json.dumps({"completed_attempts": stats["completed_attempts"],
                    "completed_problems": stats["completed_problems"], "fraction": stats["fraction"],
                    "eta_seconds": stats["eta_seconds"]}), flush=True)
                last_log = time.monotonic()
            time.sleep(0.5)
    finally:
        stop_workers(workers)
        failure = sys.exc_info()[0]
        state = "complete" if success else "interrupted" if failure in (SystemExit, KeyboardInterrupt) else "failed"
        write_json(run / "status.json", {"state": state, "complete": success})
        # Archive even if verification/export detects corrupted committed data.
        try:
            stats = export_results(run, cfg, selected, writer)
            assert not success or stats["complete"]
        finally:
            writer.close()
            target = archive_run(run, root)
            print(json.dumps({"archive": str(target), "status": state}), flush=True)
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

if __name__ == "__main__":
    modes = argparse.ArgumentParser(add_help=False)
    modes.add_argument("--mode", choices=("prepare", "worker", "run"), default="run")
    mode, remaining = modes.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    {"prepare": prepare_main, "worker": worker_main, "run": controller_main}[mode.mode]()
