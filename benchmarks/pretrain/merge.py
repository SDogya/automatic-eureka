"""Combine disjoint GSM8K GPU shards with the first completed original draws.

Run from the remote project root::

    ../.venv/bin/python benchmarks/pretrain/merge.py --watch

All source runs remain separate. Completed chunk markers are hash checked once;
the first complete 16-draw group for each global problem ID is claimed. Two
original full runs race over IDs 0..63; shard offsets cover IDs 64..1318.
"""

import argparse
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from torch.utils.tensorboard import SummaryWriter

from artifacts import aggregate, bootstrap, sha256, write_json, write_npz


def source_run(root):
    """Return the one live run directory below a configured worker root."""
    runs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists())
    return runs[-1] if runs else None


def source_definitions(project, shards):
    """Describe exact, nonoverlapping global coverage and prefix precedence."""
    definitions = [
        {"name": "gpu1", "gpu": 1, "start": 0, "stop": 64,
         "root": project / "runs/pretrain_gsm8k"},
        {"name": "gpu6", "gpu": 6, "start": 0, "stop": 64,
         "root": project / "runs/pretrain_gsm8k_gpu6"},
    ]
    for shard in shards["shards"]:
        gpu = shard["gpu"]
        definitions.append({
            "name": f"gpu{gpu}", "gpu": gpu, "start": shard["start"],
            "stop": shard["stop"],
            "root": project / f"runs/pretrain_gsm8k_shards/gpu{gpu}",
            "dataset_sha256": shard["sha256"],
        })
    return definitions


def verify_source(source, canonical, required):
    """Reject any worker with changed weights, generation or reward protocol."""
    manifest = json.loads((source["run"] / "manifest.json").read_text())
    config, identity = manifest["config"], manifest["identity"]
    for key in required:
        assert config[key] == canonical["config"][key], f"{source['name']} changes {key}"
    assert identity["model_files"] == canonical["identity"]["model_files"]
    assert identity["native_sources"] == canonical["identity"]["native_sources"]
    assert identity["benchmark_sources"] == canonical["identity"]["benchmark_sources"]
    assert identity["gpu_uuid"] != ""
    if "dataset_sha256" in source:
        assert identity["dataset_sha256"] == source["dataset_sha256"]
    else:
        assert identity["dataset_sha256"] == canonical["identity"]["dataset_sha256"]
    assert config["expected_problems"] == source["stop"] - source["start"] or source["start"] == 0
    if (source["run"] / "model_load.json").exists():
        model = json.loads((source["run"] / "model_load.json").read_text())
        assert model["peft"] is False and model["trainable_parameters"] == 0
    source["verified"] = True


def new_chunks(source, seen, expected_rows):
    """Read newly committed chunks once and restore original dataset IDs."""
    groups = {}
    run = source["run"]
    for marker in sorted((run / "chunks").glob("*.complete.json")):
        key = (source["name"], marker.name)
        if key in seen:
            continue
        entry = json.loads(marker.read_text())
        global_id = source["start"] + entry["example_id"]
        if not source["start"] and global_id >= source["stop"]:
            seen.add(key)
            continue
        assert source["start"] <= global_id < source["stop"]
        for name, digest in entry["sha256"].items():
            assert sha256(run / "chunks" / name) == digest, f"Corrupt chunk {marker}"
        payload = json.loads((run / "chunks" / entry["answers"]).read_text())
        assert len(payload["samples"]) == entry["count"]
        records = []
        for original in payload["samples"]:
            assert original["question"] == expected_rows[global_id]["question"]
            assert original["gold_answer"] == expected_rows[global_id]["answer"]
            assert original["example_id"] == entry["example_id"]
            record = original | {
                "example_id": global_id,
                "local_example_id": original["example_id"],
                "source_gpu": source["gpu"],
                "source_run": str(run.resolve()),
                "trace_path": str((run / "chunks" / entry["trace"]).resolve()),
            }
            records.append(record)
        groups.setdefault(global_id, []).extend(records)
        seen.add(key)
    return groups


def main():
    parser = argparse.ArgumentParser(description="Merge pure pretrained GPU shards")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    shards = json.loads((project / "datasets/gsm8k/shards/shard_manifest.json").read_text())
    original = project / "datasets/gsm8k/gsm8k/main/test-00000-of-00001.parquet"
    assert sha256(original) == shards["source_sha256"]
    expected_rows = pq.read_table(original).to_pylist()
    assert len(expected_rows) == 1319
    sources = source_definitions(project, shards)
    out = project / "runs/pretrain_gsm8k_merged"
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out / "tensorboard"))
    seen, pending, claimed = set(), {}, {}
    required = (
        "model_id", "model_revision", "samples", "dtype", "attention_implementation",
        "max_new_tokens", "steps", "steps_per_block", "block_size", "temperature",
        "remasking", "stochastic_transfer", "cfg_scale", "cfg_keep_tokens",
        "right_shift_logits", "prompt_template", "enable_thinking", "reward_mode",
        "bootstrap_repetitions", "bootstrap_seed", "confidence_level", "pass_k",
    )
    canonical = json.loads((source_run(sources[0]["root"]) / "manifest.json").read_text())
    write_json(out / "manifest.json", {
        "purpose": "union of 1319 original GSM8K problems; one complete 16-draw group per ID",
        "dataset_sha256": shards["source_sha256"], "shard_manifest": shards,
        "source_priority": [s["name"] for s in sources],
        "model_revision": canonical["config"]["model_revision"],
        "protocol": "stochastic T=0.6 boxed prompt; distinct from official greedy gsm8k_cot",
        "bootstrap": "10000 task-level percentile draws, shared across k, conditional on collected generations",
    })
    last_count = -1
    while True:
        for source in sources:
            source["run"] = source_run(source["root"])
            if source["run"] is None:
                continue
            if not source.get("verified"):
                verify_source(source, canonical, required)
            for global_id, records in new_chunks(source, seen, expected_rows).items():
                pending.setdefault((source["name"], global_id), []).extend(records)
        for source in sources:
            for global_id in range(source["start"], source["stop"]):
                records = pending.get((source["name"], global_id), [])
                if global_id not in claimed and len(records) == 16:
                    assert sorted(r["sample_id"] for r in records) == list(range(16))
                    claimed[global_id] = records
        if len(claimed) != last_count:
            records = [record for key in sorted(claimed) for record in claimed[key]]
            stats, _, _ = aggregate(records, 16, canonical["config"]["pass_k"])
            stats.update({
                "complete": len(claimed) == len(expected_rows),
                "coverage_fraction": len(claimed) / len(expected_rows),
                "missing_problem_ids": [i for i in range(len(expected_rows)) if i not in claimed],
                "source_runs": {s["name"]: str(s["run"]) if s["run"] else None for s in sources},
                "source_problem_counts": {s["name"]: sum(claimed[i][0]["source_gpu"] == s["gpu"] for i in claimed) for s in sources},
            })
            write_json(out / "progress.json", stats)
            if claimed:
                writer.add_scalar("parallel/covered_problems", len(claimed), len(claimed))
                writer.add_scalar("parallel/coverage_fraction", stats["coverage_fraction"], len(claimed))
                for k, value in stats["pass_at_k"].items():
                    writer.add_scalar(f"parallel/pass@{k}", value, len(claimed))
                writer.flush()
            print(json.dumps({"covered": len(claimed), "missing": len(expected_rows) - len(claimed),
                              "sources": stats["source_problem_counts"]}), flush=True)
            last_count = len(claimed)
        if len(claimed) == len(expected_rows):
            records = [record for key in sorted(claimed) for record in claimed[key]]
            stats, scores, ids = aggregate(records, 16, canonical["config"]["pass_k"])
            draws, bounds = bootstrap(scores, canonical["config"]["bootstrap_repetitions"],
                                      canonical["config"]["bootstrap_seed"], canonical["config"]["confidence_level"])
            write_npz(out / "bootstrap.npz", distributions=draws, problem_ids=ids,
                      task_scores=scores, pass_k=canonical["config"]["pass_k"])
            stats["complete"] = True
            stats["confidence_intervals"] = {
                str(k): {"lower": float(bounds[0, i]), "upper": float(bounds[1, i]),
                         "degenerate": bool(bounds[0, i] == bounds[1, i])}
                for i, k in enumerate(canonical["config"]["pass_k"])
            }
            for k, interval in stats["confidence_intervals"].items():
                for bound in ("lower", "upper"):
                    writer.add_scalar(f"ci/pass@{k}/{bound}", interval[bound], len(claimed))
            with (out / "answers.jsonl.tmp").open("w") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            (out / "answers.jsonl.tmp").replace(out / "answers.jsonl")
            write_json(out / "metrics.json", stats)
            write_json(out / "complete.json", {
                "problems": len(claimed), "samples": len(records),
                "metrics_sha256": sha256(out / "metrics.json"),
                "answers_sha256": sha256(out / "answers.jsonl"),
            })
            writer.close()
            print(json.dumps({"complete": str(out), "pass_at_k": stats["pass_at_k"]}), flush=True)
            return
        if not args.watch:
            writer.close()
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
