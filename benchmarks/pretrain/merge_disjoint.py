"""Merge exactly one 16-draw pretrained group for every original GSM8K row.

The assignment in datasets/gsm8k/disjoint_v1/plan.json is a permutation of
all 1319 original IDs. Each of seven workers owns a unique subset and keeps its
native sampler traces separately. Run with --watch to finalize automatically.
"""

import argparse
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from torch.utils.tensorboard import SummaryWriter

from artifacts import aggregate, bootstrap, sha256, write_json, write_npz
from merge import source_run, verify_source


def main():
    parser = argparse.ArgumentParser(description="Merge seven disjoint pretrained GSM8K runs")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    plan = json.loads((project / "datasets/gsm8k/disjoint_v1/plan.json").read_text())
    original = Path(plan["source"])
    assert sha256(original) == plan["source_sha256"]
    expected = pq.read_table(original).to_pylist()
    assert len(expected) == 1319
    all_ids = [id for worker in plan["workers"] for id in worker["global_ids"]]
    assert len(all_ids) == len(set(all_ids)) == 1319 and sorted(all_ids) == list(range(1319))
    sources = [{
        "name": f"gpu{worker['gpu']}", "gpu": worker["gpu"],
        "ids": worker["global_ids"], "start": 0, "stop": worker["count"],
        "dataset_sha256": worker["sha256"],
        "root": project / f"runs/pretrain_disjoint/gpu{worker['gpu']}",
    } for worker in plan["workers"]]
    while source_run(sources[0]["root"]) is None:
        time.sleep(3)
    canonical = json.loads((source_run(sources[0]["root"]) / "manifest.json").read_text())
    required = (
        "model_id", "model_revision", "samples", "dtype", "attention_implementation",
        "max_new_tokens", "steps", "steps_per_block", "block_size", "temperature",
        "remasking", "stochastic_transfer", "cfg_scale", "cfg_keep_tokens",
        "right_shift_logits", "prompt_template", "enable_thinking", "reward_mode",
        "bootstrap_repetitions", "bootstrap_seed", "confidence_level", "pass_k",
    )
    output = project / "runs/pretrain_disjoint_merged"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", {
        "assignment": plan,
        "model_revision": canonical["config"]["model_revision"],
        "protocol": "pure pretrained BD3LM, boxed prompt, stochastic T=0.6; differs from official greedy gsm8k_cot",
        "selection": "all 1319 original IDs exactly once; all 16 draws for an ID from the same GPU run",
        "bootstrap": "10000 task-level percentile draws; individual 95% intervals conditional on collected generations",
    })
    writer = SummaryWriter(str(output / "tensorboard"))
    seen, pending, claimed = set(), {}, {}
    previous_count = -1
    while True:
        for source in sources:
            source["run"] = source_run(source["root"])
            if source["run"] is None:
                continue
            if not source.get("verified"):
                verify_source(source, canonical, required)
                assert len(source["ids"]) == source["stop"]
            for marker in sorted((source["run"] / "chunks").glob("*.complete.json")):
                marker_key = (source["name"], marker.name)
                if marker_key in seen:
                    continue
                entry = json.loads(marker.read_text())
                local_id = entry["example_id"]
                assert 0 <= local_id < len(source["ids"])
                global_id = source["ids"][local_id]
                for filename, digest in entry["sha256"].items():
                    assert sha256(source["run"] / "chunks" / filename) == digest
                payload = json.loads((source["run"] / "chunks" / entry["answers"]).read_text())
                assert len(payload["samples"]) == entry["count"]
                group = pending.setdefault(global_id, [])
                for saved in payload["samples"]:
                    assert saved["example_id"] == local_id
                    assert saved["question"] == expected[global_id]["question"]
                    assert saved["gold_answer"] == expected[global_id]["answer"]
                    group.append(saved | {
                        "example_id": global_id,
                        "local_example_id": local_id,
                        "source_gpu": source["gpu"],
                        "source_run": str(source["run"].resolve()),
                        "trace_path": str((source["run"] / "chunks" / entry["trace"]).resolve()),
                    })
                seen.add(marker_key)
                if len(group) == 16:
                    assert sorted(record["sample_id"] for record in group) == list(range(16))
                    claimed[global_id] = group
        if len(claimed) != previous_count:
            records = [record for id in sorted(claimed) for record in claimed[id]]
            stats, _, _ = aggregate(records, 16, canonical["config"]["pass_k"])
            stats.update({
                "complete": len(claimed) == 1319,
                "coverage_fraction": len(claimed) / 1319,
                "missing_problem_ids": [id for id in range(1319) if id not in claimed],
                "source_problem_counts": {source["name"]: sum(id in claimed for id in source["ids"]) for source in sources},
                "source_runs": {source["name"]: str(source["run"]) if source["run"] else None for source in sources},
            })
            write_json(output / "progress.json", stats)
            if claimed:
                writer.add_scalar("parallel/covered_problems", len(claimed), len(claimed))
                writer.add_scalar("parallel/coverage_fraction", len(claimed) / 1319, len(claimed))
                for k, value in stats["pass_at_k"].items():
                    writer.add_scalar(f"parallel/pass@{k}", value, len(claimed))
                writer.flush()
            print(json.dumps({"covered": len(claimed), "by_gpu": stats["source_problem_counts"]}), flush=True)
            previous_count = len(claimed)
        if len(claimed) == 1319:
            records = [record for id in sorted(claimed) for record in claimed[id]]
            stats, scores, ids = aggregate(records, 16, canonical["config"]["pass_k"])
            draws, bounds = bootstrap(scores, canonical["config"]["bootstrap_repetitions"],
                                      canonical["config"]["bootstrap_seed"], canonical["config"]["confidence_level"])
            write_npz(output / "bootstrap.npz", distributions=draws, problem_ids=ids,
                      task_scores=scores, pass_k=canonical["config"]["pass_k"])
            stats["complete"] = True
            stats["confidence_intervals"] = {
                str(k): {"lower": float(bounds[0, i]), "upper": float(bounds[1, i]),
                         "degenerate": bool(bounds[0, i] == bounds[1, i])}
                for i, k in enumerate(canonical["config"]["pass_k"])
            }
            for k, interval in stats["confidence_intervals"].items():
                for bound in ("lower", "upper"):
                    writer.add_scalar(f"ci/pass@{k}/{bound}", interval[bound], 1319)
            with (output / "answers.jsonl.tmp").open("w") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            (output / "answers.jsonl.tmp").replace(output / "answers.jsonl")
            write_json(output / "metrics.json", stats)
            write_json(output / "complete.json", {
                "problems": 1319, "samples": len(records),
                "metrics_sha256": sha256(output / "metrics.json"),
                "answers_sha256": sha256(output / "answers.jsonl"),
            })
            writer.close()
            print(json.dumps({"complete": str(output), "pass_at_k": stats["pass_at_k"]}), flush=True)
            return
        if not args.watch:
            writer.close()
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
