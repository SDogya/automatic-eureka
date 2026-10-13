"""CPU checks for seed identity, dynamic scheduling, restart and result archives.

Run with the installed benchmark environment: python pretrain_a100/tests.py.
Controller integration uses subprocess workers that emit synthetic artifacts;
it exercises real scheduling and failure recovery without occupying any GPU.
Native forward/repeatability checks run separately in worker preflight on A100.
"""

import json
import fcntl
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from artifacts import parse_answer, reconstruct_trace, sha256, write_json, write_npz
from artifacts import archive_run, attempt_seed, committed, export_results, pending_tasks


def fake_commit(run, cfg, selected, task, attempt, gpu):
    """Publish one complete synthetic sample using the production artifact format."""
    run = Path(run)
    name = f"problem-{task:04d}-attempt-{attempt:04d}"
    answers, trace = name + ".json", name + ".npz"
    text = "\\boxed{2}" if attempt % 2 == 0 else "\\boxed{3}"
    sample = {"example_id": task, "sample_id": attempt, "seed": attempt_seed(cfg["seed_start"], task, attempt),
        "question": selected[task]["question"], "gold_answer": selected[task]["answer"],
        "answer_text": text, "answer_length": 2, "has_eos": True, "has_mask": False, "has_pad": False,
        "gpu": gpu, "trace": "attempts/" + trace} | parse_answer(text, selected[task]["answer"])
    perf = {"gpu": gpu, "generation_seconds": 1.0, "model_forwards": 2, "generated_tokens": 2,
            "revealed_tokens": 2, "confidence_sum": 1.5, "log_confidence_sum": -0.6, "peak_allocated_bytes": 128}
    write_npz(run / "attempts" / trace, test=np.arange(2))
    write_json(run / "attempts" / answers, {"sample": sample, "performance": perf})
    write_json(run / "attempts" / (name + ".complete.json"), {"example_id": task, "sample_id": attempt,
        "answers": answers, "trace": trace,
        "sha256": {n: sha256(run / "attempts" / n) for n in (answers, trace)}})


FAKE_WORKER = '''
import argparse, json, os, time
from pathlib import Path
import yaml
from artifacts import write_json
from artifacts import committed
from tests import fake_commit
p = argparse.ArgumentParser()
p.add_argument("--run"); p.add_argument("--gpu", type=int)
a = p.parse_args(); run = Path(a.run)
cfg = yaml.safe_load((run / "config.yaml").read_text())
selected = json.loads((run / "selected_examples.json").read_text())
d = run / "workers" / f"gpu{a.gpu}"
if os.environ.get("FAKE_ALWAYS_FAIL") and a.gpu == 0: raise SystemExit(17)
_, _, done = committed(run, cfg, selected)
write_json(d / "state.json", {"state": "ready"})
while True:
 job = d / "job.json"
 if not job.exists(): time.sleep(.02); continue
 task = json.loads(job.read_text()); job.unlink()
 if task is None: break
 write_json(d / "state.json", {"state": "running", "example_id": task})
 for attempt in range(cfg["attempts"]):
  if (task, attempt) not in done:
   fake_commit(run, cfg, selected, task, attempt, a.gpu)
   done.add((task, attempt))
   if os.environ.get("FAKE_CRASH_ONCE") and a.gpu == 0 and not (run / "crashed-once.json").exists():
    write_json(run / "crashed-once.json", {"task": task}); raise SystemExit(17)
 time.sleep(.1)
 write_json(d / "state.json", {"state": "done", "example_id": task})
'''


class BookkeepingTests(unittest.TestCase):
    def test_unique_schedule_independent_seeds(self):
        seeds = {attempt_seed(1, i, a) for i in range(1319) for a in range(16)}
        self.assertEqual(len(seeds), 1319 * 16)
        self.assertEqual(attempt_seed(1, 0, 0), 1)
        # Increase total attempts or reassign GPUs without changing old seeds.
        for i in (0, 1, 1318):
            self.assertEqual([attempt_seed(1, i, a) for a in range(2)],
                             [attempt_seed(1, i, a) for a in range(16)][:2])
        cfg = {"assignment_seed": 42, "attempts": 2}
        self.assertEqual(set(pending_tasks(cfg, [None] * 4, {(0, 0), (0, 1), (2, 0)})), {1, 2, 3})

    def test_trace_extension(self):
        arrays = {"initial_ids": np.array([[10, 11]]), "history_lengths": np.array([2, 4, 4]),
                  "mask_id": np.array(99), "delta": np.array([[1, 0, 2, 20], [2, 0, 3, 21]])}
        np.testing.assert_array_equal(reconstruct_trace(arrays), [[10, 11, 20, 21]])

    def fixture(self, root, attempts):
        cfg = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
        cfg.update(attempts=attempts, pass_k=[k for k in cfg["pass_k"] if k <= attempts],
                   smoke=True, bootstrap_repetitions=64, progress_interval=.1)
        selected = [{"example_id": i, "question": f"Q{i}", "answer": "1+1=2\n#### 2"} for i in range(6)]
        run = root / "runs" / "test"
        for name in ("attempts", "workers", "logs", "tensorboard", "source/pretrain_a100"):
            (run / name).mkdir(parents=True, exist_ok=True)
        (root / "models").mkdir()
        (root / "models" / "not-in-archive.bin").write_text("weights")
        (root / "credentials.txt").write_text("not-a-real-token")
        (run / "config.yaml").write_text(yaml.safe_dump(cfg))
        write_json(run / "selected_examples.json", selected)
        write_json(run / "manifest.json", {"synthetic_test": True})
        return run, cfg, selected

    def test_hash_verification_and_uncommitted_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run, cfg, selected = self.fixture(root, 2)
            fake_commit(run, cfg, selected, 0, 0, 0)
            (run / "attempts" / "unfinished.json").write_text("{}")
            self.assertEqual(committed(run, cfg, selected)[2], {(0, 0)})
            p = run / "attempts" / "problem-0000-attempt-0000.json"
            p.write_text(p.read_text() + " ")
            with self.assertRaisesRegex(AssertionError, "Corrupt artifact"):
                committed(run, cfg, selected)

    def test_controller_recovery_exports_and_archive(self):
        for count in (1, 2, 16):
            with self.subTest(attempts=count), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                run, cfg, selected = self.fixture(root, count)
                code = run / "source/pretrain_a100"
                for name in ("run.py", "artifacts.py", "tests.py", "config.yaml"):
                    shutil.copyfile(Path(__file__).with_name(name), code / name)
                (code / "fake_worker.py").write_text(FAKE_WORKER)
                runtime = (code / "run.py").read_text().replace('str(Path(__file__))', 'str(Path(__file__).with_name("fake_worker.py"))').replace('"--mode", "worker", ', '')
                (code / "run.py").write_text(runtime)
                env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "FAKE_CRASH_ONCE": "1"}
                result = subprocess.run([sys.executable, str(code / "run.py"), "--root", str(root),
                                         "--run", str(run)], env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((run / "crashed-once.json").exists())
                records, _, pairs = committed(run, cfg, selected)
                self.assertEqual(len(records), len(pairs))
                self.assertEqual(len(pairs), len(selected) * count)
                metrics = json.loads((run / "metrics.json").read_text())
                self.assertEqual(set(metrics["pass_at_k"]), {str(k) for k in cfg["pass_k"]})
                self.assertAlmostEqual(metrics["pass_at_k"]["1"], (count + 1) // 2 / count)
                self.assertEqual(len((run / "answers.jsonl").read_text().splitlines()), len(pairs))
                self.assertTrue((run / "bootstrap.npz").exists())
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                events = EventAccumulator(str(run / "tensorboard")).Reload()
                self.assertIn("eval/pass@1", events.Tags()["scalars"])
                archive = root / "archives" / "test.tar.gz"
                with tarfile.open(archive) as tar:
                    names = tar.getnames()
                self.assertIn("test/metrics.json", names)
                self.assertFalse(any("not-in-archive" in n or "credentials" in n for n in names))
                self.assertEqual(sha256(archive), archive.with_name(archive.name + ".sha256").read_text().split()[0])
                # A second controller sees only committed tasks and adds no attempts.
                result = subprocess.run([sys.executable, str(code / "run.py"), "--root", str(root),
                                         "--run", str(run)], env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(committed(run, cfg, selected)[2], pairs)

                # Holding the real run lock rejects another controller before it
                # can overwrite status, export results or launch any worker.
                with (run / "controller.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    blocked = subprocess.run([sys.executable, str(code / "run.py"), "--root", str(root),
                        "--run", str(run)], env=env, capture_output=True, text=True, timeout=30)
                    self.assertNotEqual(blocked.returncode, 0)
                    self.assertIn("BlockingIOError", blocked.stderr)

    def test_three_worker_failures_preserve_incomplete_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run, cfg, selected = self.fixture(root, 2)
            code = run / "source/pretrain_a100"
            for name in ("run.py", "artifacts.py", "tests.py"):
                shutil.copyfile(Path(__file__).with_name(name), code / name)
            (code / "fake_worker.py").write_text(FAKE_WORKER)
            runtime = (code / "run.py").read_text().replace('str(Path(__file__))', 'str(Path(__file__).with_name("fake_worker.py"))').replace('"--mode", "worker", ', '')
            (code / "run.py").write_text(runtime)
            result = subprocess.run([sys.executable, str(code / "run.py"), "--root", str(root),
                "--run", str(run)], env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "FAKE_ALWAYS_FAIL": "1"},
                capture_output=True, text=True, timeout=90)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads((run / "failure.json").read_text())["failures"], 3)
            self.assertFalse((run / "complete.json").exists())
            self.assertTrue((root / "archives/test-incomplete.tar.gz").exists())
            self.assertFalse(json.loads((run / "metrics.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
