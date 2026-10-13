# Frozen pretrained BD3LM / GSM8K on four A100s

This benchmark loads `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1` directly.
It does not train or load TraFL, adapters, a reference model or a log-Z head.
The pinned native `BD3LMSampler` generates one answer per call. Four independent
workers own distinct tasks and take the next task from a dynamic queue.

The working code is `run.sh`, `run.py`, `artifacts.py` and `config.yaml`.
`run.py` contains preparation, inference and the four-process controller.

## One-command launch

```bash
git clone --branch pretrain-a100 --single-branch \
  https://github.com/SDogya/automatic-eureka.git
cd automatic-eureka
bash pretrain_a100/run.sh --attempts 16 --gpus 0,1,2,3 \
  --output "$HOME/bd3lm_benchmark"
```

Requires Linux x86_64, Bash, git, curl, tar, flock, four accessible full A100
devices, internet and an NVIDIA driver compatible with the CUDA 12.6 wheel.
The script installs uv if needed, Python 3.12 and a private virtual environment
inside the output directory. It never uses sudo or modifies system Python.
Installation can download several GB of Torch/CUDA libraries.

The command prints the launcher PID, run directory and log path and returns.
Installation and inference continue under nohup after disconnecting SSH. Check
the log because starting the launcher does not mean preflight has passed.
For visible synchronous execution add `--foreground`.

```bash
# Short target-machine check: still requires four A100s.
bash pretrain_a100/run.sh --smoke --output "$HOME/bd3lm_smoke"
# Custom generation settings: copy/edit the flat config, then pass --config.
bash pretrain_a100/run.sh --config my-config.yaml --attempts 8
# Resume from a prepared run; saved source and settings are used unchanged.
bash pretrain_a100/run.sh --resume "$HOME/bd3lm_benchmark/runs/RUN_ID"
```

A failure during installation/preparation has no manifest yet: start a new run
after fixing its cause. Shared downloaded files and the environment are reused.
The script blocks simultaneous launches using the same output directory.

## Protocol and precision

All 1319 original `gsm8k/main/test` rows from Kaggle dataset version 3 are used.
Model revision: `60f77bba67ba94231c16f5fb29bc66c500b038f1`.
Native dllm revision: `ca176752fbceec49c6b4777a2c18ae88e4eb10ed`.
Torch: `2.8.0+cu126`; Transformers: `4.57.0`; all runtime dependencies are
pinned with SHA256 hashes in `requirements.lock`. The dllm repository is
installed editable at the pinned commit without its optional DeepSpeed/CUDA
training extensions. Its namespace imports still require the locked evaluation
and trainer support libraries; they do not participate in generation.

Default: 16 independent attempts, temperature 0.6, 256 new tokens, 256 steps,
32-token blocks, low-confidence remasking, no CFG, no logit shift, thinking
disabled and the existing boxed math prompt. Exact numeric reward uses the last
balanced `\\boxed{...}` and the gold answer after `####`. Missing or invalid
boxed answers receive zero; raw output and parsing decisions are retained.

This stochastic boxed protocol differs from the model card's greedy
`gsm8k_cot` evaluation, so its accuracy is not directly comparable.

A100 has native BF16 support. Model parameters/forward use BF16 with autocast;
native Gumbel noise remains FP64, observed confidence is stored as FP32 and
statistics use FP64. Math SDPA, deterministic algorithms, disabled TF32 and
cuDNN benchmarking, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are fixed. No compile,
quantization, batching or replacement sampler is introduced. Single-answer
sampling prioritizes the original sampler and can underutilize an A100.

Every worker checks CUDA 12.6, compute capability 8.0, hardware BF16 support,
finite real-model forward logits, repeated same-seed native output and exact
trace reconstruction before tasks are assigned. Failure stops the run without
silently falling back to another dtype.

## Seeds and restart

For original zero-based task ID `i` and zero-based attempt index `a`:

```python
seed = seed_start + (i << 32) + a
```

`seed_start` defaults to 1 and can be set through `--seed-start` or config.
Each task-attempt pair has a distinct seed; requesting more attempts preserves
existing seeds. Neither time, scheduling nor GPU index enters this formula.
Python, NumPy and Torch RNGs are reset before every sample. NumPy receives
`seed % 2**32`; it is not used for native GPU sampling.

Each answer/trace pair is published with a final SHA256 completion marker.
An unfinished attempt is regenerated; committed attempts are verified and
skipped. A crashed worker's task returns to the queue. Three failures on one
GPU stop automatic recovery and preserve the run with incomplete status.
Locks reject a second controller or launcher for the same active run.

Resume verifies configuration, original data, model weights, native sources,
benchmark sources and installed versions. Reproducibility is conditional on
the same hardware/software/config; bitwise agreement across different platforms
or releases is not promised. No old seven-GPU benchmark results are imported.

## Progress and data

```bash
tail -f "$HOME/bd3lm_benchmark/runs/RUN_ID/logs/launcher.log"
cat "$HOME/bd3lm_benchmark/runs/RUN_ID/status.json"
cat "$HOME/bd3lm_benchmark/runs/RUN_ID/progress.json"
"$HOME/bd3lm_benchmark/.venv/bin/tensorboard" \
  --logdir "$HOME/bd3lm_benchmark/runs" --host 127.0.0.1 --port 6006
```

Worker logs and startup checks are under `workers/gpuN/`. `progress.json`
records completed tasks/attempts, per-GPU throughput and a generation-based ETA.
The estimate excludes setup and serialization, and answer lengths vary.
TensorBoard records Pass@k on completed tasks, parsing/special-token diagnostics,
lengths, reveal confidence, forwards, GPU memory, throughput and final CI bounds.

`attempts/` retains every response, raw/trimmed token sequence, prompt, gold
answer, seed, assigned device, parsing result, timing, memory and compressed
native history deltas. The `.npz` files reconstruct each history and include
revealed-token confidence and transfer quotas; they do not store vocabulary
logits. `source/` snapshots the executable benchmark and native Python sources.
Manifests record model/data/source hashes, versions, hardware and launch commit.

At full coverage `answers.jsonl`, `metrics.json`, `bootstrap.npz` and
`complete.json` are written. Pass@k uses the combinatorial estimator for values
from 1/2/4/5/8/16 that do not exceed the attempt count. Confidence intervals use
10,000 task-level percentile bootstrap draws shared across k. They are individual
95% intervals conditional on collected answers; degenerate bounds are flagged.
Incomplete runs export saved answers and partial metrics without final CIs.
Smoke runs are labeled explicitly and contain only two tasks.

## Archives and stopping

Results are archived automatically to `archives/RUN_ID.tar.gz` with an external
SHA256 file. On controller/preparation failure the suffix is `-incomplete`.
Archives contain answers, traces, TensorBoard events, logs, source snapshots,
effective config, manifests and versions. Weights, dataset files, the Python
environment and credentials live outside the run and are excluded.

To stop cleanly after the controller has started:

```bash
# Read controller_pid from status.json, then:
kill -TERM CONTROLLER_PID
```

The controller terminates its workers, exports committed answers and archives
the partial result. SIGKILL, power loss and a full disk cannot execute cleanup;
committed attempts remain resumable. On successful completion download the
archive and its `.sha256` file and run `sha256sum -c RUN_ID.tar.gz.sha256`.

The complete four-A100 run has not yet been tested on the inaccessible target
machine. The launcher contains the hardware/model checks to run there first.
