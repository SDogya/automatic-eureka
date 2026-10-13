# Pure pretrained BD3LM / GSM8K

Independent inference entrypoint; it imports no `trafl` modules and loads no
adapters, reference model, log-Z head or optimizer. Uses the installed native
`dllm.core.samplers.bd3lm.BD3LMSampler` unchanged, with a read-only trace observer.

From the remote project root:

```bash
../.venv/bin/python benchmarks/pretrain/run.py --config benchmarks/pretrain/config.yaml
../.venv/bin/python benchmarks/pretrain/run.py --resume runs/pretrain_gsm8k/RUN
../.venv/bin/tensorboard --logdir runs/pretrain_gsm8k --host 127.0.0.1 --port 6006
```

All 1319 main/test problems, 16 stochastic answers each, 256-token limit. The
boxed prompt and temperature 0.6 deliberately differ from the model's official
greedy `gsm8k_cot` protocol. Its reported accuracy is not directly comparable.

`selected_examples.json` stores full questions, gold answers and prompts.
`chunks/*.json` stores every raw/trimmed answer, IDs, parsing decision, seed and
performance. `*.npz` stores native history deltas and revealed-token confidence.
Initial IDs plus each `history_lengths` extension filled with MASK plus the
step-indexed `delta` reproduce native histories. `reveal_index` columns are
history step, sample row, absolute padded position. Confidence is native-dtype
softmax at the revealed token; logs use an FP32 tiny clamp. Transfer quotas are
scheduled quotas, not necessarily the number of successful non-MASK reveals.

Only chunks with hash-verified `*.complete.json` markers count as completed.
Resume enforces identical config, GPU, model/data hashes, dependency versions and
native/benchmark sources. The unfinished chunk is regenerated from its stable
seed. Floating-point results are not promised bit-identical across GPU/software
changes. Microbatch changes require a new run, because they change RNG grouping
and native batch EOS stopping. An OOM leaves prior chunks intact.

Final `metrics.json`, `answers.jsonl`, `bootstrap.npz` and `complete.json` appear
after all draws. Pass@k uses the combinatorial estimator. The 95% intervals use
10,000 problem-level bootstrap draws, shared across k; these are individual,
conditional-on-collected-draws intervals. A degenerate interval is flagged and
does not establish certainty. Raw per-problem data support later paired analysis.

TensorBoard records completed-problem Pass@k, parsing/special-token diagnostics,
lengths, reveal confidence, throughput, model calls, GPU memory and final CI
bounds. Partial progress is explicitly marked in `progress.json`. There is no
training loss or learning rate for a frozen pretrained benchmark.
