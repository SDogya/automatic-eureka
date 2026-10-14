# BD3LM TraFL before sampler/performance changes

This commit preserves the existing implementation before optimization.

- Policy/reference: pinned Qwen3-0.6B BD3LM; LoRA rank/alpha 128/128.
- TraFL: G=5, K=4, group reward centering, average scores before squared
  residual, frozen reference, detached prompt features, one optimizer update.
- Rollouts: nucleus top-p .91, confidence threshold .9, prompt-aligned blocks,
  full-canvas recomputation without KV cache, no early EOS block stopping.
- Scores: four sequential corruption forwards per completion, no KV cache.
- Terminal response: trim after first EOS; EOS stays visible and is excluded
  from score masks. No post-EOS context is added to the score.
- Compilation: only masked-token log-probability reduction.

The optimization must keep the objective and score normalization. Cache changes
must preserve the original block grid, including an incomplete prompt block.
Before/after comparisons must use identical weights, inputs and decoder settings.
The separate fast pretrained benchmark has no nucleus cutoff and is not a
matched baseline for the updated training decoder.
