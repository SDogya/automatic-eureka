"""Cached BD3LM nucleus rollouts with the original prompt-aligned block grid.

Only completed blocks enter the prefix cache. The unfinished prompt suffix is
recomputed with the current completion block. Rollouts have no gradient; score
forwards use a separate, uncached path. The nucleus distribution, raw confidence
threshold and forced best position are retained, including the .91 crossing token.
"""

from collections import OrderedDict
from types import SimpleNamespace

import torch

from .attention import staircase_attention


class PrefixCache:
    """Read-only completed prefix; append current KV without modifying the cache."""

    def __init__(self, cache):
        self.layers = cache.layers
        self.length = cache.get_seq_length()

    def get_seq_length(self, layer_idx=0):
        return self.length

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        layer = self.layers[layer_idx]
        return torch.cat((layer.keys, keys), -2), torch.cat((layer.values, values), -2)


class CachedForward:
    """Replay model forward for a bounded set of rollout shapes.

LoRA weights are read from their live storage: in-place optimizer updates are
visible on the next replay. Cache/input buffers refresh at each block boundary.
Graphs are used only in eval/no_grad mode and are never used for scoring.
"""

    def __init__(self, model, limit):
        self.model, self.limit = model, limit
        self.entries = OrderedDict()

    def __call__(self, block, attention, positions, cache):
        prefix_length = cache.length if cache is not None else 0
        key = (block.shape[0], block.shape[1], prefix_length)
        if key not in self.entries:
            if len(self.entries) >= self.limit:
                self.entries.popitem(last=False)
            ids, mask, pos = block.clone(), attention.clone(), positions.clone()
            fixed = None
            if cache is not None:
                fixed = PrefixCache.__new__(PrefixCache)
                fixed.length = prefix_length
                fixed.layers = [SimpleNamespace(keys=l.keys.clone(), values=l.values.clone()) for l in cache.layers]
            def forward():
                return self.model(ids, attention_mask=mask, position_ids=pos,
                                  past_key_values=fixed, use_cache=False).logits
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = forward()
            self.entries[key] = ids, mask, pos, fixed, graph, output
            self.current_cache = cache
        self.entries.move_to_end(key)
        ids, mask, pos, fixed, graph, output = self.entries[key]
        ids.copy_(block)
        # Refresh completed-prefix buffers only when a new block/prompt starts.
        if getattr(self, "current_cache", None) is not cache:
            mask.copy_(attention)
            pos.copy_(positions)
            if fixed is not None:
                for target, source in zip(fixed.layers, cache.layers):
                    target.keys.copy_(source.keys)
                    target.values.copy_(source.values)
            self.current_cache = cache
        graph.replay()
        return output


def nucleus_step(logits, noise, block, slots, live, temperature, top_p, threshold, mask_id, suppress_mask):
    """Exact nucleus support followed by FP64 exponential-race categorical sampling.

Noise is unit exponential, drawn outside compilation with the saved rollout RNG.
Sampling p/E is categorical with probabilities proportional to p. Temperature
is already applied to p, before nucleus truncation. This changes RNG consumption
from the previous multinomial implementation; protocol versioning rejects old
full-resume states rather than silently changing their random stream.
"""
    logits = logits.float()
    raw = logits.softmax(-1)
    proposals = logits.masked_fill(slots.new_tensor(suppress_mask) &
                                   (torch.arange(logits.shape[-1], device=logits.device) == mask_id), -torch.inf)
    if temperature == 0:
        proposed = proposals.argmax(-1)
    else:
        probs, order = (proposals / temperature).softmax(-1).sort(-1, descending=True)
        probs = probs.masked_fill(probs.cumsum(-1) - probs >= top_p, 0)
        draw = (probs.double() / noise.clamp_min(torch.finfo(torch.float64).tiny)).argmax(-1)
        proposed = order.gather(-1, draw[..., None]).squeeze(-1)
    active = (block == mask_id) & slots[None] & live[:, None]
    confidence = raw.gather(-1, proposed[..., None]).squeeze(-1).masked_fill(~active, -torch.inf)
    forced = torch.zeros_like(active).scatter(1, confidence.argmax(-1, keepdim=True), True)
    transfer = active & ((confidence >= threshold) | forced)
    return torch.where(transfer, proposed, block)


class CachedSampler:
    """One persistent decoder per policy; cache is rebuilt for each rollout/block."""

    def __init__(self, model, cfg):
        self.model = model
        self.step = torch.compile(nucleus_step, fullgraph=True, dynamic=True,
                                  backend=cfg["compile_backend"]) if cfg["generation_compile"] else nucleus_step
        self.graph = CachedForward(model, cfg["generation_graph_limit"]) if cfg["generation_cuda_graph"] else None

    @torch.no_grad()
    def sample(self, prompt, count, length, block_size, temperature, top_p, threshold,
               mask_id, suppress_mask, rng, dtype, eos_id, stop_on_eos):
        """Finish an EOS block, then freeze that row; return the fixed-size canvas.

Unused positions after a stopped row stay MASK and are removed by EOS trimming
before scoring/reward. Current blocks retain the original absolute coordinates;
no left padding or changes to the score's attention geometry are introduced.
"""
        model = self.model
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        prefix = prompt.numel()
        canvas = torch.cat((prompt[None].expand(count, -1),
            torch.full((count, length), mask_id, device=prompt.device, dtype=torch.long)), 1)
        live = torch.ones(count, device=prompt.device, dtype=torch.bool)
        start = 0
        # Autocast weight caching could freeze an old BF16 copy of FP32 LoRA
        # inside a CUDA graph. Capture/replay must read current adapter weights.
        with torch.autocast(prompt.device.type, dtype=dtype, cache_enabled=False):
            while start < length:
                cache_end = (prefix + start) // block_size * block_size
                end = min(cache_end + block_size, prefix + length)
                mask, pos = staircase_attention(end, block_size, prompt.device)
                cache = None
                if cache_end:
                    cached = base.model(canvas[:, :cache_end], attention_mask=mask[:, :, :cache_end, :cache_end],
                        position_ids=pos[:, :cache_end], use_cache=True)
                    cache = PrefixCache(cached.past_key_values)
                    del cached
                block = canvas[:, cache_end:end].contiguous()
                attention, positions = mask[:, :, cache_end:end, :], pos[:, cache_end:end]
                slots = torch.arange(cache_end, end, device=prompt.device) >= prefix
                for _ in range(end - prefix - start):
                    logits = self.graph(block, attention, positions, cache) if self.graph is not None else model(
                        block, attention_mask=attention, position_ids=positions,
                        past_key_values=cache, use_cache=False).logits
                    noise = torch.empty_like(logits, dtype=torch.float64).exponential_(generator=rng)
                    block = self.step(logits, noise, block, slots, live, temperature,
                                      top_p, threshold, mask_id, suppress_mask)
                    # One group-wide control sync, not per row/per token/quota.
                    if not ((block == mask_id) & slots[None] & live[:, None]).any():
                        break
                canvas[:, cache_end:end] = block
                if stop_on_eos:
                    live &= ~((block == eos_id) & slots[None]).any(-1)
                    if not live.any():
                        break
                start = end - prefix
        return canvas[:, prefix:]
