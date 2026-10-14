"""Eq. (4) masked-token scores; здесь живёт kernel, заменяемый torch.compile."""

import torch

from .attention import staircase_attention


def masked_token_mean(logits, targets):
    r"""Усреднить log probabilities исходных target tokens в FP32.
    
    Args:
        logits: Tensor [M, V] для выбранных masked positions, M>0.
        targets: Long tensor [M] с оригинальными, ещё не corrupted token IDs.
    
    Returns:
        FP32 scalar mean(log_softmax(logits)[position, target]). Denominator
        равен числу masked positions M, а не полной длине completion.
    
    Это численный kernel Eq. (4), который runner заменяет compiled-версией именно
    в этом модуле. Cast к float сохраняет gradients к исходным logits."""
    return logits.float().log_softmax(dim=-1).gather(-1, targets[..., None]).mean()



def masked_score(model, prompt, completion, masks, mask_id, dtype, empty_score, block_size):
    r"""Оценить Eq. (4): mean по masked positions, затем mean по K masks.
    
    Args:
        model: Policy/reference с Qwen-compatible forward и logits_to_keep.
        prompt: Unpadded long tensor [P] на device модели; не маскируется.
        completion: Long tensor [L] на том же device, обрезанный после первого EOS.
        masks: CPU bool tensor [K, L]; caller передаёт одинаковые masks двум моделям.
        mask_id: Token ID mask конкретного tokenizer.
        dtype: Autocast dtype neural forward; log-softmax/reductions остаются FP32.
        empty_score: Scalar convention для mask без выбранных позиций.
        block_size: BD3LM staircase block size из train/eval YAML settings.
    
    Returns:
        Scalar surrogate score с gradients, если caller не отключил autograd.
        Каждая corruption создаётся из исходного prompt+completion; target IDs
        берутся из исходной completion, logits вычисляются только для masked positions.
    
    Reference freezing/no_grad задаются caller. EOS eligibility определяется masks.
    Score является surrogate, а не точным likelihood используемого decoder.
    Во всех corruptions используется одна block-causal geometry полного
    prompt+completion; target tokens и Eq. (4) reductions не изменены."""
    attention_mask, position_ids = staircase_attention(
        prompt.numel() + completion.numel(), block_size, prompt.device
    )
    scores = []
    for mask in masks:
        positions = mask.nonzero().flatten().to(prompt.device, non_blocking=True)
        if positions.numel() == 0:
            scores.append(torch.full((), empty_score, device=prompt.device))
            continue
        tokens = torch.cat((prompt, completion)).clone()
        tokens[prompt.numel() + positions] = mask_id
        with torch.autocast(device_type=prompt.device.type, dtype=dtype):
            logits = model(
                tokens[None], attention_mask=attention_mask, position_ids=position_ids,
                use_cache=False, logits_to_keep=prompt.numel() + positions
            ).logits[0]
        scores.append(masked_token_mean(logits, completion[positions]))
    return torch.stack(scores).mean()


def batched_token_scores(logits, targets, selected, empty_score):
    """FP32 per-corruption mean; unselected/padded positions have zero weight."""
    logp = logits.float().log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    counts = selected.sum(-1)
    means = (logp * selected).sum(-1) / counts.clamp_min(1)
    return torch.where(counts > 0, means, torch.full_like(means, empty_score))


def masked_scores(model, prompt, completions, masks, mask_id, dtype, empty_score,
                  block_size, pad_id, microbatch):
    """Batch C completions × K corruptions without changing Eq.(4).

Right padding preserves the original block grid and RoPE positions. Each row's
true extent is used in both query and key masks. Only the union of corrupted
positions reaches lm_head; loss weights exclude all unselected positions.
Scores are averaged over K before the caller squares its residual. Prefix KV
from generation is never reused in this differentiable masked-context forward.
"""
    device = prompt.device
    c, k = len(completions), masks[0].shape[0]
    longest = max(x.numel() for x in completions)
    mask_cpu = torch.zeros(c * k, longest, dtype=torch.bool)
    tokens = torch.full((c * k, prompt.numel() + longest), pad_id, device=device, dtype=torch.long)
    tokens[:, :prompt.numel()] = prompt
    lengths = []
    for i, (completion, corruptions) in enumerate(zip(completions, masks)):
        length = completion.numel()
        tokens[i*k:(i+1)*k, prompt.numel():prompt.numel()+length] = completion
        mask_cpu[i*k:(i+1)*k, :length] = corruptions
        lengths.extend([prompt.numel() + length] * k)
    positions = mask_cpu.any(0).nonzero().flatten().to(device)
    if not positions.numel():
        return torch.full((c,), empty_score, device=device)
    chosen = mask_cpu.to(device, non_blocking=True)
    targets = tokens[:, prompt.numel()+positions].clone()
    tokens[:, prompt.numel():] = torch.where(chosen, mask_id, tokens[:, prompt.numel():])
    attention, ids = staircase_attention(tokens.shape[1], block_size, device)
    valid = torch.arange(tokens.shape[1], device=device)[None] < torch.tensor(lengths, device=device)[:, None]
    scores = []
    for start in range(0, c*k, microbatch):
        stop = min(start + microbatch, c*k)
        row_valid = valid[start:stop]
        row_mask = attention & row_valid[:, None, :, None] & row_valid[:, None, None, :]
        with torch.autocast(device.type, dtype=dtype):
            logits = model(tokens[start:stop], attention_mask=row_mask, position_ids=ids,
                           use_cache=False, logits_to_keep=prompt.numel()+positions).logits
        scores.append(batched_token_scores(logits, targets[start:stop], chosen[start:stop, positions], empty_score))
    return torch.cat(scores).reshape(c, k).mean(-1)

