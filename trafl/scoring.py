"""Eq. (4) masked-token scores; здесь живёт kernel, заменяемый torch.compile."""

import torch


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



def masked_score(model, prompt, completion, masks, mask_id, dtype, empty_score):
    r"""Оценить Eq. (4): mean по masked positions, затем mean по K masks.
    
    Args:
        model: Policy/reference с Qwen-compatible forward и logits_to_keep.
        prompt: Unpadded long tensor [P] на device модели; не маскируется.
        completion: Long tensor [L] на том же device, обрезанный после первого EOS.
        masks: CPU bool tensor [K, L]; caller передаёт одинаковые masks двум моделям.
        mask_id: Token ID mask конкретного tokenizer.
        dtype: Autocast dtype neural forward; log-softmax/reductions остаются FP32.
        empty_score: Scalar convention для mask без выбранных позиций.
    
    Returns:
        Scalar surrogate score с gradients, если caller не отключил autograd.
        Каждая corruption создаётся из исходного prompt+completion; target IDs
        берутся из исходной completion, logits вычисляются только для masked positions.
    
    Reference freezing/no_grad задаются caller. EOS eligibility определяется masks.
    Score является surrogate, а не точным likelihood используемого decoder."""
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
                tokens[None], use_cache=False, logits_to_keep=prompt.numel() + positions
            ).logits[0]
        scores.append(masked_token_mean(logits, completion[positions]))
    return torch.stack(scores).mean()

