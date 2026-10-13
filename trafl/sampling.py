"""Qwen block diffusion с nucleus proposals и threshold раскрытием Fast-dLLM."""

import torch


def nucleus_tokens(logits, temperature, top_p, generator):
    r"""Предложить токены из temperature-scaled nucleus distribution.
    
    Args:
        logits: Tensor [..., V] с vocabulary на последней оси.
        temperature: Перед softmax logits делятся на это значение; 0 выбирает argmax.
        top_p: Nucleus threshold. Токен, пересекающий порог cumulative mass,
            сохраняется; удаляются токены, перед которыми mass уже >= top_p.
        generator: Torch generator на device logits для categorical multinomial.
    
    Returns:
        Long tensor [...] sampled token IDs. Ветка temperature=0 не потребляет RNG.
    
    Logits/softmax вычисляются в FP32. Удалённая mass не участвует в multinomial;
    оставшиеся weights нормализуются самим multinomial."""
    logits = logits.float()
    if temperature == 0:
        return logits.argmax(dim=-1)
    probs = (logits / temperature).softmax(dim=-1)
    sorted_probs, order = probs.sort(dim=-1, descending=True)
    remove = sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0)
    draw = torch.multinomial(
        sorted_probs.reshape(-1, sorted_probs.shape[-1]), 1, generator=generator
    ).reshape(*logits.shape[:-1], 1)
    return order.gather(-1, draw).squeeze(-1)



@torch.no_grad()
def generate(
    model,
    prompt,
    count,
    length,
    block_size,
    temperature,
    top_p,
    threshold,
    mask_id,
    suppress_mask,
    generator,
    dtype,
):
    r"""Сгенерировать group completions через no-cache block diffusion.
    
    Args:
        model: Qwen-compatible denoiser с logits_to_keep; caller выбирает eval mode.
        prompt: Unpadded long tensor [P] на device модели.
        count: Число независимых draws в общей batched canvas.
        length: Число позиций completion, изначально заполненных mask_id.
        block_size: Размер последовательно обрабатываемых blocks (paper evaluation: 32).
        temperature, top_p: Параметры nucleus proposals.
        threshold: Раскрывать active positions с confidence >= этому значению.
        mask_id: Model-specific ID mask.
        suppress_mask: Убирать mask из proposal logits; confidence до suppression.
        generator: Выделенный device generator stochastic proposals.
        dtype: Autocast dtype forward; proposals/confidence используют FP32.
    
    Returns:
        Long tensor [count, length] без prompt. EOS пока не обрезается.
    
    На каждом step сначала оцениваются raw probabilities, потом proposals.
    Confidence — raw probability предложенного токена; раскрывается threshold set
    и наиболее уверенная active position каждой строки. Cache выключен, EOS не
    завершает denoising досрочно; каждый block допускает максимум его длину steps.
    Это адаптация threshold-ветки Fast-dLLM v1 к Qwen с nucleus согласно TraFL;
    threshold, mask suppression и EOS conventions — OWN G10/G20."""
    canvas = torch.cat(
        (
            prompt[None].expand(count, -1),
            torch.full((count, length), mask_id, device=prompt.device),
        ),
        dim=1,
    )
    prefix = prompt.numel()
    for start in range(0, length, block_size):
        end = min(start + block_size, length)
        positions = torch.arange(prefix + start, prefix + end, device=prompt.device)
        for step in range(end - start):
            active = canvas[:, positions] == mask_id
            if not active.any():
                break  # Required control sync for confidence-aware parallel decoding.
            with torch.autocast(device_type=prompt.device.type, dtype=dtype):
                logits = model(
                    canvas, use_cache=False, logits_to_keep=positions
                ).logits.float()
            probabilities = logits.softmax(dim=-1)
            if suppress_mask:
                logits[..., mask_id] = -torch.inf
            proposed = nucleus_tokens(logits, temperature, top_p, generator)
            confidence = probabilities.gather(-1, proposed[..., None]).squeeze(-1)
            confidence = confidence.masked_fill(~active, -torch.inf)
            transfer = active & (confidence >= threshold)
            forced = torch.zeros_like(active).scatter_(
                1, confidence.argmax(dim=1)[:, None], True
            )
            transfer |= forced & active  # Highest-confidence token is always accepted.
            canvas[:, positions] = torch.where(transfer, proposed, canvas[:, positions])
    return canvas[:, prefix:]



def sample_group(model, prompt, count, length, prefix, cfg, tokenizer, rng, dtype):
    r"""Передать отдельные train/eval YAML settings в diffusion sampler.
    
    Args:
        model, prompt: Denoiser и unpadded prompt на device модели.
        count, length: Число group draws и длина completion.
        prefix: Строка train или eval, выбирающая соответствующие YAML keys.
        cfg: Flat config с block_size/temperature/top_p/confidence_threshold.
        tokenizer: Источник model-specific mask_token_id.
        rng: Device generator rollout/evaluation stream.
        dtype: Autocast dtype.
    
    Returns:
        Long tensor [count, length] после полного denoising, до EOS trimming.
    
    Wrapper не изменяет sampler и не смешивает training settings с evaluation."""
    return generate(
        model,
        prompt,
        count,
        length,
        cfg[f"{prefix}_block_size"],
        cfg[f"{prefix}_temperature"],
        cfg[f"{prefix}_top_p"],
        cfg[f"{prefix}_confidence_threshold"],
        tokenizer.mask_token_id,
        cfg["generation_suppress_mask"],
        rng,
        dtype,
    )

