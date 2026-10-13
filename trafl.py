"""TraFL equations and Qwen diffusion kernels. Tests: python -m pytest test_trafl.py."""

import math
import re
from fractions import Fraction

import torch
from torch import nn


def numeric_answer(text):
    text = re.sub(r"\\(?:dfrac|tfrac)", r"\\frac", text.strip())
    text = text.replace(",", "").replace("$", "").replace(" ", "")
    text = re.sub(r"\\(?:left|right)", "", text)
    fraction = re.fullmatch(r"(-?)\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
    if fraction:
        sign, numerator, denominator = fraction.groups()
        if not (re.fullmatch(r"[+-]?\d+(?:\.\d+)?", numerator)
                and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", denominator)):
            return None
        divisor = Fraction(denominator)
        return (-1 if sign else 1) * Fraction(numerator) / divisor if divisor else None
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return Fraction(text)
    return None


def boxed_reward(completion, answer, invalid_reward=0.0):
    starts = list(re.finditer(r"\\boxed\s*\{", completion))
    if not starts:
        return invalid_reward
    start = starts[-1].end()
    depth, end = 1, start
    while end < len(completion) and depth:
        depth += (completion[end] == "{") - (completion[end] == "}")
        end += 1
    if depth:
        return invalid_reward
    prediction = numeric_answer(completion[start:end - 1])
    target = numeric_answer(answer.rsplit("####", 1)[-1])
    return float(prediction is not None and target is not None and prediction == target)


def trim_completion(ids, eos_id):
    """Keep the first EOS visible to scoring; everything after it is padding."""
    return ids[:ids.index(eos_id) + 1] if eos_id in ids else ids


def antithetic_masks(eligible, replicates, generator, endpoint="repeat_full"):
    """OWN G11: uniform 1..N; complement except full->full at the empty endpoint."""
    positions = eligible.nonzero().flatten()
    count = positions.numel()
    masks = torch.zeros(replicates, eligible.numel(), dtype=torch.bool)
    if count == 0:
        return masks
    for pair in range(replicates // 2):
        level = int(torch.randint(1, count + 1, (), generator=generator))
        selected = positions[torch.randperm(count, generator=generator)[:level]]
        masks[2 * pair, selected] = True
        complement = eligible & ~masks[2 * pair]
        masks[2 * pair + 1] = complement if complement.any() else (
            eligible if endpoint == "repeat_full" else torch.zeros_like(eligible))
    return masks


def center_rewards(rewards):
    return rewards - rewards.mean(dim=-1, keepdim=True)


def balance_residual(policy_score, reference_score, centered_reward, log_z, beta):
    return policy_score.float() - reference_score.float() - beta * centered_reward + log_z


def trajectory_loss(policy_score, reference_score, rewards, log_z, beta):
    residual = balance_residual(policy_score, reference_score,
                                center_rewards(rewards), log_z[:, None], beta)
    return residual.square().mean()


def masked_token_mean(logits, targets):
    return logits.float().log_softmax(dim=-1).gather(-1, targets[..., None]).mean()


def masked_score(model, prompt, completion, masks, mask_id, dtype, empty_score):
    """Eq.(4): mean masked-token logp, then mean over K shared corruptions."""
    scores = []
    for mask in masks:
        positions = mask.nonzero().flatten().to(prompt.device, non_blocking=True)
        if positions.numel() == 0:
            scores.append(torch.full((), empty_score, device=prompt.device))
            continue
        tokens = torch.cat((prompt, completion)).clone()
        tokens[prompt.numel() + positions] = mask_id
        with torch.autocast(device_type=prompt.device.type, dtype=dtype):
            logits = model(tokens[None], use_cache=False,
                           logits_to_keep=prompt.numel() + positions).logits[0]
        scores.append(masked_token_mean(logits, completion[positions]))
    return torch.stack(scores).mean()


class LogZ(nn.Module):
    def __init__(self, dimension, hidden, activation, bias, zero_output):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, hidden, bias=bias),
                                    getattr(nn, activation)(), nn.Linear(hidden, 1, bias=bias))
        if zero_output:
            nn.init.zeros_(self.layers[-1].weight)
            if bias:
                nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        return self.layers(features.detach().float()).squeeze(-1)


@torch.no_grad()
def prompt_features(policy, prompt, dtype):
    """OWN G09: final current-policy states of a clean, unpadded prompt."""
    with torch.autocast(device_type=prompt.device.type, dtype=dtype):
        hidden = policy.get_base_model().model(prompt[None], use_cache=False).last_hidden_state
    return hidden.float().mean(dim=1).detach()


def nucleus_tokens(logits, temperature, top_p, generator):
    """Keep the probability-mass crossing token, sample from the remaining nucleus."""
    logits = logits.float()
    if temperature == 0:
        return logits.argmax(dim=-1)
    probs = (logits / temperature).softmax(dim=-1)
    sorted_probs, order = probs.sort(dim=-1, descending=True)
    remove = sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0)
    draw = torch.multinomial(sorted_probs.reshape(-1, sorted_probs.shape[-1]), 1,
                             generator=generator).reshape(*logits.shape[:-1], 1)
    return order.gather(-1, draw).squeeze(-1)


@torch.no_grad()
def generate(model, prompt, count, length, block_size, temperature, top_p,
             threshold, mask_id, suppress_mask, generator, dtype):
    """Fast-dLLM v1 no-cache confidence-threshold scheme, with true nucleus proposals.

    At least one highest-confidence active token is committed per row and step.
    EARLY EOS stopping is disabled (OWN G10); trim only after the final denoising.
    """
    canvas = torch.cat((prompt[None].expand(count, -1),
                        torch.full((count, length), mask_id, device=prompt.device)), dim=1)
    prefix = prompt.numel()
    for start in range(0, length, block_size):
        end = min(start + block_size, length)
        positions = torch.arange(prefix + start, prefix + end, device=prompt.device)
        for step in range(end - start):
            active = canvas[:, positions] == mask_id
            if not active.any():
                break  # Required control sync for confidence-aware parallel decoding.
            with torch.autocast(device_type=prompt.device.type, dtype=dtype):
                logits = model(canvas, use_cache=False, logits_to_keep=positions).logits.float()
            probabilities = logits.softmax(dim=-1)
            if suppress_mask:
                logits[..., mask_id] = -torch.inf
            proposed = nucleus_tokens(logits, temperature, top_p, generator)
            confidence = probabilities.gather(-1, proposed[..., None]).squeeze(-1)
            confidence = confidence.masked_fill(~active, -torch.inf)
            transfer = active & (confidence >= threshold)
            forced = torch.zeros_like(active).scatter_(1, confidence.argmax(dim=1)[:, None], True)
            transfer |= forced & active  # Highest-confidence token is always accepted.
            canvas[:, positions] = torch.where(transfer, proposed, canvas[:, positions])
    return canvas[:, prefix:]


def learning_rate(step, updates, peak, minimum, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(updates - warmup - 1, 1)
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2


def pass_at_k(correct, total, k):
    return 1.0 - (math.comb(total - correct, k) if total - correct >= k else 0) / math.comb(total, k)
