"""TraFL group centering/residual/loss и исходный cosine/warmup schedule."""

import math


def center_rewards(rewards):
    r"""Вычесть среднюю награду внутри каждой prompt group.
    
    Args:
        rewards: Tensor [..., G] с reward каждого rollout; последняя ось — группа.
    
    Returns:
        Tensor той же формы/device: rewards - mean(rewards, last axis).
    
    Algorithm 1 использует только centering, без деления на стандартное отклонение.
    All-equal groups получают нулевые centered rewards и не удаляются из batch."""
    return rewards - rewards.mean(dim=-1, keepdim=True)



def balance_residual(policy_score, reference_score, centered_reward, log_z, beta):
    r"""Вычислить practical TraFL residual в FP32.
    
    Args:
        policy_score: Score текущей policy, с сохранённым autograd graph.
        reference_score: Score frozen reference на тех же corruptions.
        centered_reward: Reward после centering внутри prompt group.
        log_z: Выход prompt-dependent головы; формы должны broadcast с scores.
        beta: Коэффициент reward, конкретное значение — OWN G01.
    
    Returns:
        policy_score.float() - reference_score.float()
        - beta * centered_reward + log_z. Квадрат и batch reduction здесь
        не применяются; gradient policy/head сохраняется.
    
    Это residual Algorithm 1, использующий surrogate score Eq. (4)."""
    return (
        policy_score.float() - reference_score.float() - beta * centered_reward + log_z
    )



def trajectory_loss(policy_score, reference_score, rewards, log_z, beta):
    r"""Вычислить полный batch TraFL loss для уже оценённых scores.
    
    Args:
        policy_score, reference_score: Tensors [B, G]; mean по K masks уже выполнен.
        rewards: Tensor [B, G] исходных rewards, до group centering.
        log_z: Tensor [B] с prompt-dependent scalar для каждого prompt.
        beta: Reward coefficient.
    
    Returns:
        Scalar: mean_{B,G}((policy - reference - beta * centered_reward + log_z)^2).
    
    Mean scores по corruptions выполняется до возведения residual в квадрат.
    Training loop использует эквивалентные последовательные backward с divisor B*G;
    эта функция также используется независимыми численными тестами."""
    residual = balance_residual(
        policy_score, reference_score, center_rewards(rewards), log_z[:, None], beta
    )
    return residual.square().mean()



def learning_rate(step, updates, peak, minimum, warmup):
    r"""Вернуть LR для указанного нулевого номера optimizer update.
    
    Args:
        step: Next update index, начиная с 0; при resume берётся сохранённая позиция.
        updates: Общий исходный horizon M, не число remaining updates.
        peak, minimum: Cosine endpoints; paper defaults 6e-5 и 6e-6.
        warmup: Число linear warmup updates; собственный default равен 0.
    
    Returns:
        Python float, применяемый перед optimizer.step. Без warmup первый update
        получает peak, последний — minimum; horizon denominator защищён max(..., 1).
    
    Cosine/endpoints опубликованы; scheduler phase/placement/horizon — OWN G05."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(updates - warmup - 1, 1)
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2

