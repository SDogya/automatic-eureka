"""Один Algorithm 1 online update с streamed backward и single optimizer.step."""

import itertools
import json

from .masking import antithetic_masks
from .model import prompt_features
from .objective import balance_residual
from .rewards import boxed_reward, trim_completion
from .sampling import sample_group
from .scoring import masked_score


def scored_completions(generated, tokenizer, answer, cfg):
    # One host transfer at the generation boundary: reward extraction requires text.
    r"""Обрезать/декодировать group outputs и вычислить string-based rewards.

    Args:
        generated: Device long tensor [G, L] полного denoising.
        tokenizer: EOS ID и batch_decode(skip_special_tokens=True).
        answer: Одна GSM8K target string для всех completions этого prompt.
        cfg: Config с invalid_reward.

    Returns:
        (token_lists, texts, rewards): token IDs до первого EOS включительно,
        decoded strings и Python float rewards по каждому rollout.

    Один GPU->CPU transfer нужен для text parsing. Token lists сохраняют EOS
    для scoring context, а decoded strings скрывают special tokens. No autograd."""
    token_lists = [
        trim_completion(ids, tokenizer.eos_token_id) for ids in generated.cpu().tolist()
    ]
    texts = tokenizer.batch_decode(token_lists, skip_special_tokens=True)
    rewards = [boxed_reward(text, answer, cfg["invalid_reward"]) for text in texts]
    return token_lists, texts, rewards


def update_batch(
    policy,
    forward,
    reference,
    head,
    rows,
    cfg,
    tokenizer,
    rng,
    mask_rng,
    dtype,
    device,
    optimizer,
    record,
):
    r"""Выполнить один полный online Algorithm 1 optimizer update.

    Args:
        policy, forward: Current trainable PEFT policy и callable её forward.
        reference: Frozen reference; scores вычисляются под no_grad.
        head: Trainable scalar log Z head с detached prompt features.
        rows: Actual prompt batch; denominator равен len(rows)*cfg.rollouts.
        cfg, tokenizer: Flat settings и token/reward decoding.
        rng, mask_rng: Раздельные rollout/device и masking/CPU generators.
        dtype, device: AMP dtype и torch device.
        optimizer: Один joint optimizer policy/head.
        record: Открытый JSONL stream для фактических completions/rewards.

    Returns:
        Device FP32 tensor [5]: loss, mean reward, mixed-group fraction,
        mean policy-reference score difference и mean log Z.

    Zero_grad вызывается один раз. На prompt сначала fresh rollout и group centering,
    затем K shared masks policy/reference на completion, mean scores до square.
    Каждый completion backward делится на B*G; optimizer.step ровно один после
    всех rows/rollouts. Optional gradient clipping применяется перед этим update.
    Группы с одинаковыми rewards сохраняются. Reward/rollout gradients отсутствуют."""
    import torch

    optimizer.zero_grad(set_to_none=True)
    stats = torch.zeros(5, device=device)
    denominator = len(rows) * cfg["rollouts"]
    for row in rows:
        prompt = row["input_ids"].to(device, non_blocking=True)
        policy.eval()
        generated = sample_group(
            forward,
            prompt,
            cfg["rollouts"],
            cfg["train_max_new_tokens"],
            "train",
            cfg,
            tokenizer,
            rng,
            dtype,
        )
        token_lists, texts, reward_list = scored_completions(
            generated, tokenizer, row["answer"], cfg
        )
        rewards = torch.tensor(reward_list, device=device)
        centered = rewards - rewards.mean()
        features = prompt_features(policy, prompt, dtype, cfg["train_block_size"])
        policy.train()
        stats[1] += rewards.sum() / denominator
        stats[2] += (rewards.max() != rewards.min()).float() / len(rows)
        for g, ids in enumerate(token_lists):
            completion = generated[g, : len(ids)]
            eligible = torch.tensor([token != tokenizer.eos_token_id for token in ids])
            masks = antithetic_masks(
                eligible, cfg["mask_replicates"], mask_rng, cfg["mask_endpoint"]
            )
            score = masked_score(
                forward,
                prompt,
                completion,
                masks,
                tokenizer.mask_token_id,
                dtype,
                cfg["empty_score"],
                cfg["train_block_size"],
            )
            with torch.no_grad():
                ref_score = masked_score(
                    reference,
                    prompt,
                    completion,
                    masks,
                    tokenizer.mask_token_id,
                    dtype,
                    cfg["empty_score"],
                    cfg["train_block_size"],
                )
            log_z = head(features).squeeze(0)
            residual = balance_residual(
                score, ref_score, centered[g], log_z, cfg["beta"]
            )
            loss = residual.square() / denominator
            loss.backward()
            stats[0] += loss.detach()
            stats[3] += (score - ref_score).detach() / denominator
            stats[4] += log_z.detach() / denominator
        record.write(
            json.dumps(
                {
                    "example_id": row["example_id"],
                    "question": row["question"],
                    "completions": texts,
                    "rewards": reward_list,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    if cfg["grad_clip"] is not None:
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(policy.parameters(), head.parameters()), cfg["grad_clip"]
        )
    optimizer.step()  # Exactly ONE update after the complete prompt/rollout batch.
    return stats
