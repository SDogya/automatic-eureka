"""Problem-level unbiased Pass@k, diagnostic validation loss и eval artifacts."""

import json
import math

from .checkpoint import write_json
from .masking import antithetic_masks
from .model import prompt_features
from .objective import balance_residual
from .sampling import sample_group
from .scoring import masked_score
from .training import scored_completions


def pass_at_k(correct, total, k):
    r"""Вычислить unbiased problem-level Pass@k из n независимых draws.
    
    Args:
        correct: Число correct completions c.
        total: Число draws n; поддержанный protocol предполагает 1 <= k <= n.
        k: Sampling budget для метрики.
    
    Returns:
        1 - comb(n-c, k)/comb(n, k), с нулём для numerator при n-c<k.
    
    Benchmark evaluation усредняет этот estimator по problems. Это не boolean
    наличие хотя бы одного correct среди первых k фактически сгенерированных draws."""
    return 1.0 - (
        math.comb(total - correct, k) if total - correct >= k else 0
    ) / math.comb(total, k)



def evaluate(
    policy,
    forward,
    reference,
    head,
    rows,
    cfg,
    tokenizer,
    dtype,
    device,
    writer,
    out,
    step,
):
    r"""Посчитать Pass@k и diagnostic TraFL validation loss на held-out rows.
    
    Args:
        policy, forward, reference, head: Current policy, callable forward,
            frozen reference и log Z head; policy временно переключается в eval.
        rows: Evaluation rows официального test split, без train examples.
        cfg, tokenizer: Eval lengths/settings, n samples, k budgets и reward decoder.
        dtype, device: Autocast dtype и device.
        writer: Существующий TensorBoard SummaryWriter.
        out: Run directory для answers-stepS-L.json и metrics-stepS.json.
        step: Completed optimizer updates, общий logging step.
    
    Returns:
        Dict length-as-string -> metrics. Каждая метрика усреднена по problems.
    
    Для каждого length rollout generator заново seeded eval_seed+length,
    mask generator — mask_seed: before/after evaluation использует общий protocol.
    Pass@k считает все n draws; validation loss — первые G draws с centering
    внутри этой группы и K одинаковыми policy/reference masks, всё под no_grad.
    Сохраняются ответы/метрики, TensorBoard flush; на выходе policy.train()."""
    import torch

    policy.eval()
    results = {}
    for length in cfg["eval_max_new_tokens"]:
        rng = torch.Generator(device=device).manual_seed(cfg["eval_seed"] + length)
        mask_rng = torch.Generator().manual_seed(cfg["mask_seed"])
        totals, answers = {}, []
        validation_loss = torch.zeros((), device=device)
        for row in rows:
            prompt = row["input_ids"].to(device, non_blocking=True)
            generated = sample_group(
                forward,
                prompt,
                cfg["eval_samples"],
                length,
                "eval",
                cfg,
                tokenizer,
                rng,
                dtype,
            )
            ids, texts, reward_list = scored_completions(
                generated, tokenizer, row["answer"], cfg
            )
            correct = int(sum(reward_list))
            for k in cfg["eval_pass_k"]:
                name = f"pass@{k}"
                totals[name] = totals.get(name, 0.0) + pass_at_k(
                    correct, cfg["eval_samples"], k
                )
            # Validation loss uses its own G=5 group (first five independent evaluation draws).
            rewards = torch.tensor(reward_list[: cfg["rollouts"]], device=device)
            centered = rewards - rewards.mean()
            with torch.no_grad():
                log_z = head(prompt_features(policy, prompt, dtype)).squeeze(0)
                losses = []
                for g, tokens in enumerate(ids[: cfg["rollouts"]]):
                    eligible = torch.tensor(
                        [token != tokenizer.eos_token_id for token in tokens]
                    )
                    masks = antithetic_masks(
                        eligible, cfg["mask_replicates"], mask_rng, cfg["mask_endpoint"]
                    )
                    completion = generated[g, : len(tokens)]
                    score = masked_score(
                        forward,
                        prompt,
                        completion,
                        masks,
                        tokenizer.mask_token_id,
                        dtype,
                        cfg["empty_score"],
                    )
                    ref_score = masked_score(
                        reference,
                        prompt,
                        completion,
                        masks,
                        tokenizer.mask_token_id,
                        dtype,
                        cfg["empty_score"],
                    )
                    losses.append(
                        balance_residual(
                            score, ref_score, centered[g], log_z, cfg["beta"]
                        ).square()
                    )
                validation_loss += torch.stack(losses).mean()
            answers.append(
                {
                    "example_id": row["example_id"],
                    "question": row["question"],
                    "completions": texts,
                    "rewards": reward_list,
                }
            )
        totals["loss/val"] = validation_loss.cpu().item()
        metrics = {name: value / len(rows) for name, value in totals.items()}
        for name, value in metrics.items():
            tag = f"{name}/{length}" if name == "loss/val" else f"eval/{length}/{name}"
            writer.add_scalar(tag, value, step)
        write_json(out / f"answers-step{step}-{length}.json", answers)
        results[str(length)] = metrics
    writer.add_scalar(
        "loss/val", sum(m["loss/val"] for m in results.values()) / len(results), step
    )
    writer.add_scalar(
        "reward/val", sum(m["pass@1"] for m in results.values()) / len(results), step
    )
    writer.flush()
    write_json(out / f"metrics-step{step}.json", results)
    print(json.dumps({"step": step, "evaluation": results}), flush=True)
    policy.train()
    return results

