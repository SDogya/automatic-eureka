"""Original numerical checks, relocated; runtime tests were not executed for this split."""

import itertools
import math
from collections import defaultdict
from types import SimpleNamespace
from io import StringIO

import pytest
import torch
import yaml
from torch import nn

from trafl import training as run
from trafl.evaluation import pass_at_k
from trafl.masking import antithetic_masks
from trafl.model import LogZ
from trafl.objective import center_rewards, learning_rate, trajectory_loss
from trafl.rewards import boxed_reward, trim_completion
from trafl.sampling import generate, nucleus_tokens
from trafl.scoring import masked_score, masked_token_mean


def test_paper_loss_and_analytic_gradients():
    r"""Сверить Algorithm 1 loss и его gradients с независимыми численными ожиданиями.
    
    Включает group centering, beta, log Z broadcasting и BG reduction."""
    policy = torch.tensor([[-1., -2., -3., -4., -5.], [-.2, -.4, -.6, -.8, -1.]], requires_grad=True)
    reference = torch.tensor([[-1.] * 5, [0.] * 5])
    rewards = torch.tensor([[1., 0., 1., 0., 0.], [0.] * 5])
    log_z = torch.tensor([.3, -.2], requires_grad=True)
    expected_delta = torch.tensor([[-.9, .1, -2.9, -1.9, -2.9], [-.4, -.6, -.8, -1., -1.2]])
    loss = trajectory_loss(policy, reference, rewards, log_z, 2.)
    assert loss.item() == pytest.approx(2.485)
    loss.backward()
    torch.testing.assert_close(policy.grad, expected_delta / 5)
    torch.testing.assert_close(log_z.grad, expected_delta.sum(dim=1) / 5)
    assert reference.grad is None


def test_centering_and_all_equal_groups():
    r"""Проверить mean centering и сохранение all-equal reward groups в loss.
    
    Нулевые centered rewards не должны приводить к пропуску group."""
    rewards = torch.tensor([[1., 0., 1., 0., 0.], [1.] * 5, [0.] * 5])
    torch.testing.assert_close(center_rewards(rewards),
                              torch.tensor([[.6, -.4, .6, -.4, -.4], [0.] * 5, [0.] * 5]))
    policy = torch.tensor([[-.2] * 5, [-.3] * 5], requires_grad=True)
    loss = trajectory_loss(policy, torch.zeros_like(policy), rewards[1:], torch.zeros(2), 1.)
    assert loss.item() == pytest.approx(.065)  # Equal rewards do not skip either group.


def test_masked_mean_uses_original_targets_and_mask_count():
    r"""Проверить FP32 target log-probability mean по выбранным masked positions.
    
    Expected value вычислен напрямую из двух заданных вероятностей."""
    logits = torch.tensor([[math.log(.25), math.log(.75)], [math.log(.8), math.log(.2)]])
    expected = (math.log(.75) + math.log(.8)) / 2
    assert masked_token_mean(logits, torch.tensor([1, 0])).item() == pytest.approx(expected)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_exhaustive_antithetic_uniform_marginals(count, monkeypatch):
    r"""Исчерпывающе проверить antithetic mask marginals на коротких completions.
    
    Перебирает levels/subsets, eligibility и full->full endpoint для N=1..4."""
    eligible = torch.tensor([True] * count + [False])  # One unscored position.
    masses = [defaultdict(float), defaultdict(float)]
    for level in range(1, count + 1):
        for selected in itertools.combinations(range(count), level):
            permutation = list(selected) + [i for i in range(count) if i not in selected]
            monkeypatch.setattr(torch, "randint", lambda *a, **k: torch.tensor(level))
            monkeypatch.setattr(torch, "randperm", lambda *a, **k: torch.tensor(permutation))
            masks = antithetic_masks(eligible, 2, torch.Generator())
            weight = 1 / (count * math.comb(count, level))
            for replica in range(2):
                key = tuple(masks[replica].nonzero().flatten().tolist())
                masses[replica][key] += weight
            assert not masks[:, -1].any()
            assert masks.all(dim=1).sum() == 0  # Ineligible position never corrupted.
            if level == count:
                assert torch.equal(masks[0], masks[1])
            else:
                assert torch.equal(masks[0] ^ masks[1], eligible)
    for replica in range(2):
        for level in range(1, count + 1):
            for selected in itertools.combinations(range(count), level):
                assert masses[replica][selected] == pytest.approx(1 / (count * math.comb(count, level)))


def test_four_masks_empty_completion_and_eos():
    r"""Проверить K=4 masks, empty eligibility и EOS-prefix trimming.
    
    EOS остаётся видимым в token list, а suffix после него отбрасывается."""
    masks = antithetic_masks(torch.tensor([True, False, True]), 4, torch.Generator().manual_seed(43))
    assert masks.shape == (4, 3)
    assert not masks[:, 1].any()
    assert masks.any(dim=1).all()
    assert not antithetic_masks(torch.tensor([False]), 4, torch.Generator()).any()
    assert trim_completion([1, 2, 9, 3], 9) == [1, 2, 9]
    assert trim_completion([9, 1], 9) == [9]
    assert trim_completion([1, 2], 9) == [1, 2]


class TinyPolicy(nn.Module):
    r"""Маленький differentiable denoiser для реальных scoring/update paths.
    
    Один scalar parameter управляет logits; clean-prompt features используют
    тот же parameter, чтобы проверять detachment к backbone."""
    def __init__(self, value=.2):
        r"""Создать один trainable scalar parameter с заданным initial value."""
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))

    def forward(self, input_ids, use_cache=False, logits_to_keep=0, attention_mask=None, position_ids=None):
        r"""Выдать synthetic logits нужных positions, сохранив gradient к weight.
        
        Поддерживает тот же use_cache/logits_to_keep interface, что Qwen scorer."""
        positions = logits_to_keep if torch.is_tensor(logits_to_keep) else torch.arange(input_ids.shape[1])
        logits = self.weight * torch.arange(4.).expand(input_ids.shape[0], positions.numel(), 4)
        return SimpleNamespace(logits=logits)

    def get_base_model(self):
        r"""Вернуть synthetic backbone с last_hidden_state для prompt feature extraction."""
        return SimpleNamespace(model=lambda ids, use_cache=False, attention_mask=None, position_ids=None:
                               SimpleNamespace(last_hidden_state=self.weight.expand(1, ids.shape[1], 2)))


def test_actual_shared_mask_scoring_and_average_before_square():
    r"""Сверить настоящий masked_score с независимо рассчитанными mask means.
    
    Проверяет mean-before-square, trainable score gradient и empty-mask convention."""
    policy = TinyPolicy()
    prompt, completion = torch.tensor([0, 0]), torch.tensor([1, 2, 3])
    masks = torch.tensor([[True, False, False], [False, True, False],
                          [True, True, False], [True, False, False]])
    score = masked_score(policy, prompt, completion, masks, 3, torch.bfloat16, 0., 32)
    normalizer = math.log(sum(math.exp(.2 * i) for i in range(4)))
    per_mask = [.2 - normalizer, .4 - normalizer, .3 - normalizer, .2 - normalizer]
    assert score.item() == pytest.approx(sum(per_mask) / 4, abs=1e-6)
    assert score.square().item() != pytest.approx(sum(s * s for s in per_mask) / 4, abs=1e-6)
    score.backward()
    assert policy.weight.grad is not None
    assert masked_score(policy, prompt, torch.tensor([3]), torch.zeros(4, 1, dtype=torch.bool),
                        3, torch.bfloat16, 0., 32).item() == 0


def test_head_features_detached_but_head_trains():
    r"""Проверить отсутствие gradients к features при наличии gradients головы."""
    head = LogZ(2, 3, "SiLU", True, False)
    features = torch.ones(1, 2, requires_grad=True)
    head(features).sum().backward()
    assert features.grad is None
    assert head.layers[-1].weight.grad.abs().sum() > 0


def test_streamed_gradients_equal_full_batch():
    r"""Сравнить gradients full batch loss и streamed backward с divisor B*G.
    
    Это независимая численная проверка эквивалентности accumulation."""
    rewards = torch.tensor([[1., 0., 1., 0., 0.], [0., 1., 0., 0., 1.]])
    score = torch.arange(10.).reshape(2, 5) / 10
    parameter, intercept = torch.tensor(.3, requires_grad=True), torch.tensor(.2, requires_grad=True)
    trajectory_loss(parameter * score, torch.zeros_like(score), rewards, intercept.expand(2), 1.).backward()
    full = (parameter.grad.clone(), intercept.grad.clone())
    parameter.grad, intercept.grad = None, None
    for b, g in itertools.product(range(2), range(5)):
        delta = parameter * score[b, g] - (rewards[b, g] - rewards[b].mean()) + intercept
        (delta.square() / 10).backward()
    torch.testing.assert_close(parameter.grad, full[0])
    torch.testing.assert_close(intercept.grad, full[1])


def test_real_update_one_step_and_frozen_reference(monkeypatch):
    r"""Проверить production update_batch на synthetic rollout/rewards.
    
    Monkeypatch применяется к training.sample_group, где update_batch
    ищет callable после переноса. Проверяется one step, frozen reference
    и независимое аналитическое ожидание policy gradient."""
    cfg = yaml.safe_load(open("config.yaml"))
    cfg.update(optimizer_fused=False, grad_clip=None)
    policy, reference = TinyPolicy(), TinyPolicy()
    reference.requires_grad_(False)
    head = LogZ(2, 4, "SiLU", True, True)
    optimizer = torch.optim.AdamW(list(policy.parameters()) + list(head.parameters()), lr=1e-3,
                                 weight_decay=cfg["weight_decay"])
    before = reference.weight.detach().clone()
    policy_before = policy.weight.detach().clone()
    generated = torch.tensor([[2, 2, 3], [1, 1, 3], [2, 2, 3], [1, 1, 3], [1, 1, 3]])
    monkeypatch.setattr(run, "sample_group", lambda *a: generated)
    tokenizer = SimpleNamespace(mask_token_id=3, eos_token_id=3,
        batch_decode=lambda ids, skip_special_tokens: [r"\boxed{2}", r"\boxed{0}", r"\boxed{2}", "", ""])
    rows = [{"input_ids": torch.tensor([0, 0]), "answer": "#### 2", "question": "test", "example_id": 0}]
    steps, original = [], optimizer.step
    monkeypatch.setattr(optimizer, "step", lambda: (steps.append(1), original())[1])
    stats = run.update_batch(policy, policy, reference, head, rows, cfg, tokenizer,
        torch.Generator(), torch.Generator().manual_seed(43), torch.bfloat16,
        torch.device("cpu"), optimizer, StringIO())
    assert len(steps) == 1
    torch.testing.assert_close(reference.weight, before)
    assert reference.weight.grad is None
    # 2/5 * (2*(-.6)*2 + 3*(.4)*1) = -.48; softmax normalizer cancels.
    assert policy.weight.grad.item() == pytest.approx(-.48, abs=1e-5)
    assert (policy.weight.detach() - policy_before).abs().item() > 1e-4
    assert stats[1].item() == pytest.approx(.4)
    assert stats[2].item() == 1.


def test_nucleus_crossing_token_and_temperature():
    r"""Проверить nucleus crossing-token rule и temperature-before-top-p.
    
    Частоты сравниваются с conditional categorical distribution."""
    logits = torch.tensor([[math.log(.6), math.log(.25), math.log(.15)]]).expand(4000, -1)
    draws = nucleus_tokens(logits, 1., .7, torch.Generator().manual_seed(7))
    assert not (draws == 2).any()
    assert (draws == 1).any()  # Token crossing .7 must survive.
    assert (draws == 0).float().mean().item() == pytest.approx(.6 / .85, abs=.03)
    draws = nucleus_tokens(logits, .5, .7, torch.Generator().manual_seed(7))
    assert (draws == 0).all()  # Temperature changes mass BEFORE selecting nucleus.
    assert (nucleus_tokens(logits, 0., .7, torch.Generator()) == 0).all()


def test_sampler_completes_without_mask_or_prompt_changes():
    r"""Проверить завершение block denoising без leftover mask и изменения prompt."""
    prompt = torch.tensor([0, 1])
    generated = generate(TinyPolicy(), prompt, 3, 6, 2, .6, .91, .9,
                         3, True, torch.Generator().manual_seed(44), torch.bfloat16)
    assert generated.shape == (3, 6)
    assert not (generated == 3).any()
    assert prompt.tolist() == [0, 1]


def test_compiled_eager_values_and_gradients():
    r"""Сравнить eager и compiled TraFL loss, включая gradients по scores."""
    score = torch.tensor([[-.3, -.1, -.4, -.5, -.6]], requires_grad=True)
    reference, rewards, log_z = torch.zeros_like(score), torch.tensor([[1., 0., 0., 1., 0.]]), torch.tensor([.2])
    eager = trajectory_loss(score, reference, rewards, log_z, 1.)
    expected = torch.autograd.grad(eager, score)[0]
    compiled = torch.compile(trajectory_loss, backend="inductor", fullgraph=True)
    actual = compiled(score, reference, rewards, log_z, 1.)
    torch.testing.assert_close(actual, eager)
    torch.testing.assert_close(torch.autograd.grad(actual, score)[0], expected)


def test_production_compiled_score_kernel_values_and_gradients():
    r"""Сравнить production score kernel eager/compiled по outputs и gradients."""
    logits = torch.linspace(-2, 2, 55).reshape(5, 11).requires_grad_()
    targets = torch.tensor([1, 3, 4, 2, 10])
    eager = masked_token_mean(logits, targets)
    expected = torch.autograd.grad(eager, logits)[0]
    compiled = torch.compile(masked_token_mean, backend="inductor", dynamic=True)
    actual = compiled(logits, targets)
    torch.testing.assert_close(actual, eager)
    torch.testing.assert_close(torch.autograd.grad(actual, logits)[0], expected)


def test_schedule_endpoints_and_pass_k_known_values():
    r"""Проверить cosine endpoints и известные combinatorial Pass@k значения."""
    assert learning_rate(0, 100, 6e-5, 6e-6, 0) == pytest.approx(6e-5)
    assert learning_rate(99, 100, 6e-5, 6e-6, 0) == pytest.approx(6e-6)
    assert pass_at_k(0, 16, 5) == 0
    assert pass_at_k(16, 16, 5) == 1
    assert pass_at_k(2, 4, 2) == pytest.approx(5 / 6)


@pytest.mark.parametrize("text,answer,result", [
    (r"<answer>\boxed{1,200}</answer>", "#### 1200", 1),
    (r"\boxed{\frac{1}{2}}", "#### 0.5", 1),
    (r"\boxed{2} then \boxed{3}", "#### 2", 0),
    (r"\boxed{\frac{1}{0}}", "#### 2", 0),
    (r"\boxed{\frac{a}{2}}", "#### 2", 0),
    (r"\boxed{2", "#### 2", 0),
    ("answer is 2", "#### 2", 0),
])
def test_binary_reward(text, answer, result):
    r"""Проверить boxed numeric reward на exact, invalid и malformed completions.
    
    Параметризация включает nested frac, zero denominator и last-boxed rule."""
    assert boxed_reward(text, answer) == result
