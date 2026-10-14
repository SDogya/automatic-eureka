"""Independent numerical comparisons of the optimized and scalar paths."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trafl.cached_sampling import CachedSampler, nucleus_step
from trafl.sampling import generate
from trafl.scoring import masked_score, masked_scores


def tiny_qwen():
    """Use the actual pinned Qwen architecture with a small random configuration."""
    from dllm.pipelines.a2d.models.qwen3.modeling_qwen3 import A2DQwen3LMHeadModel
    from dllm.pipelines.a2d.models.qwen3.modeling_qwen3 import A2DQwen3Config
    cfg = A2DQwen3Config(vocab_size=16, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=8)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(11)
    return A2DQwen3LMHeadModel(cfg).eval()


@pytest.mark.parametrize("prompt_length", [3, 4, 7])
def test_cached_greedy_matches_full_canvas(prompt_length):
    model = tiny_qwen()
    prompt = torch.arange(prompt_length) % 8
    old = generate(model, prompt, 2, 9, 4, 0., .91, .9, 15, True,
                   torch.Generator().manual_seed(5), torch.bfloat16)
    sampler = CachedSampler(model, dict(generation_compile=False, generation_cuda_graph=False))
    new = sampler.sample(prompt, 2, 9, 4, 0., .91, .9, 15, True,
                         torch.Generator().manual_seed(5), torch.bfloat16, 14, False)
    torch.testing.assert_close(new, old)


@pytest.mark.parametrize("microbatch", [1, 3, 8])
def test_ragged_batched_scores_and_gradients_match_scalar(microbatch):
    model = tiny_qwen()
    prompt = torch.tensor([2, 3, 4])
    completions = [torch.tensor([1, 2, 3, 4, 14]), torch.tensor([6, 7, 14]), torch.tensor([14])]
    masks = [torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]], dtype=torch.bool),
             torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool),
             torch.zeros(2, 1, dtype=torch.bool)]
    expected = torch.stack([masked_score(model, prompt, x, m, 15, torch.bfloat16, 0., 4)
                            for x, m in zip(completions, masks)])
    parameter = model.lm_head.weight
    expected_gradient = torch.autograd.grad(expected.square().sum(), parameter)[0]
    actual = masked_scores(model, prompt, completions, masks, 15, torch.bfloat16, 0., 4, 0, microbatch)
    # BF16 matmuls may differ slightly when batch shapes change.
    torch.testing.assert_close(actual, expected, atol=.015, rtol=.005)
    gradient = torch.autograd.grad(actual.square().sum(), parameter)[0]
    torch.testing.assert_close(gradient, expected_gradient, atol=.04, rtol=.04)


def test_exponential_race_nucleus_support_and_frequency():
    n = 8000
    logits = torch.tensor([.6, .25, .15]).log().expand(n, 1, 3)
    noise = torch.empty_like(logits, dtype=torch.float64).exponential_(generator=torch.Generator().manual_seed(7))
    tokens = nucleus_step(logits, noise, torch.full((n, 1), 2), torch.ones(1, dtype=torch.bool),
                          torch.ones(n, dtype=torch.bool), 1., .7, .9, 2, False)
    assert not (tokens == 2).any()
    assert (tokens == 0).float().mean().item() == pytest.approx(.6/.85, abs=.025)


def test_compiled_batched_reduction_values_and_gradients():
    from trafl.scoring import batched_token_scores
    logits = torch.randn(4, 5, 9, requires_grad=True)
    targets = torch.randint(0, 9, (4, 5))
    selected = torch.rand(4, 5) > .5
    selected[0] = False
    expected = batched_token_scores(logits, targets, selected, 0.)
    gradient = torch.autograd.grad(expected.square().sum(), logits)[0]
    compiled = torch.compile(batched_token_scores, backend="inductor", fullgraph=True)
    actual = compiled(logits, targets, selected, 0.)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), logits)[0], gradient)


def test_cold_sampler_reproduces_saved_rollout_rng():
    model = tiny_qwen()
    cfg = dict(generation_compile=False, generation_cuda_graph=False)
    rng = torch.Generator().manual_seed(44)
    args = (torch.tensor([2, 3, 4]), 2, 9, 4, .6, .91, .9, 15, True)
    first = CachedSampler(model, cfg)
    first.sample(*args, rng, torch.bfloat16, 14, False)
    saved_rng = rng.get_state().clone()
    expected = first.sample(*args, rng, torch.bfloat16, 14, False)
    rng.set_state(saved_rng)
    actual = CachedSampler(model, cfg).sample(*args, rng, torch.bfloat16, 14, False)
    torch.testing.assert_close(actual, expected)


class EosDenoiser(nn.Module):
    """Force row zero to EOS and row one to ordinary tokens in every block."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, **kwargs):
        self.calls.append(input_ids.clone())
        logits = torch.full((*input_ids.shape, 4), -20.)
        logits[0, :, 2] = 20.
        logits[1, :, 1] = 20.
        return SimpleNamespace(logits=logits)

    @property
    def model(self):
        def backbone(ids, **kwargs):
            cache = SimpleNamespace(layers=[], get_seq_length=lambda: ids.shape[1])
            return SimpleNamespace(past_key_values=cache)
        return backbone


def test_eos_freezes_row_after_finishing_current_block():
    model = EosDenoiser()
    sampler = CachedSampler(model, dict(generation_compile=False, generation_cuda_graph=False))
    result = sampler.sample(torch.tensor([0]), 2, 7, 4, 0., .91, .9, 3, True,
                            torch.Generator(), torch.bfloat16, 2, True)
    assert result[0].tolist() == [2, 2, 2, 3, 3, 3, 3]
    assert result[1].tolist() == [1] * 7
    assert len(model.calls) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires GPU")
def test_cuda_graph_reads_updated_lora_weights():
    from peft import LoraConfig, get_peft_model
    from trafl.attention import staircase_attention
    from trafl.cached_sampling import CachedForward
    model = get_peft_model(tiny_qwen().to("cuda", dtype=torch.float16),
                          LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"]))
    model.eval()
    ids = torch.tensor([[2, 3, 15, 15]], device="cuda")
    attention, positions = staircase_attention(4, 4, ids.device)
    graph = CachedForward(model, 2)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
        before = graph(ids, attention, positions, None).clone()
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.add_(2.)
        after = graph(ids, attention, positions, None).clone()
        eager = model(ids, attention_mask=attention, position_ids=positions, use_cache=False).logits
    assert not torch.equal(before, after)
    torch.testing.assert_close(after, eager, atol=.002, rtol=.002)
