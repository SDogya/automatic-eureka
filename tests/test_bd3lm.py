"""BD3LM-specific checks for staircase attention and prompt-aligned generation.

These tests are supplied for a later explicit test run; they were not executed
while preparing this branch.
"""

from types import SimpleNamespace

import torch
from torch import nn

from trafl.attention import staircase_attention
from trafl.sampling import generate


def test_staircase_attention_known_matrix():
    """Check an independent literal attention matrix for three physical blocks."""
    mask, positions = staircase_attention(5, 2, torch.device("cpu"))
    expected = torch.tensor([
        [True, True, False, False, False],
        [True, True, False, False, False],
        [True, True, True, True, False],
        [True, True, True, True, False],
        [True, True, True, True, True],
    ])
    torch.testing.assert_close(mask, expected[None, None])
    torch.testing.assert_close(positions, torch.tensor([[0, 1, 2, 3, 4]]))


class RecordingDenoiser(nn.Module):
    """Record decoder positions/masks while returning one confident ordinary token."""

    def __init__(self):
        """Initialize an empty forward trace for checking actual block boundaries."""
        super().__init__()
        self.calls = []

    def forward(self, input_ids, attention_mask, position_ids, use_cache, logits_to_keep):
        """Record explicit BD3LM attention and propose token 2 for selected positions."""
        self.calls.append((logits_to_keep.clone(), attention_mask.clone(), position_ids.clone()))
        logits = torch.full((input_ids.shape[0], logits_to_keep.numel(), 4), -30.)
        logits[..., 2] = 30.
        return SimpleNamespace(logits=logits)


def test_generation_aligns_partial_prompt_block():
    """Check that P=3, block=4 fills one token, a full block, then a final token."""
    model = RecordingDenoiser()
    result = generate(model, torch.tensor([0, 1, 0]), 2, 6, 4,
                      .6, .91, .9, 3, True,
                      torch.Generator().manual_seed(44), torch.bfloat16)
    assert [entry[0].tolist() for entry in model.calls] == [[3], [4, 5, 6, 7], [8]]
    assert result.shape == (2, 6)
    assert (result == 2).all()
    for _, mask, positions in model.calls:
        assert mask.shape == (1, 1, 9, 9)
        assert positions.tolist() == [list(range(9))]
        assert not mask[0, 0, :4, 4:].any()
        assert mask[0, 0, 4:8, :8].all()


def test_generation_starts_full_block_at_aligned_prompt():
    """Check the zero-offset boundary and a shorter final block."""
    model = RecordingDenoiser()
    generate(model, torch.tensor([0, 1, 0, 1]), 1, 5, 4,
             .6, .91, .9, 3, True,
             torch.Generator().manual_seed(44), torch.bfloat16)
    assert [entry[0].tolist() for entry in model.calls] == [[4, 5, 6, 7], [8]]
