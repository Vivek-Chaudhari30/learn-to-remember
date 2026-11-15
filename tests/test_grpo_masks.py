# Origin: ported from GRPO-Zero `tests/test_grpo.py` (main @ ce51ed9), plus new
# position-id tests.
"""Model-free tests for the mask helpers. The masks decide which tokens count
toward the loss/KL, so a bug here silently corrupts training."""

import torch

from learn_to_remember.grpo.core import make_completion_mask, position_ids_from_mask

EOS = 0


def test_mask_eos_in_middle():
    ids = torch.tensor([[5, 6, EOS, 9, 9]])
    assert make_completion_mask(ids, EOS).tolist() == [[1, 1, 1, 0, 0]]


def test_mask_no_eos_all_valid():
    # row hit max_new_tokens with no EOS -> every token is real
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    assert make_completion_mask(ids, EOS).tolist() == [[1, 1, 1, 1, 1]]


def test_mask_eos_first_token():
    ids = torch.tensor([[EOS, 9, 9, 9, 9]])
    assert make_completion_mask(ids, EOS).tolist() == [[1, 0, 0, 0, 0]]


def test_mask_eos_last_token():
    ids = torch.tensor([[1, 2, 3, 4, EOS]])
    assert make_completion_mask(ids, EOS).tolist() == [[1, 1, 1, 1, 1]]


def test_mask_only_first_eos_counts():
    # when pad_id == eos_id, trailing pads are EOS too; only the first ends it
    ids = torch.tensor([[7, EOS, EOS, EOS, EOS]])
    assert make_completion_mask(ids, EOS).tolist() == [[1, 1, 0, 0, 0]]


def test_mask_batched_rows_independent():
    ids = torch.tensor([
        [5, 6, EOS, 9, 9],   # eos mid
        [1, 2, 3, 4, 5],     # no eos
        [EOS, 1, 1, 1, 1],   # eos first
    ])
    expected = [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
    ]
    assert make_completion_mask(ids, EOS).tolist() == expected


def test_position_ids_left_padding_starts_at_first_real_token():
    attn = torch.tensor([[0, 0, 1, 1, 1],
                         [1, 1, 1, 1, 1]])
    assert position_ids_from_mask(attn).tolist() == [[0, 0, 0, 1, 2],
                                                     [0, 1, 2, 3, 4]]


def test_position_ids_right_padding_is_identity_on_real_tokens():
    attn = torch.tensor([[1, 1, 1, 0, 0]])
    assert position_ids_from_mask(attn)[0, :3].tolist() == [0, 1, 2]
