"""The structural statistics, and the swap-direction convention the swap depends on.

Both are places where a wrong answer looks entirely plausible: a cosine against the wrong
lens vector is still a number in [-1, 1], and a swap driven along ``'S'`` instead of
``' Soccer'`` still produces a success rate. These pin the semantics instead.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from src.config import LensSpec
from src.heads import head_columns
from src.readout import target_direction_id, target_token_ids
from src.structural import excess_kurtosis, workspace_loading
from tests.tiny_llama import D_MODEL, N_HEADS, N_LAYERS


def _spec(head_dim: int | None = None) -> LensSpec:
    return LensSpec(
        key="tiny",
        hf_model_id="tiny",
        lens_file="",
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim=head_dim or D_MODEL // N_HEADS,
        n_prompts_fitted=0,
    )


class _BPEStub:
    """Enough tokenizer to pin the direction convention.

    Reproduces the shape of the real defect on Llama-3.1's vocabulary: for some words the
    *bare* spelling's first token carries a smaller vocabulary id than the space-prefixed
    spelling, so sorting candidates by id picks a bare letter or fragment.
    """

    _VOCAB = {
        " Soccer": [38633],
        "Soccer": [50, 12345],       # 'S' + 'occer'; 50 < 38633
        " Brazil": [16327],
        "Brazil": [68299],           # here the space-prefixed form already wins
        " elephant": [46840],
        "elephant": [10274, 762],    # 'ele' + 'phant'
        " Baseball": [38258],
        "Baseball": [4066, 999],     # 'Base' + 'ball'
        " ": [],
        "": [],
    }

    def __call__(self, text, *, add_special_tokens=True):
        return SimpleNamespace(input_ids=list(self._VOCAB[text]))


@pytest.fixture
def tokenizer():
    return _BPEStub()


# ---------------------------------------------------------- the swap direction


@pytest.mark.parametrize(
    ("form", "fragment_id", "concept_id"),
    [("Soccer", 50, 38633), ("elephant", 10274, 46840), ("Baseball", 4066, 38258)],
)
def test_direction_is_the_concept_not_the_smallest_id(
    tokenizer, form, fragment_id, concept_id
):
    """The defect and the fix, side by side.

    ``target_token_ids`` sorts numerically over vocabulary ids and exists for *grading*,
    so its first element is a BPE accident. Driving a swap along it moves the activation
    towards a bare letter's lens vector rather than the concept's.
    """
    assert target_token_ids(tokenizer, form)[0] == fragment_id
    assert target_direction_id(tokenizer, form) == concept_id


def test_direction_agrees_with_grading_when_the_word_is_one_token(tokenizer):
    """Where the space-prefixed form already sorts first, nothing changes."""
    assert target_token_ids(tokenizer, "Brazil")[0] == 16327
    assert target_direction_id(tokenizer, "Brazil") == 16327


def test_direction_is_always_a_graded_candidate(tokenizer):
    """The chosen direction must be in the accepted-success set, or grading and the
    intervention would be pointing at different things."""
    for form in ("Soccer", "Brazil", "elephant", "Baseball"):
        assert target_direction_id(tokenizer, form) in target_token_ids(tokenizer, form)


def test_direction_rejects_an_untokenisable_form(tokenizer):
    with pytest.raises(ValueError, match="space-prefixed"):
        target_direction_id(tokenizer, "")


# ------------------------------------------------------------- workspace loading


def test_workspace_loading_matches_the_per_head_definition():
    """``cos(z_{l,h}, J_{l,h}^T row)``, head by head, against the batched form."""
    spec = _spec()
    torch.manual_seed(0)
    jz = torch.randn(spec.d_model, spec.d_source)
    z = torch.randn(spec.d_source)
    row = torch.randn(spec.d_model)

    got = workspace_loading(jz, z, spec, row)
    assert got.shape == (spec.n_heads,)
    for head in range(spec.n_heads):
        vector = head_columns(jz, head, spec).T @ row
        expected = torch.nn.functional.cosine_similarity(
            z[spec.head_slice(head)], vector, dim=0
        )
        assert float(got[head]) == pytest.approx(float(expected), abs=1e-5)


def test_workspace_loading_is_scale_free_in_z():
    """A cosine, so RMSNorm's scale invariance cannot leak in as a magnitude signal --
    which is the whole reason legibility and magnitude are kept separate."""
    spec = _spec()
    torch.manual_seed(1)
    jz = torch.randn(spec.d_model, spec.d_source)
    z = torch.randn(spec.d_source)
    row = torch.randn(spec.d_model)

    torch.testing.assert_close(
        workspace_loading(jz, z, spec, row),
        workspace_loading(jz, 7.5 * z, spec, row),
        atol=1e-5,
        rtol=1e-4,
    )




def test_workspace_loading_handles_an_oblong_geometry():
    """``n_heads * head_dim != d_model`` is the case that catches a slice taken on the
    wrong axis; Llama-3.1-8B's equality is a coincidence, not a guarantee."""
    spec = _spec(head_dim=6)
    assert spec.d_source != spec.d_model
    torch.manual_seed(4)
    jz = torch.randn(spec.d_model, spec.d_source)
    z = torch.randn(spec.d_source)
    row = torch.randn(spec.d_model)

    got = workspace_loading(jz, z, spec, row)
    assert got.shape == (spec.n_heads,)
    assert torch.isfinite(got).all()
    assert (got.abs() <= 1.0 + 1e-5).all()


def test_workspace_loading_rejects_a_mismatched_transport():
    spec = _spec()
    with pytest.raises(ValueError, match="transport is"):
        workspace_loading(
            torch.randn(spec.d_model, spec.d_source + 1),
            torch.randn(spec.d_source),
            spec,
            torch.randn(spec.d_model),
        )


def test_workspace_loading_batches_over_concepts():
    """The threshold null scores 256 concepts at every site. The batched form must
    equal the looped one exactly, or the per-site quantile is a quantile of something
    else."""
    spec = _spec()
    torch.manual_seed(6)
    jz = torch.randn(spec.d_model, spec.d_source)
    z = torch.randn(spec.d_source)
    rows = torch.randn(7, spec.d_model)

    batched = workspace_loading(jz, z, spec, rows)
    assert batched.shape == (7, spec.n_heads)
    for token in range(7):
        torch.testing.assert_close(batched[token], workspace_loading(jz, z, spec, rows[token]))



def test_workspace_loading_rejects_a_rank_three_row_stack():
    spec = _spec()
    with pytest.raises(ValueError, match=r"\[d_model\] or \[n_tokens, d_model\]"):
        workspace_loading(
            torch.randn(spec.d_model, spec.d_source),
            torch.randn(spec.d_source),
            spec,
            torch.randn(2, 2, spec.d_model),
        )


def test_workspace_loading_rejects_a_batch_of_positions():
    spec = _spec()
    with pytest.raises(ValueError, match="one position"):
        workspace_loading(
            torch.randn(spec.d_model, spec.d_source),
            torch.randn(4, spec.d_source),
            spec,
            torch.randn(spec.d_model),
        )


# ------------------------------------------------------------ per-head kurtosis


def test_excess_kurtosis_runs_per_head_over_the_vocabulary():
    """``scripts.read`` feeds it ``[n_heads, vocab]``; it must reduce the vocabulary
    axis only."""
    torch.manual_seed(5)
    logits = torch.randn(N_HEADS, 512)
    values = excess_kurtosis(logits)
    assert values.shape == (N_HEADS,)
    # Gaussian logits sit near zero excess kurtosis; a spiked head must stand out.
    spiked = logits.clone()
    spiked[0, 0] = 200.0
    assert float(excess_kurtosis(spiked)[0]) > float(values[0]) + 10


def test_excess_kurtosis_is_zero_for_gaussian():
    """Excess kurtosis is defined so a Gaussian scores 0; that fixes the -3 convention."""
    generator = torch.Generator().manual_seed(0)
    sample = torch.randn(200_000, generator=generator)
    assert abs(float(excess_kurtosis(sample))) < 0.1


def test_excess_kurtosis_is_scale_and_shift_invariant():
    """Standardised, so it compares across arms without any scale matching.

    Load-bearing for the project: RMSNorm makes every readout scale-free, and the J-head
    and Logit-head arms have wildly different natural magnitudes.
    """
    generator = torch.Generator().manual_seed(1)
    sample = torch.randn(50_000, generator=generator)
    base = float(excess_kurtosis(sample))
    assert abs(float(excess_kurtosis(sample * 37.0 + 5.0)) - base) < 1e-3


def test_excess_kurtosis_rises_with_spikiness():
    diffuse = torch.zeros(10_000)
    diffuse[:5000] = 1.0
    spiky = torch.zeros(10_000)
    spiky[0] = 100.0
    assert float(excess_kurtosis(spiky)) > float(excess_kurtosis(diffuse))


def test_excess_kurtosis_batches():
    assert excess_kurtosis(torch.randn(3, 7, 500)).shape == (3, 7)


def test_excess_kurtosis_handles_real_scale_logits():
    """Sanity at the magnitudes a 128k-vocab readout actually produces."""
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(128_256, generator=generator) * 8.0
    value = float(excess_kurtosis(logits))
    assert math.isfinite(value) and abs(value) < 0.2
