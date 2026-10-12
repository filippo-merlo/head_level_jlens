"""Magnitude statistics and the lens-agnostic ranking, on CPU.

The batched ablation is the risky part: it zeroes a different head in each batch element,
so an indexing slip silently ablates the wrong head and every downstream number moves
without anything failing. It is checked against the obvious one-at-a-time implementation.
"""

from __future__ import annotations

import pytest
import torch

from src.config import LensSpec
from src.hooks import ZinPatcher
from src.magnitude import ablation_kl, relative_write_norms
from tests.tiny_llama import D_MODEL, N_HEADS, N_LAYERS, tiny_llama

PROMPT = "the quick brown fox jumps over the lazy dog and keeps running for a while"


def _spec() -> LensSpec:
    return LensSpec(
        key="tiny",
        hf_model_id="tiny",
        lens_file="",
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim=D_MODEL // N_HEADS,
        n_prompts_fitted=0,
    )


@pytest.fixture(scope="module")
def model():
    return tiny_llama()


def test_relative_write_norms_shape_and_range(model):
    spec = _spec()
    input_ids = model.encode(PROMPT)
    ratios = relative_write_norms(model, spec, input_ids, [0, 1, 2])
    assert set(ratios) == {0, 1, 2}
    for layer, values in ratios.items():
        assert values.shape == (spec.n_heads,), layer
        assert (values >= 0).all() and torch.isfinite(values).all()


def test_relative_write_norms_skip_the_attention_sink_prefix(model):
    """Averaging over "all valid positions" must mean the project-wide convention.

    The first ``SKIP_FIRST_N_POSITIONS`` act as attention sinks and were excluded when the
    lenses were fitted. It matters specifically for rho because Llama's sink positions
    carry anomalously large ``||x_mid,l||``, which is this ratio's *denominator* -- so
    including them deflates the number against every other readout in the project, and
    against the eval-position rho a reader will compare it to.
    """
    spec = _spec()
    input_ids = model.encode(PROMPT)
    n_positions = int(input_ids.shape[-1])
    skip = 4
    assert n_positions > skip, "prompt too short to exercise the skip"

    skipped = relative_write_norms(model, spec, input_ids, [1], skip_first=skip)[1]
    everything = relative_write_norms(model, spec, input_ids, [1], skip_first=0)[1]
    assert not torch.allclose(skipped, everything)

    by_hand = torch.stack(
        [
            relative_write_norms(model, spec, input_ids, [1], position=p)[1]
            for p in range(skip, n_positions)
        ]
    ).mean(dim=0)
    torch.testing.assert_close(skipped, by_hand, atol=1e-5, rtol=1e-4)


def test_relative_write_norms_ignore_skip_when_a_position_is_named(model):
    """``position`` names an already-chosen readout point, so the skip must not refilter."""
    spec = _spec()
    input_ids = model.encode(PROMPT)
    torch.testing.assert_close(
        relative_write_norms(model, spec, input_ids, [1], position=2, skip_first=16)[1],
        relative_write_norms(model, spec, input_ids, [1], position=2, skip_first=0)[1],
    )


def test_relative_write_norms_sum_bounds_the_stream(model):
    """Each head's ratio must be below the summed write's ratio.

    A ratio above 1 for a single head would mean it alone outweighs the stream it joins,
    which the triangle inequality forbids given the sum is what forms that stream.
    """
    spec = _spec()
    input_ids = model.encode(PROMPT)
    ratios = relative_write_norms(model, spec, input_ids, [1])[1]
    assert float(ratios.sum()) > 0
    for head in range(spec.n_heads):
        assert float(ratios[head]) < float(ratios.sum()) + 1e-6


def test_ablation_kl_matches_one_at_a_time(model):
    """The batched ablation must equal ablating each head in its own forward pass.

    This is the test that matters here: the batched version zeroes head ``h`` in batch
    element ``h``, and getting that index wrong produces plausible KLs for the wrong
    heads.
    """
    spec = _spec()
    input_ids = model.encode(PROMPT)
    layer, position = 1, -1
    batched = ablation_kl(model, spec, input_ids, [layer], position=position)[layer]

    with torch.no_grad():
        clean = model.unembed(model.forward(input_ids).last_hidden_state[0, position])
        clean_logprobs = torch.log_softmax(clean.float(), dim=-1)
        for head in range(spec.n_heads):

            def zero_this_head(z, _h=head):
                edited = z.clone()
                edited[:, :, spec.head_slice(_h)] = 0.0
                return edited

            with ZinPatcher(model.layers, {layer: zero_this_head}):
                ablated = model.forward(input_ids).last_hidden_state[0, position]
            logprobs = torch.log_softmax(model.unembed(ablated).float(), dim=-1)
            expected = float(
                (clean_logprobs.exp() * (clean_logprobs - logprobs)).sum()
            )
            assert float(batched[head]) == pytest.approx(expected, abs=1e-5), f"head {head}"


def test_ablation_kl_is_non_negative_and_nonzero(model):
    """KL is non-negative by definition, and ablating a real head must do *something*."""
    spec = _spec()
    input_ids = model.encode(PROMPT)
    values = ablation_kl(model, spec, input_ids, [0, 1])
    for layer, kl in values.items():
        assert (kl >= -1e-6).all(), layer
        assert float(kl.max()) > 1e-6, f"layer {layer}: no head changed the output"
