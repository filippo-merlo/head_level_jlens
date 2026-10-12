"""The lens-coordinate swap, on constructed frames where the answer is known.

The additive form is the whole technique, so the property that matters most is what the
swap leaves *alone*: everything orthogonal to the two concept directions.
"""

from __future__ import annotations

import pytest
import torch

from src.config import LensSpec
from src.heads import head_columns
from src.hooks import ResidualPatcher
from src.swap import (
    apply_swap,
    clamped_swap_delta,
    head_frames,
    lens_vector,
    matched_norm_delta,
    swap_coefficients,
    swap_delta,
    swap_frame,
    swap_pseudoinverse,
)
from tests.tiny_llama import D_MODEL, N_LAYERS, tiny_llama


def test_swap_exchanges_coordinates_in_an_orthonormal_frame():
    """With orthonormal lens vectors the swap is exactly a coordinate exchange."""
    d = 8
    v_s, v_t = torch.zeros(d), torch.zeros(d)
    v_s[0], v_t[1] = 1.0, 1.0
    frame = swap_frame(v_s, v_t)

    x = torch.zeros(d)
    x[0], x[1], x[5] = 3.0, 7.0, 2.0
    out = apply_swap(x, frame)

    assert out[0] == pytest.approx(7.0)
    assert out[1] == pytest.approx(3.0)
    assert out[5] == pytest.approx(2.0), "an unrelated coordinate moved"


def test_precomputed_pseudoinverse_changes_nothing():
    """Hoisting ``pinv`` out of the inner loop is a speed change only.

    ``scripts.write`` calls ``swap_delta`` once per head per layer per forward pass --
    352 times a pass -- on frames that depend only on (source, target). The
    precomputed path
    must be bit-comparable with the inline one or the optimisation is a silent edit to
    every causal number.
    """
    torch.manual_seed(7)
    d = 24
    frame = swap_frame(torch.randn(d), torch.randn(d))
    x = torch.randn(5, d)

    torch.testing.assert_close(
        swap_delta(x, frame, alpha=1.7),
        swap_delta(x, frame, alpha=1.7, pseudo=swap_pseudoinverse(frame)),
    )


def test_pseudoinverse_batches_over_heads():
    """A whole layer's ``[n_heads, d, 2]`` stack inverts in one call, and each slice
    matches the frame it came from."""
    torch.manual_seed(8)
    n_heads, d = 4, 16
    frames = torch.stack(
        [swap_frame(torch.randn(d), torch.randn(d)) for _ in range(n_heads)]
    )
    batched = swap_pseudoinverse(frames)
    assert batched.shape == (n_heads, 2, d)
    for head in range(n_heads):
        torch.testing.assert_close(batched[head], swap_pseudoinverse(frames[head]))


def test_matched_norm_control_redraws_per_seed():
    """The control asks for "a perturbation of this size *anywhere*", which needs a fresh draw per
    trial. Seeding on (layer, head) alone gave one direction reused across every trial
    whose prompt tokenised to the same length -- a sample of size one dressed as 140."""
    torch.manual_seed(9)
    delta = torch.randn(3, 16)
    first = matched_norm_delta(delta, seed=100_003 * 1 + 5)
    second = matched_norm_delta(delta, seed=100_003 * 2 + 5)
    assert not torch.allclose(first, second)
    # Same seed still reproduces, so a run stays deterministic.
    torch.testing.assert_close(first, matched_norm_delta(delta, seed=100_003 * 1 + 5))


def test_swap_preserves_the_orthogonal_complement():
    """The additive form's defining property, on a random non-orthogonal frame.

    ``x <- V sigma(c)`` would zero everything outside the two-dimensional span; this must
    not.
    """
    torch.manual_seed(0)
    d = 32
    v_s, v_t = torch.randn(d), torch.randn(d)
    frame = swap_frame(v_s, v_t)
    x = torch.randn(d)
    out = apply_swap(x, frame)

    span = torch.linalg.qr(frame)[0]                  # [d, 2] orthonormal basis
    projector = torch.eye(d) - span @ span.T
    assert torch.allclose(projector @ out, projector @ x, atol=1e-4)


def test_swap_moves_along_the_direction_difference():
    """The update is rank-one along ``v_s - v_t``, as the algebra says."""
    torch.manual_seed(1)
    d = 16
    v_s, v_t = torch.randn(d), torch.randn(d)
    frame = swap_frame(v_s, v_t)
    x = torch.randn(d)
    delta = swap_delta(x, frame)

    difference = (v_s - v_t).to(delta.dtype)
    cosine = torch.nn.functional.cosine_similarity(delta, difference, dim=0).abs()
    assert float(cosine) == pytest.approx(1.0, abs=1e-4)


def test_swap_is_idempotent_under_repetition_in_an_orthonormal_frame():
    """Swapping twice returns the original: sigma is an involution."""
    d = 8
    v_s, v_t = torch.zeros(d), torch.zeros(d)
    v_s[0], v_t[1] = 1.0, 1.0
    frame = swap_frame(v_s, v_t)
    x = torch.randn(d)
    assert torch.allclose(apply_swap(apply_swap(x, frame), frame), x, atol=1e-5)


def test_alpha_scales_the_update():
    torch.manual_seed(2)
    frame = swap_frame(torch.randn(12), torch.randn(12))
    x = torch.randn(12)
    assert torch.allclose(
        swap_delta(x, frame, alpha=2.0), 2 * swap_delta(x, frame, alpha=1.0), atol=1e-5
    )
    assert torch.allclose(swap_delta(x, frame, alpha=0.0), torch.zeros(12), atol=1e-6)


@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_clamped_swap_is_the_swap_on_the_clean_activation(alpha):
    """At the first edited layer nothing upstream has moved, so the two modes coincide."""
    torch.manual_seed(3)
    frame = swap_frame(torch.randn(20), torch.randn(20))
    pseudo = swap_pseudoinverse(frame)
    x = torch.randn(3, 5, 20)
    clean = swap_coefficients(x, pseudo)
    torch.testing.assert_close(
        clamped_swap_delta(x, frame, clean, alpha=alpha, pseudo=pseudo),
        swap_delta(x, frame, alpha=alpha, pseudo=pseudo),
    )


@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_clamped_swap_sets_the_swapped_clean_coordinates(alpha):
    """Whatever an earlier edit did to the two coordinates is overwritten with
    ``sigma_alpha(c_clean)``; the orthogonal complement of the edited activation stays."""
    torch.manual_seed(4)
    d = 24
    frame = swap_frame(torch.randn(d), torch.randn(d))
    pseudo = swap_pseudoinverse(frame)
    x_clean = torch.randn(6, d)
    x_edited = x_clean + torch.randn(6, d)
    clean = swap_coefficients(x_clean, pseudo)
    out = x_edited + clamped_swap_delta(x_edited, frame, clean, alpha=alpha, pseudo=pseudo)

    target = clean + alpha * (clean.flip(-1) - clean)
    torch.testing.assert_close(swap_coefficients(out, pseudo), target, atol=1e-4, rtol=1e-4)
    span = torch.linalg.qr(frame)[0]
    projector = torch.eye(d) - span @ span.T
    assert torch.allclose(out @ projector, x_edited @ projector, atol=1e-4)


def test_clamping_holds_under_repetition_where_recomputing_undoes_itself():
    """Two consecutive layers that carry the same coordinates: recomputing swaps back
    (``alpha=1``) or triples the gap with a sign flip (``alpha=2``); clamping holds."""
    d = 8
    v_s, v_t = torch.zeros(d), torch.zeros(d)
    v_s[0], v_t[1] = 1.0, 1.0
    frame = swap_frame(v_s, v_t)
    x = torch.zeros(d)
    x[0], x[1] = 3.0, 1.0
    clean = swap_coefficients(x, swap_pseudoinverse(frame))

    twice = apply_swap(apply_swap(x, frame), frame)
    assert twice[:2].tolist() == pytest.approx([3.0, 1.0])
    twice_strong = apply_swap(apply_swap(x, frame, alpha=2.0), frame, alpha=2.0)
    assert (twice_strong[1] - twice_strong[0]).item() == pytest.approx(9 * (1.0 - 3.0))

    once = x + clamped_swap_delta(x, frame, clean)
    held = once + clamped_swap_delta(once, frame, clean)
    assert held[:2].tolist() == pytest.approx([1.0, 3.0])


def test_swap_applies_per_position_when_batched():
    d = 8
    v_s, v_t = torch.zeros(d), torch.zeros(d)
    v_s[0], v_t[1] = 1.0, 1.0
    frame = swap_frame(v_s, v_t)
    x = torch.zeros(2, 3, d)
    x[..., 0], x[..., 1] = 1.0, 5.0
    out = apply_swap(x, frame)
    assert torch.allclose(out[..., 0], torch.full((2, 3), 5.0))
    assert torch.allclose(out[..., 1], torch.full((2, 3), 1.0))


def test_swap_rejects_a_mismatched_frame():
    with pytest.raises(ValueError, match="width"):
        swap_delta(torch.zeros(10), swap_frame(torch.zeros(8), torch.zeros(8)))
    with pytest.raises(ValueError, match="matching 1-D"):
        swap_frame(torch.zeros(8), torch.zeros(9))


def test_matched_norm_control_matches_norms_but_not_direction():
    torch.manual_seed(3)
    delta = torch.randn(4, 16)
    control = matched_norm_delta(delta, seed=0)
    assert torch.allclose(control.norm(dim=-1), delta.norm(dim=-1), atol=1e-4)
    cosine = torch.nn.functional.cosine_similarity(control, delta, dim=-1).abs()
    assert float(cosine.max()) < 0.9, "control landed on the swap direction"


def test_head_frames_match_the_per_head_construction():
    """The batched builder must equal the per-head loop exactly: every site's swap frame
    in the causal experiment comes out of it."""
    spec = LensSpec(
        key="tiny", hf_model_id="tiny", lens_file="", n_layers=4,
        d_model=12, n_heads=4, head_dim=3, n_prompts_fitted=0,
    )
    torch.manual_seed(11)
    jz = torch.randn(spec.d_model, spec.d_source)
    rows = torch.randn(2, spec.d_model)

    batched = head_frames(jz, spec, rows)
    assert batched.shape == (spec.n_heads, spec.head_dim, 2)
    for head in range(spec.n_heads):
        j_head = head_columns(jz, head, spec)
        v = rows.to(j_head.dtype) @ j_head
        torch.testing.assert_close(batched[head], swap_frame(v[0], v[1]))


def test_head_frames_rejects_a_mismatched_row_stack():
    spec = LensSpec(
        key="tiny", hf_model_id="tiny", lens_file="", n_layers=4,
        d_model=12, n_heads=4, head_dim=3, n_prompts_fitted=0,
    )
    with pytest.raises(ValueError, match="rows are"):
        head_frames(torch.randn(12, 12), spec, torch.randn(3, 12))


def test_lens_vector_pulls_back_through_the_jacobian():
    torch.manual_seed(4)
    j = torch.randn(16, 8)          # [d_model, d_source]
    row = torch.randn(16)
    v = lens_vector(j, row)
    assert v.shape == (8,)
    # Reading z along v equals reading J z along the unembedding row.
    z = torch.randn(8)
    assert float(v @ z) == pytest.approx(float(row @ (j @ z)), rel=1e-4)


# ---------------------------------------------------------------- residual patching


def test_residual_patcher_changes_the_output():
    model = tiny_llama()
    input_ids = model.encode("hello world this is a test of the patcher")
    with torch.no_grad():
        clean = model.forward(input_ids).last_hidden_state.clone()
        with ResidualPatcher(model.layers, {1: lambda x: x + 1.0}):
            patched = model.forward(input_ids).last_hidden_state
        assert not torch.allclose(clean, patched)

        # and a no-op edit must leave the forward pass untouched
        with ResidualPatcher(model.layers, {1: lambda x: x}):
            identity = model.forward(input_ids).last_hidden_state
        assert torch.allclose(clean, identity, atol=1e-6)


def test_residual_patcher_removes_its_hooks():
    model = tiny_llama()
    input_ids = model.encode("hello world")
    with torch.no_grad():
        with ResidualPatcher(model.layers, {0: lambda x: x * 5.0}):
            pass
        after = model.forward(input_ids).last_hidden_state
        clean = model.forward(input_ids).last_hidden_state
        assert torch.allclose(after, clean)


def test_residual_patcher_rejects_a_shape_change():
    model = tiny_llama()
    input_ids = model.encode("hello world")
    with torch.no_grad(), pytest.raises(ValueError, match="expected"):
        with ResidualPatcher(model.layers, {0: lambda x: x[..., :-1]}):
            model.forward(input_ids)


def test_residual_patcher_layer_bounds():
    model = tiny_llama()
    with pytest.raises(ValueError, match="out of range"):
        ResidualPatcher(model.layers, {N_LAYERS: lambda x: x})


def test_swap_through_the_patcher_preserves_unrelated_structure():
    """End to end: a swap at a real block output changes the run without breaking it."""
    model = tiny_llama()
    input_ids = model.encode("hello world this is a test of the patcher")
    torch.manual_seed(5)
    frame = swap_frame(torch.randn(D_MODEL), torch.randn(D_MODEL))

    with torch.no_grad():
        clean = model.forward(input_ids).last_hidden_state
        with ResidualPatcher(model.layers, {1: lambda x: apply_swap(x, frame).to(x.dtype)}):
            swapped = model.forward(input_ids).last_hidden_state
    assert not torch.allclose(clean, swapped)
    assert torch.isfinite(swapped).all()
