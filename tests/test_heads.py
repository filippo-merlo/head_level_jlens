"""The per-head algebra, checked exactly on a real (tiny) Llama in fp32.

Everything the per-head decomposition rests on is checked here, on CPU, in seconds:

* **Additivity** -- ``sum_h c_{l,h}`` is ``o_proj``'s output, so a
  head's write is a genuine term of a sum rather than an attribution estimate.
* **Anchor consistency** -- ``J^z = J_attn @ W_O``. Nothing
  depends on this identity any more, since ``J^z`` is fitted at its own anchor; it is
  kept because it is the sharpest check available on the fit, being exact rather than
  convergence-limited. Checked three independent ways: against the shared estimator,
  against raw autograd at a single position pair, and against central finite
  differences, which involve no autograd at all.

The fixture is a genuine ``LlamaForCausalLM`` -- real RMSNorm, real RoPE, real GQA, real
``o_proj`` -- so these exercise the production hook points rather than a stand-in.
"""

from __future__ import annotations

import pytest
import torch

from src.config import LensSpec
from src.heads import (
    COMPUTE_DTYPE,
    attention_write,
    effective_unembed,
    head_columns,
    head_contribution,
    head_lens_vectors,
    o_proj_weight,
)
from src.hooks import AnchorRecorder, ZinPatcher
from src.jacobian import jacobian_for_prompt
from src.validation import (
    additivity_error,
    anchor_consistency,
    anchor_consistency_controls,
)
from tests.tiny_llama import (
    D_MODEL,
    N_HEADS,
    N_KV_HEADS,
    N_LAYERS,
    VOCAB_SIZE,
    tiny_llama,
)

PROMPT = "the quick brown fox jumps over the lazy dog and keeps running for a while"


def _spec(head_dim: int) -> LensSpec:
    """A :class:`LensSpec` describing the fixture model."""
    return LensSpec(
        key="tiny",
        hf_model_id="tiny",
        lens_file="",
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim=head_dim,
        n_prompts_fitted=0,
    )


@pytest.fixture(scope="module")
def square_model():
    """``n_heads * head_dim == d_model``, as on Llama-3.1-8B."""
    return tiny_llama()


@pytest.fixture(scope="module")
def oblong_model():
    """``d_source != d_model``.

    Llama-3.1-8B's ``32 * 128 == 4096`` is a coincidence of that architecture, and a
    fixture that shares it cannot catch code which conflates the two widths. This one
    has ``4 * 6 = 24`` against ``d_model = 32``.
    """
    return tiny_llama(head_dim=6)


def _models(square_model, oblong_model):
    return [(square_model, _spec(D_MODEL // N_HEADS)), (oblong_model, _spec(6))]


# --------------------------------------------------------------------------- shapes


def test_fixture_widths(square_model, oblong_model):
    """The two fixtures really do differ in ``d_source``, or the suite proves nothing."""
    assert _spec(D_MODEL // N_HEADS).d_source == D_MODEL
    assert _spec(6).d_source == 24 != D_MODEL
    for model, spec in _models(square_model, oblong_model):
        assert o_proj_weight(model, 0).shape == (spec.d_model, spec.d_source)


def test_head_columns_rejects_wrong_axis(square_model):
    """A matrix indexed by ``d_model`` on its column axis is not head-sliceable."""
    spec = _spec(6)  # d_source = 24, but the matrix below has 32 columns
    with pytest.raises(ValueError, match="d_source"):
        head_columns(torch.zeros(D_MODEL, D_MODEL), 0, spec)


def test_head_slice_bounds():
    spec = _spec(8)
    assert spec.head_slice(0) == slice(0, 8)
    assert spec.head_slice(N_HEADS - 1) == slice(8 * (N_HEADS - 1), 8 * N_HEADS)
    with pytest.raises(ValueError, match="out of range"):
        spec.head_slice(N_HEADS)


# ------------------------------------------------------------------------- additivity


def test_additivity(square_model, oblong_model):
    """``sum_h c_{l,h}`` equals the layer's ``o_proj`` output, exactly.

    Elhage et al.'s "attention heads are independent and additive". Exact in fp32 up to
    summation order, which is why the tolerance is float-epsilon rather than a fudge.
    """
    for model, spec in _models(square_model, oblong_model):
        errors = additivity_error(model, spec, PROMPT)
        assert set(errors) == set(range(spec.n_layers))
        worst = max(errors.values())
        assert worst < 1e-5, f"d_source={spec.d_source}: additivity off by {worst:.2e}"


def test_head_contribution_upcasts(square_model):
    """The per-head algebra runs in fp32 even when the activations are lower precision.

    On the real bf16 checkpoint, recomposing in bf16 leaves a ~2e-2 residual against
    ``o_proj``'s own output while fp32 leaves ~1e-6 on the identical activations
    (recorded in ``results/fit_checks/``, a legacy artifact; :mod:`scripts.fit_lens`
    records the pair itself now). That is bf16's mantissa, not a bug -- but carrying
    2% error in every head's write would corrupt every readout downstream, so
    :data:`src.heads.COMPUTE_DTYPE` governs rather than the activation's dtype.
    """
    spec = _spec(D_MODEL // N_HEADS)
    model = square_model
    input_ids = model.encode(PROMPT)
    with torch.no_grad(), AnchorRecorder(model.layers, zin_at=[1]) as recorder:
        model.forward(input_ids)
    z = recorder.activations[("zin", 1)].to(torch.bfloat16)
    w_o = o_proj_weight(model, 1).to(torch.bfloat16)

    assert head_contribution(z, w_o, 0, spec).dtype is COMPUTE_DTYPE
    assert attention_write(z, w_o).dtype is COMPUTE_DTYPE
    # and the escape hatch still measures what the model itself represents
    assert head_contribution(z, w_o, 0, spec, dtype=torch.bfloat16).dtype is torch.bfloat16

    # Upcasting is not cosmetic: it must actually reduce the recomposition residual.
    def residual(dtype):
        total = sum(
            head_contribution(z, w_o, h, spec, dtype=dtype) for h in range(spec.n_heads)
        )
        reference = attention_write(z, w_o, dtype=dtype)
        return float((total - reference).abs().max() / reference.abs().max())

    assert residual(torch.float32) < residual(torch.bfloat16) / 100


def test_additivity_detects_a_wrong_width(square_model):
    """A ``spec`` whose ``d_source`` contradicts the model is caught, not averaged over.

    The failure this guards is silent: with the wrong ``head_dim`` the head slices still
    index legally, still sum to something, and still look like a residual-stream vector.
    """
    wrong = _spec(6)  # the square fixture has head_dim 8, so d_source is 32, not 24
    with pytest.raises(ValueError, match="d_source"):
        additivity_error(square_model, wrong, PROMPT)


def test_head_contribution_is_a_true_restriction(square_model):
    """Zeroing every head but ``h`` reproduces ``c_{l,h}`` through the real ``o_proj``.

    Guards the column-block convention: slicing the *wrong* axis of ``W_O`` would still
    produce a plausible ``[.., d_model]`` vector on the square fixture.
    """
    spec = _spec(D_MODEL // N_HEADS)
    model, layer, head = square_model, 1, 2
    input_ids = model.encode(PROMPT)
    with torch.no_grad(), AnchorRecorder(model.layers, zin_at=[layer]) as recorder:
        model.forward(input_ids)
    z = recorder.activations[("zin", layer)]

    masked = torch.zeros_like(z)
    masked[..., spec.head_slice(head)] = z[..., spec.head_slice(head)]
    through_o_proj = o_proj_weight(model, layer).new_tensor(0) + (
        masked @ o_proj_weight(model, layer).T
    )
    assert torch.allclose(
        through_o_proj, head_contribution(z, o_proj_weight(model, layer), head, spec),
        atol=1e-6,
    )


# ----------------------------------------------------------- anchor consistency


def test_anchor_consistency_via_estimator(square_model, oblong_model):
    """``J^z == J_attn @ W_O`` on estimates from the *same* backward passes.

    Both anchors are read out of one set of backward passes, so the two estimates saw
    identical prompts, positions and cotangents. Any residual difference is the identity
    failing, not estimator noise -- hence the tight tolerance.

    Runs through :func:`src.fit.fit`, the loop that produces the shipped lens, so it
    covers the production accumulation path rather than a second implementation of it.
    """
    for model, spec in _models(square_model, oblong_model):
        layers = list(range(spec.n_layers))
        result = anchor_consistency(
            model, spec, [PROMPT], layers=layers, dim_batch=4, skip_first=2
        )
        assert result["n_prompts"] == 1
        for layer in layers:
            error = result["relative"][layer]
            assert error < 1e-5, (
                f"d_source={spec.d_source} layer {layer}: anchors disagree by "
                f"{error:.2e}"
            )
            assert result["row_cosine_min"][layer] > 1 - 1e-6


def test_anchor_consistency_controls_discriminate(square_model):
    """The correct transport must beat every wrong one by orders of magnitude.

    Without this, the tolerance in the test above is arbitrary: it would pass equally on
    a comparison that cannot fail. Each control is a transport that could be written by
    accident, and each must score at order unity while the truth sits at roundoff.
    """
    spec = _spec(D_MODEL // N_HEADS)
    scores = anchor_consistency_controls(
        square_model, spec, PROMPT, layer=1, dim_batch=4, skip_first=2
    )
    assert scores["_J_z_norm"] > 1e-3, "degenerate fit: both sides are ~zero"

    correct = scores["correct__J_attn_W_O"]
    assert correct < 1e-5, f"correct transport scored {correct:.2e}"
    for name, value in scores.items():
        if name.startswith("wrong__"):
            assert value > 0.1, f"{name} scored {value:.2e}; the check has no power"
            assert value > 1000 * correct, f"{name} is only {value / correct:.0f}x worse"


def test_anchor_consistency_raw_autograd(square_model):
    """The same identity from bare ``autograd.grad``, independent of :mod:`src.jacobian`.

    A bug in the shared estimator would cancel between the two anchors in the test
    above. Here the Jacobians are built one output dimension at a time at a single
    ``(source, target)`` position pair, using nothing from this project but the hooks.
    """
    spec = _spec(D_MODEL // N_HEADS)
    model, layer, source_pos, target_pos = square_model, 1, 5, 9
    input_ids = model.encode(PROMPT)

    with AnchorRecorder(
        model.layers,
        zin_at=[layer],
        attn_at=[layer],
        out_at=[N_LAYERS - 1],
        root_graph=True,
    ) as recorder, torch.enable_grad():
        model.forward(input_ids)
        target = recorder.activations[("out", N_LAYERS - 1)]
        z = recorder.activations[("zin", layer)]
        x_mid = recorder.activations[("attn", layer)]

        j_z = torch.zeros(D_MODEL, spec.d_source)
        j_attn = torch.zeros(D_MODEL, D_MODEL)
        for dim in range(D_MODEL):
            cotangent = torch.zeros_like(target)
            cotangent[0, target_pos, dim] = 1.0
            grad_z, grad_mid = torch.autograd.grad(
                target, [z, x_mid], grad_outputs=cotangent, retain_graph=True
            )
            j_z[dim] = grad_z[0, source_pos]
            j_attn[dim] = grad_mid[0, source_pos]

    predicted = j_attn @ o_proj_weight(model, layer).float()
    error = (j_z - predicted).norm() / predicted.norm()
    assert error < 1e-5, f"raw-autograd anchor consistency off by {error:.2e}"
    # Guard against the degenerate pass where both sides are ~0 and agree trivially.
    assert j_z.norm() > 1e-3


def test_finite_difference_grad_check(square_model):
    """Finite-difference check: perturb ``z_{l,h}`` and watch ``h_final`` move.

    Involves no autograd at all, so it validates the hook wiring against the model
    itself. A sampled grid of ``(head, head-dim, output-dim)`` entries against central
    differences, with the tolerance stated rather than tuned.
    """
    spec = _spec(D_MODEL // N_HEADS)
    model, layer, source_pos, target_pos = square_model, 1, 5, 9
    eps = 1e-3
    input_ids = model.encode(PROMPT)

    def final_residual(delta: torch.Tensor | None) -> torch.Tensor:
        edits = {} if delta is None else {layer: lambda z: z + delta}
        with torch.no_grad(), ZinPatcher(model.layers, edits), AnchorRecorder(
            model.layers, out_at=[N_LAYERS - 1]
        ) as recorder:
            model.forward(input_ids)
        return recorder.activations[("out", N_LAYERS - 1)][0, target_pos].clone()

    with AnchorRecorder(
        model.layers, zin_at=[layer], out_at=[N_LAYERS - 1], root_graph=True
    ) as recorder, torch.enable_grad():
        model.forward(input_ids)
        target = recorder.activations[("out", N_LAYERS - 1)]
        z = recorder.activations[("zin", layer)]
        analytic = torch.zeros(D_MODEL, spec.d_source)
        for dim in range(D_MODEL):
            cotangent = torch.zeros_like(target)
            cotangent[0, target_pos, dim] = 1.0
            (grad_z,) = torch.autograd.grad(
                target, [z], grad_outputs=cotangent, retain_graph=True
            )
            analytic[dim] = grad_z[0, source_pos]

    failures = []
    for head in range(N_HEADS):
        for within_head in (0, spec.head_dim // 2):
            column = spec.head_dim * head + within_head
            delta = torch.zeros(1, input_ids.shape[1], spec.d_source)
            delta[0, source_pos, column] = eps
            plus = final_residual(delta)
            minus = final_residual(-delta)
            numeric = (plus - minus) / (2 * eps)
            for out_dim in (0, D_MODEL // 2, D_MODEL - 1):
                a = float(analytic[out_dim, column])
                n = float(numeric[out_dim])
                if abs(a - n) > 1e-3 * max(1.0, abs(a)):
                    failures.append((head, within_head, out_dim, a, n))
    assert not failures, f"finite-difference mismatches (analytic vs numeric): {failures}"


def test_readout_arms_read_the_same_vector(square_model):
    """``J_{l,h} z_{l,h}`` equals ``J_attn c_{l,h}`` -- the third consequence.

    The two ways of writing the J-head arm must agree, or the claim that the arms
    "differ by exactly one matrix on the same vector" is not what the code computes.
    """
    spec = _spec(D_MODEL // N_HEADS)
    model, layer = square_model, 1
    jacobians, _, _ = jacobian_for_prompt(
        model,
        PROMPT,
        anchors=[("zin", layer), ("attn", layer)],
        d_source=spec.d_source,
        dim_batch=4,
        skip_first=2,
    )
    j_z, j_attn = jacobians[("zin", layer)], jacobians[("attn", layer)]

    input_ids = model.encode(PROMPT)
    with torch.no_grad(), AnchorRecorder(model.layers, zin_at=[layer]) as recorder:
        model.forward(input_ids)
    z = recorder.activations[("zin", layer)].float()
    w_o = o_proj_weight(model, layer).float()

    for head in range(N_HEADS):
        via_head_space = z[..., spec.head_slice(head)] @ head_columns(j_z, head, spec).T
        via_residual = head_contribution(z, w_o, head, spec) @ j_attn.T
        assert torch.allclose(via_head_space, via_residual, atol=1e-4), f"head {head}"


# ----------------------------------------------------------------------- lens vectors


def test_effective_unembed_reproduces_model_logits(square_model):
    """``(w_t * g) . x / rms(x)`` is what ``model.unembed`` computes.

    Folding the final norm's learned gain into the unembedding rows is what makes a lens
    *vector* the direction a token actually reads along. Getting this wrong would leave
    every workspace-loading cosine and every swap direction subtly off, while still
    producing plausible numbers.
    """
    model = square_model
    x = torch.randn(4, D_MODEL)
    reference = model.unembed(x)

    rows = effective_unembed(model)
    assert rows.shape == (VOCAB_SIZE, D_MODEL)
    rms = x.float().pow(2).mean(-1, keepdim=True).add(model._final_norm.variance_epsilon).sqrt()
    predicted = (x.float() / rms) @ rows.float().T
    assert torch.allclose(predicted, reference.float(), atol=1e-4)


def test_head_lens_vectors_shape_and_pullback(square_model, oblong_model):
    """``v_{t,l,h} = J_{l,h}^T (w_t * g)`` -- a pullback into head-native coordinates."""
    for model, spec in _models(square_model, oblong_model):
        j_z = torch.randn(spec.d_model, spec.d_source)
        token_ids = torch.tensor([1, 7, 13])
        rows = effective_unembed(model, token_ids)
        vectors = head_lens_vectors(j_z, 2, spec, rows)
        assert vectors.shape == (len(token_ids), spec.head_dim)

        # The defining property: reading z along v_t equals reading J z along (w_t * g).
        z_head = torch.randn(spec.head_dim)
        along_head = vectors @ z_head
        along_residual = rows.float() @ (head_columns(j_z, 2, spec) @ z_head)
        assert torch.allclose(along_head, along_residual, atol=1e-4)


def test_gqa_does_not_break_the_decomposition(square_model):
    """Heads sharing a ``W_V`` group still decompose additively (the GQA note).

    The decomposition is a statement about ``W_O`` alone, so KV sharing is irrelevant to
    it -- but the fixture has real GQA (4 query heads over 2 KV heads), so assert it.
    """
    assert square_model.hf_model.config.num_key_value_heads == N_KV_HEADS
    assert N_HEADS % N_KV_HEADS == 0 and N_HEADS != N_KV_HEADS


# ------------------------------------------------- the W_O gate's counterfactual arm


def test_attn_from_jz_inverts_the_anchor_identity():
    """``J^z = J_attn W_O`` is exactly undone by the solve, so the recovery is not a fit."""
    from src.validation import attn_from_jz

    torch.manual_seed(0)
    w_o = torch.randn(D_MODEL, D_MODEL)
    j_attn = torch.randn(D_MODEL, D_MODEL)
    recovered = attn_from_jz(j_attn @ w_o, w_o)
    error = (recovered - j_attn).norm() / j_attn.norm()
    assert error < 1e-4, f"recovery off by {error:.2e}"


def test_attn_from_jz_refuses_a_non_square_w_o():
    """With ``d_source != d_model`` there is no inverse, and a wrong answer is worse."""
    from src.validation import attn_from_jz

    with pytest.raises(ValueError, match="square"):
        attn_from_jz(torch.randn(D_MODEL, 24), torch.randn(D_MODEL, 24))


def test_wo_gate_counterfactual_shows_the_two_numbers_swapping():
    """The claim S1 makes in prose, as an assertion: the arms trade places.

    Built so the truth is known -- ``J^z`` is *defined* as ``W_O`` here, which is what
    the top block approaches -- and then the recovered ``attn`` matrix must look like
    the identity instead. Without this, "the numbers would swap" is untested.
    """
    from src.validation import wo_gate_counterfactual

    torch.manual_seed(0)
    w_o = torch.randn(D_MODEL, D_MODEL)
    scores = wo_gate_counterfactual(w_o.clone(), w_o)

    assert scores["jz_median_row_cosine_vs_W_O"] > 0.99
    assert abs(scores["jz_median_row_cosine_vs_identity"]) < 0.2
    # J_attn = J^z W_O^-1 = I, so the arms trade places.
    assert scores["attn_median_row_cosine_vs_identity"] > 0.99
    assert abs(scores["attn_median_row_cosine_vs_W_O"]) < 0.2
    assert scores["roundtrip_relative_residual"] < 1e-4
