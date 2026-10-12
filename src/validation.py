"""Measurements that check the method's structural claims, as numbers.

Separated from both the tests that assert on them and the scripts that record them, so
one implementation produces the number in either context -- a claim in the writeup and
the assertion guarding it can never drift apart.

Three claims live here, and the project's design rests on all three:

* **Additivity**: ``sum_h c_{l,h}`` is the layer's attention write. Without it, a
  per-head readout is an attribution estimate rather than a term of a sum.
* **Anchor consistency**: the ``zin``-anchored fit agrees with the
  ``attn``-anchored fit times ``W_O``. Both estimates come out of the *same* backward
  passes, so this is an exact identity about the estimator rather than a claim about
  convergence -- which is what licenses a tight tolerance on it. It presumes a linear
  path from ``o_proj`` to the residual add, so it is not defined on Gemma-3's blocks.
* **The two-sided ``W_O`` gate**: at the top layer ``J^z`` must look like ``W_O``
  and *not* like the identity. It is the check that catches the one wiring mistake with
  no other symptom -- a ``zin`` hook that slipped to ``o_proj``'s output. The identity
  arm exists only when ``W_O`` is square; the ``W_O`` reference is
  :func:`src.heads.residual_o_proj_weight`, which is ``W_O`` itself on a pre-norm block.

:func:`anchor_consistency_controls` is the reason to believe the second check means
anything: it scores the same comparison against deliberately wrong transports. A test
that cannot fail is not evidence, and on Llama-3.1-8B every quantity in sight is a
4096x4096 matrix of plausible-looking numbers.
"""

from __future__ import annotations

import logging

import torch

from src.config import LensSpec
from src.heads import attention_write, head_contribution, o_proj_weight
from src.hooks import AnchorRecorder
from src.jacobian import jacobian_for_prompt

logger = logging.getLogger(__name__)

#: Gate thresholds. The gap between them is the whole point: on the current estimate the
#: two numbers are 0.867 and -0.0002, and at the ``attn`` anchor they would be ~0 and
#: 0.904. Anything in between means neither hook point is being read cleanly.
WO_GATE_MIN = 0.80
IDENTITY_GATE_MAX = 0.10


def _relative(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    return float((estimate - reference).norm() / reference.norm())


def additivity_error(
    model,
    spec: LensSpec,
    prompt: str,
    *,
    layers: list[int] | None = None,
    dtype: torch.dtype | None = None,
) -> dict[int, float]:
    """Max relative error of ``sum_h c_{l,h}`` against ``o_proj``'s output, per layer.

    Reported as a max over entries rather than a norm ratio: additivity is an exact
    identity, so the interesting quantity is the worst single element, not an average
    that a few good entries could carry.

    Args:
        dtype: Precision to recompose at. ``None`` uses
            :data:`~src.heads.COMPUTE_DTYPE` (fp32), which is what the project actually
            computes in and therefore what tests the *wiring*. Passing the activation's
            own dtype instead measures the model's representational floor -- on a bf16
            checkpoint that is ~2e-2 and says nothing about correctness.
    """
    layers = spec.jz_source_layers if layers is None else layers
    input_ids = model.encode(prompt)
    with torch.no_grad(), AnchorRecorder(model.layers, zin_at=layers) as recorder:
        model.forward(input_ids)

    errors: dict[int, float] = {}
    for layer in layers:
        z = recorder.activations[("zin", layer)]
        if z.shape[-1] != spec.d_source:
            raise ValueError(
                f"layer {layer}: o_proj input is {z.shape[-1]} wide, but spec says "
                f"d_source={spec.d_source} (n_heads={spec.n_heads} x "
                f"head_dim={spec.head_dim})"
            )
        w_o = o_proj_weight(model, layer)
        kwargs = {} if dtype is None else {"dtype": dtype}
        total = sum(
            head_contribution(z, w_o, h, spec, **kwargs) for h in range(spec.n_heads)
        )
        reference = attention_write(z, w_o, **kwargs)
        errors[layer] = float(
            (total - reference).abs().max() / reference.abs().max()
        )
    return errors


def attn_from_jz(jz_layer: torch.Tensor, w_o_layer: torch.Tensor) -> torch.Tensor:
    """Recover ``J_attn_l`` from a fitted ``J^z_l``, without refitting anything.

    The anchor identity ``J^z_l = J_attn_l @ W_O`` is exact, and ``W_O`` is square
    whenever ``d_source == d_model``, so ``J_attn_l = J^z_l @ W_O^{-1}``. Solved rather
    than inverted -- ``W_O^T X^T = (J^z)^T`` -- which is better conditioned and avoids
    forming an explicit inverse.

    Raises:
        ValueError: If ``W_O`` is not square. The recovery has no meaning then: ``W_O``
            maps a wider or narrower source space and cannot be undone.
    """
    d_model, d_source = w_o_layer.shape
    if d_model != d_source:
        raise ValueError(
            f"W_O is {d_model}x{d_source}; recovering J_attn needs a square W_O "
            f"(d_source == d_model)"
        )
    return torch.linalg.solve(w_o_layer.float().T, jz_layer.float().T).T


def wo_gate_counterfactual(
    jz_layer: torch.Tensor, w_o_layer: torch.Tensor
) -> dict[str, float]:
    """Score *both* arms of the two-sided gate, on ``J^z`` and on the ``attn`` anchor.

    :func:`wo_gate` scores only the fitted ``J^z``, so the claim that a slipped hook
    would **swap** the two numbers has to be taken on trust. Recovering ``J_attn`` with
    :func:`attn_from_jz` supplies the missing arm from the shipped lens itself: at the
    top block ``J^z`` should sit near ``W_O`` and at ~0 against the identity, while
    ``J_attn`` does the reverse.

    ``roundtrip_relative_residual`` guards the recovery: it re-applies ``W_O`` and
    compares with the input, so an ill-conditioned solve shows up as a number rather
    than as a plausible-looking cosine.
    """
    from src.metrics import cosine_rows

    j_attn = attn_from_jz(jz_layer, w_o_layer)
    j_z = jz_layer.float()
    w_o = w_o_layer.float()
    eye = torch.eye(w_o.shape[0], device=j_z.device, dtype=j_z.dtype)
    return {
        "attn_median_row_cosine_vs_identity": float(cosine_rows(j_attn, eye).median()),
        "attn_median_row_cosine_vs_W_O": float(cosine_rows(j_attn, w_o).median()),
        "jz_median_row_cosine_vs_identity": float(cosine_rows(j_z, eye).median()),
        "jz_median_row_cosine_vs_W_O": float(cosine_rows(j_z, w_o).median()),
        "roundtrip_relative_residual": float(
            (j_attn @ w_o - j_z).norm() / j_z.norm()
        ),
    }


def wo_gate(
    jz_top: torch.Tensor, w_o_top: torch.Tensor, *, reference: str = "W_O"
) -> dict[str, object]:
    """The two-sided gate: ``J^z`` at the top layer against ``W_O`` and the identity.

    At the final block almost no network remains between ``z_l`` and the target, so
    ``d h_final / d z_31 ~= W_O^(31)``. Scored by **row cosine**, not relative Frobenius:
    the Frobenius distance to ``W_O`` is 0.711 even when the fit is right, so it admits
    no tight bound, while orientation is both sharp here and the thing a softmax keeps.

    ``w_o_top`` is whatever matrix the top block actually adds to the stream per unit of
    ``z``: ``W_O`` on a pre-norm block, ``diag(g) W_O`` under Gemma-3's post-attention
    norm (:func:`src.heads.residual_o_proj_weight`); ``reference`` names it in the record.

    The identity arm is what makes it a test. Had the ``zin`` pre-hook slipped to
    ``o_proj``'s *output*, the fitted object would be the ``attn``-anchored one, whose
    top-layer row cosine against the identity is 0.904 and against ``W_O`` is ~0: the two
    numbers swap, so no single-sided threshold could tell the cases apart. This function
    scores only ``J^z``; :func:`wo_gate_counterfactual` scores the other arm off the same
    shipped lens, and ``scripts/check_wo_counterfactual.py`` records it.
    """
    from src.metrics import cosine_rows

    vs_w_o = float(cosine_rows(jz_top, w_o_top).median())
    vs_identity = None
    if jz_top.shape[0] == jz_top.shape[1]:
        vs_identity = float(cosine_rows(jz_top, torch.eye(jz_top.shape[0])).median())
    return {
        "reference": reference,
        "median_row_cosine_vs_W_O": vs_w_o,
        "median_row_cosine_vs_identity": vs_identity,
        "min_vs_W_O": WO_GATE_MIN,
        "max_vs_identity": IDENTITY_GATE_MAX,
        "PASS": vs_w_o > WO_GATE_MIN
        and (vs_identity is None or vs_identity < IDENTITY_GATE_MAX),
    }


def anchor_consistency(
    model,
    spec: LensSpec,
    prompts: list[str],
    *,
    layers: list[int],
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int | None = None,
) -> dict[str, dict]:
    """Fit both sub-layer anchors together, then score ``zin`` against ``attn @ W_O``.

    ``o_proj`` is linear with constant weights and its output reaches the final residual
    only through ``x_mid,l``, so the chain rule makes the two anchors' estimates equal by
    construction. Both come out of the *same* backward passes -- identical prompts,
    positions and cotangents -- so any difference is the identity failing rather than
    estimator noise, which is what makes a tight tolerance meaningful here.

    It runs through :func:`src.fit.fit`, the same loop that produces the shipped lens, so
    it exercises the production accumulation path and not a second implementation of it.
    Asking for the second anchor costs no extra backward passes, only one 67 MB
    accumulator per layer.

    Returns:
        ``{"relative": {layer: err}, "row_cosine_min": {layer: c}, "n_prompts": n}``.
        Row cosine is reported beside relative error because orientation is what survives
        a softmax.
    """
    from src.fit import fit

    kwargs = {} if skip_first is None else {"skip_first": skip_first}
    result = fit(
        model,
        prompts,
        anchors=[("zin", l) for l in layers] + [("attn", l) for l in layers],
        d_source=spec.d_source,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        checkpoint_every=None,
        **kwargs,
    )
    return score_anchor_consistency(result.jacobians, model, layers=layers,
                                    n_prompts=result.n_done)


def score_anchor_consistency(
    jacobians: dict[tuple, torch.Tensor], model, *, layers: list[int], n_prompts: int
) -> dict[str, dict]:
    """Score an already-fitted pair of anchors, so one fit can serve several checks.

    Split out from :func:`anchor_consistency` so a caller that has fitted more anchors
    than these two -- they all come out of the *same* backward passes -- can score this
    identity without refitting.
    """
    relative: dict[int, float] = {}
    row_cosine: dict[int, float] = {}
    for layer in layers:
        j_z = jacobians[("zin", layer)]
        j_attn = jacobians[("attn", layer)]
        predicted = j_attn @ o_proj_weight(model, layer).float().cpu()
        relative[layer] = _relative(j_z, predicted)
        row_cosine[layer] = float(
            torch.nn.functional.cosine_similarity(j_z, predicted, dim=-1).min()
        )
    return {"relative": relative, "row_cosine_min": row_cosine, "n_prompts": n_prompts}


def anchor_consistency_controls(
    model, spec: LensSpec, prompt: str, *, layer: int, dim_batch: int = 8,
    skip_first: int | None = None,
) -> dict[str, float]:
    """Score ``J^z`` against the correct transport and against wrong ones.

    Every control is a transport someone could plausibly write by accident: the wrong
    ``W_O`` orientation, a neighbouring layer's weights, no transport at all. The
    correct identity should sit at float-roundoff while all of these sit at order unity;
    if they do not, the check has no power and the tolerance is arbitrary.

    Controls whose shapes do not even line up at this model's geometry are reported
    under ``_unreachable`` rather than scored -- when ``d_source != d_model`` several of
    these mistakes raise instead of computing, which is the point of keeping the two
    widths distinct.
    """
    kwargs = {} if skip_first is None else {"skip_first": skip_first}
    # The nearest other layer that has a head anchor: on a hybrid the neighbour block
    # may have no o_proj at all.
    other_layer = min(
        (l for l in spec.jz_source_layers if l != layer), key=lambda l: abs(l - layer)
    )
    anchors = [("zin", layer), ("attn", layer), ("attn", other_layer)]
    jacobians, _, _ = jacobian_for_prompt(
        model, prompt, anchors=anchors, d_source=spec.d_source, dim_batch=dim_batch,
        **kwargs,
    )
    j_z = jacobians[("zin", layer)]
    j_attn = jacobians[("attn", layer)]
    w_o = o_proj_weight(model, layer).float().cpu()

    candidates = {
        "correct__J_attn_W_O": lambda: j_attn @ w_o,
        "wrong__W_O_transposed": lambda: j_attn @ w_o.T,
        "wrong__other_layer_W_O": lambda: j_attn
        @ o_proj_weight(model, other_layer).float().cpu(),
        "wrong__other_layer_J_attn": lambda: jacobians[("attn", other_layer)] @ w_o,
        "wrong__no_transport": lambda: w_o,
        "wrong__identity_transport": lambda: j_attn,
    }

    scores: dict[str, float] = {}
    unreachable: list[str] = []
    for name, build in candidates.items():
        try:
            matrix = build()
        except RuntimeError:
            # Shape-incompatible, so this particular mistake cannot be made silently at
            # this geometry -- which is the whole argument for keeping d_source distinct
            # from d_model. Recorded, not scored.
            unreachable.append(name)
            continue
        if matrix.shape != j_z.shape:
            unreachable.append(name)
            continue
        scores[name] = _relative(j_z, matrix)

    if "correct__J_attn_W_O" not in scores:
        raise RuntimeError("the correct transport did not even produce the right shape")
    scores["_J_z_norm"] = float(j_z.norm())
    scores["_unreachable"] = unreachable
    return scores
