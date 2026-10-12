"""The Jacobian estimator, at any of the three anchors.

Same estimator as :func:`jlens.fitting.jacobian_for_prompt` -- for each output dimension,
inject a one-hot cotangent at *every* valid target position at once, backpropagate, and
average the resulting gradient over valid source positions::

    J_anchor,l = E_{prompts, p} [ sum_{p' >= p} d h_final[p'] / d anchor_l[p] ]

-- generalised over :mod:`src.hooks`'s three anchors, and with the source width taken
from the anchor rather than assumed to be ``d_model``. A ``zin`` source is
``d_source = n_heads * head_dim`` wide, which merely *happens* to equal ``d_model`` on
Llama-3.1-8B.

**This module is the estimator, not the fit.** :func:`jacobian_for_prompt` is what runs
per prompt and is reused byte-unchanged by :mod:`src.fit`, which adds everything a
multi-hour run needs -- accumulation, checkpointing and resume. :func:`fit` here is the
naive average over a handful of prompts: no checkpoint, no trace, useful for tests and
one-off probes.

Nothing here is specific to one anchor. Because the backward count depends only on the
*output* width, several anchors and several layers come out of one set of passes for the
price of an accumulator each -- which is what makes
:func:`src.validation.anchor_consistency` free.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import torch
from jlens.fitting import SKIP_FIRST_N_POSITIONS, valid_position_mask
from jlens.protocol import LensModel

from src.hooks import Anchor, AnchorRecorder

logger = logging.getLogger(__name__)


def _source_width(kind: str, d_model: int, d_source: int) -> int:
    """Width of an anchor's activation. Only ``zin`` differs from the residual stream."""
    return d_source if kind == "zin" else d_model


def jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    *,
    anchors: Sequence[Anchor],
    d_source: int,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[Anchor, torch.Tensor], int, int]:
    """Per-prompt Jacobian estimates at the requested anchors.

    One forward pass on the prompt replicated ``dim_batch`` times, then
    ``ceil(d_model / dim_batch)`` backward passes against the retained graph; batch
    element ``b`` carries the cotangent for output dimension ``dim_start + b``. The
    backward count depends only on the *output* width, which is why fitting several
    anchors, or several layers, costs no extra passes.

    Returns:
        ``(jacobians, seq_len, n_valid_positions)``. ``jacobians`` maps each anchor to a
        ``[d_model, width]`` fp32 CPU tensor whose row ``i`` holds
        ``d h_final[i] / d anchor[:]`` -- upstream's orientation, so
        :meth:`jlens.lens.JacobianLens.transport` applies it unchanged.
    """
    d_model = model.d_model
    target_layer = model.n_layers - 1
    anchors = sorted(set(anchors))
    if not anchors:
        raise ValueError("no anchors requested")
    for kind, layer in anchors:
        if kind == "out" and layer >= target_layer:
            raise ValueError(
                f"('out', {layer}) is the fit target, not a source; valid out layers "
                f"are 0..{target_layer - 1}"
            )
        if not 0 <= layer < model.n_layers:
            raise ValueError(f"({kind}, {layer}) out of range for {model.n_layers} blocks")

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid_positions = int(position_mask.sum())

    jacobians = {
        anchor: torch.zeros(
            d_model, _source_width(anchor[0], d_model, d_source), dtype=torch.float32
        )
        for anchor in anchors
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        AnchorRecorder(
            model.layers,
            zin_at=[l for k, l in anchors if k == "zin"],
            attn_at=[l for k, l in anchors if k == "attn"],
            out_at=[*(l for k, l in anchors if k == "out"), target_layer],
            root_graph=True,
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids.expand(dim_batch, -1))
        target_activation = recorder.activations[("out", target_layer)]
        source_activations = [recorder.activations[a] for a in anchors]

        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(
            target_activation.device
        )
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims = min(dim_batch, d_model - dim_start)
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for anchor, grad in zip(anchors, grads, strict=True):
                positions = valid_positions.to(grad.device, non_blocking=True)
                rows = grad[:n_dims, positions, :].float().mean(dim=1)
                jacobians[anchor][dim_start : dim_start + n_dims, :] = rows.cpu()
            del grads

    return jacobians, seq_len, n_valid_positions


def fit(
    model: LensModel,
    prompts: Sequence[str],
    *,
    anchors: Sequence[Anchor],
    d_source: int,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[Anchor, torch.Tensor], int]:
    """Average :func:`jacobian_for_prompt` over ``prompts``.

    Prompts too short to have any valid position (after ``skip_first``) are skipped, so
    the divisor is the number actually used.

    Returns:
        ``(jacobians, n_prompts_used)``.
    """
    totals: dict[Anchor, torch.Tensor] = {}
    n_used = 0
    for index, prompt in enumerate(prompts):
        per_prompt, seq_len, n_valid = jacobian_for_prompt(
            model,
            prompt,
            anchors=anchors,
            d_source=d_source,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
        )
        if n_valid == 0:
            logger.warning(
                "prompt %d: seq_len=%d has no valid positions after skip_first=%d",
                index,
                seq_len,
                skip_first,
            )
            continue
        for anchor, matrix in per_prompt.items():
            if anchor in totals:
                totals[anchor] += matrix
            else:
                totals[anchor] = matrix
        n_used += 1
    if n_used == 0:
        raise RuntimeError("no prompt yielded a valid position; nothing was fitted")
    return {anchor: total / n_used for anchor, total in totals.items()}, n_used
