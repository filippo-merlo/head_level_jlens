"""The lens-coordinate swap: exchanging two concepts inside an activation.

The paper's intervention (Methods, Fig. 4C). Given a source token ``s``, a target token
``t`` and their lens vectors, form ``V = [v_s  v_t]``, read the activation's coordinates
in that two-dimensional frame with the pseudoinverse, and swap them::

    c = V^+ x
    x <- x + alpha * V (sigma(c) - c)          sigma exchanges the two entries

**The additive form is the point.** It leaves the component of ``x`` orthogonal to
``span{v_s, v_t}`` untouched, so everything else the activation carries survives. Writing
``x <- V sigma(c)`` instead would project the activation onto a two-dimensional subspace
and destroy the rest -- which would change the model's behaviour for reasons having
nothing to do with the concepts being swapped.

Note the update reduces to a rank-one edit along ``v_s - v_t``::

    V (sigma(c) - c) = (c_1 - c_0) (v_s - v_t)

so the swap moves the activation along the difference of the two concept directions, by
however much of ``v_t`` it currently contains minus however much of ``v_s``. It is
implemented in the general ``V`` form anyway: that is the form the paper states, it
generalises to more than two coordinates, and it keeps the pseudoinverse's handling of
non-orthogonal lens vectors explicit rather than assumed away.

**Across several layers the swap is clamped** (:func:`clamped_swap_delta`), as the paper
applies it: each edited layer sets the activation's coordinates to the swapped *clean-pass*
coordinates, ``V^+ x <- sigma(c_clean)``. Recomputing ``c`` from the edited pass instead
(:func:`swap_delta` at every layer) swaps whatever the previous edit left behind. The
residual stream carries each layer's edit into the next, so at ``alpha=1`` consecutive
layers undo one another (``sigma`` is an involution), and at ``alpha=2`` the coordinate gap
is multiplied by -3 per layer and overflows within about twenty layers. At a single edited
site the two coincide.
"""

from __future__ import annotations

import logging

import torch

from src.heads import COMPUTE_DTYPE

logger = logging.getLogger(__name__)


def swap_frame(v_source: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """``V = [v_s  v_t]`` as ``[d, 2]``."""
    if v_source.shape != v_target.shape or v_source.ndim != 1:
        raise ValueError(
            f"lens vectors must be matching 1-D tensors, got {tuple(v_source.shape)} "
            f"and {tuple(v_target.shape)}"
        )
    return torch.stack([v_source, v_target], dim=1).to(COMPUTE_DTYPE)


def swap_pseudoinverse(frame: torch.Tensor) -> torch.Tensor:
    """``V^+`` for a frame (or a batch of them), ``[..., 2, d]``.

    Split out of :func:`swap_delta` so it can be hoisted. A frame depends only on
    ``(source, target)``, not on the arm, the strength or the activation, but
    ``swap_delta`` is called once per head per layer per forward pass -- 352 times a pass
    once per condition -- so computing the pseudoinverse inside it repeats identical work
    hundreds of times per prompt. ``torch.linalg.pinv`` batches over leading dimensions,
    so a whole layer's ``[n_heads, d, 2]`` stack inverts in one call.
    """
    return torch.linalg.pinv(frame.to(COMPUTE_DTYPE))


def swap_delta(
    x: torch.Tensor,
    frame: torch.Tensor,
    *,
    alpha: float = 1.0,
    pseudo: torch.Tensor | None = None,
) -> torch.Tensor:
    """``alpha * V (sigma(V^+ x) - V^+ x)`` -- what the swap adds to ``x``.

    Args:
        x: ``[..., d]``. Any leading shape; the swap is applied per position.
        frame: ``[d, 2]`` from :func:`swap_frame`.
        pseudo: ``V^+`` from :func:`swap_pseudoinverse`, if it has already been computed
            for this frame. Purely a speed argument -- the result is identical.

    Returns:
        The additive update, same shape as ``x``.
    """
    if frame.shape[0] != x.shape[-1]:
        raise ValueError(f"frame is {tuple(frame.shape)} but x has width {x.shape[-1]}")
    frame = frame.to(COMPUTE_DTYPE)
    if pseudo is None:
        pseudo = swap_pseudoinverse(frame)         # [2, d]
    coefficients = x.to(COMPUTE_DTYPE) @ pseudo.to(COMPUTE_DTYPE).T  # [..., 2]
    swapped = coefficients.flip(-1)
    return alpha * ((swapped - coefficients) @ frame.T)


def swap_coefficients(x: torch.Tensor, pseudo: torch.Tensor) -> torch.Tensor:
    """``V^+ x``: the activation's coordinates in the frame, ``[..., 2]``."""
    return x.to(COMPUTE_DTYPE) @ pseudo.to(COMPUTE_DTYPE).T


def clamped_swap_delta(
    x: torch.Tensor,
    frame: torch.Tensor,
    clean: torch.Tensor,
    *,
    alpha: float = 1.0,
    pseudo: torch.Tensor | None = None,
) -> torch.Tensor:
    """``V (sigma_alpha(c_clean) - V^+ x)``: clamp ``x``'s coordinates to the swapped clean ones.

    ``sigma_alpha(c) = c + alpha (sigma(c) - c)``, so ``alpha=1`` is the exchange and
    ``alpha=2`` the overshoot of :func:`swap_delta`. Whatever earlier edits did to ``x``'s
    two coordinates is overwritten; its component orthogonal to the frame's span is left
    alone. When ``x`` is the clean activation this is :func:`swap_delta` exactly.

    Args:
        x: ``[..., d]``, the activation in the edited pass.
        clean: ``V^+ x_clean`` from :func:`swap_coefficients`, broadcastable to
            ``x``'s leading shape plus ``[2]``.
    """
    if frame.shape[0] != x.shape[-1]:
        raise ValueError(f"frame is {tuple(frame.shape)} but x has width {x.shape[-1]}")
    frame = frame.to(COMPUTE_DTYPE)
    if pseudo is None:
        pseudo = swap_pseudoinverse(frame)
    clean = clean.to(COMPUTE_DTYPE)
    target = clean + alpha * (clean.flip(-1) - clean)
    return (target - swap_coefficients(x, pseudo)) @ frame.T


def apply_swap(
    x: torch.Tensor, frame: torch.Tensor, *, alpha: float = 1.0
) -> torch.Tensor:
    """``x + swap_delta(x, frame, alpha)``."""
    return x.to(COMPUTE_DTYPE) + swap_delta(x, frame, alpha=alpha)


def matched_norm_delta(
    delta: torch.Tensor, *, seed: int = 0, generator: torch.Generator | None = None
) -> torch.Tensor:
    """A random direction with the same per-position norm as ``delta``.

    The third control. It separates "this specific pair of lens directions" from "a
    perturbation of this size somewhere in the activation" -- without it, a swap that
    works could simply be one that pushed hard enough to derail the model.
    """
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(delta.shape, generator=generator).to(delta.device, delta.dtype)
    scale = delta.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return noise * scale


def head_frames(jz_layer: torch.Tensor, spec, rows: torch.Tensor) -> torch.Tensor:
    """Every head's swap frame at one layer, ``[n_heads, head_dim, 2]``.

    The pullback ``v_{t,l,h} = J_{l,h}^T (w_t * g)`` of both concept directions into each
    head's own coordinates, stacked into the frames :func:`swap_delta` consumes. The
    predecessor pipeline built these in a per-head Python loop; this is the same
    computation in one einsum, and the test that it matches the per-head construction
    exactly is what pins the two together.

    Args:
        jz_layer: ``J^z_l``, ``[d_model, d_source]``.
        rows: ``[2, d_model]`` -- the effective-unembedding rows of the source and target
            concepts, in that order.

    Returns:
        ``[n_heads, head_dim, 2]``; pair it with :func:`swap_pseudoinverse`, which batches
        over the leading axis.
    """
    if rows.shape != (2, spec.d_model):
        raise ValueError(f"rows are {tuple(rows.shape)}, expected (2, {spec.d_model})")
    if jz_layer.shape != (spec.d_model, spec.d_source):
        raise ValueError(
            f"transport is {tuple(jz_layer.shape)}, expected "
            f"({spec.d_model}, {spec.d_source})"
        )
    blocks = jz_layer.to(COMPUTE_DTYPE).view(spec.d_model, spec.n_heads, spec.head_dim)
    # [n_heads, 2, head_dim] -> the [head_dim, 2] frame swap_frame would have built.
    vectors = torch.einsum("tm,mhd->htd", rows.to(COMPUTE_DTYPE), blocks)
    return vectors.transpose(1, 2).contiguous()


def lens_vector(jacobian: torch.Tensor, unembed_row: torch.Tensor) -> torch.Tensor:
    """``v_t = J^T (w_t * g)`` -- a lens vector in the source activation's coordinates.

    Works for either anchor: pass the Jacobian lens's ``J_l`` to get a ``d_model`` vector in
    residual-stream coordinates (the ``layer_wide`` condition), or a head's ``J_{l,h}`` to
    get a ``head_dim`` vector in head-native coordinates (every ``zin`` condition).

    ``unembed_row`` must already carry the final norm's learned gain; see
    :func:`src.heads.effective_unembed` for why that matters.
    """
    return jacobian.to(COMPUTE_DTYPE).T @ unembed_row.to(COMPUTE_DTYPE)
