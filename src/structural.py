"""Structural characterisation of a head's readout, at head granularity.

Two statistics per (layer, head), computed on the J-head and Logit-head arms over
identical activations:

* **Excess kurtosis** of the readout's *logit* distribution over the vocabulary -- the
  paper's Fig. 28(b) statistic applied per head instead of per layer. Both arms are
  scale-invariant under RMSNorm and kurtosis is standardised, so the arms are directly
  comparable without any scale matching.
* **Workspace loading** -- ``cos(z_{l,h}, v_{t,l,h})`` for the concept ``t`` the item
  *specifies*, following the paper's Fig. 19. Emphatically **not**
  the cosine against the head's own top-ranked lens vector: that is a
  maximum-over-vocabulary statistic and is positively biased by construction, so it would
  score a head highly for having any spiky readout at all.

Neither needs a forward pass of its own -- both are functions of activations the pass@k
readout already reads -- which is why :mod:`scripts.read` computes them in the same pass
rather than in a driver of their own.

**Why they are here at all.** These are the project's only *non-causal* line of evidence on
whether a head carries standalone content. The causal experiment has a detection threshold
set by its trial count; a structural statistic has a different failure mode, so the two
together say more than either alone. They are recorded, never adjudicated.
"""

from __future__ import annotations

import logging

import torch

from src.heads import COMPUTE_DTYPE

logger = logging.getLogger(__name__)

__all__ = ["excess_kurtosis", "workspace_loading"]


def excess_kurtosis(logits: torch.Tensor) -> torch.Tensor:
    """Excess kurtosis of the logit distribution over the vocabulary.

    Standardised by construction, so it is comparable across lenses, layers and arms
    without any scale matching -- which matters because RMSNorm makes every readout
    scale-free anyway.

    Args:
        logits: ``[..., vocab]``.

    Returns:
        ``[...]`` -- one value per position.
    """
    centred = logits.float() - logits.float().mean(dim=-1, keepdim=True)
    variance = centred.pow(2).mean(dim=-1)
    fourth = centred.pow(4).mean(dim=-1)
    return fourth / variance.pow(2) - 3.0


def workspace_loading(
    jz_layer: torch.Tensor,
    z: torch.Tensor,
    spec,
    unembed_row: torch.Tensor,
) -> torch.Tensor:
    """``cos(z_{l,h}, v_{t,l,h})`` for every head at one layer.

    How much of the *specified* concept's direction head ``h`` actually wrote, in the
    head's own 128 coordinates. The lens vector is the pullback
    ``v_{t,l,h} = J_{l,h}^T (w_t * g)``, so this asks whether the head's write
    points along the concept rather than merely producing a confident-looking readout.

    Args:
        jz_layer: ``J^z_l``, ``[d_model, d_source]``.
        z: ``[d_source]`` -- one position's ``o_proj`` input.
        spec: The :class:`~src.config.LensSpec`, for the head geometry.
        unembed_row: ``[d_model]`` for one concept token, or ``[n_tokens, d_model]`` for
            several, **with the final norm's gain already folded in** (see
            :func:`src.heads.effective_unembed`). Using a bare ``w_t`` here is a subtly
            wrong direction. The batched form exists for the per-site null of
            :mod:`src.sites`, which scores 256 concepts at every site and would
            otherwise repeat the same transport 256 times.

    Returns:
        ``[n_heads]`` cosine similarities in ``[-1, 1]``, or ``[n_tokens, n_heads]`` if
        ``unembed_row`` carried a token axis.
    """
    if z.ndim != 1:
        raise ValueError(f"expected one position's z, got shape {tuple(z.shape)}")
    if jz_layer.shape != (spec.d_model, spec.d_source):
        raise ValueError(
            f"transport is {tuple(jz_layer.shape)}, expected "
            f"({spec.d_model}, {spec.d_source})"
        )
    if unembed_row.ndim not in (1, 2):
        raise ValueError(
            f"expected [d_model] or [n_tokens, d_model] rows, got "
            f"{tuple(unembed_row.shape)}"
        )

    rows = unembed_row.to(COMPUTE_DTYPE)
    batched = rows.ndim == 2
    if not batched:
        rows = rows.unsqueeze(0)

    # v_{t,l,h} = J_{l,h}^T row, for every head and every concept at once.
    blocks = jz_layer.to(COMPUTE_DTYPE).view(spec.d_model, spec.n_heads, spec.head_dim)
    vectors = torch.einsum("mhd,tm->thd", blocks, rows)   # [n_tokens, n_heads, head_dim]
    z_heads = z.to(COMPUTE_DTYPE).view(spec.n_heads, spec.head_dim)
    loading = torch.nn.functional.cosine_similarity(z_heads, vectors, dim=-1)
    return loading if batched else loading.squeeze(0)
