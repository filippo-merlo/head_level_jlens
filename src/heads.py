"""The per-head object: decomposing an attention write, and transporting one head's part.

Notation. For layer ``l``:

* ``z_l`` -- ``o_proj``'s input, the concatenated head results, width ``d_source``.
* ``z_{l,h}`` -- head ``h``'s block of it, width ``head_dim``.
* ``c_{l,h} = W_O^h z_{l,h}`` -- head ``h``'s contribution to the residual stream,
  width ``d_model``. The layer's whole attention write is ``sum_h c_{l,h}``, exactly
  (Elhage et al.'s "attention heads are independent and additive").

The transport. ``J^z_l`` is fitted directly at ``z_l`` by :mod:`src.fit`, and a head's
transport is a **column slice** of it::

    J_{l,h} = J^z_l[:, head columns]

That is why there is one fit per *layer* rather than one per head: the head index is a
slice of ``o_proj``'s input axis, which ``J^z`` is already indexed by, so all 32 heads of
a layer come out of the one estimate at no extra cost.

The same linearity gives an identity across anchors -- ``o_proj`` is linear with constant
weights and reaches the final residual only through ``x_mid,l = x_l + a_l``, so
``J^z_l = J_attn,l @ W_O`` exactly, per source/target position pair and hence through the
estimator's averaging. Nothing depends on it any more, but it is the sharpest available
check on the fit: :func:`src.validation.anchor_consistency` estimates both anchors out of
one set of backward passes and compares.

Gemma-3 breaks the second half of that: ``a_l = norm(W_O z_l)``, an RMSNorm between
``o_proj`` and the residual add. ``c_{l,h}`` still sums to ``o_proj``'s output and
``J^z_l`` is still the transport from ``z_l``, but a head's write *into the stream* is its
share of the normalised sum, :func:`head_residual_writes`, and the vector the identity
transport should read is that write's direction, ``g_post * c_{l,h}`` up to a positive
scalar (:func:`residual_o_proj_weight`).

Two consequences worth keeping in view, both of which shape how results are read
rather than how they are computed:

* ``rank J_{l,h} <= head_dim``, so a per-head readout points into a ``head_dim``-slice
  of logit space. This constrains **both** lens arms identically -- ``c_{l,h}`` also
  lives in the image of ``W_O^h`` -- so it limits per-head analysis in general and does
  not differentiate the lenses.
* RMSNorm is scale-invariant, so a readout reports the *direction* a head pushes and
  nothing about how hard. Every legibility number needs a magnitude statistic beside it;
  see :mod:`src.magnitude`.
"""

from __future__ import annotations

import logging

import torch

from src.config import LensSpec
from src.hooks import o_proj, post_sublayer_norm

logger = logging.getLogger(__name__)


def o_proj_weight(model, layer: int) -> torch.Tensor:
    """``W_O`` at ``layer``: ``[d_model, d_source]``.

    Torch stores ``nn.Linear`` weights as ``[out_features, in_features]``, so this is
    already the paper's ``W_O`` orientation and head ``h`` owns a block of its *columns*.
    """
    return o_proj(model.layers[layer]).weight


def head_columns(matrix: torch.Tensor, head: int, spec: LensSpec) -> torch.Tensor:
    """The ``head_dim`` columns of ``matrix`` belonging to ``head``.

    Applies unchanged to ``W_O`` (giving ``W_O^h``) and to ``J^z_l`` (giving
    ``J_{l,h}``), since both are indexed by ``o_proj``'s input on their column axis.
    """
    if matrix.shape[-1] != spec.d_source:
        raise ValueError(
            f"expected {spec.d_source} columns (d_source), got {matrix.shape[-1]} -- "
            "this matrix is not indexed by o_proj's input"
        )
    return matrix[..., spec.head_slice(head)]


#: Precision the per-head algebra runs at, regardless of the forward pass's dtype.
#:
#: Recomposing ``sum_h c_{l,h}`` at the checkpoint's own bf16 is limited by bf16's 8-bit
#: mantissa rather than by the wiring, and an error of that size in every head's write
#: would propagate silently into every readout, cosine and swap downstream. The forward
#: pass stays bf16, matching what the lenses were fitted under; only the arithmetic on
#: captured activations is upcast. ``tests/test_heads.py::test_additivity`` asserts the
#: fp32 bound on the CPU fixtures.
COMPUTE_DTYPE = torch.float32


def head_contribution(
    z: torch.Tensor,
    w_o: torch.Tensor,
    head: int,
    spec: LensSpec,
    *,
    dtype: torch.dtype = COMPUTE_DTYPE,
) -> torch.Tensor:
    """``c_{l,h} = W_O^h z_{l,h}``, head ``h``'s write into the residual stream.

    Computed in ``dtype`` (fp32 by default; see :data:`COMPUTE_DTYPE` for why). Pass the
    activation's own dtype to measure what the model itself represents.

    Args:
        z: ``[..., d_source]`` -- ``o_proj``'s input, from a ``zin`` anchor.
        w_o: ``[d_model, d_source]`` -- that layer's :func:`o_proj_weight`.

    Returns:
        ``[..., d_model]`` in ``dtype``.
    """
    if z.shape[-1] != spec.d_source:
        raise ValueError(f"z has width {z.shape[-1]}, expected d_source={spec.d_source}")
    z_head = z[..., spec.head_slice(head)].to(dtype)
    return z_head @ head_columns(w_o, head, spec).to(dtype).T


def attention_write(
    z: torch.Tensor, w_o: torch.Tensor, *, dtype: torch.dtype = COMPUTE_DTYPE
) -> torch.Tensor:
    """The layer's whole attention write ``a_l = W_O z_l``, i.e. ``o_proj``'s output.

    The reference that :func:`head_contribution` must sum to; see
    ``tests/test_heads.py::test_additivity``.
    """
    return z.to(dtype) @ w_o.to(dtype).T


def norm_gain(norm: torch.nn.Module) -> torch.Tensor:
    """The per-dimension gain an RMSNorm applies after rescaling, ``[d]`` in fp32.

    Read off the module by evaluating it on a vector of ones, whose RMS is exactly 1:
    the output *is* the gain, whatever the parametrisation. Llama stores it as
    ``weight``; Gemma-3 and Qwen3.5 store ``weight - 1`` and apply ``1 + weight``. Asking
    the module rather than its parameter means neither convention has to be known here,
    and a third one would be read correctly too. The ``rsqrt(1 + eps)`` the ones-vector
    picks up is one positive scalar over all dimensions and cancels wherever the gain is
    used as a direction.
    """
    weight = next(norm.parameters())
    ones = torch.ones(weight.shape[-1], dtype=torch.float32, device=weight.device)
    with torch.no_grad():
        return norm(ones).float()


def effective_unembed(model, token_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Rows of the unembedding, with the final norm's learned gain folded in.

    ``model.unembed`` is ``lm_head(final_norm(x))``, and an RMSNorm applies a learned
    per-dimension gain ``g`` after rescaling::

        logit_t(x) = w_t . ((x / rms(x)) * g) = ((w_t * g) . x) / rms(x)

    So the residual-stream direction token ``t`` actually reads along is ``w_t * g``, not
    ``w_t``. The ``1/rms(x)`` factor is a positive scalar and drops out of every ranking
    and cosine this project computes. Folding ``g`` in matters wherever a lens *vector*
    is used as a direction rather than a score -- :mod:`src.structural`'s workspace loading
    and :mod:`src.swap`'s swaps -- where bare ``w_t`` would be a subtly wrong direction.
    ``g`` comes from :func:`norm_gain`, so the parametrisation of the norm is irrelevant.

    Args:
        token_ids: Restrict to these vocabulary rows; ``None`` returns all of them.

    Returns:
        ``[n_tokens, d_model]``.
    """
    rows = model._lm_head.weight
    if token_ids is not None:
        rows = rows[token_ids]
    if not any(True for _ in model._final_norm.parameters()):
        return rows
    return rows * norm_gain(model._final_norm).to(rows.dtype)


def post_attention_gain(model, layer: int) -> torch.Tensor | None:
    """Gain of the norm between ``o_proj`` and the residual add, or ``None`` without one."""
    norm = post_sublayer_norm(model.layers[layer])
    return None if norm is None else norm_gain(norm)


def residual_o_proj_weight(model, layer: int) -> torch.Tensor:
    """``W_O`` scaled so that its columns are directions in the residual stream.

    On a pre-norm block this is :func:`o_proj_weight` itself: ``o_proj``'s output is
    added to the stream as it is. On Gemma-3 the output first passes through an RMSNorm
    with gain ``g``, so a lone head's write would be ``g * c_{l,h} / rms(c_{l,h})``, and
    the direction the identity transport should read is ``g * c_{l,h}`` --
    ``diag(g) W_O^h z_{l,h}``. The positive scalar is dropped: RMSNorm at the readout
    removes it anyway. This is the matrix the Logit-head arm and the top-layer ``W_O``
    gate use; :func:`head_contribution` keeps the unscaled ``W_O``, because additivity is
    a statement about ``o_proj``'s output.

    Returns:
        ``[d_model, d_source]`` in ``W_O``'s dtype.
    """
    w_o = o_proj_weight(model, layer)
    gain = post_attention_gain(model, layer)
    if gain is None:
        return w_o
    return w_o * gain.to(w_o.dtype)[:, None]


def head_residual_writes(
    z: torch.Tensor,
    w_o: torch.Tensor,
    spec: LensSpec,
    *,
    norm: torch.nn.Module | None,
    dtype: torch.dtype = COMPUTE_DTYPE,
) -> torch.Tensor:
    """What each head adds to the residual stream, in context: ``[..., n_heads, d_model]``.

    On a pre-norm block that is ``c_{l,h}`` itself. On a block with a norm between
    ``o_proj`` and the residual add (``norm`` not ``None``), the write is
    ``norm(sum_h c_h) - norm(sum_h c_h - c_h)``: the change in the sub-layer's output
    when head ``h`` is removed, which is the only quantity that both reduces to ``c_h``
    without the norm and sums to nothing in particular with it. Used for the relative
    write norm in :mod:`src.magnitude`; :func:`src.magnitude.ablation_kl` makes the same
    removal through the model itself.

    Args:
        z: ``[..., d_source]`` at the ``zin`` anchor.
        w_o: ``[d_model, d_source]``, the unscaled :func:`o_proj_weight`.
        norm: :func:`src.hooks.post_sublayer_norm` of that block.
    """
    contributions = torch.stack(
        [head_contribution(z, w_o, h, spec, dtype=dtype) for h in range(spec.n_heads)],
        dim=-2,
    )
    if norm is None:
        return contributions
    total = contributions.sum(dim=-2, keepdim=True)
    with torch.no_grad():
        return norm(total).to(dtype) - norm(total - contributions).to(dtype)


def head_lens_vectors(
    jz_layer: torch.Tensor,
    head: int,
    spec: LensSpec,
    unembed_rows: torch.Tensor,
) -> torch.Tensor:
    """``v_{t,l,h} = J_{l,h}^T (w_t * g)`` -- lens vectors in head-native space.

    The pullback of a layer-level lens vector into head ``h``'s ``head_dim`` coordinates.
    These are the directions the swap operates on, and the ones :mod:`src.structural`
    measures workspace loading against.

    Args:
        jz_layer: ``J^z_l``, ``[d_model, d_source]``.
        unembed_rows: ``[n_tokens, d_model]`` from :func:`effective_unembed`.

    Returns:
        ``[n_tokens, head_dim]``.
    """
    j_head = head_columns(jz_layer, head, spec)
    return unembed_rows.to(j_head.dtype) @ j_head
