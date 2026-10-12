"""The three readout arms, and the token-rank machinery they are scored with.

Every arm ends the same way -- ``model.unembed``, i.e. the model's own final norm and
unembedding -- and differs only in what is transported into final-layer coordinates
first. That is the whole comparison:

======================  ==========================  =========================
Arm                     Transport                   Vector read
======================  ==========================  =========================
``jhead``               ``J_{l,h}``                 ``z_{l,h}``
``logithead``           identity                    ``c_{l,h}``
``jlayer`` (reference)  Jacobian lens ``J_l``       full residual ``x_{l+1}``
======================  ==========================  =========================

The first two read the *same vector* ``c_{l,h}`` and differ by exactly one matrix, since
``J_{l,h} z_{l,h} = J_attn,l c_{l,h}``. That is a cleaner contrast than the layer-level
J-lens-vs-logit-lens comparison, where the two arms differ by ``J_l`` vs ``I`` on a
residual stream of very different composition.

"""

from __future__ import annotations

import logging
from typing import Literal

import torch

from src.config import LensSpec
from src.heads import COMPUTE_DTYPE, head_columns, head_contribution, o_proj_weight

logger = logging.getLogger(__name__)

Arm = Literal["jhead", "logithead", "jlayer"]
HEAD_ARMS: tuple[Arm, ...] = ("jhead", "logithead")


def head_transported(
    jz_layer: torch.Tensor,
    z: torch.Tensor,
    head: int,
    spec: LensSpec,
    *,
    arm: Arm,
    w_o: torch.Tensor | None = None,
) -> torch.Tensor:
    """One head's write, in final-layer coordinates, under ``arm``.

    Args:
        jz_layer: ``J^z_l``, ``[d_model, d_source]``.
        z: ``[..., d_source]`` -- ``o_proj``'s input at the readout position(s).
        w_o: Required for ``logithead`` (which needs ``c_{l,h}`` untransported).

    Returns:
        ``[..., d_model]``, ready for ``model.unembed``.
    """
    if arm == "logithead":
        if w_o is None:
            raise ValueError("the logithead arm needs w_o to form c_{l,h}")
        return head_contribution(z, w_o, head, spec)

    j_head = head_columns(jz_layer, head, spec).to(COMPUTE_DTYPE)
    z_head = z[..., spec.head_slice(head)].to(COMPUTE_DTYPE)
    transported = z_head @ j_head.T
    if arm == "jhead":
        return transported
    raise ValueError(f"unknown head arm {arm!r}; expected one of {HEAD_ARMS}")


def head_logits(
    model,
    jz_layer: torch.Tensor,
    z: torch.Tensor,
    head: int,
    spec: LensSpec,
    *,
    arm: Arm,
    layer: int | None = None,
) -> torch.Tensor:
    """Vocabulary logits for one head under ``arm``.

    ``layer`` is only needed for ``logithead``, to fetch that layer's ``W_O``.
    """
    w_o = None
    if arm == "logithead":
        if layer is None:
            raise ValueError("the logithead arm needs the layer index to fetch W_O")
        w_o = o_proj_weight(model, layer)
    vector = head_transported(
        jz_layer, z, head, spec, arm=arm, w_o=w_o
    )
    return model.unembed(vector).float()


def all_heads_transported(
    jz_layer: torch.Tensor,
    z: torch.Tensor,
    spec: LensSpec,
    *,
    arm: Arm,
    w_o: torch.Tensor | None = None,
) -> torch.Tensor:
    """Every head at one layer, transported, as ``[n_heads, d_model]``.

    The batched form of :func:`head_transported`, and the one the experiments actually
    use. Unembedding is memory-bound -- one call streams the whole 4096x128256 head --
    so doing 32 heads in one ``[n_heads, vocab]`` call rather than 32 separate ones is
    the difference between seconds and minutes per prompt.

    Args:
        z: ``[d_source]`` -- one position's ``o_proj`` input.

    Returns:
        ``[n_heads, d_model]``, in the same head order as ``spec.head_slice``.
    """
    if z.ndim != 1:
        raise ValueError(f"expected one position's z, got shape {tuple(z.shape)}")
    if arm == "logithead" and w_o is None:
        raise ValueError("the logithead arm needs w_o to form c_{l,h}")
    matrix = (w_o if arm == "logithead" else jz_layer).to(COMPUTE_DTYPE)
    if matrix.shape != (spec.d_model, spec.d_source):
        raise ValueError(
            f"transport is {tuple(matrix.shape)}, expected "
            f"({spec.d_model}, {spec.d_source})"
        )

    # [d_model, n_heads, head_dim] x [n_heads, head_dim] -> [n_heads, d_model]
    blocks = matrix.view(spec.d_model, spec.n_heads, spec.head_dim)
    z_heads = z.to(COMPUTE_DTYPE).view(spec.n_heads, spec.head_dim)
    transported = torch.einsum("mhd,hd->hm", blocks, z_heads)

    if arm not in HEAD_ARMS:
        raise ValueError(f"unknown head arm {arm!r}; expected one of {HEAD_ARMS}")
    return transported


def layer_logits(model, lens, residual: torch.Tensor, layer: int) -> torch.Tensor:
    """The ``jlayer`` reference arm: the Jacobian lens on the whole residual stream."""
    return model.unembed(lens.transport(residual.to(COMPUTE_DTYPE), layer)).float()


# ------------------------------------------------------------------------- ranking


def target_token_ids(
    tokenizer, forms: str | tuple[str, ...] | list[str], *, single_token_only: bool = False
) -> list[int]:
    """Candidate vocabulary ids for a target, over its accepted surface forms.

    A lens readout scores single vocabulary tokens, but the eval sets name targets as
    words ("Brazil", "small") and, for order-ops, as synonym sets. Both the
    space-prefixed and bare spellings of each form are taken, because which one a BPE
    tokenizer treats as the natural word-initial token varies by word, and scoring only
    one would penalise the lens for a tokenisation accident rather than a readout
    failure. Ranks are minimised over the candidates.

    Args:
        single_token_only: Keep only forms the tokenizer represents as exactly one token,
            per the README's "min over single-token synonyms" for order-ops. Otherwise
            the *first* token of each form is used -- a multi-token word has no single
            direction to read, so its first token is the standard proxy.

    Raises:
        ValueError: If ``single_token_only`` leaves nothing. Returning an empty candidate
            list would make the item unscoreable while looking like a lens failure.
    """
    if isinstance(forms, str):
        forms = (forms,)
    candidates: list[int] = []
    for form in forms:
        for spelling in (f" {form}", form):
            ids = tokenizer(spelling, add_special_tokens=False).input_ids
            if not ids:
                continue
            if single_token_only:
                if len(ids) == 1:
                    candidates.append(int(ids[0]))
            else:
                candidates.append(int(ids[0]))
    if not candidates:
        raise ValueError(
            f"no {'single-token ' if single_token_only else ''}candidate for {forms!r}"
        )
    return sorted(set(candidates))


def target_direction_id(tokenizer, form: str) -> int:
    """The single vocabulary id a swap should be driven along.

    The space-prefixed spelling's first token. In a completion-style prompt the model is
    choosing a *word-initial continuation*, so that is the token whose lens vector names
    the concept.

    Why this is not :func:`target_token_ids`'s first element. That function returns
    ``sorted(set(candidates))`` -- sorted **numerically over vocabulary ids** -- and is a
    *scoring* set, built so a rank can be minimised over every accepted surface form.
    Taking its first element picks a direction by BPE accident: on this tokenizer
    ``Soccer`` -> ``'S'`` (id 50, beating ``' Soccer'`` at 38633), ``Lemon`` -> ``'L'``,
    ``elephant`` -> ``'ele'``. The swap is then driven along the lens vector for a bare
    letter or word fragment rather than for the concept, which depresses the J-lens arm
    while leaving the already-floored controls untouched.

    Grading still uses :func:`target_token_ids`: success should accept any surface form
    of the target, but the *direction* has to be the one that names it.
    """
    ids = tokenizer(f" {form}", add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"{form!r} has no space-prefixed tokenisation")
    return int(ids[0])


def token_ranks(logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    """Full-vocabulary rank of each token id (0 = top-1).

    Computed by counting strictly-greater logits rather than by sorting: the vocabulary
    is 128k wide and only a handful of targets are ever scored.
    """
    if logits.ndim != 1:
        raise ValueError(f"expected a single position's logits, got {tuple(logits.shape)}")
    ids = torch.tensor(token_ids, device=logits.device)
    target = logits[ids]
    return (logits.unsqueeze(0) > target.unsqueeze(1)).sum(dim=1)


def best_rank(logits: torch.Tensor, token_ids: list[int]) -> int:
    """The best (lowest) rank achieved by any candidate form of a target."""
    return int(token_ranks(logits, token_ids).min())


def top_tokens(logits: torch.Tensor, tokenizer, k: int = 8) -> list[tuple[str, float]]:
    """The ``k`` highest-scoring tokens, decoded -- for qualitative inspection only."""
    values, indices = logits.topk(k)
    return [
        (tokenizer.decode([int(i)]), float(v))
        for v, i in zip(values, indices, strict=True)
    ]
