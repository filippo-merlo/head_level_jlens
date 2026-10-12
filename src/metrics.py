"""Comparison metrics for lens readouts and for the Jacobians themselves.

Kept separate from the code that produces activations so a metric can be re-derived
from saved logits without a GPU, and so changing how a number is *reported* cannot
change the number.
"""

from __future__ import annotations

import torch


def kl_divergence(
    reference_logits: torch.Tensor, other_logits: torch.Tensor
) -> torch.Tensor:
    """``KL(softmax(reference) || softmax(other))``, per position.

    Argument order matters and is the one the fidelity check wants: the *model's* own
    distribution is the reference, so the result reads "how much information is lost by
    believing the lens instead of the model".

    Args:
        reference_logits: ``[..., vocab_size]``.
        other_logits: Same shape.

    Returns:
        ``[...]`` -- the leading dimensions of the inputs.
    """
    if reference_logits.shape != other_logits.shape:
        raise ValueError(
            f"shape mismatch: {tuple(reference_logits.shape)} vs "
            f"{tuple(other_logits.shape)}"
        )
    reference = torch.log_softmax(reference_logits.float(), dim=-1)
    other = torch.log_softmax(other_logits.float(), dim=-1)
    return (reference.exp() * (reference - other)).sum(-1)


def top1_agreement(a_logits: torch.Tensor, b_logits: torch.Tensor) -> float:
    """Fraction of positions where the two readouts pick the same top token."""
    return float(
        (a_logits.argmax(-1) == b_logits.argmax(-1)).float().mean()
    )


def top_k_overlap(
    a_logits: torch.Tensor, b_logits: torch.Tensor, k: int = 5
) -> torch.Tensor:
    """Per-position ``|top_k(a) & top_k(b)| / k``.

    A softer companion to :func:`top1_agreement`: two readouts can disagree on the
    winner while ranking the same handful of tokens, which is a much milder failure
    than disagreeing on the whole shortlist.
    """
    a_top = a_logits.topk(k, dim=-1).indices
    b_top = b_logits.topk(k, dim=-1).indices
    matches = (a_top.unsqueeze(-1) == b_top.unsqueeze(-2)).any(-1)
    return matches.sum(-1).float() / k


def relative_frobenius(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """``||estimate - reference||_F / ||reference||_F``.

    The scale-free way to ask whether two Jacobians are the same matrix -- used to
    compare a refitted ``J_out`` against the Jacobian lens's own.
    """
    if estimate.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: {tuple(estimate.shape)} vs {tuple(reference.shape)}"
        )
    reference = reference.float()
    denominator = reference.norm()
    if denominator == 0:
        raise ValueError("reference matrix is all zeros; relative error is undefined")
    return float((estimate.float() - reference).norm() / denominator)


def cosine_rows(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-row cosine similarity between two matrices.

    Complements :func:`relative_frobenius`: a lens can be badly scaled but perfectly
    oriented, and only the orientation survives the softmax.
    """
    return torch.nn.functional.cosine_similarity(
        estimate.float(), reference.float(), dim=-1
    )
