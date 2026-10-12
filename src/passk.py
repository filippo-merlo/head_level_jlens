"""pass@k at head granularity, and the paired J-head minus Logit-head difference.

The documented metric (``data/evaluations/README.md``) is: the mean over items of the
fraction of that item's intermediates whose **min-over-layers** lens rank is at most
``k``, read at one documented position per set.

**A head has no depth.** Head ``h`` of layer 4 and head ``h`` of layer 5 are unrelated
parameters that share a column index and nothing else, so nothing may be minimised or
averaged "per head" across layers. The unit is the site, one ``(layer, head)`` pair. At
one layer, the head-level rank of an intermediate is the **minimum over that layer's 32
heads** -- the best single head -- which gives a head arm one rank per (item,
intermediate, layer), the same shape as the whole-layer reference. The package's min over
layers then applies to both alike: the all-layers aggregate of a head arm is its best of
1,024 sites, a per-layer value its best of 32 heads.

Every statistic here is a mean over items of a per-item quantity; the minimum is taken
inside each item, before averaging. A difference of two arms' pass@k is therefore the
mean of the per-item differences.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Padding value in every rank array; a slot with no intermediate.
PAD = -1


def best_site_rank(ranks: np.ndarray, layers: list[int] | None = None) -> np.ndarray:
    """Best (lowest) rank over the heads of the chosen layers; padding stays padding.

    Args:
        ranks: ``[n_items, n_intermediates, n_layers, n_heads]``, zero-based ranks with
            ``PAD`` where an item has no intermediate in that slot. The whole-layer
            reference passes ``n_heads = 1``.
        layers: indices into the layer axis to minimise over; all of them by default.

    Returns:
        ``[n_items, n_intermediates]``.
    """
    block = ranks if layers is None else ranks[:, :, layers, :]
    valid = block != PAD
    masked = np.where(valid, block, np.iinfo(np.int64).max)
    best = masked.min(axis=(2, 3))
    return np.where(valid.any(axis=(2, 3)), best, PAD)


def best_site_agreement(
    ranks_a: np.ndarray, ranks_b: np.ndarray, layers: list[int] | None = None
) -> float:
    """How often two arms' best sites coincide: the fraction of (item, intermediate) pairs
    for which some site is a minimiser under both arms.

    Each arm takes its own minimum, so J-head's rank on an item is the rank at the head
    J-head reads best and Logit-head's is the rank at the head *it* reads best; the two
    arms share the candidate sites and the items, not the chosen site. This says how often
    they nonetheless land on the same one. Counting a shared minimiser, rather than
    comparing first arg-minima, makes the statistic symmetric and indifferent to ties.

    Args:
        ranks_a, ranks_b: ``[n_items, n_intermediates, n_layers, n_heads]``, aligned as
            :func:`paired_difference` requires.
        layers: indices into the layer axis to minimise over; all of them by default.
    """
    if ranks_a.shape != ranks_b.shape:
        raise ValueError(f"paired arms must align: {ranks_a.shape} vs {ranks_b.shape}")
    if layers is not None:
        ranks_a, ranks_b = ranks_a[:, :, layers, :], ranks_b[:, :, layers, :]
    valid = (ranks_a != PAD).any(axis=(2, 3))
    best_a = best_site_rank(ranks_a)[..., None, None]
    best_b = best_site_rank(ranks_b)[..., None, None]
    shared = ((ranks_a == best_a) & (ranks_b == best_b)).any(axis=(2, 3))
    return float(shared[valid].mean())


def scoreable_items(ranks: np.ndarray) -> np.ndarray:
    """Indices of the items with at least one scoreable intermediate.

    An item whose every intermediate the candidate rule could not score under this
    tokenizer (:func:`src.evals.candidate_coverage`) is all ``PAD``. It is no evidence
    about any arm, so it is neither a hit nor a miss: pass@k averages over the others.
    """
    return np.flatnonzero((ranks != PAD).any(axis=1))


def passk(ranks: np.ndarray, k: int) -> float:
    """pass@k from an item x intermediate rank array.

    Args:
        ranks: ``[n_items, n_intermediates]``, zero-based as :func:`src.readout.token_ranks`
            returns them (0 = top-1), so a rank of ``r`` means ``r`` tokens score strictly
            higher and "within the top ``k``" is ``r < k``. ``PAD`` entries are ignored,
            and an item with nothing but padding is left out of the mean
            (:func:`scoreable_items`); ``nan`` if no item is scoreable.
    """
    scored = scoreable_items(ranks)
    if scored.size == 0:
        return float("nan")
    valid = ranks[scored] != PAD
    hit = (ranks[scored] < k) & valid
    # Fraction of *this item's* intermediates that were hit, then mean over items --
    # not a pooled fraction over all intermediates, which would weight items by how many
    # intermediates they happen to carry.
    per_item = hit.sum(axis=1) / valid.sum(axis=1)
    return float(per_item.mean())


def sites_hit(ranks: np.ndarray, k: int) -> float:
    """Mean number of sites at which an intermediate ranks within the top ``k``.

    The best-site reduction keeps one site per intermediate; this counts all of them --
    how many heads, at any layer, read the concept that high. Weighted as :func:`passk`
    is: the mean over an item's intermediates, then over scoreable items.

    Args:
        ranks: ``[n_items, n_intermediates, n_layers, n_heads]``, zero-based, ``PAD``
            where an item has no intermediate in that slot.
    """
    valid = (ranks != PAD).any(axis=(2, 3))
    scored = np.flatnonzero(valid.any(axis=1))
    if scored.size == 0:
        return float("nan")
    count = ((ranks < k) & (ranks != PAD)).sum(axis=(2, 3))
    per_item = count[scored].sum(axis=1) / valid[scored].sum(axis=1)
    return float(per_item.mean())


def paired_difference(ranks_a: np.ndarray, ranks_b: np.ndarray, k: int) -> float:
    """``pass@k(a) - pass@k(b)`` on the same items: the mean per-item difference.

    Both arms read the same activations on the same items, so they must align in shape
    and in which slots are padding.
    """
    if ranks_a.shape != ranks_b.shape or ((ranks_a == PAD) != (ranks_b == PAD)).any():
        raise ValueError(
            f"paired arms must align in shape and padding: {ranks_a.shape} vs {ranks_b.shape}"
        )
    return passk(ranks_a, k) - passk(ranks_b, k)
