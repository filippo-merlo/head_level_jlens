"""Source loading against the source's readout rank, over head sites.

Experiment 2 edits the sites with the highest source loading ``cos(z_{l,h}, v_{s,l,h})``;
Experiment 1 reads a concept at a site by its rank in that site's readout. The two share a
numerator: the readout scores token ``t`` by ``v_{t,l,h} . z_{l,h}`` up to a positive factor
common to every token, so the source's rank at a site is set by ``|v_t| cos(z, v_t)`` over
the vocabulary, and loading is the source's own term with its length dropped. They can
disagree only through those lengths, so how far they agree is an empirical question about
the lens, which this module measures on recorded ``[n_layers, n_heads]`` maps.

Both measures are rank-based, because a readout rank has no scale to compare a cosine to.
Ties are real: at late heads the source is often rank 0 at several sites, so the best-read
set at a cut is ambiguous, and :func:`share_best_read` shares the tied sites out evenly
rather than letting a sort order decide.
"""

from __future__ import annotations

import numpy as np

#: Experiment 2's fixed site counts.
SITE_COUNTS = (1, 8, 16, 32, 64)


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks of a flat array, ties given the mean of the ranks they span."""
    x = np.asarray(x).reshape(-1)
    order = np.argsort(x, kind="stable")
    ordered = x[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], x.size]
    group = np.repeat(np.arange(starts.size), ends - starts)
    ranks = np.empty(x.size)
    ranks[order] = ((starts + ends + 1) / 2)[group]
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rho over sites, with tied values averaged."""
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"{a.size} values against {b.size}")
    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def top_mask(loading: np.ndarray, m: int) -> np.ndarray:
    """The ``m`` highest-loading sites, as a mask of ``loading``'s shape."""
    flat = np.asarray(loading).reshape(-1)
    if not 0 < m <= flat.size:
        raise ValueError(f"m={m} out of range for {flat.size} sites")
    mask = np.zeros(flat.size, dtype=bool)
    mask[np.argsort(-flat, kind="stable")[:m]] = True
    return mask.reshape(np.shape(loading))


def share_best_read(selected: np.ndarray, rank: np.ndarray) -> float:
    """Share of the selected sites that are among as many sites where the source reads best.

    With ``m`` selected sites, the best-read set is the ``m`` sites of lowest rank. Sites
    tied at the cut each count as a member with probability ``(places left) / (sites
    tied)``, which is the expectation over every way of breaking the tie. A uniformly
    random selection of ``m`` sites scores ``m / n_sites`` in expectation.

    Returns:
        NaN for an empty selection.
    """
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    rank = np.asarray(rank).reshape(-1)
    if selected.size != rank.size:
        raise ValueError(f"{selected.size} sites selected against {rank.size} ranks")
    m = int(selected.sum())
    if m == 0:
        return float("nan")
    cut = np.sort(rank)[m - 1]
    better, tied = rank < cut, rank == cut
    places = m - int(better.sum())
    overlap = (selected & better).sum() + (selected & tied).sum() * places / tied.sum()
    return float(overlap / m)


def summarise(
    loading: np.ndarray,
    rank: np.ndarray,
    tau: np.ndarray,
    groups: list[str],
    readouts: tuple[str, ...],
) -> dict:
    """Per group and readout: the rank correlation and the best-read share at each count.

    Args:
        loading, rank, tau: ``[n_prompts, n_readouts, n_layers, n_heads]`` -- the source
            loading, the source's 0-based readout rank, and the active-site cut.
        groups: Each prompt's group (the task), in prompt order.

    The correlation is with the negated rank, so agreement is positive. ``active`` uses
    each prompt's own active set against the same number of best-read sites. Shares are
    means over prompts, correlations are summarised by their quartiles.
    """
    if not loading.shape == rank.shape == tau.shape:
        raise ValueError(f"shapes differ: {loading.shape}, {rank.shape}, {tau.shape}")
    n_prompts, n_readouts = loading.shape[:2]
    if len(groups) != n_prompts or len(readouts) != n_readouts:
        raise ValueError("groups or readouts do not match the arrays")
    n_sites = loading[0, 0].size
    out = {}
    for group in dict.fromkeys(groups):
        rows = [i for i, g in enumerate(groups) if g == group]
        out[group] = {"n_prompts": len(rows)}
        for r, readout in enumerate(readouts):
            rho = np.array([spearman(loading[i, r], -rank[i, r]) for i in rows])
            share = {
                str(m): float(np.mean([share_best_read(top_mask(loading[i, r], m), rank[i, r])
                                       for i in rows]))
                for m in SITE_COUNTS
            }
            active = [share_best_read(loading[i, r] > tau[i, r], rank[i, r]) for i in rows]
            active = [v for v in active if not np.isnan(v)]  # a prompt with no active site
            share["active"] = float(np.mean(active)) if active else float("nan")
            out[group][readout] = {
                "spearman": [float(v) for v in rho],
                "spearman_quartiles": [float(v) for v in np.quantile(rho, [0.25, 0.5, 0.75])],
                "share_best_read": share,
                "chance": {str(m): m / n_sites for m in SITE_COUNTS},
            }
    return out
