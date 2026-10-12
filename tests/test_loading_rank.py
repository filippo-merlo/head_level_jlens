"""Loading-against-rank agreement, on hand-built maps where the answer is known."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from src.loading_rank import (
    SITE_COUNTS,
    average_ranks,
    share_best_read,
    spearman,
    summarise,
    top_mask,
)


def test_average_ranks_are_one_based_and_average_ties():
    assert average_ranks(np.array([30, 10, 20])).tolist() == [3, 1, 2]
    assert average_ranks(np.array([10, 20, 20, 30])).tolist() == [1, 2.5, 2.5, 4]
    assert average_ranks(np.zeros(4)).tolist() == [2.5] * 4


def test_spearman_is_one_for_monotone_and_minus_one_for_reversed():
    a = np.arange(10.0)
    assert spearman(a, a**3) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)


def test_spearman_averages_ties():
    # Ranks of b are [1, 2, 3.5, 5, 3.5]; Pearson of those with 1..5 is 8 / sqrt(95).
    assert spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) == pytest.approx(8 / np.sqrt(95))


def test_spearman_rejects_a_size_mismatch():
    with pytest.raises(ValueError):
        spearman(np.zeros(3), np.zeros(4))


def test_top_mask_takes_the_highest_and_keeps_the_shape():
    loading = np.array([[0.1, 0.9], [0.5, -0.2]])
    assert top_mask(loading, 2).tolist() == [[False, True], [True, False]]
    with pytest.raises(ValueError):
        top_mask(loading, 5)


def test_share_is_one_when_the_selection_is_the_best_read_set():
    rank = np.array([4, 0, 7, 1, 9])
    assert share_best_read(np.array([0, 1, 0, 1, 0]), rank) == 1.0
    assert share_best_read(np.array([1, 0, 1, 0, 0]), rank) == 0.0


def test_share_splits_sites_tied_at_the_cut_evenly():
    # Two places, four sites tied at rank 0: each is in the best-read set half the time.
    rank = np.array([0, 0, 0, 0, 5, 6])
    assert share_best_read(np.array([1, 1, 0, 0, 0, 0]), rank) == pytest.approx(0.5)
    assert share_best_read(np.array([1, 0, 0, 0, 1, 0]), rank) == pytest.approx(0.25)


def test_share_of_a_random_selection_is_m_over_n_in_expectation():
    rank = np.array([0, 0, 2, 2, 2, 7])
    for m in (1, 2, 3):
        shares = []
        for chosen in combinations(range(rank.size), m):
            mask = np.zeros(rank.size, dtype=bool)
            mask[list(chosen)] = True
            shares.append(share_best_read(mask, rank))
        assert np.mean(shares) == pytest.approx(m / rank.size)


def test_share_of_an_empty_selection_is_nan():
    assert np.isnan(share_best_read(np.zeros(3, dtype=bool), np.arange(3)))


def test_summarise_on_perfect_and_reversed_agreement():
    rng = np.random.default_rng(0)
    rank = np.stack([rng.permutation(100).reshape(10, 10) for _ in range(3)])
    # Readout 0's loading orders sites exactly as the rank does; readout 1's reverses it.
    loading = np.stack([-rank, rank], axis=1).astype(float)
    rank = np.stack([rank, rank], axis=1)
    tau = np.full(loading.shape, np.inf)
    tau[:, 0] = -10.5  # readout 0: active = the ten best-read sites
    out = summarise(loading, rank, tau, ["a", "a", "b"], ("J", "L"))
    assert out["a"]["n_prompts"] == 2 and out["b"]["n_prompts"] == 1
    agree, reverse = out["a"]["J"], out["a"]["L"]
    assert agree["spearman"] == pytest.approx([1.0, 1.0])
    assert reverse["spearman"] == pytest.approx([-1.0, -1.0])
    assert all(agree["share_best_read"][str(m)] == 1.0 for m in SITE_COUNTS)
    assert agree["share_best_read"]["active"] == 1.0
    assert reverse["share_best_read"]["8"] == 0.0
    assert np.isnan(reverse["share_best_read"]["active"])  # nothing beats an infinite cut
    assert agree["chance"]["8"] == 0.08
