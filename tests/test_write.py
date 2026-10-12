"""The causal driver's site axis: a count crossed with a selection rule.

The driver itself needs the checkpoint, but the table it fills is fixed by two pure pieces
-- the condition layout and the per-trial site sets -- and both decide *what gets edited*.
These pin that every count gets its test row and both controls at exactly that count, and
that the sets are nested across counts under every rule, which is what lets the count axis
be read as a dose.
"""

from __future__ import annotations

import torch

from scripts.write import (
    CONDITIONS,
    COUNTS,
    RESIDUAL_CONDITION,
    RULES,
    SELECTED_CONDITIONS,
    SITE_COUNTS,
    ZIN_CONDITIONS,
    condition_name,
    site_sets,
)

#: 8 x 16 = 128 sites, so the largest fixed count still leaves a disjoint bottom set.
LAYERS = list(range(8))
N_HEADS = 16


def _loading(seed: int = 0) -> torch.Tensor:
    """Distinct values, so every top / bottom set is unambiguous."""
    n_sites = len(LAYERS) * N_HEADS
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n_sites, generator=generator).float().reshape(len(LAYERS), N_HEADS)


def _tau(loading: torch.Tensor, n_active: int) -> torch.Tensor:
    """A per-site cut admitting exactly the ``n_active`` highest sites."""
    ranked = loading.reshape(-1).sort(descending=True).values
    cut = ranked[n_active - 1] if n_active else loading.max() + 1
    return torch.full_like(loading, float(cut) - 0.5)


# ------------------------------------------------------------------ layout


def test_every_count_has_a_test_row_and_both_controls():
    assert CONDITIONS[0] == RESIDUAL_CONDITION
    assert CONDITIONS[1:] == ZIN_CONDITIONS
    assert len(set(CONDITIONS)) == len(CONDITIONS)
    assert len(ZIN_CONDITIONS) == len(COUNTS) * len(RULES)
    for count in COUNTS:
        assert {condition_name(rule, count) for rule in RULES} <= set(ZIN_CONDITIONS)


def test_the_threshold_row_keeps_its_own_name():
    """It is not the top-``n_active`` by raw loading -- the cut is per site -- so it must
    not be named as if it were."""
    assert condition_name("top", "active") == "active"
    assert condition_name("random", "active") == "random_active"
    assert condition_name("top", 8) == "top_8"
    assert SELECTED_CONDITIONS == tuple(f"top_{k}" for k in SITE_COUNTS) + ("active",)


# ------------------------------------------------------------------ site sets


def test_every_row_has_the_count_it_claims():
    loading = _loading()
    sites = site_sets(loading, LAYERS, _tau(loading, 13), seed=7)
    assert set(sites) == set(ZIN_CONDITIONS)
    for k in SITE_COUNTS:
        for rule in RULES:
            assert len(sites[condition_name(rule, k)]) == k
    assert len(sites["active"]) == 13
    assert len(sites["random_active"]) == 13
    assert len(sites["bottom_active"]) == 13


def test_sets_are_nested_across_counts_under_every_rule():
    """The count axis is a dose along one fixed ordering: a step up adds sites and
    exchanges none. Random inherits this from sharing one seed across counts."""
    loading = _loading()
    sites = site_sets(loading, LAYERS, _tau(loading, 13), seed=7)
    for rule in RULES:
        for small, large in zip(SITE_COUNTS, SITE_COUNTS[1:], strict=False):
            assert set(sites[condition_name(rule, small)]) <= set(
                sites[condition_name(rule, large)]
            )


def test_top_and_bottom_select_by_loading_and_random_does_not():
    loading = _loading()
    sites = site_sets(loading, LAYERS, _tau(loading, 13), seed=7)
    ranked = loading.reshape(-1).argsort(descending=True)
    to_site = lambda i: (LAYERS[int(i) // N_HEADS], int(i) % N_HEADS)  # noqa: E731
    for k in SITE_COUNTS:
        assert sites[f"top_{k}"] == sorted(to_site(i) for i in ranked[:k])
        assert sites[f"bottom_{k}"] == sorted(to_site(i) for i in ranked[-k:])
        assert not set(sites[f"top_{k}"]) & set(sites[f"bottom_{k}"])
    # The threshold set is what the cut admits, here the 13 highest by construction of tau.
    assert sites["active"] == sorted(to_site(i) for i in ranked[:13])
    # A random set of 64 out of 128 that coincided with the top would be a seed accident;
    # the point is that it is not *built* from the loading.
    assert sites["random_64"] != sites["top_64"]
    reseeded = site_sets(loading, LAYERS, _tau(loading, 13), seed=8)
    assert reseeded["random_8"] != sites["random_8"]


def test_a_trial_with_nothing_active_gets_empty_controls_too():
    loading = _loading()
    sites = site_sets(loading, LAYERS, _tau(loading, 0), seed=7)
    assert sites["active"] == []
    assert sites["random_active"] == []
    assert sites["bottom_active"] == []
    assert len(sites["top_1"]) == 1
