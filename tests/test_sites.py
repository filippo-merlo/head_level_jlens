"""Site selection, on hand-built loading maps where the answer is known.

Every rule here decides *what gets edited*, so a wrong one produces a perfectly plausible
success rate against a silently different experiment. These pin the semantics: which sites
each rule returns, that the count-matched controls really are count-matched, and that the
seeded control reproduces.
"""

from __future__ import annotations

import pytest
import torch

from src.config import LensSpec
from src.sites import (
    apply_site_deltas,
    loading_map,
    null_threshold,
    plan_edits,
    select_sites,
    site_spread,
)
from src.structural import workspace_loading

LAYERS = [15, 16, 17]
N_HEADS = 4


def _spec(head_dim: int = 3) -> LensSpec:
    return LensSpec(
        key="tiny",
        hf_model_id="tiny",
        lens_file="",
        n_layers=8,
        d_model=N_HEADS * head_dim,
        n_heads=N_HEADS,
        head_dim=head_dim,
        n_prompts_fitted=0,
    )


def _loading() -> torch.Tensor:
    """A map whose ordering is obvious by inspection: value == 100*row + col."""
    return torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]
    )


# ------------------------------------------------------------------ selection rules


def test_top_takes_the_highest_loading_sites():
    got = select_sites(_loading(), LAYERS, "top", k=3)
    assert got == [(17, 1), (17, 2), (17, 3)]


def test_bottom_takes_the_lowest_and_is_disjoint_from_top():
    top = select_sites(_loading(), LAYERS, "top", k=3)
    bottom = select_sites(_loading(), LAYERS, "bottom", k=3)
    assert bottom == [(15, 0), (15, 1), (15, 2)]
    assert not set(top) & set(bottom), "anti-selection overlapped the selection"


def test_top1_is_the_single_best_site():
    assert select_sites(_loading(), LAYERS, "top", k=1) == [(17, 3)]


def test_threshold_selects_per_site_against_its_own_cut():
    """A per-site tau, not a global one -- the whole point of :func:`null_threshold`."""
    tau = torch.full_like(_loading(), 100.0)
    tau[0, 0] = -1.0          # a site with a low bar admits its concept
    tau[2, 3] = 1000.0        # the highest-loading site does not clear its own bar
    got = select_sites(_loading(), LAYERS, "threshold", tau=tau)
    assert got == [(15, 0)]


def test_threshold_is_strict():
    """Equality is not activity; a concept must *beat* the null, not tie it."""
    assert select_sites(_loading(), LAYERS, "threshold", tau=_loading()) == []


def test_threshold_accepts_a_scalar_cut():
    got = select_sites(_loading(), LAYERS, "threshold", tau=12.5)
    assert got == [(16, 3), (17, 0), (17, 1), (17, 2), (17, 3)]


def test_random_is_count_matched_and_reproducible():
    """The site-count control is worthless if it is not exactly count-matched, and
    unreproducible if the seed does not fix it."""
    first = select_sites(_loading(), LAYERS, "random", k=5, seed=7)
    again = select_sites(_loading(), LAYERS, "random", k=5, seed=7)
    other = select_sites(_loading(), LAYERS, "random", k=5, seed=8)
    assert len(first) == 5
    assert first == again
    assert first != other, "two seeds drew the same control set"


def test_random_draws_without_replacement():
    got = select_sites(_loading(), LAYERS, "random", k=len(LAYERS) * N_HEADS, seed=0)
    assert len(got) == len(LAYERS) * N_HEADS, "a site was drawn twice"


def test_sites_are_sorted_by_layer_then_head():
    got = select_sites(_loading(), LAYERS, "top", k=6)
    assert got == sorted(got)


def test_selection_uses_the_supplied_layer_labels():
    """Rows carry the caller's layer labels, not 0..n -- an off-by-one in that mapping
    would edit the wrong depth."""
    got = select_sites(_loading(), [30, 31, 32], "top", k=1)
    assert got == [(32, 3)]


@pytest.mark.parametrize(
    ("rule", "kwargs", "match"),
    [
        ("top", {}, "needs k"),
        ("random", {"k": 2}, "needs a seed"),
        ("threshold", {}, "needs tau"),
        ("top", {"k": 0}, "out of range"),
        ("top", {"k": 99}, "out of range"),
        ("nonsense", {}, "unknown rule"),
    ],
)
def test_selection_rejects_an_underspecified_request(rule, kwargs, match):
    with pytest.raises(ValueError, match=match):
        select_sites(_loading(), LAYERS, rule, **kwargs)


def test_selection_rejects_a_layer_label_mismatch():
    with pytest.raises(ValueError, match="loading rows"):
        select_sites(_loading(), [15, 16], "top", k=1)


# ------------------------------------------------------------------ the loading map


def test_loading_map_stacks_layers_in_the_given_order():
    spec = _spec()
    torch.manual_seed(0)
    jz = {l: torch.randn(spec.d_model, spec.d_source) for l in LAYERS}
    z = {l: torch.randn(spec.d_source) for l in LAYERS}
    row = torch.randn(spec.d_model)

    got = loading_map(jz, z, spec, row, LAYERS)
    assert got.shape == (len(LAYERS), spec.n_heads)
    for index, layer in enumerate(LAYERS):
        torch.testing.assert_close(
            got[index], workspace_loading(jz[layer], z[layer], spec, row)
        )


def test_loading_map_reversed_layers_reverse_the_rows():
    """Row order follows ``layers``, not dict insertion, or every recorded site is wrong."""
    spec = _spec()
    torch.manual_seed(1)
    jz = {l: torch.randn(spec.d_model, spec.d_source) for l in LAYERS}
    z = {l: torch.randn(spec.d_source) for l in LAYERS}
    row = torch.randn(spec.d_model)

    forward = loading_map(jz, z, spec, row, LAYERS)
    backward = loading_map(jz, z, spec, row, list(reversed(LAYERS)))
    torch.testing.assert_close(forward, backward.flip(0))


def test_loading_map_batches_the_concept_population():
    """The 256-token null must equal 256 single-token maps, or the threshold is not the
    quantile it claims to be."""
    spec = _spec()
    torch.manual_seed(2)
    jz = {l: torch.randn(spec.d_model, spec.d_source) for l in LAYERS}
    z = {l: torch.randn(spec.d_source) for l in LAYERS}
    rows = torch.randn(5, spec.d_model)

    batched = loading_map(jz, z, spec, rows, LAYERS)
    assert batched.shape == (5, len(LAYERS), spec.n_heads)
    for token in range(5):
        torch.testing.assert_close(
            batched[token], loading_map(jz, z, spec, rows[token], LAYERS)
        )


def test_loading_map_rejects_a_missing_layer():
    spec = _spec()
    jz = {15: torch.randn(spec.d_model, spec.d_source)}
    z = {15: torch.randn(spec.d_source)}
    with pytest.raises(ValueError, match="no J\\^z or z for layers"):
        loading_map(jz, z, spec, torch.randn(spec.d_model), [15, 16])


# ------------------------------------------------------------------ the null threshold


def test_null_threshold_is_per_site():
    null = torch.zeros(100, 2, 2)
    null[:, 0, 0] = torch.linspace(0.0, 1.0, 100)
    null[:, 1, 1] = torch.linspace(0.0, 10.0, 100)
    tau = null_threshold(null, q=0.99)
    assert tau.shape == (2, 2)
    assert float(tau[0, 0]) == pytest.approx(0.99, abs=0.02)
    assert float(tau[1, 1]) == pytest.approx(9.9, abs=0.2)
    assert float(tau[0, 1]) == pytest.approx(0.0, abs=1e-6)


def test_null_threshold_admits_about_the_complement_of_q():
    """The rule's selectivity is its whole justification: at q=0.99 an arbitrary concept
    clears ~1% of sites, so a 1024-site grid admits ~10 by chance."""
    torch.manual_seed(3)
    null = torch.randn(4000, 8, 8)
    tau = null_threshold(null, q=0.99)
    admitted = (null > tau).float().mean()
    assert float(admitted) == pytest.approx(0.01, abs=0.003)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.5, 2.0])
def test_null_threshold_rejects_a_degenerate_quantile(q):
    with pytest.raises(ValueError, match="quantile"):
        null_threshold(torch.randn(4, 2, 2), q=q)


def test_null_threshold_rejects_a_map_without_a_token_axis():
    with pytest.raises(ValueError, match="n_tokens"):
        null_threshold(torch.randn(2, 2), q=0.9)


# ------------------------------------------------------------------ GQA independence


# ------------------------------------------------------------------ the batched edit


def test_plan_edits_groups_elements_by_layer_arm_and_head():
    plan = plan_edits([
        ([(15, 0), (16, 1)], "head_jlens"),
        ([(15, 0)], "head_jlens"),
        ([(15, 0)], "matched_norm"),
    ])
    assert plan[15]["head_jlens"][0] == [0, 1], "elements sharing a site must share the slice"
    assert plan[15]["matched_norm"][0] == [2], "arms must not be merged; they apply different updates"
    assert plan[16] == {"head_jlens": {1: [0]}}
    assert 17 not in plan


def test_plan_edits_of_nothing_is_empty():
    assert plan_edits([([], "head_jlens")]) == {}


def _edit_spec() -> LensSpec:
    return _spec(head_dim=3)


def test_apply_site_deltas_touches_only_the_selected_element_and_head():
    """The property the whole design rests on: an element that selected no site here must
    come back untouched, or every condition quietly becomes the same condition."""
    spec = _edit_spec()
    torch.manual_seed(20)
    z = torch.randn(3, 4, spec.d_source)
    plan = plan_edits([([(15, 1)], "head_jlens"), ([], "head_jlens"), ([], "head_jlens")])

    out = apply_site_deltas(
        z, spec, plan[15], lambda arm, head, block, index: torch.ones_like(block)
    )
    torch.testing.assert_close(out[1], z[1])
    torch.testing.assert_close(out[2], z[2])
    head_slice = spec.head_slice(1)
    torch.testing.assert_close(out[0, :, head_slice], z[0, :, head_slice] + 1.0)
    # every coordinate outside head 1 is untouched, including inside the edited element
    untouched = [i for i in range(spec.d_source) if not (3 <= i < 6)]
    torch.testing.assert_close(out[0][:, untouched], z[0][:, untouched])


def test_apply_site_deltas_matches_a_per_element_loop():
    """Batching over elements is an optimisation; it must equal editing them one at a
    time, or the site conditions are not the conditions they claim to be."""
    spec = _edit_spec()
    torch.manual_seed(21)
    z = torch.randn(4, 5, spec.d_source)
    elements = [
        ([(15, 0), (15, 3)], "head_jlens"),
        ([(15, 3)], "matched_norm"),
        ([], "head_jlens"),
        ([(15, 0), (15, 1), (15, 2)], "head_jlens"),
    ]

    def delta_for(arm, head, block, index):
        # deterministic, arm- and head-dependent, so a mixed-up grouping cannot pass
        return block * (0.5 if arm == "head_jlens" else -2.0) + float(head)

    batched = apply_site_deltas(z, spec, plan_edits(elements)[15], delta_for)
    for position, (sites, arm) in enumerate(elements):
        alone = apply_site_deltas(
            z[position : position + 1],
            spec,
            plan_edits([(sites, arm)]).get(15, {}),
            delta_for,
        )
        torch.testing.assert_close(batched[position : position + 1], alone)


def test_apply_site_deltas_accumulates_total_norms_per_element():
    """Pass 2's scale factor is this number, so it has to be the total over every site an
    element edited -- summed across layers, not per layer."""
    spec = _edit_spec()
    z = torch.zeros(2, 2, spec.d_source)
    totals = torch.zeros(2)
    plan = plan_edits([([(15, 0), (15, 1)], "head_jlens"), ([(15, 0)], "head_jlens")])

    apply_site_deltas(
        z, spec, plan[15], lambda arm, head, block, index: torch.ones_like(block),
        totals=totals,
    )
    # element 0 edited two heads (2 * 2 positions * 3 dims), element 1 edited one.
    assert float(totals[0]) == pytest.approx(12.0)
    assert float(totals[1]) == pytest.approx(6.0)


def test_apply_site_deltas_rejects_a_misshapen_delta():
    spec = _edit_spec()
    z = torch.zeros(2, 2, spec.d_source)
    with pytest.raises(ValueError, match="expected"):
        apply_site_deltas(
            z,
            spec,
            plan_edits([([(15, 0)], "head_jlens")])[15],
            lambda arm, head, block, index: torch.ones_like(block)[..., :-1],
        )


def test_site_spread_counts_shared_value_circuits():
    """Llama-3.1-8B shares one W_V per 4 query heads, so heads 0-3 of a layer are one
    group and a set inside it is not four independent sites."""
    spread = site_spread([(15, 0), (15, 1), (15, 2)], n_heads=32, n_kv_heads=8)
    assert spread == {
        "n_sites": 3,
        "n_layers": 1,
        "n_layer_groups": 1,
        "max_sites_per_layer_group": 3,
    }


def test_site_spread_separates_groups_and_layers():
    sites = [(15, 0), (15, 4), (16, 0)]
    spread = site_spread(sites, n_heads=32, n_kv_heads=8)
    assert spread["n_layer_groups"] == 3
    assert spread["max_sites_per_layer_group"] == 1
    assert spread["n_layers"] == 2


def test_site_spread_handles_an_empty_set():
    """The threshold rule can select nothing, and the record still has to say so."""
    assert site_spread([], n_heads=32, n_kv_heads=8)["max_sites_per_layer_group"] == 0


def test_site_spread_rejects_an_impossible_grouping():
    with pytest.raises(ValueError, match="do not divide"):
        site_spread([(15, 0)], n_heads=32, n_kv_heads=7)
