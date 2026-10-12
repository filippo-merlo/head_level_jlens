"""pass@k aggregation and the paired difference, on arrays with known answers.

Experiment 1 is read off these numbers, so the reduction (best site: min over heads, then
over layers, never "per head" across layers) is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evals import (
    NEEDS_SYNONYM_EXPANSION,
    SINGLE_TOKEN_ONLY,
    SYNONYMS,
    EvalItem,
    control_items,
    target_forms,
)
from src.passk import (
    PAD,
    best_site_agreement,
    best_site_rank,
    paired_difference,
    passk,
    sites_hit,
)


def test_passk_counts_zero_based_rank_below_k():
    """Ranks are zero-based (0 = top-1), so top-k is ``rank < k``, not ``<= k``.

    The boundary is the whole test: a token in 6th place (rank 5) is *not* within the
    top 5, and a token in 2nd place (rank 1) is not a pass@1 hit. An off-by-one here
    silently reports top-(k+1) under the name pass@k.
    """
    # 4 items, 1 intermediate each, ranks 0, 1, 5, 100
    ranks = np.array([[0], [1], [5], [100]])
    assert passk(ranks, 1) == 0.25
    assert passk(ranks, 5) == 0.5
    assert passk(ranks, 6) == 0.75
    assert passk(ranks, 101) == 1.0


def test_passk_is_fraction_of_intermediates_then_mean_over_items():
    """Items are weighted equally regardless of how many intermediates they carry.

    Pooling over all intermediates instead would let a four-intermediate item count four
    times as much as a one-intermediate item, which is not the documented metric.
    """
    # item 0 has 2 intermediates (one hit), item 1 has 1 intermediate (hit)
    ranks = np.array([[0, 50], [0, PAD]])
    # item 0: 1/2 hit at k=1; item 1: 1/1 -> mean 0.75
    assert passk(ranks, 1) == 0.75


def test_passk_ignores_padding():
    assert passk(np.array([[0, PAD, PAD]]), 1) == 1.0
    assert np.isnan(passk(np.array([[PAD, PAD]]), 1))


def test_an_item_with_no_scoreable_intermediate_is_neither_hit_nor_miss():
    """Under a tokenizer that splits numerals, nine multihop items keep no candidate at
    all; counting them as misses would scale every arm's rate by the same factor."""
    from src.passk import scoreable_items

    scored = np.array([[0, 50], [3, PAD], [0, 0]])
    with_blank = np.array([[0, 50], [PAD, PAD], [3, PAD], [PAD, PAD], [0, 0]])
    assert scoreable_items(with_blank).tolist() == [0, 2, 4]
    for k in (1, 5):
        assert passk(with_blank, k) == passk(scored, k)
    other = with_blank.copy()
    other[0, 0] = 10
    assert paired_difference(with_blank, other, 1) == pytest.approx(1 / 6)
    # Arms must pad the same slots: a blank item in one arm only is not a paired design.
    with pytest.raises(ValueError):
        paired_difference(with_blank, scored[[0, 1, 1, 1, 2]], 1)


def test_best_site_is_the_minimum_over_heads_and_layers():
    """The aggregate is the best of every site, and a per-layer value the best head there."""
    # 1 item, 1 intermediate, 2 layers x 3 heads
    ranks = np.array([[[[40, 7, 90], [3, 60, 8]]]])
    assert best_site_rank(ranks).tolist() == [[3]]
    assert best_site_rank(ranks, layers=[0]).tolist() == [[7]]
    assert best_site_rank(ranks, layers=[1]).tolist() == [[3]]


def test_best_site_does_not_pair_heads_across_layers():
    """A head has no depth: column index ``h`` names unrelated heads at different layers.

    The retired aggregation minimised over layers *per column index* and averaged the 32
    results. On this array that gives 0.5 at k=1 (index 0 is best-of-{0,9} = 0, index 1 is
    best-of-{9,9} = 9); the best site gives 1.0. The two must not coincide here.
    """
    ranks = np.array([[[[0, 9], [9, 9]]]])
    assert passk(best_site_rank(ranks), 1) == 1.0
    per_index_then_mean = np.mean([passk(ranks[:, :, :, h].min(axis=2), 1) for h in range(2)])
    assert per_index_then_mean == 0.5


def test_best_site_keeps_padding_and_the_reference_shape():
    ranks = np.full((2, 3, 4, 5), PAD)
    ranks[0, 0, 2, 1] = 12
    out = best_site_rank(ranks)
    assert out.shape == (2, 3)
    assert out[0, 0] == 12 and (out[out != 12] == PAD).all()
    # the whole-layer reference has one "head"
    ref = np.array([[[[5], [2], [PAD]]]])
    assert best_site_rank(ref).tolist() == [[2]]
    assert best_site_rank(ref, layers=[2]).tolist() == [[PAD]]


def test_best_site_agreement_counts_a_shared_minimiser_within_the_region():
    """Each arm takes its own minimum; agreement is whether some site is best under both."""
    # 2 items, 1 intermediate, 2 layers x 2 heads. Item 0: both arms best at (L1, h0).
    # Item 1: arm a best at (L0, h1), arm b at (L1, h1); at layer 0 alone both are best
    # at h1, since b ties there, and at layer 1 alone a is best at h0, b at h1.
    a = np.array([[[[5, 9], [1, 7]]], [[[8, 2], [6, 9]]]])
    b = np.array([[[[4, 9], [0, 8]]], [[[3, 3], [9, 1]]]])
    assert best_site_agreement(a, b) == 0.5
    assert best_site_agreement(a, b, layers=[0]) == 1.0
    assert best_site_agreement(a, b, layers=[1]) == 0.5
    assert best_site_agreement(a, a) == 1.0
    assert best_site_agreement(a, b) == best_site_agreement(b, a)


def test_best_site_agreement_ignores_padded_intermediates():
    a = np.full((1, 2, 1, 2), PAD)
    b = np.full((1, 2, 1, 2), PAD)
    a[0, 0] = [[3, 1]]
    b[0, 0] = [[0, 2]]
    assert best_site_agreement(a, b) == 0.0
    with pytest.raises(ValueError):
        best_site_agreement(a, b[:, :1])


def test_sites_hit_counts_every_site_within_the_top_k():
    """All sites count, not the best one: zero-based ranks, ``rank < k``, padding ignored."""
    # 1 item, 1 intermediate, 2 layers x 3 heads
    ranks = np.array([[[[0, 4, 5], [1, 90, 0]]]])
    assert sites_hit(ranks, 1) == 2.0
    assert sites_hit(ranks, 5) == 4.0  # rank 5 is 6th place
    assert sites_hit(ranks, 6) == 5.0
    padded = np.full((1, 2, 2, 3), PAD)
    padded[0, :1] = ranks[0]
    assert sites_hit(padded, 5) == 4.0


def test_sites_hit_is_weighted_as_passk():
    """Mean over an item's intermediates, then over scoreable items; a blank item is skipped."""
    ranks = np.full((3, 2, 1, 2), PAD)
    ranks[0, 0] = [[0, 0]]  # item 0: 2 sites, then 0 sites -> 1.0
    ranks[0, 1] = [[9, 9]]
    ranks[2, 0] = [[0, 9]]  # item 2: 1 site -> 1.0; item 1 is blank
    assert sites_hit(ranks, 1) == 1.0
    assert np.isnan(sites_hit(np.full((1, 1, 1, 1), PAD), 1))


def test_paired_difference_is_the_mean_per_item_difference():
    """Both arms average over the same items, so the gap is the mean of per-item gaps."""
    a = np.array([[0, 50], [3, PAD], [0, 0]])
    b = np.array([[9, 50], [0, PAD], [9, 9]])
    # per item at k=1: a = 1/2, 0, 1; b = 0, 1, 0 -> differences +1/2, -1, +1
    assert paired_difference(a, b, 1) == pytest.approx((0.5 - 1 + 1) / 3)
    assert paired_difference(a, a.copy(), 1) == 0.0


def test_paired_difference_requires_alignment():
    with pytest.raises(ValueError, match="align"):
        paired_difference(np.zeros((3, 1)), np.zeros((4, 1)), 1)


# ------------------------------------------------------------------ synonym expansion


def _item(slug: str) -> EvalItem:
    return EvalItem(name="x", prompt="p", intermediates=("5",), slug=slug)


def test_target_forms_passes_through_non_synonym_sets():
    assert target_forms(_item("lens-eval-association"), "grief") == ("grief",)


def test_target_forms_expands_order_ops():
    forms = target_forms(_item("lens-eval-order-ops"), "5")
    assert "5" in forms and "five" in forms
    ops = target_forms(_item("lens-eval-order-ops"), "multiplication")
    assert "*" in ops and "times" in ops


def test_target_forms_rejects_an_unauthored_intermediate():
    """An unmapped intermediate must raise, not fall back to the literal string.

    Falling back would score that item on a harder target than its siblings and quietly
    depress the whole set's pass@k.
    """
    with pytest.raises(ValueError, match="no authored synonym set"):
        target_forms(_item("lens-eval-order-ops"), "hypotenuse")


def test_every_order_ops_intermediate_is_authored():
    """The shipped set must be fully covered, or scoring it raises mid-run."""
    from src.evals import load_eval

    item = _item("lens-eval-order-ops")
    for real in load_eval("lens-eval-order-ops"):
        for intermediate in real.intermediates:
            assert intermediate in SYNONYMS, f"{intermediate!r} needs a synonym set"
            assert target_forms(item, intermediate)


# ------------------------------------------------------- the scoring candidate rule


def test_single_token_only_covers_the_two_sets_that_need_it():
    """order-ops because the README says so; multihop because of the bare-space artefact.

    Pinned as a set rather than derived, because the two sets are in it for different
    reasons and a derived rule would silently pick up the other four -- where it removes
    nothing and costs coverage (see :data:`src.evals.SINGLE_TOKEN_ONLY`).
    """
    assert SINGLE_TOKEN_ONLY == {"lens-eval-order-ops", "lens-eval-multihop"}
    # The two concerns are separate: multihop needs the candidate rule but not synonyms.
    assert "lens-eval-multihop" not in NEEDS_SYNONYM_EXPANSION
    assert NEEDS_SYNONYM_EXPANSION < SINGLE_TOKEN_ONLY


def test_single_token_only_drops_a_split_spelling_and_keeps_the_whole_one():
    """The artefact, in miniature: a spelling the tokenizer splits contributes nothing.

    Under the loose rule the split spelling donates its first fragment; under the strict
    rule it is dropped, and only forms that are one token survive.
    """
    from src.readout import target_token_ids

    class FakeTok:
        """`" 3"` splits into a bare space plus the digit; `"3"` is one token."""

        def __call__(self, text, add_special_tokens=False):
            table = {" 3": [220, 18], "3": [18]}
            return type("Enc", (), {"input_ids": table[text]})()

    loose = target_token_ids(FakeTok(), "3", single_token_only=False)
    strict = target_token_ids(FakeTok(), "3", single_token_only=True)
    assert 220 in loose, "the loose rule is what admits the bare space"
    assert strict == [18], "the strict rule keeps only the single-token spelling"


class _SplitsDigits:
    """A tokenizer that keeps every plain word whole and, like Qwen3.5's and Gemma-3's,
    splits a numeral into its digits and a hyphenated word at the hyphen."""

    def __init__(self, split_digits: bool):
        self.split_digits = split_digits

    def __call__(self, text, add_special_tokens=False):
        body = text.lstrip(" ")
        if self.split_digits and (body.isdigit() or "-" in body):
            pieces = list(body) if body.isdigit() else body.split("-")
            ids = ([220] if text.startswith(" ") else []) + [
                abs(hash(piece)) % 100_000 + 2000 for piece in pieces
            ]
        else:
            ids = [abs(hash(text)) % 100_000 + 2000]
        return type("Enc", (), {"input_ids": ids})()


def test_candidate_coverage_names_the_slots_a_digit_splitting_tokenizer_cannot_score():
    from src.evals import ALL_SETS, candidate_coverage

    whole = {slug: candidate_coverage(_SplitsDigits(False), slug) for slug in ALL_SETS}
    assert all(c["n_scoreable"] == c["n_intermediates"] for c in whole.values())
    assert all(c["items_without_candidates"] == [] for c in whole.values())

    split = {slug: candidate_coverage(_SplitsDigits(True), slug) for slug in ALL_SETS}
    multihop, order = split["lens-eval-multihop"], split["lens-eval-order-ops"]
    assert multihop["n_intermediates"] - multihop["n_scoreable"] == 9
    assert all(word.isdigit() for _, word in multihop["unscoreable"])
    # Each of those nine items carries that numeral alone, so the item drops out of the
    # mean rather than counting as a miss (src.passk.scoreable_items).
    assert len(multihop["items_without_candidates"]) == 9
    assert order["unscoreable"] == [("mult-div-left", "24")]
    assert order["items_without_candidates"] == []
    for slug in set(ALL_SETS) - {"lens-eval-multihop", "lens-eval-order-ops"}:
        assert split[slug]["unscoreable"] == [], slug


def test_control_item_is_another_item_sharing_no_candidate_with_the_concept():
    """The control prompt is never the item itself, nor one annotated with any of the
    concept's tokens; unscoreable slots stay unscored; the draw is fixed by the seed."""
    candidates = [
        [frozenset({1}), frozenset({2, 3})],
        [frozenset({3})],  # shares token 3 with item 0's second concept
        [frozenset({4}), None],
        [frozenset({1, 5})],  # shares token 1 with item 0's first concept
        [frozenset({6})],
    ]
    for seed in range(20):
        controls = control_items(candidates, seed=seed)
        assert controls == control_items(candidates, seed=seed)
        assert controls[2][1] is None
        for i, row in enumerate(controls):
            for ids, j in zip(candidates[i], row, strict=True):
                if ids is None:
                    continue
                assert j != i
                assert not ids & frozenset().union(*(c for c in candidates[j] if c))
    assert len({tuple(map(tuple, control_items(candidates, seed=s))) for s in range(20)}) > 1


def test_control_items_refuses_when_no_prompt_is_free_of_the_concept():
    with pytest.raises(ValueError):
        control_items([[frozenset({1})], [frozenset({1, 2})]])
