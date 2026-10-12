"""Pairing, indexing and shard merging of the multi-site swap aggregation."""
import numpy as np
import pytest

from src.swap_sites_report import (
    merge_optional,
    merge_shards,
    paired_mean,
    paper_protocol,
    summarize,
)

CONFIG = {"conditions": ["top_1", "random_1"], "arms": ["J", "L", "J_rand", "L_rand"],
          "metrics": ["success", "probability", "kl", "top_id", "edit_norm"],
          "strength": "alpha", "strengths": [1.0], "n_trials_total": 4}


def _records():
    return [{"trial_index": i, "task": "t", "prompt_key": f"p{i // 2}", "success_ids": [1, 2]}
            for i in range(4)]


def test_paired_mean_weights_trials_and_counts_prompts():
    result = paired_mean(np.array([1., 1., -1.]), ['a', 'a', 'b'])
    assert result == {"mean": pytest.approx(1 / 3), "prompts": 2, "trials": 3}
    assert paired_mean(np.array([]), [])['mean'] is None
    with pytest.raises(ValueError):
        paired_mean(np.array([1., np.nan]), ['a', 'b'])


def test_contrasts_index_selection_and_arm_correctly():
    heads = np.zeros((4, 1, 2, 2, 4, 5), dtype=np.float32)
    heads[:, 0, 0, 0, 0, 0] = 1           # top_1, J-selected, J edit: all succeed
    heads[:2, 0, 0, 1, 1, 0] = 1          # top_1, L-selected, L edit: half succeed
    heads[:, 0, 0, 1, 0, 0] = 1           # top_1, L-selected, J edit: all succeed
    wide = np.zeros((4, 1, 4, 5), dtype=np.float32)
    wide[:, 0, 1, 0] = 1
    (row,) = summarize(heads, wide, _records(), CONFIG)
    assert row["cells"]["top_1"]["J/J"]["successes"] == 4
    assert row["cells"]["top_1"]["L/L"]["successes"] == 2
    c = row["contrasts"]["top_1"]
    assert c["end_to_end"]["mean"] == pytest.approx(0.5)
    assert c["direction_at_L_sites"]["mean"] == pytest.approx(0.5)
    assert c["selection_with_J_edit"]["mean"] == 0
    assert c["direction_at_J_sites"]["mean"] == 1
    assert c["end_to_end"]["prompts"] == 2
    assert row["layer_wide_contrast"]["mean"] == -1


def test_merge_restores_order_and_rejects_gaps():
    recs = _records()
    arrays = [{"heads": np.full((2, 1), k), "layer_wide": np.full((2, 1), k)} for k in (0, 1)]
    shard0 = ({"trials": [recs[0], recs[2]], "configuration": CONFIG}, arrays[0])
    shard1 = ({"trials": [recs[1], recs[3]], "configuration": CONFIG}, arrays[1])
    heads, _, merged, _ = merge_shards([shard0, shard1])
    assert [t["trial_index"] for t in merged] == [0, 1, 2, 3]
    assert heads[:, 0].tolist() == [0, 1, 0, 1]
    with pytest.raises(ValueError):
        merge_shards([shard0])


def test_nonfinite_outputs_are_counted_apart_not_as_failures():
    heads = np.zeros((4, 1, 2, 2, 4, 5), dtype=np.float32)
    wide = np.zeros((4, 1, 4, 5), dtype=np.float32)
    wide[:, 0, 0, 2] = np.nan
    wide[:, 0, 0, 0] = 1
    wide[:, 0, 1, 4] = np.inf             # huge edit, finite outputs: still valid
    wide[:, 0, 1, 0] = 1
    (row,) = summarize(heads, wide, _records(), CONFIG)
    assert row["layer_wide"]["J"] == {**row["layer_wide"]["J"], "successes": 0, "n_nonfinite": 4, "kl": None}
    assert row["layer_wide"]["L"]["successes"] == 4 and row["layer_wide"]["L"]["n_nonfinite"] == 0


def test_uniform_outputs_are_counted_as_degenerate():
    heads = np.zeros((4, 1, 2, 2, 4, 5), dtype=np.float32)
    wide = np.zeros((4, 1, 4, 5), dtype=np.float32)
    wide[:, 0, 0, 1] = 2 / 128256
    (row,) = summarize(heads, wide, _records(), CONFIG)
    assert row["layer_wide"]["J"]["n_nonfinite"] == 4
    assert row["layer_wide"]["L"]["n_nonfinite"] == 0


def test_rows_are_keyed_by_alpha():
    config = dict(CONFIG, strengths=[1.0, 2.0])
    heads = np.zeros((4, 2, 2, 2, 4, 5), dtype=np.float32)
    heads[:, 1, 0, 0, 0, 0] = 1           # alpha=2, top_1, J/J: all succeed
    wide = np.zeros((4, 2, 4, 5), dtype=np.float32)
    rows = summarize(heads, wide, _records(), config)
    assert [r["alpha"] for r in rows] == [1.0, 2.0]
    assert [r["cells"]["top_1"]["J/J"]["successes"] for r in rows] == [0, 4]


def test_band_reference_is_merged_in_trial_order_and_optional():
    def shard(indices):
        n = len(indices)
        band = np.zeros((n, 1, 4, 5), dtype=np.float32)
        band[:, 0, 0, 0] = [i % 2 for i in indices]    # J succeeds on odd trials
        record = {"trials": [_records()[i] for i in indices],
                  "configuration": CONFIG}
        return record, {"heads": np.zeros((n, 1, 2, 2, 4, 5), dtype=np.float32),
                        "layer_wide": np.zeros((n, 1, 4, 5), dtype=np.float32),
                        "layer_wide_band": band}

    shards = [shard([1, 3]), shard([0, 2])]
    band = merge_optional(shards, "layer_wide_band")
    assert band[:, 0, 0, 0].tolist() == [0, 1, 0, 1]
    assert merge_optional([(r, {k: v for k, v in a.items() if k != "layer_wide_band"})
                           for r, a in shards], "layer_wide_band") is None

    heads, wide, records, config = merge_shards(shards)
    (row,) = summarize(heads, wide, records, config, band=band)
    assert row["layer_wide_band"]["J"]["successes"] == 2
    assert "layer_wide_band" not in summarize(heads, wide, records, config)[0]


def test_paper_protocol_scores_rank_over_all_and_outside_the_clean_top10():
    metrics = ["success", "probability", "kl", "top_id", "edit_norm", "rank"]
    config = {"metrics": metrics, "arms": ["J", "L", "J_rand", "L_rand"], "strengths": [1.0]}
    records = [dict(r, clean_rank=c) for r, c in zip(_records(), [0, 3, 10, 40], strict=True)]
    wide = np.zeros((4, 1, 4, 6), dtype=np.float32)
    wide[:, 0, 0, 5] = [0, 0, 5, 0]          # J: ranks per trial
    wide[:, 0, 0, 0] = [1, 0, 0, 1]          # trial 1: rank 0 by a tie, not a greedy success
    wide[3, 0, 0, 1] = 2 / 128256            # trial 3's J output is uniform: degenerate
    rows = paper_protocol(wide, None, records, config)
    j = next(r for r in rows if r["lens"] == "J")
    assert j["all"] == {"n": 4, "top1": 1, "top5": 2}
    assert j["outside_clean_top10"] == {"n": 2, "top1": 0, "top5": 0}
    assert j["n_degenerate"] == 1 and {r["scope"] for r in rows} == {"layer_wide"}
