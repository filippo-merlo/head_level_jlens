"""The figure driver's series: what each figure draws, pinned on a synthetic record.

The pixels are not tested. What is tested is that every series a figure plots is the
cell of the record the caption says it is -- the same guarantee ``scripts.report`` gives
its tables -- and that each figure draws from those series without touching the record
again. A drawing smoke test then runs the whole driver on the synthetic records.
"""

from __future__ import annotations

import json
import math

import pytest

from scripts import figures
from src.config import PROJECT_ROOT

SETS = ("lens-eval-multihop", "lens-eval-typo")
#: A few layers for the per-layer series, out of step so that order is tested, and a top
#: layer whose region -- like layer 31 of the shipped record -- has no J-layer reference.
LAYERS = [0, 5, 15, 25, 31]
NO_REFERENCE_LAYER = 31
K = (1, 5, 25)
COUNTS = [1, 8]
RULES = ["top", "random", "bottom"]


def _stat(value: float) -> dict:
    return {"point": value}


def _read_record() -> dict:
    """A read record with distinct values in every cell a figure reads."""
    results = {}
    for s, slug in enumerate(SETS):
        regions = {}
        for region in ("all_layers", *(f"layer_{layer}" for layer in LAYERS)):
            offset = 0.0 if region == "all_layers" else 0.001 * (LAYERS.index(int(region[6:])) + 1)
            arms = ("jhead", "logithead", "jlayer")
            if region == f"layer_{NO_REFERENCE_LAYER}":
                arms = arms[:2]
            regions[region] = {
                "k": {
                    str(k): {
                        "rates": {
                            arm: _stat(0.1 * s + 0.01 * k + 0.001 * a + offset)
                            for a, arm in enumerate(arms)
                        },
                        "paired_difference": {
                            "jhead_minus_logithead": _stat(0.05 * s + 0.002 * k + offset),

                        },
                    }
                    for k in K
                }
            }
        results[slug] = {"n_items": 3, "regions": regions}
    return {
        "meta": {"timestamp": "t", "git": {"sha": "x", "dirty": False}, "layers": LAYERS},
        "magnitude": {
            "relative_write_norm": {
                str(layer): [0.01 * (layer + 1), 0.02 * (layer + 1)] for layer in LAYERS
            }
        },
        "results": results,
    }


def _write_record() -> dict:
    """A write record whose condition names are deliberately not parseable into
    (rule, count) -- the series must come from the recorded ``conditions`` block."""
    conditions = {"ref": {"anchor": "out", "rule": None, "count": None}}
    for rule in RULES:
        for count in [*COUNTS, "active"]:
            conditions[f"{rule}-{count}"] = {"anchor": "zin", "rule": rule, "count": count}
    results = {}
    for t, task in enumerate(("verbal-report", "probe-swap")):
        entry = {
            "n_trials": 10 * (t + 1),
            "edit_norm": {},
            "selection": {"n_active": {"median": 5 + t}},
        }
        for arm in ("head_jlens", "matched_norm"):
            for alpha in (1.0, 2.0):
                table = {}
                for name in conditions:
                    rule = name.split("-")[0]
                    rate = (
                        0.9
                        if name == "ref"
                        else 0.1 * t
                        + 0.01 * alpha
                        + 0.001 * RULES.index(rule)
                        + 0.005 * (arm == "matched_norm")
                    )
                    table[name] = {"rate": rate}
                entry[f"{arm}_alpha{alpha}"] = table
                entry["edit_norm"][f"{arm}_alpha{alpha}"] = {
                    name: 1.0 + i for i, name in enumerate(conditions) if name != "ref"
                }
        results[task] = entry
    return {
        "meta": {
            "timestamp": "t",
            "git": {"sha": "y", "dirty": False},
            "alphas": [1.0, 2.0],
            "site_counts": COUNTS,
            "rules": RULES,
        },
        "conditions": conditions,
        "results": results,
    }


# ------------------------------------------------------------------ Experiment 1


def test_per_layer_follows_the_recorded_layer_order():
    read = _read_record()
    series = figures.per_layer(read, 5, "logithead")
    assert len(series["typo"]) == len(LAYERS)
    for i, layer in enumerate(LAYERS):
        cell = read["results"]["lens-eval-typo"]["regions"][f"layer_{layer}"]["k"]["5"]
        assert series["typo"][i] == cell["rates"]["logithead"]["point"]
    assert figures.per_layer(read, 25, "logithead")["typo"] != series["typo"]


def test_per_layer_is_nan_only_where_the_arm_has_no_rate():
    read = _read_record()
    reference = figures.per_layer(read, 5, "jlayer")["typo"]
    for i, layer in enumerate(LAYERS):
        if layer == NO_REFERENCE_LAYER:
            assert math.isnan(reference[i])
        else:
            cell = read["results"]["lens-eval-typo"]["regions"][f"layer_{layer}"]["k"]["5"]
            assert reference[i] == cell["rates"]["jlayer"]["point"]
    for arm in ("jhead", "logithead"):
        assert not any(math.isnan(v) for v in figures.per_layer(read, 5, arm)["typo"])


def test_per_layer_difference_is_jhead_minus_logithead():
    read = _read_record()
    series = figures.per_layer_difference(read, 25)
    assert len(series["multihop"]) == len(LAYERS)
    for i, layer in enumerate(LAYERS):
        cell = read["results"]["lens-eval-multihop"]["regions"][f"layer_{layer}"]["k"]["25"]
        expected = cell["paired_difference"]["jhead_minus_logithead"]
        assert series["multihop"][i] == expected["point"]
    assert figures.per_layer_difference(read, 5)["multihop"] != series["multihop"]


def test_write_norms_are_listed_by_layer_in_order():
    layers, values = figures.write_norms(_read_record())
    assert layers == LAYERS
    assert values[2] == [0.01 * (LAYERS[2] + 1), 0.02 * (LAYERS[2] + 1)]


# ------------------------------------------------------------------ Experiment 2


def test_condition_rows_come_from_the_record_not_the_name():
    rows, reference = figures.condition_rows(_write_record())
    assert reference == "ref"
    assert rows[("top", 8)] == "top-8"
    assert rows[("bottom", "active")] == "bottom-active"
    assert len(rows) == len(RULES) * (len(COUNTS) + 1)


def test_swap_rates_lay_out_the_count_axis():
    write = _write_record()
    grid = figures.swap_rates(write, "probe-swap", "head_jlens", 2.0)
    table = write["results"]["probe-swap"]["head_jlens_alpha2.0"]
    assert grid["counts"] == COUNTS
    assert grid["n_trials"] == 20
    assert grid["active_count"] == 6
    for rule in RULES:
        assert len(grid["rules"][rule]) == len(COUNTS)
        s = table[f"{rule}-8"]
        assert grid["rules"][rule][1] == s["rate"]
        s = table[f"{rule}-active"]
        assert grid["active"][rule] == s["rate"]
    assert grid["reference"] == table["ref"]["rate"]
    # The other arm and alpha are different tables, never a maximum over them.
    assert figures.swap_rates(write, "probe-swap", "head_jlens", 1.0) != grid
    assert figures.swap_rates(write, "probe-swap", "matched_norm", 2.0) != grid


def test_edit_norms_have_no_reference_row():
    write = _write_record()
    grid = figures.edit_norms(write, "verbal-report", "head_jlens", 2.0)
    table = write["results"]["verbal-report"]["edit_norm"]["head_jlens_alpha2.0"]
    assert grid["reference"] is None
    assert grid["rules"]["random"] == [table["random-1"], table["random-8"]]
    assert grid["active"]["bottom"] == table["bottom-active"]


# ------------------------------------------------------------------ drawing


@pytest.mark.parametrize("k", figures.MAIN_K)
def test_per_layer_figure_draws_only_the_two_head_arms_and_reference(k):
    read = _read_record()
    fig = figures.figure_per_layer(read, k)
    expected = ["J-head", "Logit-head", "J-layer (whole-layer reference)"]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == expected
    for ax, slug in zip(fig.axes, SETS, strict=True):
        assert len(ax.lines) == 3
        reference, logit, head = ax.lines
        for line, arm in [(reference, "jlayer"), (logit, "logithead"), (head, "jhead")]:
            series = figures.per_layer(read, k, arm)[slug.removeprefix("lens-eval-")]
            assert line.get_ydata().tolist() == pytest.approx(series, nan_ok=True)
    figures.plt.close(fig)


def test_factual_only_intervention_figure_uses_one_task():
    fig = figures.figure_intervention(_write_record(), tasks=("probe-swap",))
    assert len(fig.axes) == 2
    assert all("intermediate replacement" in ax.get_title(loc="left") for ax in fig.axes)
    figures.plt.close(fig)


def test_draw_all_writes_every_figure_from_the_synthetic_records(tmp_path):
    written = figures.draw_all(
        {"read": _read_record(), "write": _write_record(),
         "swap_logit": [_swap_summary(1.0), _swap_summary(2.0)]},
        tmp_path,
    )
    assert written == list(figures.FIGURES)
    for name in figures.FIGURES:
        assert (tmp_path / f"{name}.png").stat().st_size > 0
        assert (tmp_path / f"{name}.pdf").stat().st_size > 0


def test_draw_all_skips_figures_whose_record_is_missing(tmp_path):
    written = figures.draw_all({"read": _read_record(), "write": None}, tmp_path)
    assert written == [n for n, (rec, _) in figures.FIGURES.items() if rec == "read"]
    assert not (tmp_path / "fig3_intervention.png").exists()


def test_series_read_off_the_shipped_records():
    """The synthetic records mirror the shipped layout; if the drivers change it, this
    is the test that notices, so the figures cannot silently go blank."""
    read_path = PROJECT_ROOT / "results" / "read" / "llama3.1-8b-it_read.json"
    write_path = PROJECT_ROOT / "results" / "write" / "llama3.1-8b-it_write.json"
    if not (read_path.exists() and write_path.exists()):
        pytest.skip("shipped records not present")
    read = json.loads(read_path.read_text())
    write = json.loads(write_path.read_text())
    for k in figures.K_VALUES:
        assert all(len(v) == 32 for v in figures.per_layer(read, k, "jhead").values())
        assert all(len(v) == 32 for v in figures.per_layer_difference(read, k).values())
    # The J-layer reference is defined at layers 0-30 only (paper S3).
    for series in figures.per_layer(read, 5, "jlayer").values():
        assert math.isnan(series[31])
        assert not any(math.isnan(v) for v in series[:31])
    for task in write["results"]:
        for alpha in write["meta"]["alphas"]:
            grid = figures.swap_rates(write, task, figures.ARM, alpha)
            assert grid["reference"] is not None
            assert set(grid["rules"]) == set(write["meta"]["rules"])


def test_depth_ticks_never_collide_at_the_last_layer():
    from scripts.figures import _depth_ticks

    assert _depth_ticks(list(range(32))) == [0, 8, 16, 24, 31]
    assert _depth_ticks(list(range(34))) == [0, 8, 16, 24, 33]  # not 32 beside 33
    assert _depth_ticks(list(range(3, 32, 4))) == [3, 7, 11, 15, 19, 23, 27, 31]


def _swap_summary(alpha: float) -> dict:
    """A swap summary with a distinct success count in every cell a figure reads."""
    conditions = [f"{rule}_{k}" for k in (1, 8, 16, 32, 64) for rule in ("top", "random", "bottom")]
    cells = {
        c: {f"{s}/{a}": {"successes": i + 50 * j, "n": 200, "n_nonfinite": 0}
            for j, (s, a) in enumerate((("J", "J"), ("L", "L"), ("J", "L"), ("L", "J")))}
        for i, c in enumerate(conditions + ["active"])
    }
    wide = {"J": {"successes": 7, "n": 200, "n_nonfinite": 0},
            "L": {"successes": 0, "n": 200, "n_nonfinite": 200}}
    rows = [{"task": task, "alpha": alpha, "n_trials": 200, "cells": cells, "layer_wide": wide,
             "layer_wide_band": wide}
            for task in ("verbal-report", "probe-swap")]
    medians = {"J": 79.0, "L": 37.0}
    return {"rows": rows, "selection": {t: {"n_active_median": medians} for t in
                                        ("verbal-report", "probe-swap")}}


def test_swap_comparison_reads_each_lens_at_its_own_sites():
    grid = figures.swap_comparison([_swap_summary(1.0), _swap_summary(2.0)], "probe-swap", 2.0)
    assert grid["alpha"] == 2.0
    # top_k are conditions 0, 3, 6, ...; J/J adds 0 and L/L adds 50 successes (of 200).
    assert grid["rules"]["J"]["top"] == [0.0, 0.015, 0.03, 0.045, 0.06]
    assert grid["rules"]["L"]["bottom"] == [0.26, 0.275, 0.29, 0.305, 0.32]
    # Shared random sites: the J-selected rows under the Logit-head edit.
    assert grid["rules"]["L"]["random"] == [0.505, 0.52, 0.535, 0.55, 0.565]
    assert grid["active"]["L"] == (37.0, grid["active"]["L"][1])
    assert grid["active"]["L"][1] == 0.325
    assert grid["reference"]["J"] == 0.035 and grid["reference"]["L"] is None


def test_swap_comparison_is_read_by_alpha():
    summaries = [_swap_summary(1.0), _swap_summary(2.0)]
    grid = figures.swap_comparison(summaries, "verbal-report", 2.0)
    assert grid["alpha"] == 2.0
    assert grid["rules"]["L"]["top"] == [0.25, 0.265, 0.28, 0.295, 0.31]


def test_swap_comparison_reads_the_named_reference():
    summaries = [_swap_summary(1.0)]
    band = {"J": {"successes": 11, "n": 200, "n_nonfinite": 0},
            "L": {"successes": 3, "n": 200, "n_nonfinite": 0}}
    for row in summaries[0]["rows"]:
        row["layer_wide_band"] = band
    grid = figures.swap_comparison(summaries, "verbal-report", 1.0, "layer_wide_band")
    assert grid["reference"]["J"] == 0.055 and grid["reference"]["L"] == 0.015


def test_draw_all_can_redraw_a_subset(tmp_path):
    written = figures.draw_all(
        {"read": _read_record(), "write": _write_record()}, tmp_path, only=["figS3_edit_norm"]
    )
    assert written == ["figS3_edit_norm"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["figS3_edit_norm.pdf", "figS3_edit_norm.png"]
