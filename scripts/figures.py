"""[5] The paper's figures, from the read and write records.

    python -m scripts.figures

Reads only ``results/`` -- no model, no GPU -- and writes ``results/figures/``. Each
figure is one of the paper's tables drawn: the series it plots are pulled out of the
records by the pure functions in the first half of this file, and the drawing code in the
second half takes only those series. A plotting tweak therefore cannot change a number,
and the tests pin the series functions, not the pixels.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

K_VALUES = (1, 5, 25)
#: The thresholds drawn per layer; complete k = 1 rates remain in results data.
MAIN_K = (5, 25)
#: The alpha the causal figures lead with; the other is drawn beside it, never maxed over.
PRIMARY_ALPHA = 1.0
ARM = "head_jlens"
ARM_LABEL = {
    "jhead": "J-head",
    "logithead": "Logit-head",
    "jlayer": "J-layer (whole-layer reference)",
}
RULE_LABEL = {"top": "highest-loading", "random": "random", "bottom": "lowest-loading"}
#: Panel names for the task keys; "probe-swap" stays the key because the released data file
#: is named after it, but the edit uses lens directions only, so the paper calls it this.
TASK_LABEL = {"verbal-report": "verbal report", "probe-swap": "intermediate replacement"}

# The two saturated ends of the explorer's ramps (notebooks/explorer/charts.py), so a blue
# line means the same thing in the paper as in the notebook: blue for the fitted transport
# and the concept-selected sites, orange for the direct decode and the anti-aligned
# control, grey for whatever is context. Text is always ink, never a series colour.
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
GRIDLINE, AXIS_LINE = "#e1e0d9", "#c3c2b7"
#: Ordinal shades of the one blue ramp for k = 1, 5, 25: steps 250, 450 and 700.
K_SHADES = {1: "#86b6ef", 5: "#2a78d6", 25: "#0d366b"}
ARM_COLOUR = {"jhead": BLUE, "logithead": ORANGE}
RULE_COLOUR = {"top": BLUE, "random": GREY, "bottom": ORANGE}
#: The J-head vs Logit-head swap figure encodes the lens by colour and the site rule
#: by line style, so the J-head / Logit-head comparison is read within each rule.
LENS_COLOUR = {"J": BLUE, "L": ORANGE}
LENS_LABEL = {"J": "J-head", "L": "Logit-head"}
RULE_STYLE = {"top": "-", "random": (0, (4, 2)), "bottom": (0, (1, 1.5))}
#: Layer-wide references in Figure 4: dash-dot, so they never read as a random_k line.
REFERENCE_STYLE = (0, (7, 2, 1.5, 2))
REFERENCE_LABEL = {"J": "Jacobian lens", "L": "unembedding"}
#: Residual-stream references a swap summary row can hold.
REFERENCE_NAME = {"layer_wide": "layer-wide", "layer_wide_band": "workspace band"}
#: The swap figure (Figure 4) puts the primary alpha first.
SWAP_ALPHAS = (1.0, 2.0)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.edgecolor": AXIS_LINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
    }
)


# ------------------------------------------------------------------ series (pure)


def set_name(slug: str) -> str:
    return slug.replace("lens-eval-", "")


def per_layer(read: dict, k: int, arm: str) -> dict:
    """``{set: [pass@k at each layer of read['meta']['layers']]}`` -- pass@k of the best
    head of that one layer, not the best site over the depth. ``nan`` where the arm has no
    rate at a layer: the J-layer reference stops at layer 30, the top block's output being
    the layer-output lens's target, while the head arms run to 31."""
    layers = read["meta"]["layers"]

    def stat(entry: dict, layer: int) -> float:
        rates = entry["regions"][f"layer_{layer}"]["k"][str(k)]["rates"]
        return rates[arm]["point"] if arm in rates else float("nan")

    return {
        set_name(slug): [stat(entry, layer) for layer in layers]
        for slug, entry in read["results"].items()
    }


def per_layer_difference(read: dict, k: int) -> dict:
    """``{set: [J-head minus Logit-head at each layer]}``, the primary statistic (Figure S3)."""
    layers = read["meta"]["layers"]
    return {
        set_name(slug): [
            entry["regions"][f"layer_{layer}"]["k"][str(k)]["paired_difference"][
                "jhead_minus_logithead"
            ]["point"]
            for layer in layers
        ]
        for slug, entry in read["results"].items()
    }


def write_norms(read: dict) -> tuple[list[int], list[list[float]]]:
    """Relative write norm ``||c_{l,h}|| / ||x_mid,l||`` per head, listed by layer."""
    block = read["magnitude"]["relative_write_norm"]
    layers = sorted(int(layer) for layer in block)
    return layers, [list(block[str(layer)]) for layer in layers]


def condition_rows(write: dict) -> tuple[dict, str]:
    """``{(rule, count): condition name}`` for the head-level rows, found by their recorded
    rule and count rather than by parsing names, and the one block-anchored reference."""
    rows = {
        (block["rule"], block["count"]): name
        for name, block in write["conditions"].items()
        if block["anchor"] == "zin"
    }
    reference = next(
        name for name, block in write["conditions"].items() if block["anchor"] == "out"
    )
    return rows, reference


def _count_axis(write: dict, task: str, table: dict, cell) -> dict:
    """The count x rule grid of one task, ``cell`` reading a value off each entry of
    ``table``.

    ``active`` is placed at its median realised count, since its count is an outcome of
    the per-site null rather than a setting. The reference is a different patch family,
    so a table without a row for it returns ``None`` there.
    """
    rows, reference = condition_rows(write)
    rules = list(write["meta"]["rules"])
    counts = list(write["meta"]["site_counts"])
    entry = write["results"][task]
    return {
        "counts": counts,
        "rules": {
            rule: [cell(table[rows[(rule, count)]]) for count in counts] for rule in rules
        },
        "active_count": entry["selection"]["n_active"]["median"],
        "active": {rule: cell(table[rows[(rule, "active")]]) for rule in rules},
        "reference": cell(table[reference]) if reference in table else None,
        "n_trials": entry["n_trials"],
    }


def swap_rates(write: dict, task: str, arm: str, alpha: float) -> dict:
    """Swap rate along the count axis."""
    table = write["results"][task][f"{arm}_alpha{alpha}"]
    return _count_axis(write, task, table, lambda s: s["rate"])


def edit_norms(write: dict, task: str, arm: str, alpha: float) -> dict:
    """Mean total edit norm along the count axis."""
    table = write["results"][task]["edit_norm"][f"{arm}_alpha{alpha}"]
    return _count_axis(write, task, table, float)


def swap_comparison(
    summaries: list[dict], task: str, alpha: float, reference: str = "layer_wide",
) -> dict:
    """J-head and Logit-head success along the count axis under the coordinate swap at one
    alpha, from :mod:`scripts.report_swap_sites` summaries.

    Each lens edits its own top / bottom / active sites with its own direction; random
    sites are shared, so they are read off the J-selected rows under each lens's edit.
    Every cell is the success rate, or ``None`` for a degenerate (overflowed) output rather
    than a zero rate. ``summaries`` may split alphas across runs; the one holding
    ``(task, alpha)`` is used. ``reference`` names the row's residual-stream reference:
    ``"layer_wide"`` (block outputs 0-30) or ``"layer_wide_band"`` (the workspace band of
    Gurnee et al.).
    """
    summary, row = next(
        (s, r) for s in summaries for r in s["rows"]
        if r["task"] == task and r["alpha"] == alpha
    )
    counts = [1, 8, 16, 32, 64]

    def rate(cell):
        return None if cell["n_nonfinite"] else cell["successes"] / cell["n"]

    cells = row["cells"]
    medians = summary["selection"][task]["n_active_median"]
    return {
        "counts": counts,
        "alpha": alpha,
        "n_trials": row["n_trials"],
        "rules": {
            lens: {
                "top": [rate(cells[f"top_{k}"][f"{lens}/{lens}"]) for k in counts],
                "random": [rate(cells[f"random_{k}"][f"J/{lens}"]) for k in counts],
                "bottom": [rate(cells[f"bottom_{k}"][f"{lens}/{lens}"]) for k in counts],
            }
            for lens in "JL"
        },
        "active": {lens: (medians[lens], rate(cells["active"][f"{lens}/{lens}"])) for lens in "JL"},
        "reference": {lens: rate(row[reference][lens]) for lens in "JL"},
    }


# ------------------------------------------------------------------ drawing


def _tidy(ax, *, left=True, bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=0)


def _panel_letter(ax, letter: str, title: str):
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold", pad=6)


def _small_multiples(sets: list[str]):
    """One panel per evaluation set over the depth, three to a row, axes shared."""
    ncols = min(3, len(sets))
    nrows = -(-len(sets) // ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10, 2.2 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
        gridspec_kw={"wspace": 0.1, "hspace": 0.35},
    )
    for ax in axes.flat[len(sets) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("layer")
    return fig, axes


def _depth_ticks(layers) -> list[int]:
    """Every layer when there are few, else ~5 evenly spaced ones ending at the last."""
    if len(layers) <= 10:
        return list(layers)
    step = max(1, len(layers) // 4)
    ticks = list(layers[::step])
    if ticks[-1] == layers[-1]:
        return ticks
    # The last layer always gets a tick; a regular tick within a couple of layers of
    # it (32 beside 33 on a 34-layer model) would print on top of it.
    return [*(t for t in ticks if layers[-1] - t > 2), layers[-1]]


def _finish_depth_axis(ax, layers, title: str):
    ax.set_title(title, loc="left", pad=4)
    ax.set_xticks(_depth_ticks(layers))
    ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
    _tidy(ax)


def figure_per_layer(read: dict, k: int):
    """The best head at each layer for both head arms, with the whole-layer reference."""
    layers = read["meta"]["layers"]
    arms = {
        arm: per_layer(read, k, arm) for arm in ("jhead", "logithead", "jlayer")
    }
    sets = list(arms["jhead"])
    fig, axes = _small_multiples(sets)
    for ax, name in zip(axes.flat, sets, strict=False):
        # The reference goes down first, dashed and in ink, so the two series it is
        # compared against sit on top of it.
        ax.plot(
            layers,
            arms["jlayer"][name],
            color=INK_2,
            lw=1.1,
            ls=(0, (3, 2)),
            label=ARM_LABEL["jlayer"],
        )
        for arm in ("logithead", "jhead"):
            ax.plot(layers, arms[arm][name], color=ARM_COLOUR[arm], label=ARM_LABEL[arm])
        _finish_depth_axis(ax, layers, name)
    axes[0, 0].set_ylim(0, None)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"pass@{k}, best head at one layer")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[::-1], labels[::-1], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02)
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.12)
    return fig


def figure_per_layer_difference(read: dict):
    """Figure S3 -- the paired J-head minus Logit-head difference at each single layer,
    at every k, against a line at zero."""
    layers = read["meta"]["layers"]
    series = {k: per_layer_difference(read, k) for k in K_VALUES}
    sets = list(series[K_VALUES[0]])
    fig, axes = _small_multiples(sets)
    for ax, name in zip(axes.flat, sets, strict=False):
        ax.axhline(0, color=AXIS_LINE, lw=0.8, zorder=0)
        for k in K_VALUES:
            ax.plot(layers, series[k][name], color=K_SHADES[k], label=f"$k = {k}$")
        _finish_depth_axis(ax, layers, name)
    for ax in axes[:, 0]:
        ax.set_ylabel("J-head − Logit-head, paired")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.12)
    return fig


def _draw_count_axis(ax, grid: dict):
    """The three rules along the count axis, with ``active`` at its median count."""
    counts = np.asarray(grid["counts"], dtype=float)
    # A small multiplicative dodge, so the two controls' points do not sit on top
    # of each other at zero.
    dodge = {"top": 1.0, "random": 0.94, "bottom": 1.06}
    for rule, cells in grid["rules"].items():
        x = counts * dodge[rule]
        points = np.asarray(cells, dtype=float)
        ax.plot(x, points, color=RULE_COLOUR[rule], label=RULE_LABEL[rule], zorder=3)
        ax.plot(x, points, "o", color=RULE_COLOUR[rule], ms=5, mec=SURFACE, mew=0.8, zorder=4)
        active = grid["active"][rule]
        ax_x = grid["active_count"] * dodge[rule]
        ax.plot(
            ax_x, active, "D", color=RULE_COLOUR[rule], ms=5, mec=SURFACE, mew=0.8, zorder=4
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(grid["counts"])
    ax.set_xticklabels([str(c) for c in grid["counts"]])
    ax.minorticks_off()
    # ``active`` is named at the top of the axis rather than on the tick row, where its
    # median count would collide with the fixed count beside it.
    ax.text(
        grid["active_count"],
        0.99,
        f"active\n(median {grid['active_count']})",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7,
        color=INK_2,
    )
    ax.set_xlim(counts[0] * 0.75, max(counts[-1], grid["active_count"]) * 1.35)
    ax.grid(axis="x", visible=False)
    _tidy(ax)


def figure_intervention(write: dict, *, tasks: tuple[str, ...] | None = None):
    """Target-token rate by edited-site count, with the layer-wide reference as a line."""
    tasks = list(tasks) if tasks is not None else list(write["results"])
    alphas = sorted(write["meta"]["alphas"], key=lambda a: a != PRIMARY_ALPHA)
    fig, axes = plt.subplots(
        len(tasks),
        len(alphas),
        figsize=(9, 3.1 * len(tasks)),
        sharex="row",
        sharey=True,
        gridspec_kw={"wspace": 0.08, "hspace": 0.3},
    )
    axes = np.atleast_2d(axes)
    letters = iter("abcdefgh")
    top = 0.0
    for row, task in zip(axes, tasks, strict=True):
        for ax, alpha in zip(row, alphas, strict=True):
            grid = swap_rates(write, task, ARM, alpha)
            point = grid["reference"]
            cells = [c for rule in grid["rules"].values() for c in rule if c is not None]
            cells += [c for c in grid["active"].values() if c is not None]
            top = max(top, point, *cells)
            ax.axhline(point, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=1)
            ax.text(
                0.01,
                point + 0.012,
                f"residual stream {point:.3f}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=7,
                color=INK_2,
            )
            _draw_count_axis(ax, grid)
            primary = " (primary)" if alpha == PRIMARY_ALPHA else ""
            _panel_letter(
                ax,
                next(letters),
                f"{TASK_LABEL.get(task, task)}, {grid['n_trials']} trials, $\\alpha = {alpha:g}${primary}",
            )
    for ax in axes[:, 0]:
        ax.set_ylabel("swap rate")
    for ax in axes[-1]:
        ax.set_xlabel("sites edited")
    # 0.7 fits every Llama panel; a checkpoint whose swap rates run higher gets the room.
    axes[0, 0].set_ylim(-0.02, max(0.7, top * 1.08))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK_2, marker="D", ms=5, lw=0, mec=INK_2))
    labels.append("active, at its median count")
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.11)
    return fig


def _draw_comparison(ax, grid: dict, *, reference: str | None = None):
    """Both lenses under every rule, the selected rows drawn heavier.

    With ``reference``, each lens's layer-wide rate (Jacobian lens for J, bare unembedding for
    Logit) is drawn as a line across the panel in that lens's colour: the level its
    head-level rows are read against. The values are printed in ink at the top left.
    """
    counts = np.asarray(grid["counts"], dtype=float)
    dodge = {"J": 0.96, "L": 1.04}
    if reference:
        values = []
        for lens in "JL":
            cell = grid["reference"][lens]
            if cell is None:
                continue
            ax.axhline(cell, color=LENS_COLOUR[lens], lw=1.0, ls=REFERENCE_STYLE, zorder=1)
            values.append(f"{REFERENCE_LABEL[lens]} {cell:.3f}")
        ax.text(0.01, 0.98, REFERENCE_NAME[reference] + ": " + ", ".join(values), transform=ax.transAxes,
                ha="left", va="top", fontsize=7, color=INK_2)
    for lens in "JL":
        colour, x = LENS_COLOUR[lens], counts * dodge[lens]
        for rule in ("bottom", "random", "top"):
            points = np.array(grid["rules"][lens][rule], dtype=float)
            selected = rule == "top"
            ax.plot(x, points, color=colour, ls=RULE_STYLE[rule], lw=1.6 if selected else 1.1,
                    zorder=3 if selected else 2)
            ax.plot(x, points, "o", color=colour, ms=5 if selected else 3.5,
                    mfc=colour if selected else SURFACE, mec=SURFACE if selected else colour,
                    mew=0.8, zorder=4 if selected else 2)
        median, active = grid["active"][lens]
        ax.plot(median * dodge[lens], active, "D", color=colour, ms=5, mec=SURFACE, mew=0.8,
                zorder=4)
    ax.set_xscale("log", base=2)
    medians = [grid["active"][l][0] for l in "JL"]
    ax.set_xticks(grid["counts"])
    ax.set_xticklabels([str(c) for c in grid["counts"]])
    ax.minorticks_off()
    ax.set_xlim(counts[0] * 0.75, max(counts[-1], *medians) * 1.35)
    ax.grid(axis="x", visible=False)
    _tidy(ax)


def figure_swap_lenses(
    summaries: list[dict],
    values: tuple = SWAP_ALPHAS,
    *,
    reference: str | None = None,
    ymax: float = 1.02,
):
    """J-head against Logit-head under the coordinate swap at alpha = 1 and 2, with the
    workspace-band references (Figure 4)."""
    tasks = [t for t in ("verbal-report", "probe-swap")
             if any(r["task"] == t for s in summaries for r in s["rows"])]
    fig, axes = plt.subplots(
        len(tasks), len(values), figsize=(9, 3.1 * len(tasks)), sharex="row",
        sharey=True, gridspec_kw={"wspace": 0.08, "hspace": 0.3}, squeeze=False,
    )
    letters = iter("abcdefgh")
    top = 0.0
    for row, task in zip(axes, tasks, strict=True):
        for ax, value in zip(row, values, strict=True):
            grid = swap_comparison(summaries, task, value, reference or "layer_wide")
            rates = [*(v for rules in grid["rules"].values() for cells in rules.values() for v in cells),
                     *(rate for _, rate in grid["active"].values()), *grid["reference"].values()]
            top = max(top, *(v for v in rates if v is not None))
            _draw_comparison(ax, grid, reference=reference)
            _panel_letter(ax, next(letters),
                          f"{TASK_LABEL.get(task, task)}, {grid['n_trials']} trials, $\\alpha = {value:g}$")
    for ax in axes[:, 0]:
        ax.set_ylabel("target success")
    for ax in axes[-1]:
        ax.set_xlabel("sites edited")
    # ``ymax`` fits every Llama panel; a checkpoint whose rates run higher gets the room.
    axes[0, 0].set_ylim(-0.02, max(ymax, top * 1.08))
    handles = [plt.Line2D([], [], color=LENS_COLOUR[l], marker="o", ms=5, mec=SURFACE)
               for l in "JL"]
    labels = [f"{LENS_LABEL[l]} sites and edit" for l in "JL"]
    for rule, label in (("top", "highest-loading"), ("random", "random (shared sites)"),
                        ("bottom", "lowest-loading")):
        handles.append(plt.Line2D([], [], color=INK_2, ls=RULE_STYLE[rule],
                                  lw=1.6 if rule == "top" else 1.1))
        labels.append(label)
    handles.append(plt.Line2D([], [], color=INK_2, marker="D", ms=5, lw=0, mec=INK_2))
    labels.append("active, at each readout's median count")
    if reference:
        for lens in "JL":
            handles.append(plt.Line2D([], [], color=LENS_COLOUR[lens], lw=1.0, ls=REFERENCE_STYLE))
            labels.append(f"{REFERENCE_NAME[reference]}, {REFERENCE_LABEL[lens]}")
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.11)
    return fig


def figure_edit_norm(write: dict, alpha: float = PRIMARY_ALPHA):
    """Total edit norm each rule realises at each count. The
    controls are matched in count, and this is the size difference left visible."""
    tasks = list(write["results"])
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(9, 3.0), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    top = 0.0
    for ax, task, letter in zip(np.atleast_1d(axes), tasks, "abcdefgh", strict=False):
        grid = edit_norms(write, task, ARM, alpha)
        _draw_count_axis(ax, grid)
        _panel_letter(ax, letter, f"{TASK_LABEL.get(task, task)}, $\\alpha = {alpha:g}$")
        ax.set_xlabel("sites edited")
        top = max(top, *(v for cells in grid["rules"].values() for v in cells), *grid["active"].values())
    np.atleast_1d(axes)[0].set_ylabel("mean total update norm")
    # Headroom for the ``active`` label, which sits at the top of the axis.
    np.atleast_1d(axes)[0].set_ylim(0, top * 1.25)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.2)
    return fig


def figure_write_norm(read: dict):
    """How much of the residual stream each site writes, by layer: every head
    as a point, the layer median as a line."""
    layers, values = write_norms(read)
    fig, ax = plt.subplots(figsize=(8, 3.0))
    rng = np.random.default_rng(0)
    for layer, heads in zip(layers, values, strict=True):
        jitter = rng.uniform(-0.28, 0.28, size=len(heads))
        ax.plot(layer + jitter, heads, "o", color=GREY, ms=2.6, alpha=0.55, mew=0, zorder=2)
    medians = [statistics.median(heads) for heads in values]
    ax.plot(layers, medians, color=ORANGE, zorder=3, label="layer median")
    overall = statistics.median(v for heads in values for v in heads)
    ax.axhline(overall, color=INK_2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(
        layers[-1] + 0.6,
        overall,
        f"all sites, median {overall:.1%}",
        ha="right",
        va="bottom",
        fontsize=7,
        color=INK_2,
    )
    ax.set_yscale("log")
    lo, hi = min(v for heads in values for v in heads), max(v for heads in values for v in heads)
    decades = [10.0**e for e in range(-6, 1)]
    ticks = [t for d in decades for t in (d, 3 * d) if lo / 1.5 <= t <= hi * 1.5] or [0.003, 0.01, 0.03, 0.1, 0.3]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t * 100:g}%" for t in ticks])
    ax.minorticks_off()
    ax.set_xticks(_depth_ticks(layers))
    ax.set_xlim(layers[0] - 0.8, layers[-1] + 0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel(r"relative write norm $\|c_{l,h}\| \, / \, \|x^{\mathrm{mid}}_l\|$")
    ax.grid(axis="x", visible=False)
    _tidy(ax)
    ax.legend(loc="upper right")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.95, bottom=0.17)
    return fig


FIGURES = {
    **{
        f"fig{i}_per_layer_k{k}": ("read", partial(figure_per_layer, k=k))
        for i, k in enumerate(MAIN_K, start=1)
    },
    "fig3_intervention": ("write", figure_intervention),
    "fig3_probe_swap": ("write", partial(figure_intervention, tasks=("probe-swap",))),
    "fig3_swap_lenses": (
        "swap_logit",
        partial(figure_swap_lenses, reference="layer_wide_band", ymax=0.9),
    ),
    "figS1_per_layer_k1": ("read", partial(figure_per_layer, k=1)),
    "figS2_paired_per_layer": ("read", figure_per_layer_difference),
    "figS3_edit_norm": ("write", figure_edit_norm),
    "figS4_write_norm": ("read", figure_write_norm),
}


def draw_all(
    records: dict, out_dir: Path, *, dpi: int = 200, only: list[str] | None = None
) -> list[str]:
    """Every figure whose record is present (or just ``only``), as PNG for the markdown and PDF for print."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (record, draw) in FIGURES.items():
        if only is not None and name not in only:
            continue
        if records.get(record) is None:
            logger.warning("%s: no %s record, skipped", name, record)
            continue
        fig = draw(records[record])
        for suffix in ("png", "pdf"):
            fig.savefig(out_dir / f"{name}.{suffix}", dpi=dpi)
        plt.close(fig)
        written.append(name)
        logger.info("wrote %s", out_dir / f"{name}.png")
    return written


# ------------------------------------------------------------------ driver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lens", default=None, help="LensSpec key (default: DEFAULT_LENS)")
    parser.add_argument("--read-name", default="read")
    parser.add_argument("--write-name", default="write")
    parser.add_argument("--out-name", default="figures")
    parser.add_argument(
        "--swap-summaries",
        nargs="+",
        default=["results/write_swap_sites_clamp/summary.json"],
        help="J-head vs Logit-head swap summaries; drawn only for the model they were run on",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(FIGURES),
        help="redraw only these figures; their provenance is appended to meta.json's "
        "'updates' instead of replacing the record of the other figures",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from src.config import DEFAULT_LENS, LENSES, PROJECT_ROOT, Paths
    from src.provenance import meta_block

    spec = LENSES[args.lens or DEFAULT_LENS]
    paths = Paths.default()
    records = {}
    for name, run in (("read", args.read_name), ("write", args.write_name)):
        path = paths.run_dir(run) / f"{spec.key}_{name}.json"
        records[name] = json.loads(path.read_text()) if path.exists() else None

    summaries = [
        json.loads((PROJECT_ROOT / path).read_text())
        for path in args.swap_summaries
        if (PROJECT_ROOT / path).exists()
    ]
    summaries = [s for s in summaries if s["shards"][0]["model"] == spec.hf_model_id]
    records["swap_logit"] = summaries or None

    out_dir = paths.run_dir(args.out_name)
    written = draw_all(records, out_dir, only=args.only)

    # Which records the pixels came from: each input's own git block and timestamp, so a
    # figure can be matched to the run that produced its numbers.
    meta = meta_block(
        model=spec.hf_model_id,
        lens_key=spec.key,
        inputs={
            name: {"timestamp": record["meta"]["timestamp"], "git": record["meta"]["git"]}
            for name, record in records.items()
            if record is not None and name != "swap_logit"
        },
        swap_summaries=args.swap_summaries if records["swap_logit"] else None,
        figures=written,
    )
    meta_path = out_dir / "meta.json"
    if args.only and meta_path.exists():
        existing = json.loads(meta_path.read_text())
        existing.setdefault("updates", []).append(meta)
        meta = existing
    meta_path.write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
