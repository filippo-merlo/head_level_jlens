"""Every chart and table in the explorer, built from plain arrays.

Nothing here touches the model or computes a statistic: each function takes arrays that
:mod:`~explorer.readings`, :mod:`~explorer.editing` or :mod:`~explorer.records` already
produced and only lays them out. That is the processing / visualisation split
the project keeps -- a colour can change without any risk of moving a number.

Colour does one job per chart. Identity: J-head blue, Logit-head orange, J-layer grey,
the paper's own assignment (``scripts/figures.py``), so a blue line means the same thing
here as in Figures 2 and 4. Magnitude: one blue ramp, darker = better rank or larger
write, on the head grids. Polarity: blue (aligned with the source) through grey to red
(anti-aligned) on the source-loading maps. Dark mode gets its own steps of the same
ramps rather than an inversion. Ranks are drawn 1-based, as the paper states them.
"""

from __future__ import annotations

import math

import altair as alt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"
_DARK = {BLUE: "#3987e5", ORANGE: "#d95926", GREY: "#898781"}
ARM_LABEL = {"jhead": "J-head", "logithead": "Logit-head", "jlayer": "J-layer"}
LENS_LABEL = {"J": "J-head", "L": "Logit-head"}
RULE_LABEL = {"top": "highest-loading", "random": "random", "bottom": "lowest-loading"}
#: Line style carries the site rule, colour the lens -- as in the paper's Figure 4.
RULE_DASH = {"highest-loading": [1, 0], "random": [5, 3], "lowest-loading": [1, 3]}

#: The blue ramp, best/largest first. Light mode runs dark -> near-surface; dark mode runs
#: bright -> near-surface, so the uninteresting end recedes in both.
_RAMP = {
    False: ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6", "#eef4fc"],
    True: ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#104281", "#1e2733"],
}
#: Diverging, anti-aligned -> neutral -> aligned.
_DIVERGING = {
    False: ["#a82a2a", "#e9928f", "#f0efec", "#86b6ef", "#184f95"],
    True: ["#e66767", "#7a3434", "#383835", "#1c5cab", "#86b6ef"],
}
_INK = {False: "#0b0b0b", True: "#f0efec"}
_SURFACE = {False: "#fcfcfb", True: "#1a1a19"}
_GRID = {False: "#e1e0d9", True: "#2c2c2a"}


def colour(name: str, dark: bool) -> str:
    """A series colour, stepped for the dark surface when ``dark``."""
    return _DARK[name] if dark else name


def arm_scale(arms: list[str], dark: bool) -> alt.Scale:
    by_label = {"J-head": BLUE, "Logit-head": ORANGE, "J-layer": GREY}
    return alt.Scale(domain=arms, range=[colour(by_label[a], dark) for a in arms])


def _finish(chart, dark: bool):
    """Recessive chrome: no view border, hairline grid, surface-coloured background."""
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            labelFontSize=10, titleFontSize=11, titleFontWeight="normal",
            gridColor=_GRID[dark], domainColor=_GRID[dark], tickColor=_GRID[dark],
        )
        .configure_legend(labelFontSize=10, titleFontSize=10, titleFontWeight="normal")
        .configure_header(labelFontSize=12, labelFontWeight="bold", titleFontSize=11)
        .properties(background="transparent")
    )


def tok(text: str) -> str:
    """A token as it would be typed: quotes keep a leading space visible."""
    return repr(text)


def printable(text: str) -> str:
    """A token with control characters escaped (``\\x0c`` rather than a form feed).

    The vocabulary holds them, and a raw one is invalid in the SVG a saved chart goes
    through.
    """
    visible = {"\n": "↵", "\t": "⇥", "\xa0": "⍽"}
    return "".join(
        visible.get(ch, ch if ch.isprintable() else repr(ch)[1:-1]) for ch in text
    )


def _site_axes(data: pd.DataFrame):
    """Head and layer axes, every fourth index labelled."""
    heads, layers = sorted(data["head"].unique()), sorted(data["layer"].unique())
    return (
        alt.X("head:O", title="head", axis=alt.Axis(labelAngle=0, values=[int(h) for h in heads[::4]])),
        alt.Y("layer:O", title="layer", axis=alt.Axis(values=[int(l) for l in layers[::4]])),
    )


def _rank_scale(vocab_size: int, dark: bool) -> alt.Scale:
    """Log ranks, piecewise across the ramp: 1 vs 10 matters, 40k vs 50k does not."""
    stops = [vocab_size ** (i / (len(_RAMP[dark]) - 1)) for i in range(len(_RAMP[dark]))]
    return alt.Scale(type="log", domain=stops, range=_RAMP[dark], clamp=True, interpolate="lab")


# ------------------------------------------------------------------ reading (Section 3)


def head_frame(readings, decode, concept, top_k: int = 5) -> pd.DataFrame:
    """One row per (arm, layer, head): what the cell reads, and how well it reads ``concept``."""
    rows = []
    for arm in ("jhead", "logithead"):
        grid = readings.grids[arm]
        ranks = concept.ranks[arm] if concept is not None else None
        for li, layer in enumerate(grid.rows):
            for head in grid.cols:
                ids = grid.top_ids[li, head, :top_k].tolist()
                rows.append(
                    {
                        "arm": ARM_LABEL[arm],
                        "layer": layer,
                        "head": head,
                        "site": f"L{layer} h{head}",
                        "top-1": printable(decode(ids[0])),
                        "top tokens": "  ".join(tok(decode(i)) for i in ids),
                        "rank": int(ranks[li, head]) + 1 if ranks is not None else None,
                        "rho": float(readings.rho[li, head]),
                    }
                )
    return pd.DataFrame(rows)


def head_grids(
    frame: pd.DataFrame,
    *,
    colour_by: str,
    concept: str | None,
    best: dict[str, tuple[int, int]],
    vocab_size: int,
    labels: bool,
    dark: bool,
):
    """The head grid of both readouts, layer x head, side by side (stacked with labels).

    Clicking a cell selects that (layer, head) in both panels -- the same site read two
    ways. Each panel's best site for the concept is ringed.

    Args:
        colour_by: ``"rank"`` (of ``concept``) or ``"rho"`` (write norm / stream norm).
        best: ``{arm label: [(layer, head), ...]}`` every site tied for the best rank,
            ringed.
    """
    data = frame.copy()
    data["best"] = [(l, h) in set(best.get(a, ())) for a, l, h in zip(data["arm"], data["layer"], data["head"], strict=True)]
    if colour_by == "rank":
        value = alt.Color(
            "rank:Q", scale=_rank_scale(vocab_size, dark),
            legend=alt.Legend(title=f"rank of {concept!r}", orient="right", gradientLength=220,
                              values=[1, 10, 100, 1000, 10000, 100000]),
        )
        dark_text = np.log(data["rank"].clip(lower=1)) / math.log(vocab_size) < 0.45
    else:
        top = float(data["rho"].max())
        value = alt.Color(
            "rho:Q",
            scale=alt.Scale(domain=[0, top], range=list(reversed(_RAMP[dark])), interpolate="lab"),
            legend=alt.Legend(title="write norm ρ", orient="right", format=".0%", gradientLength=220),
        )
        dark_text = data["rho"] / top > 0.55
    # A condition, not a scale-free colour field: layers share the colour scale.
    data["light text"] = dark_text ^ dark
    data["label"] = data["top-1"].str.slice(0, 6)

    cell = alt.selection_point(name="cell", fields=["layer", "head"], on="click", clear="dblclick", empty=False)
    width = 1080 if labels else 470
    step_y = 17 if labels else 13
    tooltip = [
        alt.Tooltip("arm:N"), alt.Tooltip("site:N"),
        alt.Tooltip("rank:Q", title=f"rank of {concept!r}") if concept else alt.Tooltip("top-1:N"),
        alt.Tooltip("top tokens:N"), alt.Tooltip("rho:Q", title="write norm ρ", format=".2%"),
    ]
    x, y = _site_axes(data)
    base = alt.Chart().encode(x=x, y=y)
    rects = base.mark_rect(stroke=_SURFACE[dark], strokeWidth=0.6).encode(
        color=value, tooltip=tooltip
    ).add_params(cell)
    best_ring = base.mark_rect(fill=None, stroke=_INK[dark], strokeWidth=2).encode(
        tooltip=tooltip
    ).transform_filter(alt.datum.best)
    clicked_ring = base.mark_rect(fill=None, stroke=_INK[dark], strokeWidth=2, strokeDash=[3, 2]).encode(
        opacity=alt.condition(cell, alt.value(1), alt.value(0)), tooltip=tooltip
    )
    layers = [rects, best_ring, clicked_ring]
    if labels:
        layers.append(
            base.mark_text(fontSize=8, limit=width / 32 - 2).encode(
                text="label:N", color=alt.condition(alt.datum["light text"], alt.value("#fcfcfb"), alt.value("#0b0b0b")), tooltip=tooltip,
            )
        )
    facet = alt.Facet("arm:N", sort=["J-head", "Logit-head"], title=None)
    chart = (
        alt.layer(*layers, data=data)
        .properties(width=width, height=alt.Step(step_y))
        .facet(**({"row": facet} if labels else {"column": facet}))
        .resolve_scale(color="shared")
    )
    return _finish(chart, dark)


def best_by_layer(concept, layers, ref_layers, vocab_size: int, dark: bool):
    """Best rank per layer: the best of a layer's 32 heads for each head readout, the
    layer's block output for J-layer -- one item's version of the paper's Figure 2.

    A faint rule at rank 5 marks the k the paper's per-layer figure is drawn at.
    """
    rows = []
    for arm in ("jhead", "logithead", "jlayer"):
        per_layer = concept.best_per_layer(arm)
        for li, layer in enumerate(layers):
            if arm == "jlayer" and layer not in ref_layers:
                continue
            rows.append({"layer": layer, "readout": ARM_LABEL[arm], "rank": int(per_layer[li]) + 1})
    data = pd.DataFrame(rows)
    arms = ["J-head", "Logit-head", "J-layer"]
    hover = alt.selection_point(name="layer_hover", fields=["layer"], on="pointerover", nearest=True, empty=False)
    y = alt.Y(
        "rank:Q",
        scale=alt.Scale(type="log", domain=[1, vocab_size], reverse=True),
        axis=alt.Axis(values=[1, 5, 25, 100, 1000, 10000, 100000], title=f"best rank of {concept.word!r}"),
    )
    x = alt.X("layer:Q", scale=alt.Scale(domain=[0, max(layers)], nice=False), axis=alt.Axis(tickMinStep=1, title="layer"))
    colour_enc = alt.Color("readout:N", scale=arm_scale(arms, dark), sort=arms, legend=alt.Legend(title=None, orient="top", symbolOpacity=1))
    lines = alt.Chart(data).mark_line(strokeWidth=2, strokeJoin="round").encode(x=x, y=y, color=colour_enc)
    points = alt.Chart(data).mark_point(filled=True, opacity=1).encode(
        x=x, y=y, color=colour_enc,
        size=alt.condition(hover, alt.value(70), alt.value(14)),
        tooltip=["layer:Q", "readout:N", alt.Tooltip("rank:Q", title="best rank")],
    ).add_params(hover)
    rule = alt.Chart(pd.DataFrame({"rank": [5]})).mark_rule(color=GREY, strokeWidth=1).encode(y="rank:Q")
    rule_text = alt.Chart(pd.DataFrame({"rank": [5], "layer": [0]})).mark_text(
        align="left", dx=4, dy=8, fontSize=10, color=GREY
    ).encode(x="layer:Q", y="rank:Q", text=alt.value("k = 5"))
    crosshair = alt.Chart(data).mark_rule(color=GREY).encode(x="layer:Q").transform_filter(hover)
    chart = alt.layer(rule, rule_text, lines, crosshair, points).properties(width=520, height=260)
    return _finish(chart, dark)


def window_grid(grid, token_strs, decode, position: int, concept: str | None, vocab_size: int, dark: bool):
    """J-layer over the last positions of the prompt: the Jacobian lens's own slice view.

    Each cell is the top-1 token of a block output at one position, shaded by the
    concept's rank when one is set. The selected position is ringed.
    """
    rows = []
    for li, layer in enumerate(grid.rows):
        for pi, pos in enumerate(grid.cols):
            rank = int(grid.target_ranks[li, pi]) + 1 if grid.target_ranks is not None else None
            rows.append(
                {
                    "layer": layer,
                    "position": f"{pos} {tok(token_strs[pos])}",
                    "pos": pos,
                    "top-1": printable(decode(int(grid.top_ids[li, pi, 0]))),
                    "rank": rank,
                    "here": pos == position,
                }
            )
    data = pd.DataFrame(rows)
    data["label"] = data["top-1"].str.slice(0, 7)
    if concept is not None:
        shade = alt.Color("rank:Q", scale=_rank_scale(vocab_size, dark), legend=alt.Legend(title=f"rank of {concept!r}"))
        dark_text = np.log(data["rank"].clip(lower=1)) / math.log(vocab_size) < 0.45
    else:
        shade = alt.value(_RAMP[dark][-1])
        dark_text = np.zeros(len(data), dtype=bool)
    # A condition, not a scale-free colour field: layers share the colour scale.
    data["light text"] = dark_text ^ dark
    order = [f"{p} {tok(token_strs[p])}" for p in grid.cols]
    x = alt.X("position:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=90))
    y = alt.Y("layer:O", title="layer")
    tooltip = ["position:N", "layer:O", "top-1:N"] + (["rank:Q"] if concept else [])
    base = alt.Chart(data).encode(x=x, y=y)
    chart = alt.layer(
        base.mark_rect(stroke=_SURFACE[dark], strokeWidth=0.6).encode(color=shade, tooltip=tooltip),
        base.mark_rect(fill=None, stroke=_INK[dark], strokeWidth=2).encode(
            opacity=alt.condition(alt.datum.here, alt.value(1), alt.value(0))
        ),
        base.mark_text(fontSize=9, limit=64).encode(
            text="label:N", color=alt.condition(alt.datum["light text"], alt.value("#fcfcfb"), alt.value("#0b0b0b")), tooltip=tooltip
        ),
    ).properties(width=alt.Step(70), height=alt.Step(15))
    return _finish(chart, dark)


# ------------------------------------------------------------------ editing (Section 4)


def loading_frame(prep, sites: dict[str, list[tuple[int, int]]] | None = None) -> pd.DataFrame:
    """One row per (lens, layer, head): source loading, its null cut, and whether edited."""
    rows = []
    for lens in ("J", "L"):
        loading, tau = prep.loading[lens].numpy(), prep.tau[lens].numpy()
        chosen = set(sites[lens]) if sites else set()
        for li, layer in enumerate(prep.layers):
            for head in range(loading.shape[1]):
                rows.append(
                    {
                        "readout": LENS_LABEL[lens],
                        "layer": layer,
                        "head": head,
                        "site": f"L{layer} h{head}",
                        "loading": float(loading[li, head]),
                        "null 99th pct": float(tau[li, head]),
                        "active": bool(loading[li, head] > tau[li, head]),
                        "edited": (layer, head) in chosen,
                    }
                )
    return pd.DataFrame(rows)


def loading_maps(frame: pd.DataFrame, source: str, dark: bool):
    """Source loading ``cos(z_{l,h}, v_{s,l,h})`` per site, through each readout's map.

    The quantity both readouts select sites by (Section 4.3). Dots mark *active* sites,
    whose loading beats their own 99th-percentile null. Click sites to pick them for a
    manual edit (each click toggles; double-click clears) -- a clicked site is ringed in
    both panels, since a manual edit acts on the same sites through both lenses.
    """
    m = float(np.abs(frame["loading"]).max()) or 1.0
    pick = alt.selection_point(name="sites", fields=["layer", "head"], on="click", toggle="true", clear="dblclick", empty=False)
    x, y = _site_axes(frame)
    tooltip = [
        "readout:N", "site:N", alt.Tooltip("loading:Q", format="+.3f"),
        alt.Tooltip("null 99th pct:Q", format="+.3f"), "active:N",
    ]
    base = alt.Chart().encode(x=x, y=y)
    rects = base.mark_rect(stroke=_SURFACE[dark], strokeWidth=0.6).encode(
        color=alt.Color(
            "loading:Q",
            scale=alt.Scale(domain=[-m, -m / 2, 0, m / 2, m], range=_DIVERGING[dark], interpolate="lab"),
            legend=alt.Legend(title=f"loading on {source!r}", orient="right", gradientLength=220),
        ),
        tooltip=tooltip,
    ).add_params(pick)
    dots = base.mark_circle(size=9, color=_INK[dark]).encode(tooltip=tooltip).transform_filter(alt.datum.active)
    ring = base.mark_rect(fill=None, stroke=_INK[dark], strokeWidth=2).encode(
        opacity=alt.condition(pick, alt.value(1), alt.value(0)), tooltip=tooltip
    )
    chart = (
        alt.layer(rects, dots, ring, data=frame)
        .properties(width=470, height=alt.Step(13))
        .facet(column=alt.Facet("readout:N", sort=["J-head", "Logit-head"], title=None))
    )
    return _finish(chart, dark)


def edited_sites(frame: pd.DataFrame, dark: bool):
    """Which sites each readout's edit touched, side by side, so their overlap shows."""
    data = frame.copy()
    data["state"] = np.where(data["edited"], data["readout"], "not edited")
    domain = ["J-head", "Logit-head", "not edited"]
    scale = alt.Scale(domain=domain, range=[colour(BLUE, dark), colour(ORANGE, dark), _GRID[dark]])
    x, y = _site_axes(data)
    chart = (
        alt.Chart(data)
        .mark_rect(stroke=_SURFACE[dark], strokeWidth=0.6)
        .encode(
            x=x, y=y,
            color=alt.Color("state:N", scale=scale, legend=None),
            tooltip=["readout:N", "site:N", alt.Tooltip("loading:Q", format="+.3f"), "edited:N"],
        )
        .properties(width=300, height=alt.Step(8))
        .facet(column=alt.Facet("readout:N", sort=["J-head", "Logit-head"], title=None))
    )
    return _finish(chart, dark)


def dose_frame(live: dict, prep, conditions, lenses, arms, metrics, alphas, n_active) -> pd.DataFrame:
    """The paper's head-site conditions for one trial, long-form.

    Each readout's swap at its own sites (selection lens = edit lens), at every count
    and rule; ``active`` at the trial's own active count.
    """
    rows = []
    p, success, rank, norm = (metrics.index(m) for m in ("probability", "success", "rank", "edit_norm"))
    for xi, alpha in enumerate(alphas):
        for si, lens in enumerate(lenses):
            ai = arms.index(lens)
            for ci, condition in enumerate(conditions):
                rule, _, count = condition.rpartition("_")
                if condition == "active":
                    rule, count = "top", "active"
                elif count == "active":
                    continue  # random / lowest-loading at the active count: controls, not drawn
                n = n_active[lens] if count == "active" else int(count)
                values = live["heads"][xi, ci, si, ai]
                rows.append(
                    {
                        "alpha": f"α = {alpha:g}",
                        "readout": LENS_LABEL[lens],
                        "rule": RULE_LABEL[rule],
                        "set": "active" if count == "active" else f"{RULE_LABEL[rule]} {count}",
                        "active": count == "active",
                        "sites": n,
                        "P(answer)": float(values[p]),
                        "success": bool(values[success]),
                        "answer rank": int(values[rank]) + 1,
                        "update norm": float(values[norm]),
                    }
                )
    return pd.DataFrame(rows)


def dose_chart(frame: pd.DataFrame, references: pd.DataFrame, answer: str, dark: bool):
    """One trial's Figure 4: P(answer) against edited-site count, both readouts.

    Colour is the readout, line style the site rule; filled points are successes (the
    answer became the greedy token). Diamonds are the active sets at their own counts;
    horizontal rules are the layer-wide references over the workspace band.
    """
    arms = ["J-head", "Logit-head"]
    colour_enc = alt.Color("readout:N", scale=arm_scale(arms, dark), sort=arms,
                           legend=alt.Legend(title=None, orient="top", symbolOpacity=1))
    x = alt.X("sites:Q", scale=alt.Scale(type="log", base=2, domain=[1, max(64, int(frame["sites"].max()))]),
              axis=alt.Axis(values=[1, 2, 4, 8, 16, 32, 64, 128], title="edited head sites"))
    y = alt.Y("P(answer):Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", title=f"P({answer!r})"))
    tooltip = ["readout:N", "set:N", "sites:Q", alt.Tooltip("P(answer):Q", format=".1%"),
               "success:N", "answer rank:Q", alt.Tooltip("update norm:Q", format=".2f")]
    panels = []
    for alpha in sorted(frame["alpha"].unique()):
        sub, ref = frame[frame["alpha"] == alpha], references[references["alpha"] == alpha]
        fixed = sub[~sub["active"]]
        lines = alt.Chart(fixed).mark_line(strokeWidth=2, strokeJoin="round").encode(
            x=x, y=y, color=colour_enc,
            strokeDash=alt.StrokeDash("rule:N", scale=alt.Scale(domain=list(RULE_DASH), range=list(RULE_DASH.values())),
                                      legend=alt.Legend(title=None, orient="top")),
            detail="rule:N",
        )
        fill = alt.condition(alt.datum.success, alt.Color("readout:N", scale=arm_scale(arms, dark)), alt.value(_SURFACE[dark]))
        points = alt.Chart(fixed).mark_point(size=60, strokeWidth=1.5, opacity=1).encode(
            x=x, y=y, color=colour_enc, fill=fill, tooltip=tooltip,
        )
        diamonds = alt.Chart(sub[sub["active"]]).mark_point(shape="diamond", size=110, strokeWidth=1.5, opacity=1).encode(
            x=x, y=y, color=colour_enc, fill=fill, tooltip=tooltip,
        )
        refs = alt.Chart(ref).mark_rule(strokeWidth=1.5, strokeDash=[7, 2, 1.5, 2]).encode(
            y="P(answer):Q",
            color=colour_enc,
            tooltip=["reference:N", alt.Tooltip("P(answer):Q", format=".1%"), "success:N"],
        )
        panels.append(
            alt.layer(refs, lines, points, diamonds).properties(width=400, height=260, title=alpha)
        )
    return _finish(alt.hconcat(*panels).resolve_scale(color="shared", strokeDash="shared"), dark)


# ------------------------------------------------------------------ tables and text


def _md_token(text: str) -> str:
    """A token as a markdown code span: ``|`` would split a table cell and ``$`` would
    be read as LaTeX. Double-backtick fencing survives a token that is a backtick."""
    return "`` " + repr(text).replace("|", "\\|") + " ``"


def prompt_html(token_strs: list[str], position: int) -> str:
    """The prompt as its tokens, the one being read underlined, so "position 17" has a face.

    Colours are ``currentColor`` and a translucent rule, so the strip reads in both themes.
    """
    import html

    spans = []
    for i, token in enumerate(token_strs):
        text = html.escape(token).replace(" ", "&nbsp;").replace("\n", "↵")
        style = (
            "border-bottom:3px solid currentColor;font-weight:700"
            if i == position
            else "border-bottom:1px solid rgba(128,128,128,0.35)"
        )
        spans.append(f'<span title="position {i}" style="{style};margin-right:2px">{text}</span>')
    return (
        '<div style="font-family:ui-monospace,monospace;font-size:13px;line-height:2;'
        f'white-space:normal">{"".join(spans)}</div>'
    )


def best_site_md(concepts, ref_layers, k_values=(1, 5, 25)) -> str:
    """Each concept's best rank under each readout, the site that reaches it, and this
    item's pass@k -- Section 3.3 before the mean over items."""
    if not concepts:
        return "*No concept to rank. Type one above.*"
    lines = [
        "| concept | candidates | J-head | Logit-head | J-layer |",
        "|---|---|---|---|---|",
    ]
    for c in concepts:
        cells = []
        for arm in ("jhead", "logithead", "jlayer"):
            rank, site = c.best(arm)
            where = f"L{site[0]} h{site[1]}" if arm != "jlayer" else f"block {ref_layers[site[0]]}"
            ties = len(c.tied(arm)) - 1
            cells.append(f"**{rank + 1:,}** at {where}" + (f" (+{ties} tied)" if ties else ""))
        lines.append(f"| **{c.word}** | {len(c.ids)} token(s), {c.rule} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| this item | " + " | ".join(f"pass@{k}" for k in k_values) + " |")
    lines.append("|---|" + "---|" * len(k_values))
    for arm in ("jhead", "logithead", "jlayer"):
        best = [c.best(arm)[0] for c in concepts]
        lines.append(
            f"| {ARM_LABEL[arm]} | "
            + " | ".join(f"{np.mean([r < k for r in best]):.2f}" for k in k_values)
            + " |"
        )
    return "\n".join(lines)


def inspector_md(reading, decode, layer: int, head: int, concept, top_k: int = 8) -> str:
    """What one site says under both head readouts, beside J-layer at the same layer."""
    columns, headers, ranks = [], [], []
    li = reading.layers.index(layer)

    def column(grid, row, col):
        return [
            f"{_md_token(decode(int(i)))} {float(v):.1f}"
            for i, v in zip(grid.top_ids[row, col, :top_k], grid.top_values[row, col, :top_k], strict=True)
        ]

    for arm in ("jhead", "logithead"):
        columns.append(column(reading.grids[arm], li, head))
        headers.append(f"{ARM_LABEL[arm]}, L{layer} h{head}")
        ranks.append(f"{int(concept.ranks[arm][li, head]) + 1:,}" if concept else "—")
    if layer in reading.ref_layers:
        columns.append(column(reading.grids["jlayer"], reading.ref_layers.index(layer), 0))
        ranks.append(f"{int(concept.ranks['jlayer'][li]) + 1:,}" if concept else "—")
    else:
        columns.append([f"no readout at block {layer}: it is the lens's target"] + [""] * (top_k - 1))
        ranks.append("—")
    headers.append(f"J-layer, block {layer} (whole stream)")
    rows = ["| | " + " | ".join(headers) + " |", "|---|" + "---|" * len(headers)]
    rows += [f"| {i + 1} | " + " | ".join(c[i] for c in columns) + " |" for i in range(top_k)]
    if concept is not None:
        rows.append(f"| rank of {concept.word!r} | " + " | ".join(f"**{r}**" for r in ranks) + " |")
    return "\n".join(rows)


def outcome_frame(rows: list[dict]) -> pd.DataFrame:
    """Edit outcomes as a table: one row per edit, top tokens as one readable string."""
    frame = pd.DataFrame(rows).drop(columns=["lens"])
    frame["sites"] = ["—" if r["sites"] is None else str(r["sites"]) for r in rows]
    frame["success"] = np.where(frame["success"], "✓", "✗")
    frame["next token"] = frame["next token"].map(tok)
    frame["top tokens"] = [
        "  ".join(f"{tok(t)} {p:.0%}" for t, p in tokens) for tokens in frame["top tokens"]
    ]
    return frame


def picked_sites(chart_ui, frame: pd.DataFrame) -> list[tuple[int, int]]:
    """The ``(layer, head)`` sites clicked on a site chart; empty when none is.

    ``apply_selection`` returns every row when nothing is selected, so emptiness is read
    off the raw selection first: a point selection with fields arrives as
    ``{param: {field: [values], "vlPoint": {"or": [points]}}}``.
    """

    def selected(fields) -> bool:
        if not isinstance(fields, dict):
            return False
        point = fields.get("vlPoint")
        if isinstance(point, dict) and point.get("or"):
            return True
        return any(v for k, v in fields.items() if k not in ("vlPoint", "_vgsid_"))

    if not any(selected(f) for f in (chart_ui.selections or {}).values()):
        return []
    picked = chart_ui.apply_selection(frame)
    return sorted({(int(l), int(h)) for l, h in zip(picked["layer"], picked["head"], strict=True)})
