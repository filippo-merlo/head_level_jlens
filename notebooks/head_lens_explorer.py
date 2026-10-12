"""Head-JLens explorer: read and edit attention heads, one prompt at a time.

    HEAD_LENS_GPU=0 uv run marimo edit notebooks/head_lens_explorer.py --headless --port 2718
    HEAD_LENS_GPU=0 uv run marimo run  notebooks/head_lens_explorer.py     # app view

An interactive companion to ``head_jlens_paper.md``. Exploration and presentation only:
no number here is a result. The notebook computes with the experiments' own
code and checks itself against their records -- the pilot dump on load, ``results/read/``
for any evaluation item, ``results/write_swap_sites_clamp/`` for any swap trial -- and the
paper's tables are read from those records, not typed in.

The cells are prose, widgets and one call each; the loading, readouts, edits and charts
live in ``notebooks/explorer/`` (see its ``__init__``). Marimo does not reload an edited
module by default -- turn on module autoreloading, or re-run the import cell.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full", app_title="Head-JLens explorer")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reading attention heads through the Jacobian lens

    An interactive companion to the paper (`head_jlens_paper.pdf`), on **Llama-3.1-8B-Instruct**.
    It runs the paper's two experiments on one prompt at a time:

    1. **Read** (paper §3). Decode every attention head's output through Head-JLens
       (**J-head**) and through the per-head logit lens (**Logit-head**), and see which
       heads rank an unstated intermediate, such as *France* in the Lyon prompt.
    2. **Edit** (paper §4). Pick heads by how strongly each readout says they carry a
       source token, swap its coordinate for a target's, and see whether the model's
       answer follows: *Paris* → *Rome*.

    The notebook calls the experiments' own code and checks itself against their records.
    On load it reproduces `results/pilot/`. For any evaluation item it compares the live
    ranks at every site with `results/read/`. For any of the 185 swap trials it re-runs the
    paper's protocol and compares the result with `results/write_swap_sites_clamp/`.
    Numbers computed here are exploration; the paper's tables at the bottom are read
    from the records.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # The GPU pin must happen before torch is imported anywhere, so it lives in its own
    # cell and every torch-touching cell depends on `gpu_pinned`. CUDA_DEVICE_ORDER is
    # not optional: this node is heterogeneous (5x A100 + 1x H100) and CUDA's default
    # FASTEST_FIRST promotes the H100 to device 0, so an index that looks free in
    # nvidia-smi silently lands on a different, often busy, GPU. Same reasoning as
    # scripts/_gpu.py. The records were made on A100s, which is what lets a re-run match
    # them bit for bit. Changing HEAD_LENS_GPU needs a kernel restart, not a re-run.
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("HEAD_LENS_GPU", "3")
    gpu_pinned = os.environ["CUDA_VISIBLE_DEVICES"]
    return (gpu_pinned,)


@app.cell
def _(gpu_pinned):
    # `explorer` is notebooks/explorer/, which marimo puts on sys.path. Importing it
    # appends the repository root, so `scripts.write` inside explorer.editing is the
    # paper's driver and its protocol constants are imported, not copied.
    import pandas as pd
    from explorer import charts, editing, prompts, readings, records, saving
    from explorer.session import LensSession, self_check

    assert gpu_pinned is not None
    return (
        LensSession,
        charts,
        editing,
        pd,
        prompts,
        readings,
        records,
        saving,
        self_check,
    )


@app.cell
def _(LensSession, mo, self_check):
    # The one expensive cell: 8B in bf16 plus a 2 GB J^z, ~1 min. The self-check raises
    # if the lens does not reproduce the pilot dump, which stops every cell below it.
    with mo.status.spinner(title="Loading Llama-3.1-8B-Instruct, J^z and the Jacobian lens…"):
        lens = LensSession.load()
        pilot = self_check(lens)
    return lens, pilot


@app.cell
def _(mo):
    dark = mo.app_meta().theme == "dark"
    return (dark,)


@app.cell(hide_code=True)
def _(lens, mo, pilot):
    mo.hstack(
        [
            mo.stat(
                f"{len(lens.layers)} × {lens.spec.n_heads}",
                label="head sites",
                caption=f"{lens.n_sites:,} (layer, head) pairs, d_head = {lens.spec.head_dim}",
                bordered=True,
            ),
            mo.stat(
                f"{lens.jz_n_prompts:,} prompts",
                label="Head-JLens J^z",
                caption=f"fitted at o_proj's input · sha256 {lens.jz_sha256[:8]}",
                bordered=True,
            ),
            mo.stat(
                f"{lens.published.n_prompts} prompts",
                label="Jacobian lens (J-layer)",
                caption=f"block outputs {lens.ref_layers[0]}–{lens.ref_layers[-1]}, Neuronpedia",
                bordered=True,
            ),
            mo.stat(
                f"{pilot['n_cells']}/{pilot['n_cells']} cells",
                label="self-check",
                caption="J-head top-6 equals results/pilot/",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "The three readouts (paper §2, §3.2)": mo.md(r"""
    Let $z_l$ be the input to layer $l$'s attention output projection $W_O$: the head
    outputs, concatenated, before they are summed into the residual stream. Head $h$ owns
    columns $\mathcal C_h$ and writes $c_{l,h} = W_{O,l}^{(h)} z_{l,h}$ into the stream.

    | Readout | decodes | map into the final layer |
    |---|---|---|
    | **J-head** (Head-JLens) | $J_{l,h}\, z_{l,h}$, with $J_{l,h} = J^z_l[:, \mathcal C_h]$ | $J^z_l$: the Jacobian of the final residual $x_L$ with respect to $z_l$, averaged over 1,000 WikiText-103 prompts (fitted here) |
    | **Logit-head** | $c_{l,h}$ | none: the per-head logit lens |
    | **J-layer** | $J^{\mathrm{out}}_l\, x_{l+1}$ | the layer-output Jacobian lens of Gurnee et al. (2026), Neuronpedia's matrices, block outputs 0–30 |

    All three end the same way, in the model's final RMSNorm and unembedding. J-head and
    Logit-head read the same head sites and differ only in whether the per-layer map is
    applied: $J_{l,h} z_{l,h} = J^{\mathrm{attn}}_l c_{l,h}$ (§S1.1). J-layer reads the
    whole stream, one readout per layer.
    """),
            "Ranks and pass@k (paper §3.3)": mo.md(r"""
    A token's **rank** is its position in a readout's ordering of the vocabulary; rank 1
    is the top token. A concept takes the best rank among its accepted tokens. For an
    evaluation item the candidate rule is the paper's: bare and space-prefixed spellings,
    of which multihop and order-ops keep only single-token ones and the other sets score
    each spelling's first token, with order-ops' synonym lists on top. A typed concept
    gets the first token of its bare and space-prefixed spellings.

    Each readout picks its **own best site** per concept: one of 1,024 head sites for the
    head readouts, one of 31 block outputs for J-layer. This is an oracle search (the
    annotation picks the winning head per item), not a held-out test of where the
    concept lives. An item's pass@k is the fraction of its intermediates whose best rank
    is at most $k$. The paper averages it over items.

    Readouts are scale-free: the final normalisation rescales a readout's scores without
    reordering them (§2.3), so a head writing very little can still look confident. The
    grids can therefore be shaded by $\rho_{l,h} = \lVert c_{l,h}\rVert / \lVert
    x^{\mathrm{mid}}_l\rVert$, the size of the head's write against the stream it joins.
    $\rho$ is a diagnostic from the code (`src/magnitude.py`), not a quantity the paper
    reports.
    """),
            "The coordinate swap (paper §4.2–4.3)": mo.md(r"""
    For a source token $s$ and target token $t$, with gain-scaled unembedding rows
    $q_t = g \odot w_t$, each head site gets a two-direction frame
    $V = [v_s\; v_t]$ in its own coordinates: $v_{t,l,h} = J_{l,h}^\top q_t$ for J-head,
    $W_{O,l}^{(h)\top} q_t$ for Logit-head. A clean pass records each site's
    coordinates $\beta^\circ = V^+ y^\circ$ at every position. The edited pass then sets,
    at every prompt position of every selected site,

    $$y \leftarrow y + V\big(\sigma_\alpha(\beta^\circ) - V^+ y\big), \qquad
      \sigma_\alpha(\beta) = \beta + \alpha\big(\sigma(\beta) - \beta\big),$$

    where $\sigma$ exchanges the two coordinates. $\alpha = 1$ is the swap of Gurnee et
    al., clamped to clean-pass coordinates; $\alpha = 2$ overshoots it. The component
    outside the frame is left alone.

    Sites are chosen by **source loading** $\cos(z_{l,h}, v_{s,l,h})$ at the final
    position, each readout through its own map. The rules are the $m$ highest-loading
    sites, the $m$ lowest, $m$ random sites (shared by both readouts), or the **active**
    sites, whose loading beats the 99th percentile of 256 random tokens' loadings at that
    site. Each swap has a random-direction control of the same norm at the same sites.
    The reference swaps the whole residual stream at block outputs 12–28, through the
    Jacobian lens or the bare unembedding. Success means the edited model's greedy next
    token is an accepted token of the answer.
    """),
        },
        multiple=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Reading heads · paper §3

    Pick a prompt. The six evaluation sets are read where the paper reads them (poetry
    at the newline ending the couplet's first line, every other set at the final token)
    and default to their annotated intermediates. The intermediate-replacement prompts of
    §4 default to their intermediate. Gurnee et al.'s worked examples and typed prompts
    are read at the last token.
    """)
    return


@app.cell
def _(lens, mo, prompts):
    prompt_sources = prompts.library(lens.model)
    source_ui = mo.ui.dropdown(
        options=list(prompt_sources),
        value=prompts.DEFAULT_LABEL,
        searchable=True,
        label="prompt",
        full_width=True,
    )
    custom_ui = mo.ui.text_area(
        value="",
        rows=2,
        label="…or type a prompt (overrides the menu; read at its last token)",
        full_width=True,
    )
    return custom_ui, prompt_sources, source_ui


@app.cell
def _(custom_ui, lens, prompt_sources, prompts, source_ui):
    selection = prompts.select(lens.model, prompt_sources, source_ui.value, custom_ui.value)
    return (selection,)


@app.cell
def _(mo, selection):
    concepts_ui = mo.ui.text(
        value=", ".join(selection.default_targets),
        label="concepts to rank (comma-separated)",
        full_width=True,
    )
    position_ui = mo.ui.slider(
        0,
        len(selection.token_strs) - 1,
        value=selection.default_position,
        label="read at position",
        show_value=True,
        full_width=True,
    )
    return concepts_ui, position_ui


@app.cell(hide_code=True)
def _(concepts_ui, custom_ui, mo, position_ui, source_ui):
    mo.vstack(
        [
            mo.hstack([source_ui, custom_ui], widths=[1, 1], gap=2, align="start"),
            mo.hstack([concepts_ui, position_ui], widths=[1, 1], gap=2, align="end"),
        ],
        gap=1,
    ).callout(kind="neutral")
    return


@app.cell
def _(concepts_ui, lens, mo, position_ui, readings, selection):
    with mo.status.spinner(title="Reading every head…"):
        reading = readings.read_position(
            lens,
            selection,
            position=int(position_ui.value),
            words=readings.parse_concepts(concepts_ui.value),
        )
    return (reading,)


@app.cell
def _(lens, reading, readings, records, selection):
    read_check = readings.check_against_record(
        reading,
        selection,
        records.read_ranks(lens.paths.results_root, lens.spec.key, selection.item),
    )
    return (read_check,)


@app.cell(hide_code=True)
def _(charts, dark, mo, read_check, reading, selection):
    _badge = {
        "match": mo.callout(
            mo.md(f"**Matches the reading experiment.** {read_check['detail']}: every site, both "
                  "head readouts and J-layer, for each of this item's intermediates."),
            kind="success",
        ),
        "mismatch": mo.callout(mo.md(f"**Differs from the record.** {read_check['detail']}."), kind="danger"),
        "n/a": mo.md(f"<small>Record check: {read_check['detail']}.</small>"),
    }[read_check["status"]]
    _skipped = (
        mo.callout(
            mo.md("Not scoreable under the candidate rule: "
                  + "; ".join(f"**{w}** ({why})" for w, why in reading.skipped.items())),
            kind="warn",
        )
        if reading.skipped
        else mo.md("")
    )
    mo.vstack(
        [
            mo.md(f"**{selection.label}** · reading position **{reading.position}** of "
                  f"{len(selection.token_strs)}"
                  + (f" (the {selection.set_name} set's readout position)"
                     if selection.item is not None and reading.position == selection.default_position
                     else "")),
            mo.Html(charts.prompt_html(selection.token_strs, reading.position)),
            mo.md(charts.best_site_md(reading.concepts, reading.ref_layers)),
            _skipped,
            _badge,
        ],
        gap=0.75,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Every head, both readouts

    Each cell is one head's output at this position, decoded on its own. Left: through
    its slice of $J^z_l$ (J-head). Right: directly through the unembedding (Logit-head).
    Darker means the concept ranks higher. Rings mark each readout's best sites (all of
    them, when several tie), and hovering shows a cell's top tokens. Click a cell to inspect it under both readouts
    below; double-click clears.
    """)
    return


@app.cell
def _(mo, reading):
    shade_ui = mo.ui.dropdown(
        options=[c.word for c in reading.concepts] + ["write norm ρ"],
        value=reading.concepts[0].word if reading.concepts else "write norm ρ",
        label="shade by",
    )
    labels_ui = mo.ui.switch(value=False, label="print each cell's top token (stacks the panels)")
    mo.hstack([shade_ui, labels_ui], justify="start", gap=2)
    return labels_ui, shade_ui


@app.cell
def _(charts, dark, labels_ui, lens, mo, reading, shade_ui):
    shade_concept = reading.concept(shade_ui.value)
    head_cells = charts.head_frame(reading, lens.decode, shade_concept)
    head_chart = charts.head_grids(
        head_cells,
        colour_by="rank" if shade_concept else "rho",
        concept=shade_concept.word if shade_concept else None,
        best={charts.ARM_LABEL[arm]: shade_concept.tied(arm) for arm in ("jhead", "logithead")}
        if shade_concept
        else {},
        vocab_size=lens.vocab_size,
        labels=labels_ui.value,
        dark=dark,
    )
    grid_ui = mo.ui.altair_chart(head_chart, legend_selection=False)
    grid_ui
    return grid_ui, head_cells, head_chart, shade_concept


@app.cell(hide_code=True)
def _(charts, grid_ui, head_cells, lens, mo, reading, shade_concept):
    _picked = charts.picked_sites(grid_ui, head_cells)
    if _picked:
        _layer, _head = _picked[0]
        _why = "the clicked site"
    elif shade_concept is not None:
        _layer, _head = shade_concept.best("jhead")[1]
        _why = f"J-head's best site for {shade_concept.word!r} (click a cell to change it)"
    else:
        _layer, _head = max(reading.layers) // 2, 0
        _why = "a default site (click a cell to change it)"
    mo.vstack(
        [
            mo.md(f"#### One site, three readouts — L{_layer} h{_head}, {_why}"),
            mo.md(
                f"This head wrote **{reading.rho[reading.layers.index(_layer), _head]:.1%}** of "
                "the residual stream it joined. Values are logits (before the softmax). "
                "J-layer reads the whole block output at the same layer, not this head."
            ),
            mo.md(charts.inspector_md(reading, lens.decode, _layer, _head, shade_concept)),
        ],
        gap=0.5,
    )
    return


@app.cell(hide_code=True)
def _(charts, dark, lens, mo, reading, shade_concept):
    mo.stop(shade_concept is None)
    mo.vstack(
        [
            mo.md(
                f"### Layer by layer\n\nThe best rank of **{shade_concept.word!r}** at each "
                "layer: the best of that layer's 32 heads for the head readouts, the layer's "
                "block output for J-layer. This is the per-item version of the paper's "
                "Figure 2, which plots the share of items reaching rank ≤ 5 at each layer."
            ),
            charts.best_by_layer(shade_concept, reading.layers, reading.ref_layers, lens.vocab_size, dark),
        ]
    )
    return


@app.cell(hide_code=True)
def _(charts, dark, lens, mo, reading, readings, selection, shade_concept):
    _grid = readings.window(lens, selection, shade_concept.ids if shade_concept else None)
    mo.accordion(
        {
            "The Jacobian lens across positions (J-layer slice view)": mo.vstack(
                [
                    mo.md(
                        "J-layer on the whole residual stream at the last positions of the "
                        "prompt: each cell is a block output's top token, shaded by the "
                        "concept's rank. The ringed column is the position read above. This "
                        "is the view of Gurnee et al.'s lens, for orientation; the paper's "
                        "comparisons are made at the readout position only."
                    ),
                    charts.window_grid(
                        _grid, selection.token_strs, lens.decode, reading.position,
                        shade_concept.word if shade_concept else None, lens.vocab_size, dark,
                    ),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Editing heads · paper §4

    Choose a trial: the 140 verbal-report and 45 intermediate-replacement trials the paper
    runs (§4.1), or your own. Each readout scores every head site by its **source
    loading**. Pick an edit below; it runs at once, through both readouts, with each
    swap's random-direction control and the layer-wide reference. The last block re-runs
    the paper's whole protocol on this trial and checks it against the record.
    """)
    return


@app.cell
def _(editing, lens, mo):
    with mo.status.spinner(title="Staging both maps and building the paper's 185 trials…"):
        edit_context = editing.build_context(lens)
    return (edit_context,)


@app.cell
def _(edit_context, mo):
    _options = {"custom trial…": -1} | {t.label: t.index for t in edit_context.trials}
    _default = next(
        (t.label for t in edit_context.trials if "ex-city-capital-Lyon-Naples" in t.label),
        edit_context.trials[0].label,
    )
    trial_ui = mo.ui.dropdown(
        options=_options, value=_default, searchable=True, label="trial", full_width=True
    )
    custom_prompt_ui = mo.ui.text(
        value="Fact: The capital of the country where Lyon is located is",
        label="prompt",
        full_width=True,
    )
    custom_source_ui = mo.ui.text(value="France", label="source (blank = the model's next token)")
    custom_target_ui = mo.ui.text(value="Italy", label="target")
    custom_answer_ui = mo.ui.text(value="Rome", label="success word")
    return (
        custom_answer_ui,
        custom_prompt_ui,
        custom_source_ui,
        custom_target_ui,
        trial_ui,
    )


@app.cell(hide_code=True)
def _(
    custom_answer_ui,
    custom_prompt_ui,
    custom_source_ui,
    custom_target_ui,
    mo,
    trial_ui,
):
    mo.vstack(
        [trial_ui]
        + (
            [
                custom_prompt_ui,
                mo.hstack([custom_source_ui, custom_target_ui, custom_answer_ui], widths="equal", gap=1),
                mo.md("<small>A custom trial swaps along each word's space-prefixed first token "
                      "and counts the success word's first token, bare or space-prefixed, as "
                      "§4.1 does. Its random sites and directions use seed index 185, one past "
                      "the paper's trials.</small>"),
            ]
            if trial_ui.value == -1
            else []
        ),
        gap=0.75,
    ).callout(kind="neutral")
    return


@app.cell
def _(
    custom_answer_ui,
    custom_prompt_ui,
    custom_source_ui,
    custom_target_ui,
    edit_context,
    editing,
    lens,
    mo,
    trial_ui,
):
    if trial_ui.value == -1:
        mo.stop(
            not (custom_prompt_ui.value.strip() and custom_target_ui.value.strip()),
            mo.md("*Type a prompt and a target.*"),
        )
        trial_info = editing.custom_trial(
            lens,
            custom_prompt_ui.value,
            custom_source_ui.value,
            custom_target_ui.value.strip(),
            custom_answer_ui.value.strip(),
        )
    else:
        trial_info = edit_context.trials[trial_ui.value]
    with mo.status.spinner(title="Clean pass, source loadings and frames…"):
        prepared = editing.prepare(edit_context, trial_info)
    return prepared, trial_info


@app.cell(hide_code=True)
def _(charts, edit_context, editing, mo, prepared, trial_info):
    _clean = editing.clean_row(edit_context, prepared)
    _n_active = {lens: len(sites["active"]) for lens, sites in prepared.sites.items()}
    mo.vstack(
        [
            mo.md(
                f"**{trial_info.task_label}** · `{trial_info.prompt}`\n\n"
                f"The model answers **{charts.tok(_clean['next token'])}**. The edit swaps "
                f"**{trial_info.source!r}** for **{trial_info.target!r}**; it succeeds when the "
                f"greedy next token becomes **{trial_info.answer!r}**, which now has "
                f"P = {_clean['P(answer)']:.2%} and rank {_clean['answer rank']:,}."
            ),
            mo.hstack(
                [
                    mo.stat(str(_n_active["J"]), label="active sites, J-head", bordered=True,
                            caption="loading above the per-site 99th-percentile null"),
                    mo.stat(str(_n_active["L"]), label="active sites, Logit-head", bordered=True,
                            caption="each readout through its own map"),
                    mo.stat(
                        str(len(set(prepared.sites["J"]["top_64"]) & set(prepared.sites["L"]["top_64"]))),
                        label="top-64 sites shared", bordered=True,
                        caption="J-head's and Logit-head's 64 highest-loading",
                    ),
                ],
                widths="equal",
                gap=1,
            ),
        ],
        gap=0.75,
    )
    return


@app.cell(hide_code=True)
def _(mo, trial_info):
    mo.md(f"""
    ### Where each readout sees {trial_info.source!r}

    Source loading at the final position: blue sites point along the source direction,
    red sites against it. Dots mark active sites. **Click sites to build a manual edit**
    (each click toggles a site; double-click clears), then choose *the sites clicked on the
    map* below.
    """)
    return


@app.cell
def _(charts, dark, mo, prepared, trial_info):
    loading_cells = charts.loading_frame(prepared)
    loading_ui = mo.ui.altair_chart(
        charts.loading_maps(loading_cells, trial_info.source, dark), legend_selection=False
    )
    loading_ui
    return loading_cells, loading_ui


@app.cell
def _(editing, mo):
    alpha_ui = mo.ui.slider(
        steps=[0.5, 1, 1.5, 2, 3], value=1, label="strength α (paper: 1, and 2)", show_value=True
    )
    rule_ui = mo.ui.dropdown(
        options={label: key for key, label in editing.RULE_LABEL.items()},
        value=editing.RULE_LABEL["top"],
        label="sites",
    )
    count_ui = mo.ui.slider(
        steps=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        value=8,
        label="how many (paper: 1, 8, 16, 32, 64)",
        show_value=True,
    )
    reference_ui = mo.ui.switch(value=True, label="layer-wide reference (block outputs 12–28)")
    mo.hstack([alpha_ui, rule_ui, count_ui, reference_ui], justify="start", gap=2, align="center").callout(
        kind="neutral"
    )
    return alpha_ui, count_ui, reference_ui, rule_ui


@app.cell
def _(
    alpha_ui,
    charts,
    count_ui,
    edit_context,
    editing,
    loading_cells,
    loading_ui,
    mo,
    prepared,
    reference_ui,
    rule_ui,
):
    _manual = charts.picked_sites(loading_ui, loading_cells)
    if rule_ui.value == "manual":
        mo.stop(not _manual, mo.md("*Click at least one site on the map above.*"))
    edit_sites = {
        lens: editing.choose_sites(prepared, lens, rule_ui.value, int(count_ui.value), _manual)
        for lens in editing.LENSES
    }
    with mo.status.spinner(title="Editing…"):
        edit_rows = [editing.clean_row(edit_context, prepared)] + editing.custom_edit(
            edit_context, prepared, edit_sites, float(alpha_ui.value), reference=reference_ui.value
        )
    return edit_rows, edit_sites


@app.cell(hide_code=True)
def _(alpha_ui, charts, edit_rows, edit_sites, editing, mo, rule_ui, trial_info):
    _by = {r["edit"]: r for r in edit_rows}
    _j, _l = _by.get("J-head swap"), _by.get("Logit-head swap")
    _rule = editing.RULE_LABEL[rule_ui.value]

    def _tile(row, label):
        if row is None:
            return mo.stat("—", label=label, bordered=True)
        return mo.stat(
            f"{charts.tok(row['next token'])} {'✓' if row['success'] else '✗'}",
            label=label,
            caption=f"P({trial_info.answer!r}) = {row['P(answer)']:.1%} · update norm {row['update norm']:.1f}",
            bordered=True,
        )

    _shared = len(set(edit_sites["J"]) & set(edit_sites["L"]))
    mo.vstack(
        [
            mo.md(
                f"### Outcome · α = {alpha_ui.value:g}, {_rule} sites "
                f"(J-head {len(edit_sites['J'])}, Logit-head {len(edit_sites['L'])}, {_shared} shared)"
            ),
            mo.hstack(
                [_tile(_by["clean (no edit)"], "clean"), _tile(_j, "J-head swap"), _tile(_l, "Logit-head swap")]
                + [
                    _tile(_by[name], "layer-wide, Jacobian lens")
                    for name in _by
                    if name.startswith("layer-wide, Jacobian lens (")
                ],
                widths="equal",
                gap=1,
            ),
            mo.ui.table(
                charts.outcome_frame(edit_rows),
                selection=None,
                pagination=False,
                show_column_summaries=False,
                format_mapping={
                    "update norm": "{:.2f}".format,
                    "P(answer)": "{:.2%}".format,
                    "KL from clean": "{:.3f}".format,
                },
            ),
            mo.md(
                "<small>Update norm is the total head-level norm $D$ of §S3.3, taken over every "
                "edited site and position. The layer-wide rows use the residual stream's own units, "
                "so their norms are not comparable to the head rows'. Random-direction rows add a "
                "random update of the swap's norm at the same sites.</small>"
            ),
        ],
        gap=0.75,
    )
    return


@app.cell(hide_code=True)
def _(charts, dark, edit_sites, mo, prepared):
    mo.accordion(
        {
            "Which sites each edit touched": charts.edited_sites(
                charts.loading_frame(prepared, edit_sites), dark
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The paper's protocol on this trial

    Runs every condition the paper records for this trial, exactly as
    `scripts/write_swap_sites.py` runs it. That is 18 site sets (1, 8, 16, 32 and 64
    sites and the active set, each under the highest-loading, random and lowest-loading
    rule), selected by each readout, edited through each readout's frame, with and
    without a random direction. That makes 144 edits per strength, at α = 1 and 2, plus
    both layer-wide references. For a paper trial the result is compared cell by cell with
    the record. About 10–20 s.
    """)
    return


@app.cell
def _(mo):
    protocol_ui = mo.ui.run_button(label="run the paper's protocol on this trial")
    protocol_ui
    return (protocol_ui,)


@app.cell
def _(edit_context, editing, lens, mo, prepared, protocol_ui, records):
    mo.stop(not protocol_ui.value)
    with mo.status.spinner(title="288 head edits and 4 layer-wide references…"):
        protocol = editing.paper_protocol(edit_context, prepared)
    protocol_record = (
        records.recorded_trial(lens.paths.results_root, prepared.info.index)
        if prepared.info.index is not None
        else None
    )
    protocol_check = editing.compare_to_record(protocol, protocol_record)
    return protocol, protocol_check


@app.cell(hide_code=True)
def _(
    charts,
    dark,
    edit_context,
    editing,
    mo,
    pd,
    prepared,
    protocol,
    protocol_check,
    trial_info,
):
    _metrics = list(editing.SWAP_METRICS)
    _dose = charts.dose_frame(
        protocol, prepared, list(editing.ZIN_CONDITIONS), list(editing.LENSES), list(editing.ARMS),
        _metrics, list(editing.ALPHAS), {lens: len(s["active"]) for lens, s in prepared.sites.items()},
    )
    _band = f"{edit_context.band_layers[0]}–{edit_context.band_layers[-1]}"
    _refs = pd.DataFrame(
        [
            {
                "alpha": f"α = {alpha:g}",
                "readout": charts.LENS_LABEL[lens],
                "reference": f"layer-wide, {'Jacobian lens' if lens == 'J' else 'unembedding'} ({_band})",
                "P(answer)": float(protocol["layer_wide_band"][xi, editing.ARMS.index(lens), _metrics.index("probability")]),
                "success": bool(protocol["layer_wide_band"][xi, editing.ARMS.index(lens), _metrics.index("success")]),
            }
            for xi, alpha in enumerate(editing.ALPHAS)
            for lens in editing.LENSES
        ]
    )
    _check = {
        "identical": mo.callout(mo.md(f"**Bit-identical to the record.** {protocol_check['detail']}."), kind="success"),
        "match": mo.callout(mo.md(f"**Same outcomes as the record.** {protocol_check['detail']}."), kind="success"),
        "differs": mo.callout(
            mo.md(f"**Differs from the record.** {protocol_check['detail']}. bf16 kernels depend on the "
                  "GPU model; the record was made on A100s (§S3.5 bounds batch-layout flips at 16 of 7,030)."),
            kind="warn",
        ),
        "n/a": mo.md(f"<small>Record check: {protocol_check['detail']}.</small>"),
    }[protocol_check["status"]]
    mo.vstack(
        [
            _check,
            charts.dose_chart(_dose, _refs, trial_info.answer, dark),
            mo.md(
                "<small>Each readout's swap at its own sites. Solid: highest-loading; dashed: "
                "random (shared); dotted: lowest-loading; diamonds: active sets at their own "
                "counts; filled marks: the answer became the greedy token. Dash-dot rules: "
                f"the layer-wide references at block outputs {_band}. Compare the paper's "
                "Figure 4, which plots success rates over all trials.</small>"
            ),
        ],
        gap=0.75,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## The paper's numbers

    Read from the records under `results/`, so these are the paper's numbers, not this
    notebook's.
    """)
    return


@app.cell
def _(mo):
    k_ui = mo.ui.dropdown(
        options={"pass@1 (Table 1)": 1, "pass@5 (Table S2)": 5, "pass@25 (Table S3)": 25},
        value="pass@1 (Table 1)",
        label="threshold",
    )
    return (k_ui,)


@app.cell(hide_code=True)
def _(k_ui, lens, mo, records):
    _root = lens.paths.results_root
    mo.ui.tabs(
        {
            "Reading: all-layer pass@k": mo.vstack(
                [
                    k_ui,
                    mo.ui.table(
                        records.passk_table(_root, lens.spec.key, int(k_ui.value)),
                        selection=None,
                        pagination=False,
                        show_column_summaries=False,
                        format_mapping={
                            "J-head": "{:.3f}".format,
                            "Logit-head": "{:.3f}".format,
                            "J-layer": "{:.3f}".format,
                            "paired difference": "{:+.3f}".format,
                        },
                    ),
                    mo.md(
                        "<small>Each head readout searches all 1,024 sites and J-layer 31 block "
                        "outputs, so part of J-layer's gap may come from the smaller search "
                        "(§3.5). The paired difference is J-head minus Logit-head on the same "
                        "items.</small>"
                    ),
                ]
            ),
            "Editing: coordinate-swap successes (Table 3)": mo.vstack(
                [
                    mo.ui.table(
                        records.swap_table(_root),
                        selection=None,
                        pagination=False,
                        show_column_summaries=False,
                    ),
                    mo.md(
                        "<small>J-head / Logit-head. Each readout edits its own highest-loading, "
                        "lowest-loading or active sites; random sites are shared; the random "
                        "direction is a same-norm update at each readout's own 64 sites. The "
                        "update norm is the mean total head-level norm.</small>"
                    ),
                ]
            ),
            "Figures 2 and 4": mo.vstack(
                [
                    mo.image(str(_root / "figures" / "fig1_per_layer_k5.png"), width="100%",
                             caption="Figure 2: per-layer pass@5, best head per item and intermediate."),
                    mo.image(str(_root / "figures" / "fig3_swap_lenses.png"), width="100%",
                             caption="Figure 4: target-token success against edited-site count."),
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **What these pictures cannot show** (paper §2.3, §3.2, §3.5, §4.3 and Limitations).
    The best site is an oracle choice, so a good rank at some head is not evidence that
    the concept lives there. Readouts are scale-free, so a confident head can be writing
    almost nothing (shade by ρ to see it). The site controls match the number of edited
    sites, not the size of the edit: Logit-head's swaps are larger than J-head's at
    every fixed count, and highest-loading sites get larger updates than random ones. The
    results come from one model and evaluation sets adapted from Gurnee et al. (2026).
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Save what is on screen
    """)
    return


@app.cell
def _(mo):
    save_ui = mo.ui.run_button(label="save readings, edits and charts + provenance")
    save_ui
    return (save_ui,)


@app.cell(hide_code=True)
def _(
    alpha_ui,
    edit_rows,
    edit_sites,
    head_chart,
    lens,
    mo,
    reading,
    rule_ui,
    save_ui,
    saving,
    selection,
    trial_info,
):
    mo.stop(not save_ui.value, mo.md("*Not saved yet.*"))
    mo.md(
        saving.save(
            lens,
            selection=selection,
            reading=reading,
            trial=trial_info,
            edit={
                "alpha": float(alpha_ui.value),
                "rule": rule_ui.value,
                "sites": {k: [list(s) for s in v] for k, v in edit_sites.items()},
                "rows": edit_rows,
            },
            charts={"head_grids": head_chart},
        )
    )
    return


if __name__ == "__main__":
    app.run()
