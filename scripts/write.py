"""[3] The causal experiment: does the model use the lens coordinate, and how many sites carry it?

    python -m scripts.write --gpu 0

One independent variable: **which sites get edited**. Everything else -- the trials, the
arms, the alphas, the intervention -- is held fixed across every condition, so the site
axis is the only thing a difference between rows can be attributed to.

The intervention is the paper's additive two-frame swap. Stack the two concepts' lens
vectors into ``V = [v_s v_t]``, read the activation's coordinates through the
pseudoinverse and exchange them: ``x <- x + alpha V(sigma(c) - c)``. The additive form
leaves everything orthogonal to ``span{v_s, v_t}`` untouched, so the rest of what the
activation carries survives. With ``--coords clamp`` (the default) ``c`` is read from the
clean pass, so every edited site is set to the same swapped coordinates however many
earlier sites were edited; ``--coords recompute`` reads it from the edited pass, as runs
before 2026-09-28 did (see :mod:`src.swap` for why that undoes itself across layers).

Sites are every ``(layer, head)`` pair in the model: all 32 layers, 32 heads, 1024 of
them.

The site axis is a **count crossed with a selection rule**. At each count ``k`` in
:data:`SITE_COUNTS`, and at the count the data-driven ``active`` rule selects, three rows
edit the same number of sites and differ only in which::

    top_k       the k sites loading highest on the source concept    THE TEST ROWS
    random_k    k sites drawn at random, redrawn per trial            site-selection control
    bottom_k    the k sites loading lowest                            site-selection control
    active      sites loading above a per-site null; its k is an outcome, not a setting
    random_active / bottom_active    the two controls at that trial's active count

Within a rule the sets are nested across counts -- ``top_8`` is the first 8 of
``top_16``, and each trial's random sets are prefixes of one permutation -- so the count
axis is a dose along one fixed ordering under every rule: a step to the next count adds
sites without exchanging any.

**``layer_wide`` is the paper's own intervention, and a different instrument rather than
another row.** It swaps the concept in the full residual stream, patching block outputs
through the Jacobian lens with :class:`~src.hooks.ResidualPatcher`; every other row
patches ``o_proj``'s input through the fitted ``J^z`` with :class:`~src.hooks.ZinPatcher`.
It is the reference the head-level rows are read against: without a working full-strength
intervention a null at any count says nothing about the count and everything about the
setup. It carries a second patch family and keeps the Jacobian lens as a runtime
dependency, so the row is marked rather than blended in. It also runs over layers 0-30
rather than 0-31: the Jacobian lens's top block is its own target, so it has no lens there.

**The controls are matched in count, not in edit size.** Selecting by loading selects
large ``|c_s|``, and the update is ``alpha (c_t - c_s)(v_s - v_t)``, so ``top_k``'s edits
are larger than ``random_k``'s by construction. Every row's total edit norm is recorded
(``edit_norm``) so the size difference sits beside the rates instead of being equalised
away.

**Never report a maximum over alpha.** ``alpha=1`` exchanges the two lens coordinates;
``alpha=2`` sends the ``v_s`` coordinate to ``2c_1 - c_0``, an extrapolation *past* the
exchange. They are different operations, not two strengths of one. ``alpha=1`` is
primary, as in the paper, and ``alpha=2`` is reported beside it.

Per-layer localisation is behind ``--per-layer``: one head at one layer, swept over
every layer and every index. It answers a third question -- *where* in depth a single
head's coordinate is usable -- and nothing in the table above rests on it.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time

from scripts._gpu import pin_gpu

logger = logging.getLogger(__name__)

ALPHAS = (1.0, 2.0)
#: Primary throughout, as in the paper: the alpha the table is read at.
PRIMARY_ALPHA = 1.0
ARMS = ("head_jlens", "matched_norm")
TASKS = ("verbal-report", "probe-swap")

#: The one block-anchored row; patched through the Jacobian lens, not ``J^z``.
RESIDUAL_CONDITION = "layer_wide"
#: Fixed site counts, then the data-driven one whose count is an outcome.
SITE_COUNTS = (1, 8, 16, 32, 64)
COUNTS = (*SITE_COUNTS, "active")
#: Selection rules run at every count, in the order the table reads.
RULES = ("top", "random", "bottom")


def condition_name(rule: str, count: int | str) -> str:
    """``top_8``, ``random_active``, ... The threshold set keeps its own name, ``active``:
    its cut is per site, so it is not the top-``n_active`` by raw loading."""
    return "active" if (rule, count) == ("top", "active") else f"{rule}_{count}"


def condition_role(rule: str, count: int | str) -> tuple[str, str]:
    """What a row is, for a reader opening the JSON without this file."""
    if (rule, count) == ("top", "active"):
        return "zin, sites loading above a per-site null", "data-driven selection"
    n = "that trial's active count of" if count == "active" else str(count)
    if rule == "top":
        return f"zin, {n} highest-loading sites", "concept-selected"
    if rule == "random":
        return f"zin, {n} random sites", "site-selection control"
    return f"zin, {n} lowest-loading sites", "site-selection control"


#: ``zin``-anchored rows, grouped by count so each block reads test / control / control.
ZIN_CONDITIONS = tuple(condition_name(rule, count) for count in COUNTS for rule in RULES)
#: The concept-selected rows, whose site sets get the spread descriptive.
SELECTED_CONDITIONS = tuple(condition_name("top", count) for count in COUNTS)
CONDITIONS = (RESIDUAL_CONDITION, *ZIN_CONDITIONS)

#: Concepts drawn to give each site its own null loading distribution.
N_NULL_CONCEPTS = 256
NULL_SEED = 0
#: Quantile of that null a concept must beat to count as active at a site.
NULL_QUANTILE = 0.99
#: Base for the random-site control's per-trial seed. Redrawn per trial: one draw reused
#: across every trial answers "would *these* sites have done it" rather than "would any
#: sites of this count have done it".
RANDOM_SITE_SEED = 90_001


def site_sets(loading, layers: list[int], tau, *, seed: int) -> dict[str, list]:
    """Every ``zin`` row's site set for one trial: the count axis crossed with the rules.

    One seed serves every random set, so they are prefixes of one permutation and nested
    across counts, as the top and bottom sets are by construction. A trial with no active
    site gets empty controls at that count too: the three rows then all run the clean
    forward, and still differ only in which sites were edited.
    """
    from src.sites import select_sites

    active = select_sites(loading, layers, "threshold", tau=tau)
    count_of = {**{k: k for k in SITE_COUNTS}, "active": len(active)}
    sites = {}
    for count in COUNTS:
        k = count_of[count]
        for rule in RULES:
            if (rule, count) == ("top", "active"):
                chosen = active
            elif k == 0:
                chosen = []
            elif rule == "random":
                chosen = select_sites(loading, layers, "random", k=k, seed=seed)
            else:
                chosen = select_sites(loading, layers, rule, k=k)
            sites[condition_name(rule, count)] = chosen
    return sites


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", default=None, help="GPU index, as nvidia-smi numbers them")
    parser.add_argument("--lens", default=None, help="LensSpec key (default: DEFAULT_LENS)")
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=None,
        help="prompts the J^z lens was fitted on (default: the one LensSpec pins)",
    )
    parser.add_argument("--max-categories", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="also sweep each head at each layer alone, one layer at a time",
    )
    parser.add_argument(
        "--read-name",
        default="read",
        help="read.py's results directory, for the write norms of the selected sites",
    )
    parser.add_argument("--out-name", default="write")
    parser.add_argument(
        "--coords",
        choices=("clamp", "recompute"),
        default="clamp",
        help="clamp coordinates to the swapped clean pass, or swap the edited pass's "
        "(runs before 2026-09-28; see src.swap)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pin_gpu(args.gpu)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    started = time.monotonic()

    import torch

    from src.config import (
        DEFAULT_LENS,
        LENSES,
        MAX_SEQ_LEN,
        Paths,
        jz_path,
    )
    from src.heads import COMPUTE_DTYPE, effective_unembed
    from src.hooks import AnchorRecorder, ResidualPatcher, ZinPatcher
    from src.model import final_residual, load_jz, load_lens, load_model, text_config
    from src.provenance import file_digest, meta_block
    from src.sites import (
        apply_site_deltas,
        loading_map,
        null_threshold,
        plan_edits,
        site_spread,
    )
    from src.swap import (
        clamped_swap_delta,
        head_frames,
        lens_vector,
        matched_norm_delta,
        swap_coefficients,
        swap_delta,
        swap_frame,
        swap_pseudoinverse,
    )
    from src.tasks import VERBAL_REPORT_PROMPT, build_trials

    spec = LENSES[args.lens or DEFAULT_LENS]
    paths = Paths.default()
    #: Every layer the fitted ``J^z`` covers, 0..31; ``ref_layers`` is the shorter span
    #: the block-anchored Jacobian lens exists on, which only ``layer_wide`` patches.
    layers = list(spec.jz_source_layers)
    ref_layers = list(spec.source_layers)

    model = load_model(spec)
    jz_n_prompts = args.n_prompts or spec.jz_n_prompts
    jz_file = jz_path(spec, n_prompts=jz_n_prompts, paths=paths)
    jz = load_jz(spec, jz_file)
    published = load_lens(spec)
    device = model.input_device
    jz_by_layer = {l: jz.jacobians[l].to(device) for l in layers}
    # Layer l's Jacobian-lens coordinates live at x_{l+1}, so the layer_wide patch goes on
    # block l's output -- a different anchor from every other row in the table.
    published_by_layer = {l: published.jacobians[l].to(device) for l in ref_layers}
    n_heads = spec.n_heads
    config = text_config(model)
    n_kv_heads = int(config.num_key_value_heads)

    # Write norms of the selected sites, if read.py has produced them. Optional: it is a
    # descriptive beside the result.
    read_path = paths.run_dir(args.read_name) / f"{spec.key}_read.json"
    rho = (
        json.loads(read_path.read_text())["magnitude"]["relative_write_norm"]
        if read_path.exists()
        else {}
    )

    # The null concept population: fixed ids, drawn once, shared by every trial. The
    # threshold has to mean the same thing across trials for the active counts to be
    # comparable, and only z varies between them anyway.
    null_ids = torch.randperm(
        int(config.vocab_size), generator=torch.Generator().manual_seed(NULL_SEED)
    )[:N_NULL_CONCEPTS]
    null_rows = effective_unembed(model, null_ids.to(device))

    # ------------------------------------------------------------------ frames
    zin_cache: dict[tuple[int, int], dict] = {}
    residual_cache: dict[tuple[int, int], dict] = {}
    #: Clean ``zin`` and block-output activations per prompt, every position.
    clean_cache: dict[str, tuple[dict, dict]] = {}
    clamp = args.coords == "clamp"

    def rows_for(source_id: int, target_id: int) -> torch.Tensor:
        """``[2, d_model]`` effective-unembedding rows for the source and target tokens."""
        rows = effective_unembed(
            model, torch.tensor([source_id, target_id], device=device)
        )
        return rows

    def zin_frames(source_id: int, target_id: int):
        """``{layer: ([n_heads, head_dim, 2] frames, pseudoinverses)}``.

        Depends only on ``(source, target)`` -- not on the arm, the alpha or the
        activation -- so it is built once per trial and reused across both
        alphas and every condition.
        """
        key = (source_id, target_id)
        if key not in zin_cache:
            rows = rows_for(source_id, target_id)
            built = {}
            for layer in layers:
                frames = head_frames(jz_by_layer[layer], spec, rows)
                built[layer] = (frames, swap_pseudoinverse(frames))
            zin_cache[key] = built
        return zin_cache[key]

    def residual_frames(source_id: int, target_id: int):
        """``{layer: ([d_model, 2], pinv)}`` -- the same construction at the block-output anchor."""
        key = (source_id, target_id)
        if key not in residual_cache:
            rows = rows_for(source_id, target_id)
            built = {}
            for layer in ref_layers:
                frame = swap_frame(
                    lens_vector(published_by_layer[layer], rows[0]),
                    lens_vector(published_by_layer[layer], rows[1]),
                )
                built[layer] = (frame, swap_pseudoinverse(frame))
            residual_cache[key] = built
        return residual_cache[key]

    def clean_coords(source_id: int, target_id: int, prompt_key: str):
        """Clean-pass coordinates: ``zin`` ``{layer: [n_heads, seq, 2]}``, residual ``{layer: [seq, 2]}``.

        What :func:`src.swap.clamped_swap_delta` clamps each edited site to.
        """
        z_full, out_full = clean_cache[prompt_key]
        zin = {}
        for layer, (_, pseudo) in zin_frames(source_id, target_id).items():
            z = z_full[layer].to(COMPUTE_DTYPE).view(-1, spec.n_heads, spec.head_dim)
            zin[layer] = torch.einsum("shd,hkd->hsk", z, pseudo.to(COMPUTE_DTYPE))
        residual = {
            layer: swap_coefficients(out_full[layer], pseudo)
            for layer, (_, pseudo) in residual_frames(source_id, target_id).items()
        }
        return zin, residual

    # ------------------------------------------------------------------ the forwards
    def run_sites(input_ids, source_id, target_id, elements, alpha, trial_index, coords):
        """One batched ``zin`` forward; batch element ``e`` edits ``elements[e] = (sites, arm)``.

        Returns final-position logits ``[n_elements, vocab]`` and the total edit norm
        ``[n_elements]`` -- the size descriptive that sits beside every count-matched
        contrast.
        """
        totals = torch.zeros(len(elements))
        frames_by_layer = zin_frames(source_id, target_id)
        plan = plan_edits(elements)

        def make_edit(layer):
            def delta_for(arm, head, block, index):
                frame, pseudo = frames_by_layer[layer]
                if clamp:
                    delta = clamped_swap_delta(
                        block, frame[head], coords[layer][head], alpha=alpha, pseudo=pseudo[head]
                    )
                else:
                    delta = swap_delta(block, frame[head], alpha=alpha, pseudo=pseudo[head])
                if arm == "matched_norm":
                    # Drawn per element at [seq, head_dim], and seeded on the **trial**:
                    # seeding on (layer, head) alone redraws the same direction for every
                    # trial whose prompt tokenises to the same length, turning a 140-trial
                    # control into one draw reused 140 times.
                    delta = torch.stack([
                        matched_norm_delta(
                            delta[row], seed=trial_index * 100003 + layer * 100 + head
                        )
                        for row in range(delta.shape[0])
                    ])
                return delta

            return lambda z: apply_site_deltas(
                z, spec, plan.get(layer, {}), delta_for, totals=totals
            )

        batched = input_ids.expand(len(elements), -1)
        with torch.no_grad(), ZinPatcher(
            model.layers, {l: make_edit(l) for l in layers}
        ):
            hidden = final_residual(model, batched)[:, -1]
        return model.unembed(hidden).float(), totals.sqrt()

    def run_layer_wide(input_ids, source_id, target_id, arm, alpha, trial_index, coords):
        """One block-anchored forward: the Jacobian lens, every layer it has, every position."""
        frames = residual_frames(source_id, target_id)

        def make_edit(layer):
            def edit(x):
                frame, pseudo = frames[layer]
                if clamp:
                    delta = clamped_swap_delta(x, frame, coords[layer], alpha=alpha, pseudo=pseudo)
                else:
                    delta = swap_delta(x, frame, alpha=alpha, pseudo=pseudo)
                if arm == "matched_norm":
                    delta = matched_norm_delta(delta, seed=trial_index * 100003 + layer)
                return (x.to(delta.dtype) + delta).to(x.dtype)

            return edit

        with torch.no_grad(), ResidualPatcher(
            model.layers, {l: make_edit(l) for l in ref_layers}
        ):
            hidden = final_residual(model, input_ids)[0, -1]
        return model.unembed(hidden).float()

    # ------------------------------------------------------------------ trials
    trials = build_trials(
        model,
        max_seq_len=MAX_SEQ_LEN,
        max_categories=args.max_categories,
        max_items=args.max_items,
    )
    counts = {task: sum(1 for t in trials if t.task == task) for task in TASKS}

    # ------------------------------------------------------------------ the sweep
    per_trial = {
        (task, arm, a): torch.zeros(counts[task], len(CONDITIONS))
        for task in TASKS
        for arm in ARMS
        for a in ALPHAS
    }
    # Total edit norm per cell, same shape; only the zin columns are filled, because the
    # block-anchored row edits a different space and its norm would not be comparable.
    edit_norm = {key: torch.zeros_like(value) for key, value in per_trial.items()}
    row_of = {task: 0 for task in TASKS}
    condition_index = {name: i for i, name in enumerate(CONDITIONS)}
    loading_cache: dict[str, tuple] = {}
    records: dict[str, list] = {task: [] for task in TASKS}

    for trial_index, trial in enumerate(trials):
        zin_cache.clear()
        residual_cache.clear()

        # Selection: one clean forward, scored at the readout position (the final token
        # for both task sets). Trials sharing a prompt share it entirely.
        if trial.prompt_key not in loading_cache:
            with torch.no_grad(), AnchorRecorder(
                model.layers, zin_at=layers, out_at=ref_layers
            ) as rec:
                model.forward(trial.input_ids)
                z_full = {l: rec.activations[("zin", l)][0].clone() for l in layers}
                out_full = {l: rec.activations[("out", l)][0].clone() for l in ref_layers}
            clean_cache[trial.prompt_key] = (z_full, out_full)
            z_by_layer = {l: z[-1] for l, z in z_full.items()}
            source_row = effective_unembed(
                model, torch.tensor([trial.source_id], device=device)
            )[0]
            loading = loading_map(
                jz_by_layer, z_by_layer, spec, source_row, layers
            )
            tau = null_threshold(
                loading_map(jz_by_layer, z_by_layer, spec, null_rows, layers),
                q=NULL_QUANTILE,
            )
            loading_cache[trial.prompt_key] = (loading.cpu(), tau.cpu())
        loading, tau = loading_cache[trial.prompt_key]

        sites = site_sets(loading, layers, tau, seed=RANDOM_SITE_SEED + trial_index)
        active = sites["active"]

        zin_coords, residual_coords = clean_coords(
            trial.source_id, trial.target_id, trial.prompt_key
        )
        row = row_of[trial.task]
        row_of[trial.task] += 1
        for alpha in ALPHAS:
            elements = [(c, a) for c in ZIN_CONDITIONS for a in ARMS]
            logits, totals = run_sites(
                trial.input_ids,
                trial.source_id,
                trial.target_id,
                [(sites[c], a) for c, a in elements],
                alpha,
                trial_index,
                zin_coords,
            )
            top = logits.argmax(dim=-1).cpu()
            for position, (condition, arm) in enumerate(elements):
                per_trial[(trial.task, arm, alpha)][row, condition_index[condition]] = (
                    float(int(top[position]) in trial.success_ids)
                )
                edit_norm[(trial.task, arm, alpha)][row, condition_index[condition]] = (
                    float(totals[position])
                )

            # The block-anchored row, one forward per arm: a different patch family, so it
            # cannot join the batch above.
            for arm in ARMS:
                logits = run_layer_wide(
                    trial.input_ids,
                    trial.source_id,
                    trial.target_id,
                    arm,
                    alpha,
                    trial_index,
                    residual_coords,
                )
                per_trial[(trial.task, arm, alpha)][
                    row, condition_index[RESIDUAL_CONDITION]
                ] = float(int(logits.argmax()) in trial.success_ids)

        records[trial.task].append({
            "prompt_key": trial.prompt_key,
            "answer": trial.answer,
            "answer_in_class": trial.answer_in_class,
            "n_active": len(active),
            "spread": {
                condition: site_spread(
                    sites[condition], n_heads=n_heads, n_kv_heads=n_kv_heads
                )
                for condition in SELECTED_CONDITIONS
            },
            "mean_loading": {
                condition: float(
                    sum(loading[layers.index(l), h] for l, h in sites[condition])
                    / len(sites[condition])
                )
                if sites[condition]
                else 0.0
                for condition in ZIN_CONDITIONS
            },
            "mean_relative_write_norm": {
                condition: float(
                    sum(rho[str(l)][h] for l, h in sites[condition]) / len(sites[condition])
                )
                if rho and sites[condition]
                else None
                for condition in ZIN_CONDITIONS
            },
        })
        if (trial_index + 1) % 20 == 0:
            logger.info("%d/%d trials", trial_index + 1, len(trials))

    # ------------------------------------------------------------------ per-layer
    per_layer = None
    if args.per_layer:
        hits = {
            (task, layer): torch.zeros(n_heads) for task in TASKS for layer in layers
        }
        for trial_index, trial in enumerate(trials):
            zin_cache.clear()
            zin_coords, _ = clean_coords(trial.source_id, trial.target_id, trial.prompt_key)
            for layer in layers:
                logits, _ = run_sites(
                    trial.input_ids,
                    trial.source_id,
                    trial.target_id,
                    [([(layer, h)], "head_jlens") for h in range(n_heads)],
                    PRIMARY_ALPHA,
                    trial_index,
                    zin_coords,
                )
                hits[(trial.task, layer)] += torch.tensor(
                    [int(t) in trial.success_ids for t in logits.argmax(dim=-1).cpu()],
                    dtype=torch.float32,
                )
        per_layer = {
            "alpha": PRIMARY_ALPHA,
            "arm": "head_jlens",
            "note": f"one head at one layer, all {n_heads} indices per forward, at every layer. "
            "Read against the whole grid, not one head's row: any head singled out here "
            "was singled out on these same trials.",
            "hits": {
                task: {l: hits[(task, l)].tolist() for l in layers} for task in TASKS
            },
            "n_trials": counts,
        }

    # ------------------------------------------------------------------ summary
    def mean(values):
        return float(sum(values) / len(values)) if values else 0.0

    results = {}
    for task in TASKS:
        entry = {"n_trials": counts[task]}
        for arm in ARMS:
            for alpha in ALPHAS:
                hit_counts = per_trial[(task, arm, alpha)].sum(dim=0)
                entry[f"{arm}_alpha{alpha}"] = {
                    condition: {
                        "rate": int(hit_counts[i]) / counts[task],
                        "successes": int(hit_counts[i]),
                        "trials": counts[task],
                    }
                    for condition, i in condition_index.items()
                }
        entry["edit_norm"] = {
            f"{arm}_alpha{alpha}": {
                condition: float(
                    edit_norm[(task, arm, alpha)][:, condition_index[condition]].mean()
                )
                for condition in ZIN_CONDITIONS
            }
            for arm in ARMS
            for alpha in ALPHAS
        }
        entry["selection"] = {
            "n_active": {
                "mean": mean([r["n_active"] for r in records[task]]),
                "median": statistics.median(r["n_active"] for r in records[task])
                if records[task]
                else 0,
                "min": min((r["n_active"] for r in records[task]), default=0),
                "max": max((r["n_active"] for r in records[task]), default=0),
                "n_empty": sum(1 for r in records[task] if r["n_active"] == 0),
                "of_n_sites": len(layers) * n_heads,
            },
            "spread": {
                condition: {
                    "n_layers": mean([r["spread"][condition]["n_layers"] for r in records[task]]),
                    "n_layer_groups": mean([
                        r["spread"][condition]["n_layer_groups"] for r in records[task]
                    ]),
                }
                for condition in SELECTED_CONDITIONS
            },
            "mean_loading": {
                condition: mean([r["mean_loading"][condition] for r in records[task]])
                for condition in ZIN_CONDITIONS
            },
            "mean_relative_write_norm": {
                condition: mean([
                    r["mean_relative_write_norm"][condition]
                    for r in records[task]
                    if r["mean_relative_write_norm"][condition] is not None
                ])
                for condition in ZIN_CONDITIONS
            },
            # verbal-report only: on some categories the model does not name a member of
            # the concept class at all, so the direction swapped out is not a concept
            # direction. Recorded, not corrected.
            "n_trials_answer_in_class": sum(
                1 for r in records[task] if r["answer_in_class"]
            ),
            # The answer itself, per category, so a reader can see *what* was swapped
            # out when it is not a class member (a newline on the raw prompt, say).
            "answers": {
                r["prompt_key"]: {"answer": r["answer"], "in_class": r["answer_in_class"]}
                for r in records[task]
                if r["answer"] is not None
            },
        }
        entry["per_trial_n_active"] = [r["n_active"] for r in records[task]]
        results[task] = entry

    payload = {
        "meta": meta_block(
            model=spec.hf_model_id,
            lens_key=spec.key,
            jz_n_prompts=jz_n_prompts,
            jz_path=jz_file,
            jz_sha256=file_digest(jz_file),
            published_lens_n_prompts=published.n_prompts,
            layers=layers,
            reference_layers=ref_layers,
            alphas=list(ALPHAS),
            primary_alpha=PRIMARY_ALPHA,
            arms=list(ARMS),
            conditions=list(CONDITIONS),
            site_counts=list(SITE_COUNTS),
            rules=list(RULES),
            selection="cos(z_{l,h}, v_{s,l,h}) on the source concept, at the readout "
            "position; edits applied at every position on the selected sites",
            coords=args.coords,
            null_concepts=N_NULL_CONCEPTS,
            null_seed=NULL_SEED,
            null_quantile=NULL_QUANTILE,
            random_site_seed=RANDOM_SITE_SEED,
            verbal_report_prompt=VERBAL_REPORT_PROMPT,
            probe_swap_note="lens directions substituted for the shipped probe "
            "directions, and its items are not the paper's headline 50-prompt set",
            write_norms_from=str(read_path) if rho else None,
            wall_clock_seconds=round(time.monotonic() - started, 1),
            dtype="bfloat16",
        ),
        "conditions": {
            RESIDUAL_CONDITION: {
                "sites": f"block output, Jacobian lens J_l, layers {ref_layers[0]}-{ref_layers[-1]}",
                "role": "the paper's intervention: reference / positive control",
                "anchor": "out",
                "rule": None,
                "count": None,
            },
            **{
                condition_name(rule, count): {
                    "sites": condition_role(rule, count)[0],
                    "role": condition_role(rule, count)[1],
                    "anchor": "zin",
                    "rule": rule,
                    "count": count,
                }
                for count in COUNTS
                for rule in RULES
            },
        },
        "condition_index": condition_index,
        "results": results,
        # The trial axis: every condition ran on the same trials and differs only in which
        # sites were edited, so the outcomes are paired trial by trial. Accumulating
        # scalar counts would discard the pairing irrecoverably.
        # [n_trials, n_conditions] of 0/1 per (task, arm, alpha).
        "per_trial_hits": {
            task: {
                f"{arm}_alpha{a}": [
                    [int(v) for v in row] for row in per_trial[(task, arm, a)].tolist()
                ]
                for arm in ARMS
                for a in ALPHAS
            }
            for task in TASKS
        },
        "per_layer": per_layer,
        "alpha_note": "alpha=1 exchanges the two lens coordinates; alpha=2 extrapolates "
        "past the exchange (the v_s coordinate becomes 2c_1 - c_0). They are different "
        "operations, not two strengths of one. Read both, never their maximum.",
    }
    out_path = paths.run_dir(args.out_name) / f"{spec.key}_write.json"
    out_path.write_text(json.dumps(payload, indent=1))

    for task in TASKS:
        entry = results[task]
        print(f"\n{task}  (n={entry['n_trials']} trials, {len(layers) * n_heads} sites)")
        active = entry["selection"]["n_active"]
        print(
            f"  active sites per trial: mean {active['mean']:.1f}, median "
            f"{active['median']}, range {active['min']}-{active['max']} of "
            f"{active['of_n_sites']} ({active['n_empty']} trials with none)"
        )
        print(
            "  " + f"{'condition':<21}"
            + "".join(f"{arm[:6] + ' a' + str(a):>13}" for arm in ARMS for a in ALPHAS)
        )
        for condition in CONDITIONS:
            cells = "".join(
                f"{entry[f'{arm}_alpha{a}'][condition]['rate']:>13.3f}"
                for arm in ARMS
                for a in ALPHAS
            )
            mark = " *" if condition == RESIDUAL_CONDITION else ""
            print(f"  {condition:<21}{cells}{mark}")
        primary = entry[f"head_jlens_alpha{PRIMARY_ALPHA}"]
        print(
            f"  head_jlens successes by count at alpha={PRIMARY_ALPHA} "
            f"({'/'.join(RULES)}): "
            + ", ".join(
                f"{count}: "
                + "/".join(
                    str(primary[condition_name(rule, count)]["successes"])
                    for rule in RULES
                )
                for count in COUNTS
            )
        )
    print(f"\n  * {RESIDUAL_CONDITION} is block-anchored (Jacobian lens J_l), not zin")
    print(payload["alpha_note"])
    print(f"\nwrote {out_path}  ({payload['meta']['wall_clock_seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
