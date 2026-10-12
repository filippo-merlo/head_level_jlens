"""[4] One results table, from the three records the other three scripts write.

    python -m scripts.report

Reads only ``results/`` -- no model, no GPU. Every number it prints was produced by an
earlier script and is traceable through that script's ``meta`` block to a checkpoint, a
corpus seed, a lens digest and a commit.

There are no verdicts here. The evidence supports three plain statements, and this script
prints the numbers behind each:

1. **A single head's write is legible through the fitted transport**, and more legible
   than through the identity transport applied to the same vector.
2. **The lens coordinate is causally load-bearing** -- swapping it in the residual stream
   through the Jacobian lens, the paper's own intervention, changes the model's answer
   most of the time, against a control floor near zero.
3. **How many head-level sites carry it.** At each site count the concept-selected sites
   are read against the same number chosen at random or by lowest loading, and the rate
   is reported along the count axis rather than at one chosen count.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

K_VALUES = (1, 5, 25)
#: The alpha the causal table is read at; the other is printed beside it, never maxed over.
PRIMARY_ALPHA = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lens", default=None, help="LensSpec key (default: DEFAULT_LENS)")
    parser.add_argument("--read-name", default="read")
    parser.add_argument("--write-name", default="write")
    parser.add_argument("--out-name", default="report")
    return parser


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def best_site_agreement_table(ranks_path: Path, read: dict) -> dict | None:
    """``{set: {"all_layers": f, "layer_0": f, ...}}`` -- how often J-head and Logit-head
    land on the same best site, from the raw ranks ``read.py`` writes beside its record.

    Each arm takes its own minimum, so the paired comparison shares the candidate sites
    and the items, not the chosen site. The ``.npz`` is regenerable rather
    than versioned, so a checkout without it gets ``None`` and the table is skipped.
    """
    if not ranks_path.exists():
        return None
    import numpy as np

    from src.passk import best_site_agreement

    ranks = np.load(ranks_path)
    layers = read["meta"]["layers"]
    table = {}
    for slug in read["results"]:
        a, b = ranks[f"{slug}__jhead"], ranks[f"{slug}__logithead"]
        table[slug] = {"all_layers": best_site_agreement(a, b)} | {
            f"layer_{layer}": best_site_agreement(a, b, layers=[li])
            for li, layer in enumerate(layers)
        }
    return table


def _cell(stat: dict) -> str:
    return f"{stat['rate']:.3f}"


def _count_grid(labels: dict, header: list[str], rows: dict) -> list[str]:
    """A markdown table with one row per site count, ``rows[count]`` its cells."""
    lines = ["| sites | " + " | ".join(header) + " |", "|---" * (1 + len(header)) + "|"]
    for count, cells in rows.items():
        lines.append(f"| {labels[count]} | " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from src.config import DEFAULT_LENS, JZ_FIT_STAGE, LENSES, Paths
    from src.evals import ALL_SETS
    from src.provenance import meta_block

    spec = LENSES[args.lens or DEFAULT_LENS]
    paths = Paths.default()
    key = spec.key
    fit_stage = paths.run_dir(JZ_FIT_STAGE)

    # The shipped J^z may come from one run or from merged shards, which write different
    # sidecars. Prefer the merge -- when both exist the merge is the lens the experiments
    # actually loaded -- and take the validation block from a shard it names, so the
    # numbers and the bytes come from the same run.
    merge = _load(fit_stage / f"{key}_merge.json")
    fit = merge or _load(fit_stage / f"{key}_fit.json")
    shards = list((merge or {}).get("shards", {}))
    checks = (
        _load(fit_stage / f"{key}_{shards[0]}_fit.json")
        if shards
        else _load(fit_stage / f"{key}_fit.json")
    )
    read = _load(paths.run_dir(args.read_name) / f"{key}_read.json")
    agreement = (
        best_site_agreement_table(
            paths.run_dir(args.read_name) / f"{key}_read_ranks.npz", read
        )
        if read
        else None
    )
    write = _load(paths.run_dir(args.write_name) / f"{key}_write.json")

    records = {"fit": fit, "fit_checks": checks, "read": read, "write": write}
    missing = [name for name, payload in records.items() if payload is None]

    lines: list[str] = []

    def say(text: str = "") -> None:
        lines.append(text)
        print(text)

    say(f"# Head-level Jacobian lens -- {spec.hf_model_id}")
    if missing:
        say(f"\n**INCOMPLETE**: no record for {missing}. What follows covers what ran.")

    # ---------------------------------------------------------------- the instrument
    say("\n## The instrument")
    if fit:
        how = f" as {len(shards)} disjoint shards" if shards else ""
        n_fitted = len(spec.jz_source_layers)
        where = (
            f"all {n_fitted} layers"
            if n_fitted == spec.n_layers
            else f"the {n_fitted} attention layers of {spec.n_layers} "
            f"({spec.jz_source_layers[0]}, {spec.jz_source_layers[1]}, ..., "
            f"{spec.jz_source_layers[-1]})"
        )
        say(
            f"- `J^z` fitted at `o_proj`'s input on "
            f"{fit['result']['prompts_fitted']} wikitext prompts{how}, {where}. "
            f"sha256 `{fit['result']['lens_sha256'][:12]}`."
        )
    if checks:
        gate = checks["wo_gate"]
        identity = gate["median_row_cosine_vs_identity"]
        reference = gate.get("reference", "W_O")
        say(
            f"- `W_O` gate at L{spec.jz_source_layers[-1]}: row cosine "
            f"{gate['median_row_cosine_vs_W_O']:.3f} against `{reference}`"
            + (
                f" and {identity:+.4f} against the identity. A hook slipped to `o_proj`'s "
                f"*output* would swap these, which is what makes it a test rather than a "
                f"trend."
                if identity is not None
                else f". `W_O` is {spec.d_model}x{spec.d_source}, so there is no identity "
                f"arm: a hook slipped to `o_proj`'s *output* would read ~0 here instead."
            )
        )

    # ---------------------------------------------------------------- legibility
    say("\n## 1. A head's write is legible through the transport")
    say(
        f"*Best site: minimum rank over the {spec.n_heads} heads of a layer, then over the region's "
        "layers, for head arms and the whole-layer reference alike; each rate a mean over items. The paired J-head minus Logit-head difference is "
        "primary -- both arms read the same activations on the same items.*"
    )
    if read:
        mag = read["magnitude"]
        say(
            f"\nHeads write a median **{mag['median_relative_write_norm']:.4f}** of "
            f"the residual stream they join (max {mag['max_relative_write_norm']:.4f}), "
            f"with median ablation KL **{mag['median_ablation_kl']:.5f} nats** (max "
            f"{mag['max_ablation_kl']:.4f}), over {mag['n_items']} items at each one's "
            f"readout position. Every number below describes the *direction* of a write "
            f"that size."
        )
        sets = [s for s in ALL_SETS if s in read["results"]]
        # Items with no scoreable intermediate under this tokenizer are left out of the
        # mean (src.passk.scoreable_items); say so wherever it happened.
        short = {
            slug: (entry["n_items_scored"], entry["n_items"])
            for slug, entry in read["results"].items()
            if slug in sets and entry.get("n_items_scored", entry["n_items"]) != entry["n_items"]
        }
        if short:
            skipped = read["meta"].get("candidate_coverage", {})
            say(
                "\nItems scored, where the candidate rule left an item with no scoreable "
                "intermediate under this tokenizer (the rest of the set is unaffected): "
                + "; ".join(
                    f"{slug.replace('lens-eval-', '')} {n}/{total} ("
                    + ", ".join(
                        f"{item}: {word}"
                        for item, word in skipped.get(slug, {}).get("unscoreable", [])
                        if item in skipped.get(slug, {}).get("items_without_candidates", [])
                    )
                    + ")"
                    for slug, (n, total) in short.items()
                )
                + "."
            )
        for k in K_VALUES:
            say(f"\n**pass@{k}**, all layers\n")
            say("| set | J-head | Logit-head | J-layer | ratio | paired diff |")
            say("|---|---|---|---|---|---|")
            for slug in sets:
                block = read["results"][slug]["regions"]["all_layers"]["k"][str(k)]
                m = block["rates"]
                paired = block["paired_difference"]["jhead_minus_logithead"]
                ratio = block["jhead_over_logithead"]

                say(
                    f"| {slug.replace('lens-eval-', '')} | {m['jhead']['point']:.3f} | "
                    f"{m['logithead']['point']:.3f} | "
                    f"{m['jlayer']['point']:.3f} | "
                    f"{'--' if ratio is None else f'{ratio:.2f}x'} | "
                    f"{paired['point']:+.3f} |"
                )
            positive = sum(
                1
                for slug in sets
                if read["results"][slug]["regions"]["all_layers"]["k"][str(k)][
                    "paired_difference"
                ]["jhead_minus_logithead"]["point"]
                > 0
            )
            ratios = [
                read["results"][slug]["regions"]["all_layers"]["k"][str(k)][
                    "jhead_over_logithead"
                ]
                for slug in sets
            ]
            ratios = [r for r in ratios if r is not None]
            say(
                f"\nRange {min(ratios):.2f}x-{max(ratios):.2f}x over the logit lens on "
                f"the same vector; the paired difference is positive on "
                f"{positive}/{len(sets)} sets."
            )

        # The best site keeps one head per intermediate; this is how many read it that
        # high, beside the same count on a prompt the concept is not annotated on. Older
        # records predate the control and have no such block.
        if all("sites_hit" in read["results"][slug] for slug in sets):
            n_sites = read["meta"]["n_sites"]
            for k in K_VALUES:
                say(
                    f"\n**Sites within the top {k}**, of {n_sites}: mean count per "
                    "intermediate on its own prompt, and on a control prompt from the same "
                    "set not annotated with it\n"
                )
                say("| set | J-head | J-head control | Logit-head | Logit-head control | "
                    "paired diff of excess |")
                say("|---|---|---|---|---|---|")
                for slug in sets:
                    hit = read["results"][slug]["sites_hit"][str(k)]
                    j, lg = hit["jhead"], hit["logithead"]
                    excess = (j["concept"] - j["control"]) - (lg["concept"] - lg["control"])
                    say(
                        f"| {slug.replace('lens-eval-', '')} | {j['concept']:.1f} | "
                        f"{j['control']:.1f} | {lg['concept']:.1f} | {lg['control']:.1f} | "
                        f"{excess:+.1f} |"
                    )

        # The aggregate above is a summary of this table, not a scope: the same
        # comparison is available at every layer, so a reader can re-aggregate over any
        # span they like. Chunked because 32 columns do not fit a page width.
        layers = read["meta"]["layers"]
        say("\n**Per layer**, pass@5, J-head over Logit-head\n")
        chunk = 16 if len(layers) > 16 else len(layers)
        for start in range(0, len(layers), chunk):
            block_layers = layers[start : start + chunk]
            say("| set | " + " | ".join(f"L{l}" for l in block_layers) + " |")
            say("|---" * (len(block_layers) + 1) + "|")
            for slug in sets:
                cells = []
                for layer in block_layers:
                    ratio = read["results"][slug]["regions"][f"layer_{layer}"]["k"]["5"][
                        "jhead_over_logithead"
                    ]
                    cells.append("--" if ratio is None else f"{ratio:.2f}")
                say(f"| {slug.replace('lens-eval-', '')} | " + " | ".join(cells) + " |")
            say("")

        # Each arm takes its own minimum, so the paired difference compares the best
        # head *under each readout*, not one head read two ways. This is how often the
        # two nonetheless coincide: the fraction of (item, intermediate) pairs for which
        # some site is a minimiser under both.
        if agreement:
            say("\n**Best-site agreement**, J-head and Logit-head: fraction of (item, intermediate) pairs whose best site is the same under both readouts\n")
            for start in range(0, len(layers), chunk):
                block_layers = layers[start : start + chunk]
                say("| set | " + " | ".join(f"L{l}" for l in block_layers) + " |")
                say("|---" * (len(block_layers) + 1) + "|")
                for slug in sets:
                    cells = [f"{agreement[slug][f'layer_{layer}']:.2f}" for layer in block_layers]
                    say(f"| {slug.replace('lens-eval-', '')} | " + " | ".join(cells) + " |")
                say("")
            say("| set | all layers |\n|---|---|")
            for slug in sets:
                say(f"| {slug.replace('lens-eval-', '')} | {agreement[slug]['all_layers']:.2f} |")
        else:
            say("\n*Best-site agreement not computed: no `_read_ranks.npz` beside the read record.*")

    # ---------------------------------------------------------------- causal
    say("\n## 2-3. The coordinate is load-bearing, and how many sites carry it")
    say(
        "*One independent variable: which sites are edited. Same trials, same "
        "intervention, same arms throughout. At each site count three rows edit the same "
        "number of sites and differ only in which: the highest-loading on the source "
        "concept (`top_k`), a random draw (`random_k`), the lowest-loading (`bottom_k`); "
        "`active` is the per-site-null threshold set, whose count is an outcome, with its "
        "two controls drawn at that count.*"
    )
    if write:
        arms, alphas = write["meta"]["arms"], write["meta"]["alphas"]
        rules = write["meta"]["rules"]
        counts = [*write["meta"]["site_counts"], "active"]
        # Rows by (rule, count), so the grids never parse a condition name.
        by_row = {
            (block["rule"], block["count"]): name
            for name, block in write["conditions"].items()
            if block["anchor"] == "zin"
        }
        # The one block-anchored row, found by its anchor rather than its name.
        residual = next(
            name for name, block in write["conditions"].items() if block["anchor"] == "out"
        )
        ref_layers = write["meta"]["reference_layers"]
        for task in write["results"]:
            entry = write["results"][task]
            selection = entry["selection"]
            active = selection["n_active"]
            labels = {
                count: f"active (median {active['median']})" if count == "active" else count
                for count in counts
            }
            say(f"\n**{task}** -- {entry['n_trials']} trials\n")
            answers = selection.get("answers")
            if answers:
                # Whether the direction swapped out is a concept direction at all: the
                # task's premise, which the raw prompt does not guarantee on every model.
                in_class = selection["n_trials_answer_in_class"]
                outside = sorted(
                    {repr(a["answer"]) for a in answers.values() if not a["in_class"]}
                )
                say(
                    f"The greedy answer is a member of its category on {in_class} of "
                    f"{entry['n_trials']} trials"
                    + (f"; elsewhere it is {', '.join(outside)}." if outside else ".")
                    + " Where it is not, the direction swapped out is not a concept "
                    "direction and success means the target overtakes that token.\n"
                )
            reference = "; ".join(
                f"{arm} a{a:g} {_cell(entry[f'{arm}_alpha{a}'][residual])}"
                for arm in arms
                for a in alphas
            )
            say(
                f"`{residual}`, the paper's own intervention -- the Jacobian lens on the "
                f"residual stream at every block {ref_layers[0]}-{ref_layers[-1]} -- is "
                f"the reference the head-level "
                f"rows are read against: {reference}. It is block-anchored through the "
                f"Jacobian lens, not `zin` through `J^z`: a second patch family, marked "
                f"rather than blended in. Without it a null at any count would say "
                f"nothing about the count and everything about the setup."
            )

            for arm in arms:
                say(f"\n`{arm}` arm, swap rate by site count\n")
                header = [f"{rule} a{a:g}" for a in alphas for rule in rules]
                rows = {
                    count: [
                        _cell(entry[f"{arm}_alpha{a}"][by_row[(rule, count)]])
                        for a in alphas
                        for rule in rules
                    ]
                    for count in counts
                }
                for line in _count_grid(labels, header, rows):
                    say(line)

            loading = selection["mean_loading"]
            norms = selection["mean_relative_write_norm"]
            edit = entry["edit_norm"][f"head_jlens_alpha{PRIMARY_ALPHA}"]
            say(
                f"\nSelection descriptives. A concept is active at {active['mean']:.1f} "
                f"of {active['of_n_sites']} sites per trial (median {active['median']}, "
                f"range {active['min']}-{active['max']}, {active['n_empty']} trials with "
                f"none), against a per-site 99th-percentile null. Per count, "
                f"{' / '.join(rules)}: mean loading of the selected sites, their mean "
                f"relative write norm (from read.py), and the total edit norm the "
                f"`head_jlens` edit realised at alpha={PRIMARY_ALPHA:g}. The controls are "
                f"matched in count, not in size -- high loading implies a large `|c_s|` "
                f"implies a large edit -- so the last column is the size difference left "
                f"visible rather than equalised away.\n"
            )
            rows = {
                count: [
                    " / ".join(f"{loading[by_row[(rule, count)]]:.3f}" for rule in rules),
                    " / ".join(
                        f"{norms[by_row[(rule, count)]]:.3f}"
                        if norms[by_row[(rule, count)]]
                        else "--"
                        for rule in rules
                    ),
                    " / ".join(f"{edit[by_row[(rule, count)]]:.1f}" for rule in rules),
                ]
                for count in counts
            }
            header = ["mean loading", "relative write norm", "total edit norm"]
            for line in _count_grid(labels, header, rows):
                say(line)
            spread = selection["spread"]
            say(
                "\nThe concept-selected sets span (distinct layers / distinct (layer, "
                "GQA-group) pairs): "
                + ", ".join(
                    f"`{name}` {s['n_layers']:.1f}/{s['n_layer_groups']:.1f}"
                    for name, s in spread.items()
                )
                + " -- a set that kept picking group-mates would not be that many "
                "independent sites, since one `W_V` is shared within a GQA group."
            )
        say(f"\n{write['alpha_note']}")

    # ---------------------------------------------------------------- caveats
    say("\n## What this does not establish")
    # The share is read off the record rather than written in, so the caveat cannot
    # drift from the table it qualifies.
    share = (
        f"a median {read['magnitude']['median_relative_write_norm']:.1%} of"
        if read
        else "a small fraction of"
    )
    say(
        f"- **Legibility is not magnitude.** RMSNorm is scale-invariant, so a head "
        f"contributing almost nothing still yields a confident-looking top-1. The heads "
        f"above write {share} the residual stream they join."
    )
    say(
        "- **probe-swap deviates.** Its items substitute lens directions for the shipped "
        "linear-probe directions, and they are not the paper's headline 50-prompt set, "
        "so those rates are not comparable to the paper's."
    )

    payload = {
        "meta": meta_block(
            model=spec.hf_model_id,
            lens_key=spec.key,
            # The table is assembled *here*, so a reader who opens only this file must
            # still be able to tell which commit assembled it. Each input's own git block
            # closes the chain.
            inputs={
                name: (record.get("meta", {}) or {}).get("git")
                for name, record in records.items()
                if record is not None
            },
        ),
        "model": spec.hf_model_id,
        "missing_records": missing,
        "lens_sha256": fit["result"]["lens_sha256"] if fit else None,
        "best_site_agreement": agreement,
        "report_markdown": "\n".join(lines),
    }
    out_dir = paths.run_dir(args.out_name)
    (out_dir / f"{key}_report.json").write_text(json.dumps(payload, indent=1))
    (out_dir / f"{key}_report.md").write_text("\n".join(lines) + "\n")
    say(f"\nwrote {out_dir / f'{key}_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
