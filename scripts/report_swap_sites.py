"""Merge the multi-site J-head vs Logit-head swap shards into summary.json and summary.md (no GPU).

    python -m scripts.report_swap_sites results/write_swap_sites_clamp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.provenance import file_digest
from src.swap_sites_report import (
    merge_optional,
    merge_shards,
    paper_protocol,
    selection_descriptives,
    summarize,
)

COUNTS = ("1", "8", "16", "32", "64", "active")


def _diff(result):
    return f"{result['mean']:+.3f}"


def _rate(cell):
    if cell["n_nonfinite"] == cell["n"]:
        return "degenerate"
    return f"{cell['successes']}" + (f" ({cell['n_nonfinite']} degenerate)" if cell["n_nonfinite"] else "")


def _kl(cell):
    return "n/a" if cell["kl"] is None else f"{cell['kl']:.3f}"


INTRO = ["# Multi-site coordinate swap: J-head vs Logit-head", "",
         "Experiment 2's sites, scope and swap (every position of every selected head, edited",
         "pass), with the frame built through J^z (J) or W_O (Logit). The swap's size follows",
         "each lens's coordinates, so edit norms differ between lenses (last line per table)."]


def _norm(cell):
    return "n/a" if cell["edit_norm"] is None else f"{cell['edit_norm']:.1f}"


def _render_paper_protocol(rows, config) -> list[str]:
    band, ref = config["band_layers"], config["reference_layers"]
    lines = ["", "## Layer-wide references, scored as Gurnee et al. score theirs", "",
             "Target rank after the swap, top-1 and top-5, over all trials and over the trials whose",
             "target was outside the clean top 10 (the paper swaps in only such verbal-report targets).",
             f"`band` is block outputs {band[0]}-{band[-1]}, the paper's workspace depth.", "",
             "| task | alpha | scope | lens | top-1 | top-5 | top-1, outside top-10 | top-5, outside top-10 |",
             "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        a, o = r["all"], r["outside_clean_top10"]
        scope = "band" if r["scope"] == "layer_wide_band" else f"{ref[0]}-{ref[-1]}"
        lens = "Jacobian lens" if r["lens"] == "J" else "unembedding"
        lines.append(f"| {r['task']} | {r['alpha']:g} | {scope} | {lens} | {a['top1']}/{a['n']} | "
                     f"{a['top5']}/{a['n']} | {o['top1']}/{o['n']} | {o['top5']}/{o['n']} |"
                     + (f" ({r['n_degenerate']} degenerate)" if r["n_degenerate"] else ""))
    return lines


def render(rows, descriptives, config, paper=None) -> str:
    intro = list(INTRO)
    if config.get("coords") == "clamp":
        intro[2:5] = ["Experiment 2's sites and scope (every position of every selected head), with each",
                      "site's coordinates clamped to the swapped clean-pass ones and the frame built through",
                      "J^z (J) or W_O (Logit). The swap's size follows",
                      "each lens's coordinates, so edit norms differ between lenses (last line per table)."]
    lines = [*intro,
             "Cells are successes out of the task's trials. `J` / `L` name the",
             "lens; `sel` is the lens that chose the sites, `edit` the lens that built the edit.",
             "Differences are paired per trial, J minus Logit. Random sites are shared by both",
             "lenses; bottom sites use each lens's own",
             "loading; random direction is a same-norm random edit at the lens's own selected sites.",
             ""]
    for task, info in descriptives.items():
        lines.append(f"{task}: median active count J {info['n_active_median']['J']:.0f}, "
                     f"Logit {info['n_active_median']['L']:.0f}; mean J/Logit top-set overlap "
                     + ", ".join(f"{c}: {v:.1f}" for c, v in info["mean_top_overlap"].items()))
    for row in rows:
        c, k = row["cells"], row["contrasts"]
        lines += ["", f"## {row['task']}, alpha = {row['alpha']:g} ({row['n_trials']} trials)", "",
                  "| sites | J sel+edit | L sel+edit | end-to-end diff | random sites J / L "
                  "| bottom J / L | random direction J / L | direction diff at J sites "
                  "| selection diff, J edit |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for count in COUNTS:
            top = "active" if count == "active" else f"top_{count}"
            rnd, bot = f"random_{count}", f"bottom_{count}"
            lines.append(
                f"| {count} | {_rate(c[top]['J/J'])} | {_rate(c[top]['L/L'])} | "
                f"{_diff(k[top]['end_to_end'])} | "
                f"{_rate(c[rnd]['J/J'])} / {_rate(c[rnd]['J/L'])} | "
                f"{_rate(c[bot]['J/J'])} / {_rate(c[bot]['L/L'])} | "
                f"{_rate(c[top]['J/J_rand'])} / {_rate(c[top]['L/L_rand'])} | "
                f"{_diff(k[top]['direction_at_J_sites'])} | {_diff(k[top]['selection_with_J_edit'])} |")
        w = row["layer_wide"]
        lines.append(f"| layer-wide | {_rate(w['J'])} | {_rate(w['L'])} | "
                     f"{'n/a' if w['J']['n_nonfinite'] or w['L']['n_nonfinite'] else _diff(row['layer_wide_contrast'])}"
                     f" | | | {_rate(w['J_rand'])} / "
                     f"{_rate(w['L_rand'])} | | |")
        b = row.get("layer_wide_band")
        if b is not None:
            band = config["band_layers"]
            lines.append(f"| layer-wide, band {band[0]}-{band[-1]} | {_rate(b['J'])} | {_rate(b['L'])} | "
                         f"| | | {_rate(b['J_rand'])} / {_rate(b['L_rand'])} | | |")
        lines.append("")
        lines.append("Mean KL from clean, top_64 J / L: "
                     f"{_kl(c['top_64']['J/J'])} / {_kl(c['top_64']['L/L'])}; "
                     f"layer-wide J / L: {_kl(w['J'])} / {_kl(w['L'])}"
                     + (f"; band J / L: {_kl(b['J'])} / {_kl(b['L'])}." if b is not None else "."))
        lines.append("Mean total edit norm at each lens's own top sites, J / L: " + ", ".join(
            f"{k}: {_norm(c[f'top_{k}']['J/J'])} / {_norm(c[f'top_{k}']['L/L'])}" for k in COUNTS[:-1]) + ".")
    if paper:
        lines += _render_paper_protocol(paper, config)
    lines += ["", "Full cells (all selection x edit combinations, KL, edit norms): `summary.json`."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    shards = []
    for path in sorted(args.directory.glob("*_shard*of*.json")):
        record = json.loads(path.read_text())
        arrays_path = path.with_suffix(".npz")
        if file_digest(arrays_path) != record["arrays_sha256"]:
            raise ValueError(f"digest mismatch: {arrays_path}")
        shards.append((record, dict(np.load(arrays_path))))
    if not shards:
        raise ValueError("no shards found")
    heads, wide, records, config = merge_shards(shards)
    band = merge_optional(shards, "layer_wide_band")
    rows = summarize(heads, wide, records, config, band=band)
    paper = paper_protocol(wide, band, records, config) if "rank" in config["metrics"] else None
    descriptives = selection_descriptives(records)
    ranks = shards[0][0].get("ranks", {})
    summary = {"configuration": config, "selection": descriptives, "rows": rows,
               "paper_protocol": paper,
               "rank_deficient_maps": {k: v for k, v in ranks.items()
                                       if min(v) < (128 if "head" in k else 4096)},
               "shards": [record["meta"] for record, _ in shards]}
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (args.directory / "summary.md").write_text(render(rows, descriptives, config, paper))
    print((args.directory / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
