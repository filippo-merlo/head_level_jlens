"""The records the paper cites, read back for the notebook.

Two uses. The paper's headline tables are *read* from ``results/`` rather than typed into
the notebook, so the numbers beside the live pictures are the ones the paper prints and
cannot drift from them. And the per-item ranks (``results/read/``) and per-trial swap
outcomes (``results/write_swap_sites_clamp/``) are what a live reading or edit is checked
against.

Nothing here touches the model.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np

from src.evals import ALL_SETS, load_eval

#: The swap record behind the paper's Section 4.4 (Table 3, Figure 4, Table S6).
SWAP_RUN = "write_swap_sites_clamp"
#: Paper's names for the task keys; "probe-swap" stays the key because the released data
#: file is named after it.
TASK_LABEL = {"verbal-report": "verbal report", "probe-swap": "intermediate replacement"}
SET_ORDER = tuple(s.removeprefix("lens-eval-") for s in ALL_SETS)


@cache
def _json(path: Path) -> dict:
    return json.loads(path.read_text())


@cache
def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {k: arrays[k] for k in arrays.files}


# ------------------------------------------------------------------ reading (Section 3)


def read_record(results_root: Path, lens_key: str) -> dict:
    return _json(Path(results_root) / "read" / f"{lens_key}_read.json")


def read_ranks(results_root: Path, lens_key: str, item) -> dict[str, np.ndarray] | None:
    """``{arm: [n_intermediates, n_layers, n_heads]}`` recorded for one evaluation item.

    ``None`` if the raw ranks are not on disk: they are gitignored and regenerable, so a
    fresh clone has the tables but not the per-site ranks.
    """
    path = Path(results_root) / "read" / f"{lens_key}_read_ranks.npz"
    if item is None or not path.exists():
        return None
    index = next(i for i, other in enumerate(load_eval(item.slug)) if other.name == item.name)
    ranks = _npz(path)
    return {
        arm: ranks[f"{item.slug}__{arm}"][index, : len(item.intermediates)]
        for arm in ("jhead", "logithead", "jlayer")
    }


def passk_table(results_root: Path, lens_key: str, k: int) -> list[dict]:
    """Table 1 (k = 1) or Tables S2/S3 (k = 5, 25): all-layer pass@k per set."""
    results = read_record(results_root, lens_key)["results"]
    rows = []
    for slug in ALL_SETS:
        block = results[slug]["regions"]["all_layers"]["k"][str(k)]
        rates = block["rates"]
        rows.append(
            {
                "set": slug.removeprefix("lens-eval-"),
                "J-head": rates["jhead"]["point"],
                "Logit-head": rates["logithead"]["point"],
                "J-layer": rates["jlayer"]["point"],
                "paired difference": block["paired_difference"]["jhead_minus_logithead"][
                    "point"
                ],
            }
        )
    return rows


def per_layer_passk(results_root: Path, lens_key: str, set_name: str, k: int) -> list[dict]:
    """One set's per-layer pass@k for each arm -- one panel of the paper's Figure 2."""
    record = read_record(results_root, lens_key)
    entry = record["results"][f"lens-eval-{set_name}"]["regions"]
    rows = []
    for layer in record["meta"]["layers"]:
        rates = entry[f"layer_{layer}"]["k"][str(k)]["rates"]
        for arm, label in (("jhead", "J-head"), ("logithead", "Logit-head"), ("jlayer", "J-layer")):
            if arm in rates:
                rows.append({"layer": layer, "arm": label, "pass@k": rates[arm]["point"]})
    return rows


# ------------------------------------------------------------------ editing (Section 4)


def swap_summary(results_root: Path) -> dict:
    return _json(Path(results_root) / SWAP_RUN / "summary.json")


def swap_table(results_root: Path) -> list[dict]:
    """Table 3: target-token successes, J-head / Logit-head, per task and strength.

    Each readout's swap at its own sites is cell ``J/J`` or ``L/L`` (selection lens /
    edit lens); random sites are shared by construction; the random direction is the
    same-norm control at each readout's own 64 sites; the layer-wide reference is the
    workspace band, block outputs 12-28.
    """
    summary = swap_summary(results_root)
    columns = {(row["task"], row["alpha"]): row for row in summary["rows"]}
    order = [("verbal-report", 1.0), ("verbal-report", 2.0), ("probe-swap", 1.0), ("probe-swap", 2.0)]

    def pair(row, condition, j="J/J", l="L/L"):
        cells = row["cells"][condition]
        return f"{cells[j]['successes']} / {cells[l]['successes']}"

    lines = [(str(m), f"top_{m}") for m in (1, 8, 16, 32, 64)] + [("active", "active")]
    table = []
    for name, condition in lines:
        table.append({"sites": name} | {_column(t, a): pair(columns[t, a], condition) for t, a in order})
    for name, condition, j, l in (
        ("64 random sites", "random_64", "J/J", "L/L"),
        ("64 lowest-loading", "bottom_64", "J/J", "L/L"),
        ("64, random direction", "top_64", "J/J_rand", "L/L_rand"),
    ):
        table.append({"sites": name} | {_column(t, a): pair(columns[t, a], condition, j, l) for t, a in order})
    table.append(
        {"sites": "layer-wide (12–28)"}
        | {
            _column(t, a): f"{columns[t, a]['layer_wide_band']['J']['successes']} / "
            f"{columns[t, a]['layer_wide_band']['L']['successes']}"
            for t, a in order
        }
    )
    table.append(
        {"sites": "update norm, 64 sites"}
        | {
            _column(t, a): f"{columns[t, a]['cells']['top_64']['J/J']['edit_norm']:.1f} / "
            f"{columns[t, a]['cells']['top_64']['L/L']['edit_norm']:.1f}"
            for t, a in order
        }
    )
    return table


def _column(task: str, alpha: float) -> str:
    n = {"verbal-report": 140, "probe-swap": 45}[task]
    return f"{TASK_LABEL[task]}, α={alpha:g} (of {n})"


@cache
def _trial_index_map(results_root: Path) -> dict[int, tuple[Path, int]]:
    """``{trial_index: (shard stem, row)}`` over the swap record's shards."""
    out = {}
    for json_path in sorted((Path(results_root) / SWAP_RUN).glob("*_shard*of*.json")):
        for row, trial in enumerate(_json(json_path)["trials"]):
            out[trial["trial_index"]] = (json_path, row)
    return out


def recorded_trial(results_root: Path, trial_index: int) -> dict | None:
    """One trial's recorded outcomes, or ``None`` if it is not in the record.

    Returns:
        ``{"configuration", "trial", "heads", "layer_wide", "layer_wide_band"}`` where
        ``heads`` is ``[alpha, condition, selection lens, arm, metric]`` and the two
        references ``[alpha, arm, metric]``, in the record's own axis order.
    """
    found = _trial_index_map(Path(results_root)).get(trial_index)
    if found is None:
        return None
    json_path, row = found
    npz_path = json_path.with_suffix(".npz")
    if not npz_path.exists():
        return None
    record, arrays = _json(json_path), _npz(npz_path)
    return {
        "configuration": record["configuration"],
        "trial": record["trials"][row],
        "heads": arrays["heads"][row],
        "layer_wide": arrays["layer_wide"][row],
        "layer_wide_band": arrays["layer_wide_band"][row],
    }


def swap_n_trials(results_root: Path) -> int:
    return swap_summary(results_root)["configuration"]["n_trials_total"]
