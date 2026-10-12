"""Aggregation for the multi-site coordinate-swap sweep (no model, no GPU).

Contrasts are paired trial by trial, J minus Logit (:func:`paired_mean`). Indices into the head array are
``[selection lens][edit arm]``; ``J`` and ``L`` are the lenses, ``*_rand`` the matched
random directions.
"""
from __future__ import annotations

import numpy as np


def paired_mean(differences: np.ndarray, clusters: list[str]) -> dict:
    """Trial-weighted paired mean, with how many prompts and trials it averages."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) != len(clusters) or not np.isfinite(values).all():
        raise ValueError("one finite paired difference per trial required")
    if not len(values):
        return {"mean": None, "prompts": 0, "trials": 0}
    return {"mean": float(values.mean()), "prompts": len(set(clusters)), "trials": len(values)}


#: name -> ((selection, arm), (selection, arm)); the difference is first minus second.
CONTRASTS = {
    "end_to_end": (("J", "J"), ("L", "L")),
    "direction_at_J_sites": (("J", "J"), ("J", "L")),
    "direction_at_L_sites": (("L", "J"), ("L", "L")),
    "selection_with_J_edit": (("J", "J"), ("L", "J")),
    "selection_with_L_edit": (("J", "L"), ("L", "L")),
}


def merge_shards(shards: list[tuple[dict, dict]]) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Concatenate ``(record, arrays)`` shards in original trial order; refuse gaps."""
    records = [t for record, _ in shards for t in record["trials"]]
    heads = np.concatenate([arrays["heads"] for _, arrays in shards])
    wide = np.concatenate([arrays["layer_wide"] for _, arrays in shards])
    order = np.argsort([t["trial_index"] for t in records])
    total = shards[0][0]["configuration"]["n_trials_total"]
    if sorted(t["trial_index"] for t in records) != list(range(total)):
        raise ValueError("shards do not cover every trial exactly once")
    return heads[order], wide[order], [records[i] for i in order], shards[0][0]["configuration"]


def merge_optional(shards: list[tuple[dict, dict]], key: str) -> np.ndarray | None:
    """An array only some sweeps write (``layer_wide_band``), in trial order; ``None`` if absent."""
    if not all(key in arrays for _, arrays in shards):
        return None
    records = [t for record, _ in shards for t in record["trials"]]
    order = np.argsort([t["trial_index"] for t in records])
    return np.concatenate([arrays[key] for _, arrays in shards])[order]


#: Vocabulary size; an output whose target probability is exactly chance is uniform.
VOCAB = 128256


def _valid(values: np.ndarray, metrics: list[str], chance: np.ndarray) -> np.ndarray:
    """Trials whose output is finite and not uniform (see :func:`_cell`)."""
    outputs = [metrics.index(m) for m in metrics if m != "edit_norm"]
    valid = np.isfinite(values[:, outputs]).all(-1)
    return valid & ~np.isclose(values[:, metrics.index("probability")], chance, rtol=1e-6, atol=0)


def _cell(values: np.ndarray, metrics: list[str], chance: np.ndarray) -> dict:
    """A non-finite output (overflowed forward) is never a failure to elicit: counted apart.

    Only output metrics decide validity; an overflowed edit-norm accumulator with finite
    logits is a real (huge) edit, reported as inf. A uniform output (the final norm
    flattening an overflowed residual) is counted the same way.
    """
    finite = _valid(values, metrics, chance)
    success = values[:, metrics.index("success")] * finite
    return {"successes": int(success.sum()), "n": len(success), "rate": float(success.mean()),
            "n_nonfinite": int((~finite).sum()),
            "kl": float(values[finite, metrics.index("kl")].mean()) if finite.any() else None,
            "edit_norm": float(values[finite, metrics.index("edit_norm")].mean()) if finite.any() else None}


def summarize(heads: np.ndarray, wide: np.ndarray, records: list[dict], config: dict,
              band: np.ndarray | None = None) -> list[dict]:
    """One row per (task, alpha): every cell, the contrasts, and the layer-wide reference.

    ``band``, shaped like ``wide``, adds the workspace-band reference as ``layer_wide_band``.
    """
    conditions, arms, metrics = config["conditions"], config["arms"], config["metrics"]
    lenses = ("J", "L")
    success = metrics.index("success")
    rows = []
    for task in sorted({t["task"] for t in records}):
        mask = np.array([t["task"] == task for t in records])
        clusters = [t["prompt_key"] for t in records if t["task"] == task]
        chance = np.array([len(t["success_ids"]) / VOCAB for t in records if t["task"] == task])
        for bi, value in enumerate(config["strengths"]):
            block = heads[mask, bi]
            cells, contrasts = {}, {}
            for ci, condition in enumerate(conditions):
                cells[condition] = {f"{sel}/{arm}": _cell(block[:, ci, si, ai], metrics, chance)
                                    for si, sel in enumerate(lenses) for ai, arm in enumerate(arms)}
                contrasts[condition] = {}
                for name, (first, second) in CONTRASTS.items():
                    a = block[:, ci, lenses.index(first[0]), arms.index(first[1]), success]
                    b = block[:, ci, lenses.index(second[0]), arms.index(second[1]), success]
                    contrasts[condition][name] = paired_mean(a - b, clusters)
            reference = wide[mask, bi]
            layer_wide = {arm: _cell(reference[:, ai], metrics, chance) for ai, arm in enumerate(arms)}
            layer_wide_contrast = paired_mean(
                reference[:, arms.index("J"), success] - reference[:, arms.index("L"), success], clusters)
            rows.append({"task": task, "alpha": value, "n_trials": int(mask.sum()), "cells": cells,
                         "contrasts": contrasts, "layer_wide": layer_wide,
                         "layer_wide_contrast": layer_wide_contrast})
            if band is not None:
                rows[-1]["layer_wide_band"] = {arm: _cell(band[mask, bi][:, ai], metrics, chance)
                                               for ai, arm in enumerate(arms)}
    return rows


def selection_descriptives(records: list[dict]) -> dict:
    """Active-set sizes per lens and J/Logit top-set overlap, per task."""
    out = {}
    for task in sorted({t["task"] for t in records}):
        subset = [t for t in records if t["task"] == task]
        out[task] = {
            "n_active_median": {lens: float(np.median([t["n_active"][lens] for t in subset]))
                                for lens in ("J", "L")},
            "mean_top_overlap": {c: float(np.mean([t["top_overlap"][c] for t in subset]))
                                 for c in subset[0]["top_overlap"]},
        }
    return out


#: Gurnee et al. swap in only verbal-report targets outside the model's clean top 10.
CLEAN_TOP = 10


def paper_protocol(wide: np.ndarray, band: np.ndarray | None, records: list[dict],
                   config: dict) -> list[dict]:
    """The layer-wide references scored as Gurnee et al. score theirs.

    Top-1 and top-5, over every trial and over the trials whose target lies outside the
    clean top :data:`CLEAN_TOP`. Top-1 is the greedy-token ``success`` used everywhere else;
    top-5 is ``rank < 5``, where a bfloat16 tie with the target counts in its favour.
    Degenerate outputs count as failures and are tallied apart. Needs the swap sweep's
    ``rank`` metric and each trial's ``clean_rank``.
    """
    metrics, arms = config["metrics"], config["arms"]
    success, rank = metrics.index("success"), metrics.index("rank")
    rows = []
    for task in sorted({t["task"] for t in records}):
        mask = np.array([t["task"] == task for t in records])
        subset = [t for t in records if t["task"] == task]
        chance = np.array([len(t["success_ids"]) / VOCAB for t in subset])
        outside = np.array([t["clean_rank"] >= CLEAN_TOP for t in subset])
        for bi, value in enumerate(config["strengths"]):
            for scope, array in (("layer_wide", wide), ("layer_wide_band", band)):
                if array is None:
                    continue
                for lens in ("J", "L"):
                    values = array[mask, bi, arms.index(lens)]
                    valid = _valid(values, metrics, chance)
                    hits = valid & (values[:, success] == 1)
                    ranks = np.where(valid, values[:, rank], np.inf)
                    entry = {"task": task, "alpha": value, "scope": scope, "lens": lens,
                             "n_degenerate": int((~valid).sum())}
                    for name, keep in (("all", np.ones_like(outside)), ("outside_clean_top10", outside)):
                        entry[name] = {"n": int(keep.sum()), "top1": int(hits[keep].sum()),
                                       "top5": int((ranks[keep] < 5).sum())}
                    rows.append(entry)
    return rows
