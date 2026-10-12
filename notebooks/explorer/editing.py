"""The coordinate swap at head sites (paper Section 4), J-head and Logit-head.

The computation is ``scripts/write_swap_sites.py``'s, written out of that driver's
closures into functions so a notebook cell can call it on one trial. Everything that
fixes a number is imported rather than restated: the trials come from
:func:`src.tasks.build_trials`, the site sets from :func:`scripts.write.site_sets`, the
counts, alphas, null and seeds from :mod:`scripts.write`, the arms and band from
:mod:`scripts.write_swap_sites`, and the algebra from :mod:`src.swap`,
:mod:`src.sites` and :mod:`src.hooks`.

What is *not* imported -- the frame building, the clean coordinates, the batched head
edit, the residual reference and the metrics -- is restated line for line, and
:func:`paper_protocol` re-runs a paper trial as the driver ran it (all 144 condition x
lens x arm cells in one batch, both strengths, both residual references) so
:func:`compare_to_record` can demand the recorded outcomes back. A custom edit then
calls the same functions with the sites you chose.

A head's frame is ``M_h^T [q_s q_t]`` with ``M = J^z`` (J-head) or ``M = W_O``
(Logit-head); the swap clamps each edited site's two coordinates to the swapped
clean-pass ones, ``V^+ y <- sigma_alpha(V^+ y_clean)``, at every prompt position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from scripts.write import (
    ALPHAS,
    N_NULL_CONCEPTS,
    NULL_QUANTILE,
    NULL_SEED,
    RANDOM_SITE_SEED,
    ZIN_CONDITIONS,
    site_sets,
)
from scripts.write_swap_sites import (
    ARMS,
    LENSES,
    NORM_COLUMN,
    OUTPUT_COLUMNS,
    SWAP_METRICS,
    WORKSPACE_DEPTH,
)
from src.config import MAX_SEQ_LEN
from src.heads import COMPUTE_DTYPE, effective_unembed
from src.hooks import AnchorRecorder, ResidualPatcher, ZinPatcher
from src.model import final_residual, text_config
from src.readout import target_direction_id, target_token_ids
from src.sites import (
    apply_site_deltas,
    loading_map,
    null_threshold,
    plan_edits,
    select_sites,
)
from src.swap import (
    clamped_swap_delta,
    head_frames,
    lens_vector,
    swap_coefficients,
    swap_frame,
    swap_pseudoinverse,
)
from src.tasks import (
    Trial,
    build_trials,
    load_probe_swap,
    load_verbal_report,
    target_rank,
)

from .session import LensSession

LENS_LABEL = {"J": "J-head", "L": "Logit-head"}
#: The site rules a custom edit offers; the first three are the paper's.
RULE_LABEL = {
    "top": "highest-loading",
    "random": "random",
    "bottom": "lowest-loading",
    "active": "active (above the per-site null)",
    "manual": "the sites clicked on the map",
}
#: Every condition x selection lens x arm cell, in the driver's batch order.
ELEMENTS_INDEX = [
    (ci, si, ai)
    for ci in range(len(ZIN_CONDITIONS))
    for si in range(len(LENSES))
    for ai in range(len(ARMS))
]
#: The driver's forward batch (``--batch-size`` default). bf16 kernels depend on the
#: batch's shape, so reproducing a recorded outcome bit for bit means using it too.
DRIVER_BATCH = 144


@dataclass(frozen=True)
class TrialInfo:
    """One trial and the words it is about, for labels and for the record lookup.

    Attributes:
        index: Position in :func:`src.tasks.build_trials`' list -- the paper's trial
            index, which seeds the random sites and the random directions. ``None`` for
            a custom trial.
        source, target, answer: What is swapped out, swapped in, and which word counts
            as success (the target itself on verbal report, the replacement's answer on
            intermediate replacement).
    """

    index: int | None
    trial: Trial
    label: str
    prompt: str
    source: str
    target: str
    answer: str
    clean_answer: str

    @property
    def task_label(self) -> str:
        return {
            "verbal-report": "verbal report",
            "probe-swap": "intermediate replacement",
        }.get(self.trial.task, "custom")


@dataclass
class EditContext:
    """What every edit needs and nothing a widget changes: maps, null, trial list."""

    session: LensSession
    maps: dict[str, dict[int, torch.Tensor]]
    ref_maps: dict[int, torch.Tensor]
    null_rows: torch.Tensor
    band_layers: list[int]
    trials: list[TrialInfo] = field(default_factory=list)

    @property
    def custom_seed_index(self) -> int:
        """Seed index for a custom trial: one past the paper's, so it aliases none."""
        return len(self.trials)


def build_context(session: LensSession) -> EditContext:
    """Stage both lenses' maps and the null, and build the paper's 185 trials. ~10 s."""
    model, spec = session.model, session.spec
    device = model.input_device
    lo, hi = WORKSPACE_DEPTH
    null_ids = torch.randperm(
        int(text_config(model).vocab_size), generator=torch.Generator().manual_seed(NULL_SEED)
    )[:N_NULL_CONCEPTS]
    context = EditContext(
        session=session,
        maps={
            "J": {l: session.jz_by_layer[l].float() for l in session.layers},
            "L": {l: session.w_o_by_layer[l].float() for l in session.layers},
        },
        ref_maps={l: session.published.jacobians[l].to(device) for l in session.ref_layers},
        null_rows=effective_unembed(model, null_ids.to(device)),
        band_layers=[l for l in session.ref_layers if lo <= l / (spec.n_layers - 1) <= hi],
    )
    trials = build_trials(model, max_seq_len=MAX_SEQ_LEN)
    context.trials = [_describe(session, i, t) for i, t in enumerate(trials)]
    return context


def _describe(session: LensSession, index: int, trial: Trial) -> TrialInfo:
    tokenizer = session.tokenizer
    if trial.task == "probe-swap":
        item = next(
            i for i in load_probe_swap() if f"probe-swap:{i.name}" == trial.prompt_key
        )
        return TrialInfo(
            index=index,
            trial=trial,
            label=f"intermediate replacement · {item.name}: {item.intermediate} → "
            f"{item.swap_to} ({item.answer} → {item.swap_answer})",
            prompt=item.prompt,
            source=item.intermediate,
            target=item.swap_to,
            answer=item.swap_answer,
            clean_answer=item.answer,
        )
    category = trial.prompt_key.removeprefix("verbal-report:")
    word = next(
        w
        for w in load_verbal_report()[category]
        if target_direction_id(tokenizer, w) == trial.target_id
        and tuple(target_token_ids(tokenizer, w)) == trial.success_ids
    )
    return TrialInfo(
        index=index,
        trial=trial,
        label=f"verbal report · {category}: {trial.answer!r} → {word}",
        prompt=f"Think of a {category}. Answer in one word:",
        source=trial.answer,
        target=word,
        answer=word,
        clean_answer=trial.answer,
    )


def custom_trial(
    session: LensSession, prompt: str, source: str, target: str, answer: str
) -> TrialInfo:
    """A trial outside the paper's sets, built with the paper's conventions.

    The swap runs along each word's space-prefixed first token
    (:func:`src.readout.target_direction_id`), and success accepts the first token of
    the answer spelled with or without a leading space -- Section 4.1. A blank source
    means "the model's own greedy next token", as on verbal report.
    """
    model, tokenizer = session.model, session.tokenizer
    input_ids = model.encode(prompt, max_length=MAX_SEQ_LEN)
    with torch.no_grad():
        greedy = int(model.unembed(final_residual(model, input_ids)[0, -1]).float().argmax())
    source_id = target_direction_id(tokenizer, source) if source.strip() else greedy
    trial = Trial(
        task="custom",
        prompt_key="custom",
        input_ids=input_ids,
        source_id=source_id,
        target_id=target_direction_id(tokenizer, target),
        success_ids=tuple(target_token_ids(tokenizer, answer or target)),
    )
    return TrialInfo(
        index=None,
        trial=trial,
        label="custom",
        prompt=prompt,
        source=source.strip() or session.decode(greedy),
        target=target,
        answer=answer or target,
        clean_answer=session.decode(greedy),
    )


# ------------------------------------------------------------------ one trial's clean pass


@dataclass(frozen=True)
class Prepared:
    """One trial's clean pass, site selection, frames and clean coordinates.

    Attributes:
        loading: ``{lens: [n_layers, n_heads]}`` source loading ``cos(z, v_s)`` at the
            final position, each lens through its own map.
        tau: ``{lens: [n_layers, n_heads]}`` the per-site 99th percentile of the same
            cosine over 256 random tokens; a site above it is *active*.
        sites: ``{lens: {condition: [(layer, head), ...]}}`` from
            :func:`scripts.write.site_sets` -- the paper's sets for this trial.
    """

    info: TrialInfo
    seed_index: int
    layers: list[int]
    clean_logits: torch.Tensor
    z_full: dict[int, torch.Tensor]
    out_full: dict[int, torch.Tensor]
    loading: dict[str, torch.Tensor]
    tau: dict[str, torch.Tensor]
    sites: dict[str, dict[str, list[tuple[int, int]]]]
    heads: dict
    wide: dict
    head_coords: dict
    wide_coords: dict

    @property
    def trial(self) -> Trial:
        return self.info.trial


@torch.no_grad()
def prepare(context: EditContext, info: TrialInfo) -> Prepared:
    """The clean pass and everything chosen from it, as the driver does per trial."""
    session = context.session
    model, spec, device = session.model, session.spec, session.model.input_device
    layers, ref_layers = session.layers, session.ref_layers
    trial = info.trial
    seed_index = info.index if info.index is not None else context.custom_seed_index

    with AnchorRecorder(model.layers, zin_at=layers, out_at=ref_layers) as rec:
        clean_logits = model.unembed(final_residual(model, trial.input_ids)[0, -1]).float()
        z_full = {l: rec.activations["zin", l][0].clone() for l in layers}
        out_full = {l: rec.activations["out", l][0].clone() for l in ref_layers}
    z_by_layer = {l: z[-1] for l, z in z_full.items()}
    source_row = effective_unembed(model, torch.tensor([trial.source_id], device=device))[0]
    loading, tau = {}, {}
    for lens in LENSES:
        loading[lens] = loading_map(context.maps[lens], z_by_layer, spec, source_row, layers).cpu()
        tau[lens] = null_threshold(
            loading_map(context.maps[lens], z_by_layer, spec, context.null_rows, layers),
            q=NULL_QUANTILE,
        ).cpu()
    # Same seed for both lenses, so every random set is shared.
    sites = {
        lens: site_sets(loading[lens], layers, tau[lens], seed=RANDOM_SITE_SEED + seed_index)
        for lens in LENSES
    }

    rows = effective_unembed(model, torch.tensor([trial.source_id, trial.target_id], device=device))
    heads = {}
    for lens in LENSES:
        for layer in layers:
            frames = head_frames(context.maps[lens][layer], spec, rows)
            heads[lens, layer] = (frames, swap_pseudoinverse(frames))
    wide = {}
    for layer in ref_layers:
        for lens, frame in (
            (
                "J",
                swap_frame(
                    lens_vector(context.ref_maps[layer], rows[0]),
                    lens_vector(context.ref_maps[layer], rows[1]),
                ),
            ),
            ("L", swap_frame(rows[0], rows[1])),
        ):
            wide[lens, layer] = (frame, swap_pseudoinverse(frame))

    head_coords = {}
    for (lens, layer), (_, pseudo) in heads.items():
        z = z_full[layer].to(COMPUTE_DTYPE).view(-1, spec.n_heads, spec.head_dim)
        head_coords[lens, layer] = torch.einsum("shd,hkd->hsk", z, pseudo.to(COMPUTE_DTYPE))
    wide_coords = {
        key: swap_coefficients(out_full[key[1]], pseudo) for key, (_, pseudo) in wide.items()
    }
    return Prepared(
        info=info,
        seed_index=seed_index,
        layers=layers,
        clean_logits=clean_logits,
        z_full=z_full,
        out_full=out_full,
        loading=loading,
        tau=tau,
        sites=sites,
        heads=heads,
        wide=wide,
        head_coords=head_coords,
        wide_coords=wide_coords,
    )


# ------------------------------------------------------------------ the two edits


def _unit_noise(shape, seed, device):
    """What :func:`src.swap.matched_norm_delta` draws for this seed, normalised."""
    noise = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed))
    return (noise / noise.norm(dim=-1, keepdim=True).clamp(min=1e-12)).to(device, COMPUTE_DTYPE)


@torch.no_grad()
def run_heads(
    context: EditContext, prep: Prepared, elements: list[tuple[list, str]], alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edit each element's sites with its arm's frame; final-position logits and norms.

    Args:
        elements: ``[(sites, arm)]`` with ``arm`` in :data:`scripts.write_swap_sites.ARMS`:
            ``J``/``L`` swap through that lens, ``J_rand``/``L_rand`` add a random
            direction of that swap's norm at every position of every site.

    Returns:
        ``[n_elements, vocab]`` logits and ``[n_elements]`` total update norms ``D``
        (Section S3.3).
    """
    session = context.session
    model, spec, device = session.model, session.spec, session.model.input_device
    totals = torch.zeros(len(elements))
    plan = plan_edits(elements)
    noise_cache = {}

    def make_edit(layer):
        def delta_for(arm, head, block, _index):
            frame, pseudo = prep.heads[arm[0], layer]
            delta = clamped_swap_delta(
                block, frame[head], prep.head_coords[arm[0], layer][head],
                alpha=alpha, pseudo=pseudo[head],
            )
            if arm.endswith("_rand"):
                key = (layer, head)
                if key not in noise_cache:
                    noise_cache[key] = _unit_noise(
                        block.shape[1:], prep.seed_index * 100003 + layer * 100 + head, device
                    )
                delta = noise_cache[key] * delta.norm(dim=-1, keepdim=True)
            return delta

        return lambda z: apply_site_deltas(z, spec, plan.get(layer, {}), delta_for, totals=totals)

    batched = prep.trial.input_ids.expand(len(elements), -1)
    with ZinPatcher(model.layers, {l: make_edit(l) for l in session.layers}):
        hidden = final_residual(model, batched)[:, -1]
    return model.unembed(hidden).float(), totals.sqrt()


@torch.no_grad()
def run_layer_wide(
    context: EditContext, prep: Prepared, alpha: float, edit_layers: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The residual-stream reference: the same clamped swap at every position of every
    block output in ``edit_layers``, one batch row per arm of
    :data:`scripts.write_swap_sites.ARMS` (Jacobian lens, bare unembedding, and each
    one's random direction)."""
    session = context.session
    model, device = session.model, session.model.input_device
    totals = torch.zeros(len(ARMS))

    def make_edit(layer):
        def edit(x):
            out = []
            for row, arm in enumerate(ARMS):
                frame, pseudo = prep.wide[arm[0], layer]
                delta = clamped_swap_delta(
                    x[row], frame, prep.wide_coords[arm[0], layer], alpha=alpha, pseudo=pseudo
                )
                if arm.endswith("_rand"):
                    delta = _unit_noise(
                        delta.shape, prep.seed_index * 100003 + layer, device
                    ) * delta.norm(dim=-1, keepdim=True)
                totals[row] += float(delta.float().pow(2).sum())
                out.append((x[row].to(COMPUTE_DTYPE) + delta).to(x.dtype))
            return torch.stack(out)

        return edit

    batched = prep.trial.input_ids.expand(len(ARMS), -1)
    with ResidualPatcher(model.layers, {l: make_edit(l) for l in edit_layers}):
        hidden = final_residual(model, batched)[:, -1]
    return model.unembed(hidden).float(), totals.sqrt()


def metrics(logits: torch.Tensor, clean_logits: torch.Tensor, trial: Trial) -> np.ndarray:
    """``[n, 5]``: success, P(answer), KL(clean || edited), greedy id, answer's rank (0-based)."""
    logp = logits.log_softmax(-1)
    clean = clean_logits.log_softmax(-1)
    top = logp.argmax(-1)
    return np.stack(
        [
            np.isin(top.cpu().numpy(), trial.success_ids).astype(np.float32),
            logp[:, list(trial.success_ids)].exp().sum(-1).cpu().numpy(),
            (clean.exp() * (clean - logp)).sum(-1).cpu().numpy(),
            top.cpu().numpy().astype(np.float32),
            target_rank(logits, trial.success_ids).cpu().numpy(),
        ],
        -1,
    )


# ------------------------------------------------------------------ the paper's protocol


def paper_protocol(context: EditContext, prep: Prepared) -> dict[str, np.ndarray]:
    """Every cell the driver records for this trial, computed as the driver computes it.

    Returns:
        ``{"heads": [alpha, condition, selection lens, arm, metric],
        "layer_wide": [alpha, arm, metric], "layer_wide_band": [alpha, arm, metric]}``,
        the record's own layout (:func:`explorer.records.recorded_trial`).
    """
    session = context.session
    heads = np.full(
        (len(ALPHAS), len(ZIN_CONDITIONS), len(LENSES), len(ARMS), len(SWAP_METRICS)),
        np.nan,
        dtype=np.float32,
    )
    wide = np.full((len(ALPHAS), len(ARMS), len(SWAP_METRICS)), np.nan, dtype=np.float32)
    band = np.full_like(wide, np.nan)
    elements = [
        (prep.sites[LENSES[si]][ZIN_CONDITIONS[ci]], ARMS[ai]) for ci, si, ai in ELEMENTS_INDEX
    ]
    clean = prep.clean_logits[None]
    for xi, alpha in enumerate(ALPHAS):
        for start in range(0, len(elements), DRIVER_BATCH):
            chunk = elements[start : start + DRIVER_BATCH]
            logits, norms = run_heads(context, prep, chunk, alpha)
            values = metrics(logits, clean, prep.trial)
            for offset, (ci, si, ai) in enumerate(ELEMENTS_INDEX[start : start + len(chunk)]):
                heads[xi, ci, si, ai, OUTPUT_COLUMNS] = values[offset]
                heads[xi, ci, si, ai, NORM_COLUMN] = norms[offset]
        for out, edit_layers in ((wide, session.ref_layers), (band, context.band_layers)):
            logits, norms = run_layer_wide(context, prep, alpha, edit_layers)
            out[xi][:, OUTPUT_COLUMNS] = metrics(logits, clean, prep.trial)
            out[xi][:, NORM_COLUMN] = norms.numpy()
    return {"heads": heads, "layer_wide": wide, "layer_wide_band": band}


def compare_to_record(live: dict[str, np.ndarray], recorded: dict | None) -> dict:
    """How the live re-run of a paper trial compares with the recorded one.

    Greedy outcomes are compared exactly, the continuous metrics to a relative 1e-3.
    Bit-identical is expected on the hardware the record was made on (A100, same batch);
    elsewhere bf16 kernels may differ, which the paper bounds at 16 of 7,030 J-head
    outcomes between two batch layouts (Section S3.5).
    """
    if recorded is None:
        return {"status": "n/a", "detail": "this trial is not in the record"}
    success, top_id = SWAP_METRICS.index("success"), SWAP_METRICS.index("top_id")
    cells = flips = close = identical = 0
    for key in ("heads", "layer_wide", "layer_wide_band"):
        a, b = live[key], recorded[key]
        a2, b2 = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        cells += a2.shape[0]
        flips += int((a2[:, success] != b2[:, success]).sum())
        identical += int((a2 == b2).all(axis=-1).sum())
        same_top = a2[:, top_id] == b2[:, top_id]
        near = np.isclose(a2, b2, rtol=1e-3, atol=1e-6).all(axis=-1)
        close += int((same_top & near).sum())
    status = "identical" if identical == cells else ("match" if flips == 0 else "differs")
    return {
        "status": status,
        "cells": cells,
        "identical": identical,
        "close": close,
        "success_flips": flips,
        "detail": f"{identical:,} of {cells:,} recorded cells bit-identical, "
        f"{close:,} within 1e-3; {flips} success outcome(s) differ",
    }


# ------------------------------------------------------------------ a custom edit


def choose_sites(
    prep: Prepared, lens: str, rule: str, count: int, manual: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """The sites one lens edits under ``rule``, by :func:`src.sites.select_sites`.

    ``top``/``bottom``/``active`` use the lens's own source loading, ``random`` the
    trial's shared seed, ``manual`` the clicked sites (shared by both lenses). At the
    paper's counts these are exactly :func:`scripts.write.site_sets`' sets.
    """
    layers = prep.layers
    if rule == "manual":
        return sorted(set(manual))
    if rule == "active":
        return select_sites(prep.loading[lens], layers, "threshold", tau=prep.tau[lens])
    if rule == "random":
        return select_sites(
            prep.loading[lens], layers, "random", k=count, seed=RANDOM_SITE_SEED + prep.seed_index
        )
    return select_sites(prep.loading[lens], layers, rule, k=count)


def custom_edit(
    context: EditContext,
    prep: Prepared,
    sites: dict[str, list[tuple[int, int]]],
    alpha: float,
    *,
    reference: bool = True,
    top_n: int = 5,
) -> list[dict]:
    """Swap at ``sites[lens]`` through each lens, with each swap's random-direction
    control, plus (optionally) the residual-stream reference over the workspace band.

    Returns:
        One row per edit: what was edited, the update norm and the outcome.
    """
    session = context.session
    elements = [
        (sites["J"], "J"),
        (sites["J"], "J_rand"),
        (sites["L"], "L"),
        (sites["L"], "L_rand"),
    ]
    names = [
        ("J-head swap", "J"),
        ("J-head, random direction", "J"),
        ("Logit-head swap", "L"),
        ("Logit-head, random direction", "L"),
    ]
    clean = prep.clean_logits[None]
    rows = []
    if any(s for s, _ in elements):
        logits, norms = run_heads(context, prep, elements, alpha)
        values = metrics(logits, clean, prep.trial)
        for (name, lens), (edit_sites, _), v, n, row_logits in zip(
            names, elements, values, norms, logits, strict=True
        ):
            rows.append(
                _row(session, name, lens, len(edit_sites),
                     len({l for l, _ in edit_sites}), float(n), v, row_logits, top_n)
            )
    if reference:
        logits, norms = run_layer_wide(context, prep, alpha, context.band_layers)
        values = metrics(logits, clean, prep.trial)
        band = f"{context.band_layers[0]}–{context.band_layers[-1]}"
        for arm, v, n, row_logits in zip(ARMS, values, norms, logits, strict=True):
            name = {
                "J": f"layer-wide, Jacobian lens ({band})",
                "L": f"layer-wide, unembedding ({band})",
                "J_rand": "layer-wide, Jacobian lens, random direction",
                "L_rand": "layer-wide, unembedding, random direction",
            }[arm]
            rows.append(
                _row(session, name, arm[0], None, len(context.band_layers), float(n),
                     v, row_logits, top_n)
            )
    return rows


def clean_row(context: EditContext, prep: Prepared, top_n: int = 5) -> dict:
    """The unedited pass, in the same shape as :func:`custom_edit`'s rows."""
    values = metrics(prep.clean_logits[None], prep.clean_logits[None], prep.trial)[0]
    return _row(context.session, "clean (no edit)", None, 0, 0, 0.0, values,
                prep.clean_logits, top_n)


def _row(session, name, lens, n_sites, n_layers, norm, values, logits, top_n) -> dict:
    probs = logits.float().softmax(-1)
    p, ids = probs.topk(top_n)
    success, probability, kl, top_id, rank = (float(x) for x in values)
    return {
        "edit": name,
        "lens": lens,
        "sites": n_sites,
        "layers": n_layers,
        "update norm": norm,
        "next token": session.decode(int(top_id)),
        "success": bool(success),
        "P(answer)": probability,
        "answer rank": int(rank) + 1,
        "KL from clean": kl,
        "top tokens": [(session.decode(int(i)), float(q)) for i, q in zip(ids, p, strict=True)],
    }

