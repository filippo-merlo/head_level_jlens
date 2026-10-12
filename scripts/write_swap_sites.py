"""Experiment 2's coordinate swap, J-head vs Logit-head.

    python -m scripts.write_swap_sites --gpu 0 --shard 0 --n-shards 3 [--lens <key>]

:mod:`scripts.write` swaps through ``J^z`` only. Here the same swap, trials, counts, rules,
alphas and scope run through two lenses: a head's frame is ``M_h^T [q_s q_t]`` with
``M = J^z`` (J-head) or ``M = W_O`` (Logit-head: the head's direct write, no transport).
Selection and direction are crossed: each lens picks sites by its own source loading (same
per-site 99% null), both lenses' swaps run at both lenses' site sets, random sets are
shared, and each swap has a same-norm random-direction control seeded exactly as :mod:`scripts.write` seeds it. The J-selected
J-head cells therefore repeat :mod:`scripts.write`'s run, which is the check on this driver.

The swap's size is set by the activation's own coordinates in each lens's frame, so the two
lenses edit the same sites by different amounts. Every cell records its total edit norm
rather than equalising it away.

``layer_wide`` swaps in the residual stream at every block output the Jacobian lens covers
(0-30 on Llama) through that lens (J, :mod:`scripts.write`'s reference) or the bare
gain-weighted unembedding (Logit), each with its random control. ``layer_wide_band``
repeats it over the paper's workspace band only (:data:`WORKSPACE_DEPTH`, block outputs
12-28 on Llama).

``--coords clamp`` (the default) sets every edited site's coordinates to the swapped
clean-pass ones (:func:`src.swap.clamped_swap_delta`), as the paper clamps its swap.
``--coords recompute`` swaps the coordinates read from the edited pass at each site, which
is how runs before 2026-09-28 were made; across residual layers it undoes itself at
``alpha=1`` and overflows at ``alpha=2`` (see :mod:`src.swap`).
"""
from __future__ import annotations

import argparse
import json
import logging
import time

from scripts._gpu import pin_gpu
from scripts.write import ALPHAS

#: The paper's workspace layer range (L38-92 on its 0-100 depth scale), mapped onto block
#: outputs by ``layer / (n_layers - 1)``. The release defines no band for Llama.
WORKSPACE_DEPTH = (0.38, 0.92)
LENSES = ("J", "L")
ARMS = ("J", "L", "J_rand", "L_rand")
#: Per-trial outputs, with the target's rank, which Gurnee et al. grade by (top-5 on verbal report).
SWAP_METRICS = ("success", "probability", "kl", "top_id", "edit_norm", "rank")
#: Where :func:`metrics`' columns go in the metric axis; ``edit_norm`` is filled separately.
OUTPUT_COLUMNS = [SWAP_METRICS.index(m) for m in ("success", "probability", "kl", "top_id", "rank")]
NORM_COLUMN = SWAP_METRICS.index("edit_norm")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--lens", default=None, help="LensSpec key (default: DEFAULT_LENS)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--max-categories", type=int)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--out-name", default="write_swap_sites_clamp")
    parser.add_argument("--coords", choices=("clamp", "recompute"), default="clamp",
                        help="clamp coordinates to the swapped clean pass, or swap the edited pass's")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0 <= args.shard < args.n_shards:
        raise ValueError("shard out of range")
    pin_gpu(args.gpu)
    import numpy as np
    import torch

    from scripts.write import (
        N_NULL_CONCEPTS,
        NULL_QUANTILE,
        NULL_SEED,
        RANDOM_SITE_SEED,
        SELECTED_CONDITIONS,
        ZIN_CONDITIONS,
        site_sets,
    )
    from src.config import DEFAULT_LENS, MAX_SEQ_LEN, PROJECT_ROOT, Paths, jz_path
    from src.config import LENSES as SPECS
    from src.heads import COMPUTE_DTYPE, effective_unembed, residual_o_proj_weight
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
        swap_coefficients,
        swap_delta,
        swap_frame,
        swap_pseudoinverse,
    )
    from src.tasks import build_trials, target_rank

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.monotonic()
    spec = SPECS[args.lens or DEFAULT_LENS]
    paths = Paths.default()
    out_dir = paths.run_dir(args.out_name)
    name = f"{spec.key}_shard{args.shard}of{args.n_shards}"
    if (out_dir / f"{name}.json").exists():
        raise FileExistsError(f"{out_dir / name}.json exists; pick another --out-name")
    layers, ref_layers = list(spec.jz_source_layers), list(spec.source_layers)
    lo, hi = WORKSPACE_DEPTH
    band_layers = [l for l in ref_layers if lo <= l / (spec.n_layers - 1) <= hi]
    clamp = args.coords == "clamp"
    alphas = list(ALPHAS)

    model = load_model(spec)
    device = model.input_device
    jz_file = jz_path(spec, n_prompts=spec.jz_n_prompts, paths=paths)
    jz = load_jz(spec, jz_file)
    published = load_lens(spec)
    config = text_config(model)
    n_kv_heads = int(config.num_key_value_heads)
    maps = {
        "J": {l: jz.jacobians[l].to(device).float() for l in layers},
        "L": {l: residual_o_proj_weight(model, l).float() for l in layers},
    }
    ref_maps = {l: published.jacobians[l].to(device) for l in ref_layers}

    null_ids = torch.randperm(int(config.vocab_size),
                              generator=torch.Generator().manual_seed(NULL_SEED))[:N_NULL_CONCEPTS]
    null_rows = effective_unembed(model, null_ids.to(device))

    # ------------------------------------------------------------ per-trial frames
    def frames_for(source_id, target_id):
        """Head frames ``{(lens, layer): (frames, pinv)}`` and residual frames ``{(lens, layer): (V, pinv)}``."""
        rows = effective_unembed(model, torch.tensor([source_id, target_id], device=device))
        heads = {}
        for lens in LENSES:
            for layer in layers:
                frames = head_frames(maps[lens][layer], spec, rows)
                heads[lens, layer] = (frames, swap_pseudoinverse(frames))
        wide = {}
        for layer in ref_layers:
            for lens, frame in (("J", swap_frame(lens_vector(ref_maps[layer], rows[0]),
                                                 lens_vector(ref_maps[layer], rows[1]))),
                                ("L", swap_frame(rows[0], rows[1]))):
                wide[lens, layer] = (frame, swap_pseudoinverse(frame))
        return heads, wide

    def clean_coords(heads, wide, z_full, out_full):
        """Clean-pass coordinates: heads ``{(lens, layer): [n_heads, seq, 2]}``, residual ``{(lens, layer): [seq, 2]}``."""
        head_coords = {}
        for (lens, layer), (_, pseudo) in heads.items():
            z = z_full[layer].to(COMPUTE_DTYPE).view(-1, spec.n_heads, spec.head_dim)
            head_coords[lens, layer] = torch.einsum("shd,hkd->hsk", z, pseudo.to(COMPUTE_DTYPE))
        wide_coords = {key: swap_coefficients(out_full[key[1]], pseudo)
                       for key, (_, pseudo) in wide.items()}
        return head_coords, wide_coords

    def unit_noise(shape, seed):
        """What :func:`src.swap.matched_norm_delta` draws for this seed, normalised."""
        noise = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed))
        return (noise / noise.norm(dim=-1, keepdim=True).clamp(min=1e-12)).to(device, COMPUTE_DTYPE)

    def run_heads(input_ids, elements, alpha, frames, coords, trial_index):
        totals = torch.zeros(len(elements))
        plan = plan_edits(elements)
        noise_cache = {}

        def make_edit(layer):
            def delta_for(arm, head, block, _index):
                frame, pseudo = frames[arm[0], layer]
                if clamp:
                    delta = clamped_swap_delta(block, frame[head], coords[arm[0], layer][head],
                                               alpha=alpha, pseudo=pseudo[head])
                else:
                    delta = swap_delta(block, frame[head], alpha=alpha, pseudo=pseudo[head])
                if arm.endswith("_rand"):
                    key = (layer, head)
                    if key not in noise_cache:
                        noise_cache[key] = unit_noise(block.shape[1:],
                                                      trial_index * 100003 + layer * 100 + head)
                    delta = noise_cache[key] * delta.norm(dim=-1, keepdim=True)
                return delta
            return lambda z: apply_site_deltas(z, spec, plan.get(layer, {}), delta_for, totals=totals)

        batched = input_ids.expand(len(elements), -1)
        with torch.no_grad(), ZinPatcher(model.layers, {l: make_edit(l) for l in layers}):
            hidden = final_residual(model, batched)[:, -1]
        return model.unembed(hidden).float(), totals.sqrt()

    def run_layer_wide(input_ids, alpha, frames, coords, trial_index, edit_layers):
        totals = torch.zeros(len(ARMS))

        def make_edit(layer):
            def edit(x):
                out = []
                for row, arm in enumerate(ARMS):
                    frame, pseudo = frames[arm[0], layer]
                    if clamp:
                        delta = clamped_swap_delta(x[row], frame, coords[arm[0], layer],
                                                   alpha=alpha, pseudo=pseudo)
                    else:
                        delta = swap_delta(x[row], frame, alpha=alpha, pseudo=pseudo)
                    if arm.endswith("_rand"):
                        delta = unit_noise(delta.shape, trial_index * 100003 + layer) * \
                            delta.norm(dim=-1, keepdim=True)
                    totals[row] += float(delta.float().pow(2).sum())
                    out.append((x[row].to(COMPUTE_DTYPE) + delta).to(x.dtype))
                return torch.stack(out)
            return edit

        batched = input_ids.expand(len(ARMS), -1)
        with torch.no_grad(), ResidualPatcher(model.layers, {l: make_edit(l) for l in edit_layers}):
            hidden = final_residual(model, batched)[:, -1]
        return model.unembed(hidden).float(), totals.sqrt()

    def metrics(logits, clean_logits, trial):
        logp = logits.log_softmax(-1)
        clean = clean_logits.log_softmax(-1)
        top = logp.argmax(-1)
        return np.stack([
            np.isin(top.cpu().numpy(), trial.success_ids).astype(np.float32),
            logp[:, list(trial.success_ids)].exp().sum(-1).cpu().numpy(),
            (clean.exp() * (clean - logp)).sum(-1).cpu().numpy(),
            top.cpu().numpy().astype(np.float32),
            target_rank(logits, trial.success_ids).cpu().numpy(),
        ], -1)

    # ------------------------------------------------------------ sweep
    all_trials = build_trials(model, max_seq_len=MAX_SEQ_LEN,
                              max_categories=args.max_categories, max_items=args.max_items)
    indexed = list(enumerate(all_trials))[args.shard::args.n_shards]
    heads_out = np.full((len(indexed), len(alphas), len(ZIN_CONDITIONS), len(LENSES), len(ARMS),
                         len(SWAP_METRICS)), np.nan, dtype=np.float32)
    wide_out = np.full((len(indexed), len(alphas), len(ARMS), len(SWAP_METRICS)), np.nan, dtype=np.float32)
    band_out = np.full_like(wide_out, np.nan)
    elements_index = [(ci, si, ai) for ci in range(len(ZIN_CONDITIONS))
                      for si in range(len(LENSES)) for ai in range(len(ARMS))]
    prompt_cache, records = {}, []
    for row, (trial_index, trial) in enumerate(indexed):
        if trial.prompt_key not in prompt_cache:
            with torch.no_grad(), AnchorRecorder(model.layers, zin_at=layers, out_at=ref_layers) as rec:
                clean_logits = model.unembed(final_residual(model, trial.input_ids)[0, -1]).float()
                z_full = {l: rec.activations["zin", l][0].clone() for l in layers}
                out_full = {l: rec.activations["out", l][0].clone() for l in ref_layers}
            z_by_layer = {l: z[-1] for l, z in z_full.items()}
            source_row = effective_unembed(model, torch.tensor([trial.source_id], device=device))[0]
            selection = {}
            for lens in LENSES:
                loading = loading_map(maps[lens], z_by_layer, spec, source_row, layers)
                tau = null_threshold(loading_map(maps[lens], z_by_layer, spec, null_rows, layers),
                                     q=NULL_QUANTILE)
                selection[lens] = (loading.cpu(), tau.cpu())
            prompt_cache[trial.prompt_key] = (clean_logits, selection, z_full, out_full)
        clean_logits, selection, z_full, out_full = prompt_cache[trial.prompt_key]
        # Same seed for both lenses, so every random_k set is shared.
        sites = {lens: site_sets(selection[lens][0], layers, selection[lens][1],
                                 seed=RANDOM_SITE_SEED + trial_index) for lens in LENSES}
        frames_by_head, wide_frames = frames_for(trial.source_id, trial.target_id)
        head_coords, wide_coords = clean_coords(frames_by_head, wide_frames, z_full, out_full)
        elements = [(sites[LENSES[si]][ZIN_CONDITIONS[ci]], ARMS[ai]) for ci, si, ai in elements_index]
        for xi, alpha in enumerate(alphas):
            for start in range(0, len(elements), args.batch_size):
                chunk = elements[start:start + args.batch_size]
                logits, norms = run_heads(trial.input_ids, chunk, alpha, frames_by_head, head_coords,
                                          trial_index)
                values = metrics(logits, clean_logits[None], trial)
                for offset, (ci, si, ai) in enumerate(elements_index[start:start + len(chunk)]):
                    heads_out[row, xi, ci, si, ai, OUTPUT_COLUMNS] = values[offset]
                    heads_out[row, xi, ci, si, ai, NORM_COLUMN] = norms[offset]
            for out, edit_layers in ((wide_out, ref_layers), (band_out, band_layers)):
                logits, norms = run_layer_wide(trial.input_ids, alpha, wide_frames, wide_coords,
                                               trial_index, edit_layers)
                # out[row, xi] is a view; indexing it alone keeps the arm axis first.
                out[row, xi][:, OUTPUT_COLUMNS] = metrics(logits, clean_logits[None], trial)
                out[row, xi][:, NORM_COLUMN] = norms.numpy()
        top_sets = {lens: {c: sites[lens][c] for c in SELECTED_CONDITIONS} for lens in LENSES}
        records.append({
            "trial_index": trial_index, "task": trial.task, "prompt_key": trial.prompt_key,
            "source_id": trial.source_id, "target_id": trial.target_id,
            "success_ids": list(trial.success_ids), "answer_in_class": trial.answer_in_class,
            "clean_success": int(clean_logits.argmax()) in trial.success_ids,
            "clean_rank": float(target_rank(clean_logits, trial.success_ids)),
            "n_active": {lens: len(sites[lens]["active"]) for lens in LENSES},
            "top_overlap": {c: len(set(top_sets["J"][c]) & set(top_sets["L"][c])) for c in SELECTED_CONDITIONS},
            "spread": {lens: {c: site_spread(top_sets[lens][c], n_heads=spec.n_heads, n_kv_heads=n_kv_heads)
                              for c in SELECTED_CONDITIONS} for lens in LENSES},
        })
        logging.info("trial %d/%d (%s) done", row + 1, len(indexed), trial.task)

    np.savez_compressed(out_dir / f"{name}.npz", heads=heads_out, layer_wide=wide_out,
                        layer_wide_band=band_out)
    code = ["scripts/write_swap_sites.py", "scripts/write.py", "src/swap.py", "src/sites.py",
            "src/hooks.py", "src/heads.py", "src/tasks.py", "src/model.py"]
    record = {
        "meta": meta_block(stage="write_swap_sites", model=spec.hf_model_id,
                           jz_path=jz_file, jz_sha256=file_digest(jz_file),
                           gpu=torch.cuda.get_device_name(), seconds=time.monotonic() - started),
        "configuration": {
            "strength": "alpha", "strengths": alphas, "shard": args.shard, "n_shards": args.n_shards,
            "lenses": {"J": "J^z head blocks / block Jacobian lens", "L": "W_O head blocks / identity"},
            "arms": list(ARMS), "conditions": list(ZIN_CONDITIONS), "metrics": list(SWAP_METRICS),
            "layers": layers, "reference_layers": ref_layers, "band_layers": band_layers,
            "workspace_depth": list(WORKSPACE_DEPTH), "scope": "every position",
            "coords": args.coords,
            "edit": ("clamped two-frame coordinate swap, V^+ x <- c + alpha (sigma(c) - c) with c "
                     "from the clean pass" if clamp else
                     "additive two-frame coordinate swap, x += alpha (c_t - c_s)(v_s - v_t)"),
            "n_trials_total": len(all_trials),
            "code_sha256": {f: file_digest(PROJECT_ROOT / f) for f in code},
        },
        "axes": {"heads": ["trial", "alpha", "condition", "selection_lens", "arm", "metric"],
                 "layer_wide": ["trial", "alpha", "arm", "metric"],
                 "layer_wide_band": ["trial", "alpha", "arm", "metric"]},
        "trials": records,
        "arrays_sha256": file_digest(out_dir / f"{name}.npz"),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    logging.info("wrote %s in %.0fs", out_dir / name, time.monotonic() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
