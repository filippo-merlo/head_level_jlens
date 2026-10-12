"""Does a head's source loading predict how well the source reads there?

    python -m scripts.loading_rank --gpu 0 [--lens <key>]

Experiment 2 edits the sites with the highest source loading; Experiment 1 reads a concept
at a site by its rank in that site's readout. This driver puts the two side by side on the
swap trials' prompts, at every head site, under each readout, and asks whether the sites
Experiment 2 edits are the sites where the source reads best (:mod:`src.loading_rank`).

Both quantities are computed exactly as the experiments compute them, at the final
position the swap selects at: loading and its active-site cut as
:mod:`scripts.write_swap_sites` selects by (J-head through ``J^z``, Logit-head through the
residual ``W_O``), the rank as :mod:`scripts.read` scores a head (the transported write
through ``model.unembed``), for the single token the swap moves. Trials sharing a prompt
share both maps, so the unit is the prompt. The active counts and the J/Logit top-set
overlaps are checked against the swap record, which is the check that the loading maps are
the ones the swaps were selected by.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

from scripts._gpu import pin_gpu

READOUTS = ("J", "L")
#: The read.py arm each readout scores a head with.
ARMS = {"J": "jhead", "L": "logithead"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--lens", default=None, help="LensSpec key (default: DEFAULT_LENS)")
    parser.add_argument("--swap-record", default="write_swap_sites_clamp",
                        help="results/ subdirectory whose trials the loading maps are checked against")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pin_gpu(args.gpu)
    import numpy as np
    import torch

    from scripts.write import N_NULL_CONCEPTS, NULL_QUANTILE, NULL_SEED
    from src.config import (
        DEFAULT_LENS,
        LENSES,
        MAX_SEQ_LEN,
        PROJECT_ROOT,
        Paths,
        jz_path,
    )
    from src.heads import effective_unembed, residual_o_proj_weight
    from src.hooks import AnchorRecorder
    from src.loading_rank import SITE_COUNTS, summarise, top_mask
    from src.model import final_residual, load_jz, load_model, text_config
    from src.provenance import file_digest, meta_block
    from src.readout import all_heads_transported, token_ranks
    from src.sites import loading_map, null_threshold
    from src.tasks import build_trials

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.monotonic()
    spec = LENSES[args.lens or DEFAULT_LENS]
    paths = Paths.default()
    out_dir = paths.run_dir("loading_rank")
    layers = list(spec.jz_source_layers)

    model = load_model(spec)
    device = model.input_device
    jz_file = jz_path(spec, n_prompts=spec.jz_n_prompts, paths=paths)
    jz = load_jz(spec, jz_file)
    maps = {
        "J": {l: jz.jacobians[l].to(device).float() for l in layers},
        "L": {l: residual_o_proj_weight(model, l).float() for l in layers},
    }
    vocab_size = int(text_config(model).vocab_size)
    null_ids = torch.randperm(vocab_size,
                              generator=torch.Generator().manual_seed(NULL_SEED))[:N_NULL_CONCEPTS]
    null_rows = effective_unembed(model, null_ids.to(device))

    prompts = {}
    for index, trial in enumerate(build_trials(model, max_seq_len=MAX_SEQ_LEN)):
        prompts.setdefault(trial.prompt_key, (trial, []))[1].append(index)

    shape = (len(prompts), len(READOUTS), len(layers), spec.n_heads)
    loading_out = np.zeros(shape, dtype=np.float32)
    tau_out = np.zeros(shape, dtype=np.float32)
    rank_out = np.zeros(shape, dtype=np.int64)
    for p, (trial, _) in enumerate(prompts.values()):
        with torch.no_grad(), AnchorRecorder(model.layers, zin_at=layers, out_at=[]) as rec:
            final_residual(model, trial.input_ids)
            z_by_layer = {l: rec.activations["zin", l][0, -1].clone() for l in layers}
        source_row = effective_unembed(model, torch.tensor([trial.source_id], device=device))[0]
        for r, readout in enumerate(READOUTS):
            loading_out[p, r] = loading_map(maps[readout], z_by_layer, spec, source_row, layers).cpu()
            tau_out[p, r] = null_threshold(
                loading_map(maps[readout], z_by_layer, spec, null_rows, layers), q=NULL_QUANTILE
            ).cpu()
            with torch.no_grad():
                for li, layer in enumerate(layers):
                    logits = model.unembed(all_heads_transported(
                        maps["J"][layer], z_by_layer[layer], spec, arm=ARMS[readout],
                        w_o=maps["L"][layer],
                    )).float()
                    rank_out[p, r, li] = [int(token_ranks(logits[h], [trial.source_id])[0])
                                          for h in range(spec.n_heads)]
        logging.info("prompt %d/%d (%s) done", p + 1, len(prompts), trial.prompt_key)

    # The check: the maps reproduce the swap record's active counts and J/Logit top overlaps.
    recorded = {t["trial_index"]: t
                for f in sorted((paths.results_root / args.swap_record).rglob(f"{spec.key}_shard*.json"))
                for t in json.loads(f.read_text())["trials"]}
    mismatches = []
    for p, (key, (_, indices)) in enumerate(prompts.items()):
        active = loading_out[p] > tau_out[p]
        overlap = {f"top_{m}": int((top_mask(loading_out[p, 0], m) & top_mask(loading_out[p, 1], m)).sum())
                   for m in SITE_COUNTS} | {"active": int((active[0] & active[1]).sum())}
        n_active = {readout: int(active[r].sum()) for r, readout in enumerate(READOUTS)}
        for i in indices:
            if i in recorded and (recorded[i]["n_active"] != n_active
                                  or recorded[i]["top_overlap"] != overlap):
                mismatches.append(key)
    checked = sum(i in recorded for _, indices in prompts.values() for i in indices)

    groups = [trial.task for trial, _ in prompts.values()]
    np.savez_compressed(out_dir / f"{spec.key}.npz", loading=loading_out, rank=rank_out, tau=tau_out)
    record = {
        "meta": meta_block(stage="loading_rank", model=spec.hf_model_id, jz_path=jz_file,
                           jz_sha256=file_digest(jz_file), gpu=torch.cuda.get_device_name(),
                           seconds=time.monotonic() - started),
        "configuration": {
            "readouts": {"J": "J-head: J^z head blocks", "L": "Logit-head: residual W_O head blocks"},
            "layers": layers, "position": "final", "token": "the swap's source_id",
            "null": {"n_tokens": N_NULL_CONCEPTS, "seed": NULL_SEED, "quantile": NULL_QUANTILE},
            "rank": "0-based count of vocabulary tokens scoring strictly higher in the readout",
            "code_sha256": {f: file_digest(PROJECT_ROOT / f) for f in
                            ["scripts/loading_rank.py", "src/loading_rank.py", "src/sites.py",
                             "src/readout.py", "src/heads.py", "src/tasks.py"]},
        },
        "axes": {"loading": ["prompt", "readout", "layer", "head"],
                 "rank": ["prompt", "readout", "layer", "head"],
                 "tau": ["prompt", "readout", "layer", "head"]},
        "prompts": [{"prompt_key": key, "task": trial.task, "source_id": trial.source_id,
                     "source": model.tokenizer.decode([trial.source_id]), "trial_indices": indices}
                    for key, (trial, indices) in prompts.items()],
        "check_against_swap_record": {"record": args.swap_record, "trials_checked": checked,
                                      "prompts_mismatched": sorted(set(mismatches))},
        "summary": summarise(loading_out, rank_out, tau_out, groups, READOUTS),
        "arrays_sha256": file_digest(out_dir / f"{spec.key}.npz"),
    }
    (out_dir / f"{spec.key}.json").write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    logging.info("checked %d trials against %s: %d prompts differ", checked, args.swap_record,
                 len(set(mismatches)))
    logging.info("wrote %s in %.0fs", out_dir / spec.key, time.monotonic() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
