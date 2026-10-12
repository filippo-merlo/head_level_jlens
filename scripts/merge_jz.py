"""[1b] Merge fixed-budget ``J^z`` shards into the lens the experiments read.

    python -m scripts.merge_jz --shards a0 a1 a2 a3

No GPU, seconds. :meth:`jlens.lens.JacobianLens.merge` is already ``n_prompts``-weighted,
so unequal shards merge correctly -- but it does **not** check that the shards saw
disjoint prompts, and merging overlapping slices would double-count silently. That check
is this script's real job, and it is decidable only because every fit records the corpus
slice it ran on (:class:`src.fit.CorpusSlice`).

The sidecar it writes carries the merged lens's sha256, which is what
:func:`src.model.load_jz` accepts for a checkpoint whose digest is not yet recorded in
:class:`src.config.LensSpec` -- the merge is the step that vouched for every shard's bytes.
"""

from __future__ import annotations

import argparse
import json
import logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    from src.config import JZ_FIT_STAGE

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lens", default=None, help="LensSpec key (default: DEFAULT_LENS)")
    parser.add_argument("--shards", nargs="+", required=True, help="shard names to merge")
    parser.add_argument("--out-name", default=JZ_FIT_STAGE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from jlens import JacobianLens

    from src.config import DEFAULT_LENS, LENSES, Paths
    from src.model import load_jz
    from src.provenance import file_digest, meta_block

    spec = LENSES[args.lens or DEFAULT_LENS]
    paths = Paths.default()
    stage = paths.run_dir(args.out_name)

    lenses, records, seen = [], {}, {}
    for shard in args.shards:
        sidecar = json.loads((stage / f"{spec.key}_{shard}_fit.json").read_text())
        meta, result = sidecar["meta"], sidecar["result"]
        offsets = range(
            meta["prompt_offset"], meta["prompt_offset"] + meta["n_prompts_requested"]
        )
        for index in offsets:
            if index in seen:
                raise ValueError(
                    f"shards {seen[index]!r} and {shard!r} both cover corpus index "
                    f"{index}; merging them would double-count it"
                )
            seen[index] = shard
        path = result["lens"]
        digest = file_digest(path)
        if digest != result["lens_sha256"]:
            raise ValueError(
                f"{path} has sha256 {digest}, but shard {shard!r}'s sidecar recorded "
                f"{result['lens_sha256']}; the file changed after it was fitted"
            )
        lenses.append(load_jz(spec, path, verify_digest=False))
        records[shard] = {
            "corpus": (meta["corpus_size"], meta["corpus_seed"]),
            "prompt_offset": meta["prompt_offset"],
            "prompts_fitted": result["prompts_fitted"],
            "lens": path,
            "lens_sha256": digest,
        }

    # Disjoint indices only mean disjoint *prompts* if the shards indexed the same draw.
    draws = {r["corpus"] for r in records.values()}
    if len(draws) > 1:
        raise ValueError(f"shards were drawn from different corpora: {sorted(draws)}")

    merged = JacobianLens.merge(lenses)
    destination = paths.jz_fit_lens(spec, n_prompts=merged.n_prompts)
    merged.save(str(destination))

    payload = {
        "meta": meta_block(
            model=spec.hf_model_id,
            lens_key=spec.key,
            shards=args.shards,
            merge="n_prompts-weighted mean over disjoint corpus slices",
        ),
        "shards": records,
        "result": {
            "prompts_fitted": merged.n_prompts,
            "n_layers": len(merged.source_layers),
            "lens": str(destination),
            "lens_sha256": file_digest(destination),
        },
    }
    out_path = stage / f"{spec.key}_merge.json"
    out_path.write_text(json.dumps(payload, indent=1))

    print(f"merged {len(lenses)} shards ({merged.n_prompts} prompts) -> {destination}")
    print(f"  sha256: {payload['result']['lens_sha256']}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
