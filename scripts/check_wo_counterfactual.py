"""Score both arms of the two-sided ``W_O`` gate, on the shipped ``J^z``.

    python -m scripts.check_wo_counterfactual --gpu 3

:func:`src.validation.wo_gate` scores the fitted ``J^z`` against ``W_O`` and the
identity, and ``fit_lens.py`` records that pair. What it cannot score is the
*counterfactual*: that a ``zin`` pre-hook slipped to ``o_proj``'s output would have
fitted the ``attn``-anchored matrix instead, whose two numbers **swap**. That arm used to
be prose, sourced to a predecessor run.

It does not have to be. The anchor identity ``J^z_l = J_attn_l W_O`` is exact and ``W_O``
is square here, so ``J_attn`` is recoverable from the shipped lens by a linear solve --
no refit, no second fitting run. This script does that at every block and records both
arms, so the counterfactual is an artifact like everything else.

Writes ``results/fit_checks/<lens>_wo_gate_counterfactual.json``.
"""

from __future__ import annotations

import argparse
import json
import logging

from scripts._gpu import pin_gpu

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", default=None, help="GPU index, as nvidia-smi numbers them")
    parser.add_argument("--lens", default=None, help="LensSpec key (default: DEFAULT_LENS)")
    parser.add_argument("--n-prompts", type=int, default=None)
    parser.add_argument("--out-name", default="fit_checks")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pin_gpu(args.gpu)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import torch

    from src.config import DEFAULT_LENS, LENSES, Paths, jz_path
    from src.heads import o_proj_weight
    from src.model import load_jz, load_model
    from src.provenance import file_digest, meta_block
    from src.validation import wo_gate_counterfactual

    spec = LENSES[args.lens or DEFAULT_LENS]
    paths = Paths.default()
    model = load_model(spec)
    n_prompts = args.n_prompts or spec.jz_n_prompts
    jz_file = jz_path(spec, n_prompts=n_prompts, paths=paths)
    jz = load_jz(spec, jz_file)
    device = model.input_device

    per_layer: dict[int, dict[str, float]] = {}
    for layer in range(spec.n_layers):
        scores = wo_gate_counterfactual(
            jz.jacobians[layer].to(device), o_proj_weight(model, layer).to(device)
        )
        per_layer[layer] = scores
        torch.cuda.empty_cache()
        logger.info(
            "L%02d  attn-vs-I %.4f  attn-vs-W_O %+.4f  jz-vs-W_O %.4f  jz-vs-I %+.4f "
            " (roundtrip %.1e)",
            layer,
            scores["attn_median_row_cosine_vs_identity"],
            scores["attn_median_row_cosine_vs_W_O"],
            scores["jz_median_row_cosine_vs_W_O"],
            scores["jz_median_row_cosine_vs_identity"],
            scores["roundtrip_relative_residual"],
        )

    top = spec.n_layers - 1
    series = [per_layer[l]["attn_median_row_cosine_vs_identity"] for l in range(spec.n_layers)]
    # The monotone claim is made over the final eight blocks; report both scopes rather
    # than only the one that passes.
    def _decreases(values: list[float], offset: int) -> list[int]:
        return [
            offset + i + 1
            for i, (a, b) in enumerate(zip(values, values[1:], strict=False))
            if b < a
        ]

    payload = {
        "meta": meta_block(
            model=spec.hf_model_id,
            lens_key=spec.key,
            jz_path=jz_file,
            jz_sha256=file_digest(jz_file),
            jz_n_prompts=n_prompts,
            method="J_attn_l = J^z_l W_O^{-1} by linear solve; exact given the anchor "
            "identity J^z_l = J^attn_l W_O (paper, Section S1.1)",
            scored_by="median row cosine, fp32",
        ),
        "per_layer": {str(l): v for l, v in per_layer.items()},
        "summary": {
            "attn_vs_identity_block_0": series[0],
            "attn_vs_identity_top_block": series[top],
            "attn_vs_W_O_top_block": per_layer[top]["attn_median_row_cosine_vs_W_O"],
            "jz_vs_W_O_top_block": per_layer[top]["jz_median_row_cosine_vs_W_O"],
            "jz_vs_identity_top_block": per_layer[top]["jz_median_row_cosine_vs_identity"],
            "decreases_all_blocks": _decreases(series, 0),
            "decreases_over_last_8": _decreases(series[-8:], top - 8),
            "worst_roundtrip_relative_residual": max(
                v["roundtrip_relative_residual"] for v in per_layer.values()
            ),
        },
    }
    out_path = paths.run_dir(args.out_name) / f"{spec.key}_wo_gate_counterfactual.json"
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out_path}")
    print(json.dumps(payload["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
