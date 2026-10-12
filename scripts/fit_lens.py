"""[1] Build the instrument: fit ``J^z`` at ``o_proj``'s input, at every head anchor.

    python -m scripts.fit_lens --gpu 0 --n-prompts 1000            # one pass
    python -m scripts.fit_lens --gpu 2 --n-prompts 250 \
        --prompt-offset 250 --shard a1                             # one of four shards

``J^z_l`` is the transport from ``o_proj``'s input to the final residual, one matrix per
layer; a head is a column slice of it. This is the only expensive item in the project --
~53 s/prompt at ``dim_batch=64`` on an A100 80GB at this depth, ~15 GPU-hours for the
shipped lens, run as four disjoint 250-prompt shards and merged by ``scripts.merge_jz``.
The estimator is :func:`src.jacobian.jacobian_for_prompt`, reused byte-unchanged; the
loop around it is :func:`src.fit.fit`.

**Every layer with a head, non-negotiable.** :func:`src.model.load_jz` requires
``source_layers == spec.jz_source_layers`` exactly -- all 32 on Llama-3.1-8B, the 8
attention blocks of Qwen3.5-4B -- so a partial fit would raise on load. The experiments
read every one of those layers.

**The budget is fixed**, as in :func:`jlens.fitting.fit`: the loop applies no convergence
criterion, and ``mean_rel_change`` is written to the trace as a diagnostic rather than
acted on. No decision depends on that trace.

One thing is recorded here rather than in a script of its own, because the real model is
already loaded and it is a *value* the CPU tests only assert a *bound* on:

* the **two-sided ``W_O`` gate**, the one check that catches a ``zin`` hook slipped to
  ``o_proj``'s *output*. At the top layer almost no network remains between ``z_l`` and
  the final residual, so ``J^z_31`` should approach ``W_O^(31)`` -- measured median row
  cosine 0.867 on Llama, against -0.0002 for the identity. Had the hook slipped, the
  fitted object would be the ``attn``-anchored one and the two numbers would **swap**
  (0.904 against the identity, ~0 against ``W_O``). That is what makes it a test rather
  than a trend, and it is visible from the first prompt. The identity arm needs a square
  ``W_O`` and is skipped otherwise; the ``W_O`` arm reads the matrix the block actually
  adds per unit of ``z`` (:func:`src.heads.residual_o_proj_weight`).
"""

from __future__ import annotations

import argparse
import json
import logging

from scripts._gpu import pin_gpu

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    from src.config import FIT_CORPUS_SIZE, JZ_FIT_STAGE

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", default=None, help="GPU index, as nvidia-smi numbers them")
    parser.add_argument("--lens", default=None, help="LensSpec key (default: DEFAULT_LENS)")
    parser.add_argument("--n-prompts", type=int, default=1000, help="prompts in this slice")
    parser.add_argument(
        "--prompt-offset",
        type=int,
        default=0,
        help="start of this slice in the drawn corpus; shards must not overlap",
    )
    parser.add_argument("--corpus-seed", type=int, default=0)
    parser.add_argument("--corpus-size", type=int, default=FIT_CORPUS_SIZE)
    parser.add_argument(
        "--dim-batch", type=int, default=None, help="default: the spec's fit_dim_batch"
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--shard", default=None, help="name this run's artifacts apart")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--out-name", default=JZ_FIT_STAGE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pin_gpu(args.gpu)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import torch

    torch.set_num_threads(8)

    from jlens.fitting import SKIP_FIRST_N_POSITIONS

    from src.config import (
        DEFAULT_LENS,
        LENSES,
        MAX_SEQ_LEN,
        Paths,
    )
    from src.fit import CorpusSlice, fit
    from src.heads import residual_o_proj_weight
    from src.model import load_model
    from src.provenance import file_digest, meta_block
    from src.validation import wo_gate

    spec = LENSES[args.lens or DEFAULT_LENS]
    dim_batch = args.dim_batch or spec.fit_dim_batch
    paths = Paths.default()
    paths.lenses_dir.mkdir(parents=True, exist_ok=True)

    corpus = CorpusSlice(
        size=args.corpus_size,
        seed=args.corpus_seed,
        offset=args.prompt_offset,
        n_prompts=args.n_prompts,
    )
    prompts = corpus.prompts()

    model = load_model(spec)
    # ``HFLensModel`` compiles each *block*, which inlines the block's internals and
    # bypasses a pre-hook on ``self_attn.o_proj`` -- the ``zin`` anchor would silently
    # never fire. ``from_hf``'s default is already False; the released Neuronpedia run
    # used ``compile: true`` and copying its config wholesale is the obvious mistake.
    if type(model.layers[0]).__name__ == "OptimizedModule":
        raise RuntimeError("blocks are torch.compile'd; the zin pre-hook would not fire")

    top_layer = spec.jz_source_layers[-1]
    w_o_top = residual_o_proj_weight(model, top_layer).float().cpu()
    reference = "diag(g_post) W_O" if spec.post_attention_norm else "W_O"

    def monitor(jacobian_sum, n_done: int) -> float:
        """Median row cosine of the running top-layer ``J^z`` against ``W_O`` there."""
        from src.metrics import cosine_rows

        return float(cosine_rows(jacobian_sum[("zin", top_layer)] / n_done, w_o_top).median())

    anchors = [("zin", l) for l in spec.jz_source_layers]
    result = fit(
        model,
        prompts,
        anchors=anchors,
        d_source=spec.d_source,
        corpus=corpus,
        dim_batch=dim_batch,
        max_seq_len=MAX_SEQ_LEN,
        checkpoint_path=paths.jz_fit_checkpoint(spec, shard=args.shard),
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        convergence_path=paths.jz_fit_convergence(spec, shard=args.shard),
        monitor=monitor,
        monitor_name="wo_row_cosine_top",
    )

    lens = result.lens(spec.d_model)
    destination = paths.jz_fit_lens(spec, n_prompts=result.n_done, shard=args.shard)
    lens.save(str(destination))
    gate = wo_gate(result.jacobians[("zin", top_layer)], w_o_top, reference=reference)

    payload = {
        "meta": meta_block(
            model=spec.hf_model_id,
            lens_key=spec.key,
            anchor="zin (o_proj input)",
            d_model=spec.d_model,
            d_source=spec.d_source,
            n_heads=spec.n_heads,
            head_dim=spec.head_dim,
            dtype="bfloat16",
            dim_batch=dim_batch,
            max_seq_len=MAX_SEQ_LEN,
            skip_first=SKIP_FIRST_N_POSITIONS,
            layers=spec.jz_source_layers,
            shard=args.shard,
            compile=False,
            **corpus.as_meta(),
        ),
        "result": {
            "prompts_fitted": result.n_done,
            "prompts_skipped": result.n_skipped,
            "prompts_offered": len(prompts),
            "final_mean_rel_change": result.final_mean_rel_change,
            "lens": str(destination),
            "lens_sha256": file_digest(destination),
            "convergence_csv": str(paths.jz_fit_convergence(spec, shard=args.shard)),
        },
        "wo_gate": gate,
        "PASS": gate["PASS"],
    }
    name = f"{spec.key}{'_' + args.shard if args.shard else ''}_fit.json"
    out_path = paths.run_dir(args.out_name) / name
    out_path.write_text(json.dumps(payload, indent=1))

    print(f"fitted J^z on {result.n_done} prompts -> {destination}")
    print(f"  sha256: {payload['result']['lens_sha256']}")
    print(f"\ntwo-sided W_O gate at L{top_layer} (reference {reference}):")
    print(f"  median row-cos vs W_O:     {gate['median_row_cosine_vs_W_O']:.4f}  (> {gate['min_vs_W_O']})")
    identity = gate["median_row_cosine_vs_identity"]
    print(
        "  median row-cos vs identity "
        + (f"{identity:+.4f}  (< {gate['max_vs_identity']})" if identity is not None
           else "n/a (W_O is not square)")
    )
    print(f"\nwrote {out_path}")
    print("PASS" if gate["PASS"] else "FAIL")
    return 0 if gate["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
