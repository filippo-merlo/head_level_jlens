"""[2] The legibility experiment: is a single head's write legible through the transport?

    python -m scripts.read --gpu 0

For every ``(layer, head)`` in the model, transport ``z_{l,h}`` and unembed; score pass@k
against the item's intermediates, on six eval sets, at each set's documented readout
position.

**Two arms, and a reference column.** All three read the same activations at the same
position::

    J-head      J_{l,h} z_{l,h}          the thing being tested
    Logit-head  c_{l,h}, no transport    THE comparison: same vector, identity transport
    J-layer     Jacobian lens J_l on the whole residual stream

``J-layer`` is a **descriptive reference**, not an arm: it says what a whole layer reads,
for scale, and nothing is adjudicated against it. It exists at layers 0-30 only: the
Jacobian lens's top block is its own target, so it has no readout at layer 31, and the
reference column is simply absent from that layer's region.

**The paired difference is primary.** J-head and Logit-head read the same activations on
the same items, so their difference is the mean of the per-item differences. The
marginals are kept as columns.

**Aggregation: the best site.** A head has no depth -- head ``h`` of one layer and head
``h`` of the next share a column index and nothing else -- so a head arm is reduced to one
rank per layer by the minimum over that layer's 32 heads, the best single head, and the
package's min over layers then applies to head arms and the whole-layer reference alike.
The all-layers value is the best of 1,024 sites; a per-layer value is the best of 32
heads.

**The raw ranks are kept.** Every ``(item, intermediate, layer, head)`` rank is written to
an ``.npz`` beside the JSON, so any other reduction can be computed without the GPU.

**How many sites, against a control.** The best site keeps one head per intermediate; the
count of sites reading it within the top ``k`` says how many do. A readout that ranks a
token high wherever it is read would inflate that count, so each concept is also scored
on another item's prompt from the same set, one not annotated with any of its tokens
(:func:`src.evals.control_items`), at the same sites and through the same readouts.

**Every layer is reported.** Results come out per layer as well as aggregated over the
whole depth, so no number here depends on a choice of where to look. There is no
depth restriction: the scope is layers 0-31, every layer the fitted ``J^z`` covers.

**Magnitude travels with every number.** RMSNorm is scale-invariant, so a head writing
1.7% of the stream it joins still produces a confident-looking top-1. The relative write
norm and the ablation KL are measured in this same pass -- the model and the lens are
already loaded -- and carried in the payload header, because a pass@k describes the
*direction* of a small write, never its size.

The structural characterisation rides along for the same reason it always did: per-head
readout kurtosis and workspace loading are functions of readouts this script already
computes, so they cost no extra forward passes. They are recorded as a description beside
the rank-based numbers, not adjudicated against anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from scripts._gpu import pin_gpu

logger = logging.getLogger(__name__)

K_VALUES = (1, 5, 25)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", default=None, help="GPU index, as nvidia-smi numbers them")
    parser.add_argument("--lens", default=None, help="LensSpec key (default: DEFAULT_LENS)")
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=None,
        help="prompts the J^z lens was fitted on (default: the one LensSpec pins)",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--magnitude-items",
        type=int,
        default=40,
        help="items per set the write norm and ablation KL are measured on",
    )
    parser.add_argument("--out-name", default="read")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pin_gpu(args.gpu)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    started = time.monotonic()

    import numpy as np
    import torch

    from src.config import (
        DEFAULT_LENS,
        LENSES,
        MAX_SEQ_LEN,
        Paths,
        jz_path,
    )
    from src.evals import (
        ALL_SETS,
        CONTROL_SEED,
        SINGLE_TOKEN_ONLY,
        candidate_coverage,
        control_items,
        load_eval,
        readout_index,
        target_forms,
    )
    from src.heads import effective_unembed, residual_o_proj_weight
    from src.hooks import AnchorRecorder
    from src.magnitude import ablation_kl, relative_write_norms
    from src.model import load_jz, load_lens, load_model
    from src.passk import (
        PAD,
        best_site_rank,
        paired_difference,
        passk,
        scoreable_items,
        sites_hit,
    )
    from src.provenance import file_digest, meta_block
    from src.readout import (
        HEAD_ARMS,
        all_heads_transported,
        layer_logits,
        target_direction_id,
        target_token_ids,
        token_ranks,
    )
    from src.structural import excess_kurtosis, workspace_loading

    spec = LENSES[args.lens or DEFAULT_LENS]
    paths = Paths.default()
    #: Every layer the fitted ``J^z`` covers, 0..31. The block-anchored Jacobian lens
    #: stops one short of that, so the J-layer reference column runs over ``ref_layers``.
    layers = list(spec.jz_source_layers)
    ref_layers = list(spec.source_layers)
    n_layers = len(layers)

    model = load_model(spec)
    jz_n_prompts = args.n_prompts or spec.jz_n_prompts
    jz_file = jz_path(spec, n_prompts=jz_n_prompts, paths=paths)
    jz = load_jz(spec, jz_file)
    published = load_lens(spec)
    device = model.input_device
    jz_by_layer = {l: jz.jacobians[l].to(device) for l in layers}
    #: The matrix the identity transport reads a head through: ``W_O`` on a pre-norm
    #: block, ``diag(g_post) W_O`` where a norm sits between ``o_proj`` and the residual
    #: add (Gemma-3), so Logit-head reads the head's vector in residual-stream units.
    w_o_by_layer = {l: residual_o_proj_weight(model, l) for l in layers}
    #: What each set's candidate rule can score under this tokenizer. The rule is fixed;
    #: a tokenizer that splits "26" into digits leaves that slot with no single-token
    #: form, and the slot stays PAD -- an intermediate the record says it could not score.
    coverage = {slug: candidate_coverage(model.tokenizer, slug) for slug in ALL_SETS}
    for slug, cov in coverage.items():
        if cov["unscoreable"]:
            logger.warning(
                "%s: %d of %d intermediates have no candidate under this tokenizer "
                "and are skipped: %s", slug, len(cov["unscoreable"]),
                cov["n_intermediates"], cov["unscoreable"],
            )

    #: Every region reported, as ``{name: layers}``: the whole depth, then each layer on
    #: its own. The aggregate is a summary of the per-layer table, never a scope choice.
    REGIONS = {"all_layers": layers} | {f"layer_{l}": [l] for l in layers}

    def candidates(item, word: str) -> list[int] | None:
        try:
            return target_token_ids(
                model.tokenizer,
                target_forms(item, word),
                single_token_only=item.slug in SINGLE_TOKEN_ONLY,
            )
        except ValueError:
            return None  # unscoreable here; recorded in meta.coverage

    def head_ranks(logits: torch.Tensor, ids: list[int]) -> np.ndarray:
        """Each head's best rank over the candidate ids, ``[n_heads]``."""
        return (
            torch.stack([token_ranks(logits[h], ids).min() for h in range(spec.n_heads)])
            .cpu().numpy()
        )

    # Magnitude accumulators, filled from the first --magnitude-items items of each set.
    norm_totals = {l: torch.zeros(spec.n_heads) for l in layers}
    kl_totals = {l: torch.zeros(spec.n_heads) for l in layers}
    n_magnitude = 0

    results = {}
    raw_ranks = {}
    for slug in ALL_SETS:
        items = load_eval(slug)
        if args.max_items:
            items = items[: args.max_items]
        max_inter = max(len(i.intermediates) for i in items)

        targets_by_item = [[candidates(item, w) for w in item.intermediates] for item in items]
        # The (item, intermediate) pairs whose control is read on each item's prompt.
        controls_at: dict[int, list[tuple[int, int]]] = {}
        for i, row in enumerate(
            control_items([
                [None if ids is None else frozenset(ids) for ids in targets]
                for targets in targets_by_item
            ])
        ):
            for wi, j in enumerate(row):
                if j is not None:
                    controls_at.setdefault(j, []).append((i, wi))

        # [arm][n_items, max_inter, n_layers, n_heads]; PAD marks padding.
        per_layer = {
            arm: np.full((len(items), max_inter, n_layers, spec.n_heads), PAD, dtype=np.int64)
            for arm in HEAD_ARMS
        }
        per_layer["jlayer"] = np.full((len(items), max_inter, n_layers, 1), PAD, dtype=np.int64)
        # The same concepts at the same sites, read on their control item's prompt.
        control = {arm: np.full_like(per_layer[arm], PAD) for arm in HEAD_ARMS}

        kurtosis_totals = {arm: np.zeros((n_layers, spec.n_heads)) for arm in HEAD_ARMS}
        loading_totals = {
            arm: np.zeros((n_layers, spec.n_heads)) for arm in ("jhead",)
        }
        n_kurtosis = 0
        n_loading = 0

        for i, item in enumerate(items):
            input_ids = model.encode(item.prompt, max_length=MAX_SEQ_LEN)
            position = readout_index(item, input_ids, model.tokenizer)
            targets = targets_by_item[i]

            # Workspace loading is scored against the concept the item *specifies*, one
            # token per intermediate: the space-prefixed first token, the same direction
            # convention the swaps use. Not the head's own top-ranked lens vector, which
            # would be a maximum over the vocabulary and biased upward.
            concept_rows = effective_unembed(
                model,
                torch.tensor(
                    [
                        target_direction_id(model.tokenizer, w)
                        for w, ids in zip(item.intermediates, targets, strict=True)
                        if ids is not None
                    ],
                    device=device,
                    dtype=torch.long,
                ),
            )

            with torch.no_grad(), AnchorRecorder(
                model.layers, zin_at=layers, out_at=ref_layers
            ) as recorder:
                model.forward(input_ids)
                for li, layer in enumerate(layers):
                    z = recorder.activations[("zin", layer)][0, position]
                    for arm in HEAD_ARMS:
                        logits = model.unembed(
                            all_heads_transported(
                                jz_by_layer[layer], z, spec, arm=arm,
                                w_o=w_o_by_layer[layer],
                            )
                        ).float()
                        kurtosis_totals[arm][li] += excess_kurtosis(logits).cpu().numpy()
                        for wi, ids in enumerate(targets):
                            if ids is not None:
                                per_layer[arm][i, wi, li] = head_ranks(logits, ids)
                        for ci, cwi in controls_at.get(i, []):
                            control[arm][ci, cwi, li] = head_ranks(
                                logits, targets_by_item[ci][cwi]
                            )
                    for row in concept_rows:
                        loading_totals["jhead"][li] += (
                            workspace_loading(jz_by_layer[layer], z, spec, row)
                            .cpu().numpy()
                        )
                    if layer in ref_layers:
                        residual = recorder.activations[("out", layer)][0, position]
                        ref = layer_logits(model, published, residual, layer)
                        for wi, ids in enumerate(targets):
                            if ids is None:
                                continue
                            per_layer["jlayer"][i, wi, li, 0] = int(
                                token_ranks(ref, ids).min()
                            )

            n_kurtosis += 1
            n_loading += len(concept_rows)

            # Magnitude, at the same position the ranks are read at -- so the two are
            # paired rather than one being averaged over a corpus the readouts never
            # touch. Bounded per set because ablation KL costs a batched forward per
            # layer and the statistic is a distribution summary, not a per-item result.
            if i < args.magnitude_items:
                norms = relative_write_norms(
                    model, spec, input_ids, layers, position=position
                )
                kls = ablation_kl(model, spec, input_ids, layers, position=position)
                for layer in layers:
                    norm_totals[layer] += norms[layer]
                    kl_totals[layer] += kls[layer]
                n_magnitude += 1

        raw_ranks.update({f"{slug}__{arm}": per_layer[arm] for arm in per_layer})
        raw_ranks.update({f"{slug}__{arm}__control": control[arm] for arm in HEAD_ARMS})

        # Items with at least one scoreable intermediate: what every rate of this set is a mean over (src.passk.scoreable_items). Equal to n_items unless
        # meta.candidate_coverage lists an item whose every intermediate was skipped.
        n_scored = int(scoreable_items(best_site_rank(per_layer["jhead"])).size)
        entry = {"n_items": len(items), "n_items_scored": n_scored, "regions": {}}
        for region_name, layers_subset in REGIONS.items():
            reference = [l for l in layers_subset if l in ref_layers]
            scored = [*HEAD_ARMS, *(["jlayer"] if reference else [])]
            reduced = {
                arm: best_site_rank(
                    per_layer[arm],
                    [layers.index(l) for l in (reference if arm == "jlayer" else layers_subset)],
                )
                for arm in scored
            }
            region = {"layers": layers_subset, "k": {}}
            for k in K_VALUES:
                rates = {arm: {"point": passk(reduced[arm], k)} for arm in scored}
                region["k"][k] = {
                    # Primary: both arms read the same activations on the same items.
                    "paired_difference": {
                        "jhead_minus_logithead": {
                            "point": paired_difference(reduced["jhead"], reduced["logithead"], k)
                        },
                    },
                    "rates": rates,
                    # The headline ratio, recorded rather than recomputed downstream so
                    # the number in the report is the number in the record.
                    "jhead_over_logithead": (
                        rates["jhead"]["point"] / rates["logithead"]["point"]
                        if rates["logithead"]["point"] > 0
                        else None
                    ),
                }
            entry["regions"][region_name] = region

        # All layers only: a count over every site, where the best site is one of them.
        entry["sites_hit"] = {
            k: {
                arm: {
                    "concept": sites_hit(per_layer[arm], k),
                    "control": sites_hit(control[arm], k),
                }
                for arm in HEAD_ARMS
            }
            for k in K_VALUES
        }

        kurtosis_mean = {arm: kurtosis_totals[arm] / max(n_kurtosis, 1) for arm in HEAD_ARMS}
        loading_mean = {
            arm: loading_totals[arm] / max(n_loading, 1) for arm in ("jhead",)
        }
        entry["structural"] = {
            "note": "per-head excess kurtosis of the readout over the vocabulary, and "
            "workspace loading cos(z_{l,h}, v_{t,l,h}) against the concept the item "
            "specifies. Descriptive: a sharper, "
            "better-aligned readout is not a causal claim.",
            "n_items": n_kurtosis,
            "n_loading_observations": n_loading,
            "layers": layers,
            "excess_kurtosis": {arm: kurtosis_mean[arm].tolist() for arm in HEAD_ARMS},
            "workspace_loading": {
                arm: loading_mean[arm].tolist() for arm in ("jhead",)
            },
            "depth_summary": {
                "mean_excess_kurtosis": {
                    arm: float(kurtosis_mean[arm].mean()) for arm in HEAD_ARMS
                },
                # Loading is signed, so the |.| mean is
                # what says whether the head's write aligns with the concept at all; the
                # signed mean is kept beside it because a systematically negative loading
                # is a different finding from no loading.
                "mean_abs_workspace_loading": {
                    arm: float(abs(loading_mean[arm]).mean()) for arm in ("jhead",)
                },
                "mean_workspace_loading": {
                    arm: float(loading_mean[arm].mean()) for arm in ("jhead",)
                },
            },
        }
        results[slug] = entry
        logger.info("scored %s (%d items)", slug, len(items))

    write_norm = {l: norm_totals[l] / max(n_magnitude, 1) for l in layers}
    kl = {l: kl_totals[l] / max(n_magnitude, 1) for l in layers}
    all_norms = torch.cat([write_norm[l] for l in layers])
    all_kls = torch.cat([kl[l] for l in layers])
    magnitude = {
        "note": "RMSNorm is scale-invariant, so a head contributing almost nothing still "
        "produces a confident-looking top-1. Every pass@k above describes the direction "
        "of a write this size, not its size.",
        "median_relative_write_norm": float(all_norms.median()),
        "max_relative_write_norm": float(all_norms.max()),
        "median_ablation_kl": float(all_kls.median()),
        "max_ablation_kl": float(all_kls.max()),
        "n_items": n_magnitude,
        "items_per_set": args.magnitude_items,
        "measured_at": "each item's documented readout position",
        "relative_write_norm": {l: write_norm[l].tolist() for l in layers},
        "ablation_kl": {l: kl[l].tolist() for l in layers},
    }

    out_path = paths.run_dir(args.out_name) / f"{spec.key}_read.json"
    ranks_path = out_path.with_name(f"{spec.key}_read_ranks.npz")
    np.savez_compressed(ranks_path, **raw_ranks)

    payload = {
        "meta": meta_block(
            model=spec.hf_model_id,
            lens_key=spec.key,
            jz_n_prompts=jz_n_prompts,
            jz_path=jz_file,
            jz_sha256=file_digest(jz_file),
            published_lens_n_prompts=published.n_prompts,
            layers=layers,
            reference_layers=ref_layers,
            regions=list(REGIONS),
            arms=list(HEAD_ARMS),
            reference_column="jlayer (Jacobian lens J_l on the whole residual stream); "
            f"layers {ref_layers[0]}..{ref_layers[-1]} only, the Jacobian lens has no "
            "readout at the top block",
            aggregation=f"best site: min over the {spec.n_heads} heads of a layer, then "
            "min over the region's layers; a head has no depth, so nothing is reduced per "
            "column index across layers",
            n_sites=len(layers) * spec.n_heads,
            logithead_vector="diag(g_post) W_O^h z_h: the head's vector in residual-stream "
            "units, the norm between o_proj and the residual add folded in"
            if spec.post_attention_norm
            else "W_O^h z_h",
            candidate_coverage=coverage,
            raw_ranks=str(ranks_path),
            raw_ranks_layout="{set}__{arm}: [n_items, max_intermediates, n_layers, "
            f"n_heads] zero-based ranks, {PAD} = padding; jlayer has n_heads = 1; "
            "{set}__{arm}__control: the same concepts read on their control item's prompt",
            sites_hit="mean over items of the mean over an item's intermediates of the "
            "number of sites ranking the intermediate within the top k (all layers); "
            "'control' counts the same candidate tokens at the same sites on another item "
            "of the set, the next in a seeded random cycle whose own intermediates share "
            "no candidate token with the concept",
            control_seed=CONTROL_SEED,
            primary_statistic="paired J-head minus Logit-head difference: the mean over "
            "items of the per-item difference",
            single_token_only_sets=sorted(SINGLE_TOKEN_ONLY),
            candidate_rule="both space-prefixed and bare spellings; for the sets in "
            "single_token_only_sets, only forms the tokenizer represents as exactly one "
            "token, which keeps the bare-space token out of the candidate set",
            wall_clock_seconds=round(time.monotonic() - started, 1),
            dtype="bfloat16",
        ),
        "magnitude": magnitude,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=1))

    print(
        f"\nlayers {layers[0]}..{layers[-1]}; best site, "
        f"mean over items. Median relative write norm "
        f"{magnitude['median_relative_write_norm']:.4f}, median ablation KL "
        f"{magnitude['median_ablation_kl']:.5f} nats "
        f"(n={magnitude['n_items']} items)."
    )
    for k in K_VALUES:
        print(f"\n  pass@{k}")
        print(
            f"    {'set':<16} {'J-head':>8} {'Logit-head':>10} "
            f"{'J-layer':>9} {'ratio':>7}  paired diff"
        )
        for slug in ALL_SETS:
            block = results[slug]["regions"]["all_layers"]["k"][k]
            m = block["rates"]

            paired = block["paired_difference"]["jhead_minus_logithead"]
            ratio = block["jhead_over_logithead"]
            print(
                f"    {slug.replace('lens-eval-', ''):<16} {m['jhead']['point']:>8.3f} "
                f"{m['logithead']['point']:>10.3f} "
                f"{m['jlayer']['point']:>9.3f} "
                f"{'--' if ratio is None else f'{ratio:.2f}x':>7}  "
                f"{paired['point']:+.3f}"
            )

    print("\n  per layer, pass@5, J-head over Logit-head")
    chunk = 16
    for start in range(0, len(layers), chunk):
        block_layers = layers[start : start + chunk]
        print("    " + f"{'set':<16}" + "".join(f"{l:>6}" for l in block_layers))
        for slug in ALL_SETS:
            cells = "".join(
                f"{results[slug]['regions'][f'layer_{l}']['k'][5]['jhead_over_logithead'] or 0:>6.2f}"
                for l in block_layers
            )
            print(f"    {slug.replace('lens-eval-', ''):<16}{cells}")

    print(f"\nwrote {out_path}  ({payload['meta']['wall_clock_seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
