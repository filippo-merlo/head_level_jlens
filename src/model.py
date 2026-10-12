"""Load a checkpoint and its lenses. The only model-touching module.

Three things are checked at load time rather than assumed, because all three fail
silently:

* **The lens must match the checkpoint.** ``JacobianLens`` carries no model identity, so
  a lens for one model loads happily against another and produces plausible-looking
  nonsense. We verify ``d_model`` and the source-layer range against the config.
* **The lens family must match the anchor.** ``J^z`` and the Jacobian lens's ``J_l``
  are both
  :class:`~jlens.lens.JacobianLens` objects of identical shape, and nothing in the file
  format distinguishes them. Reading a head's write through the block-anchored lens
  silently skips the layer's own MLP -- the exact error this project exists to avoid.
  The source-layer *list* is the discriminator: a block-anchored file stops one short of
  the top, a ``J^z`` file covers exactly the layers with a head anchor.
* **The lens file must be the one the results cite.** Filenames get reused; contents do
  not. Every ``J^z`` is pinned by SHA-256 in :attr:`~src.config.LensSpec.jz_digests`, or,
  before a fresh fit's digest has been recorded there, by the merge sidecar that vouched
  for its shards.
"""

from __future__ import annotations

import logging

import torch

from src.config import JZ_FIT_STAGE, LensSpec, Paths, lens_path
from src.hooks import has_o_proj, post_sublayer_norm

logger = logging.getLogger(__name__)


def load_model(
    spec: LensSpec,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    attn_implementation: str | None = None,
    force_bos: bool = True,
):
    """Load the HF checkpoint and wrap it as a :class:`jlens.protocol.LensModel`.

    ``dtype`` defaults to bfloat16 to match the dtype each Neuronpedia lens was fitted
    under (see the fit ``config.yaml``); downstream arithmetic can upcast to fp32, so
    this only sets the precision of the forward pass itself.

    ``force_bos`` is upstream's default and correct for raw completion-style prompts,
    which is what this project uses throughout. It must be
    ``False`` for text that has already been through ``apply_chat_template``, since that
    string already begins with ``<|begin_of_text|>`` and would otherwise be doubled.

    Pinned to a single device: ``device_map="auto"`` would shard the model and is
    unnecessary here (8B in bf16 fits one 80 GB A100 with room to spare).

    ``AutoModelForCausalLM`` resolves a multimodal checkpoint to whatever class
    transformers maps its ``model_type`` to -- a text-only view for Qwen3.5, the
    vision-and-text wrapper for Gemma-3 -- and :func:`jlens.from_hf` finds the text stack
    under either. Model facts are therefore always read from ``get_text_config()``.
    """
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(
        "loading %s (%s, attn=%s, force_bos=%s) onto %s",
        spec.hf_model_id,
        dtype,
        attn_implementation or "default",
        force_bos,
        device,
    )
    kwargs = (
        {} if attn_implementation is None else {"attn_implementation": attn_implementation}
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        spec.hf_model_id, dtype=dtype, **kwargs
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_id)

    model = jlens.from_hf(hf_model, tokenizer, force_bos=force_bos)
    # ``HFLensModel`` keeps the HF model private, but the per-head readout needs
    # ``o_proj``'s weight at every layer. Attaching it here keeps that reach-through in
    # one place instead of at every call site.
    model.hf_model = hf_model

    if model.n_layers != spec.n_layers or model.d_model != spec.d_model:
        raise ValueError(
            f"config mismatch for {spec.key}: loaded model has n_layers="
            f"{model.n_layers} d_model={model.d_model}, spec says "
            f"{spec.n_layers}/{spec.d_model}"
        )
    check_model_facts(model, spec)
    logger.info("resolved layout: %s", model.layout)
    return model


def text_config(model):
    """The text stack's config, whether or not the checkpoint is a multimodal wrapper."""
    return model.hf_model.config.get_text_config()


def final_residual(model, input_ids: torch.Tensor) -> torch.Tensor:
    """The last block's output ``x_L``, ``[batch, seq, d_model]`` -- what ``unembed`` takes.

    ``model.forward(...).last_hidden_state`` is **not** that: HF text models apply the
    final norm before returning it, so unembedding it applies the norm twice and reads
    the stream along ``g * g`` instead of ``g``. On Llama-3.1-8B the gains are mild and
    the distortion passed unnoticed; on Gemma-3, whose final-norm gains reach the
    fifties, it turns the model's own next token into garbage. The lens target is the
    block output (``("out", n_layers - 1)`` in :mod:`src.jacobian`), and this reads it
    the same way, through a hook, so a patcher active around the call is seen.
    """
    from src.hooks import AnchorRecorder

    top = model.n_layers - 1
    with AnchorRecorder(model.layers, out_at=[top]) as recorder:
        model.forward(input_ids)
    return recorder.activations[("out", top)]


def check_model_facts(model, spec: LensSpec) -> dict[str, object]:
    """Assert the recorded model facts against the loaded checkpoint.

    The plan says to verify these "before anything else", and two of them genuinely
    cannot be read off the config: ``head_dim`` is *absent* from Llama-3.1's
    ``config.json`` and must be derived, and ``n_heads * head_dim == d_model`` is a
    coincidence of that architecture that the rest of the code deliberately does not
    assume (see :attr:`~src.config.LensSpec.d_source`).

    Two structural facts are checked against the blocks themselves rather than the
    config, because a wrong value in the spec would otherwise fit a plausible lens: the
    set of layers that have an ``o_proj`` must be exactly
    :attr:`~src.config.LensSpec.jz_source_layers`, and whether a norm sits between
    ``o_proj`` and the residual add must be what
    :attr:`~src.config.LensSpec.post_attention_norm` says.

    Returns:
        The verified quantities, for a results ``meta`` block.

    Raises:
        ValueError: On any mismatch with ``spec``.
    """
    config = text_config(model)
    derived_head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    found = {
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": derived_head_dim,
        "vocab_size": config.vocab_size,
    }
    expected = {
        "num_hidden_layers": spec.n_layers,
        "hidden_size": spec.d_model,
        "num_attention_heads": spec.n_heads,
        "head_dim": spec.head_dim,
    }
    mismatches = {k: (found[k], v) for k, v in expected.items() if found[k] != v}
    if mismatches:
        raise ValueError(f"model-facts mismatch for {spec.key}: {mismatches}")

    with_o_proj = [l for l in range(spec.n_layers) if has_o_proj(model.layers[l])]
    if with_o_proj != spec.jz_source_layers:
        raise ValueError(
            f"{spec.key}: blocks with self_attn.o_proj are {with_o_proj}, but the spec "
            f"puts the head anchor at {spec.jz_source_layers}"
        )
    for layer in spec.jz_source_layers:
        o_proj = model.layers[layer].self_attn.o_proj
        if o_proj.bias is not None:
            raise ValueError(f"layer {layer}: o_proj has a bias; the head decomposition assumes none")
        if tuple(o_proj.weight.shape) != (spec.d_model, spec.d_source):
            raise ValueError(
                f"layer {layer}: o_proj.weight is {tuple(o_proj.weight.shape)}, "
                f"expected (d_model={spec.d_model}, d_source={spec.d_source})"
            )
        normed = post_sublayer_norm(model.layers[layer]) is not None
        if normed != spec.post_attention_norm:
            raise ValueError(
                f"layer {layer}: the block {'has' if normed else 'has no'} norm between "
                f"o_proj and the residual add, but the spec says "
                f"post_attention_norm={spec.post_attention_norm}"
            )
    found["d_source"] = spec.d_source
    found["attention_layers"] = spec.jz_source_layers
    found["post_attention_norm"] = spec.post_attention_norm
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        found["layer_types"] = {t: layer_types.count(t) for t in dict.fromkeys(layer_types)}
    logger.info("model facts verified: %s", found)
    return found


def _check_lens(lens, spec: LensSpec, expected_layers: list[int], what: str, path: str):
    if lens.d_model != spec.d_model:
        raise ValueError(
            f"lens d_model={lens.d_model} but {spec.key} has d_model={spec.d_model} "
            f"-- wrong lens file for this checkpoint ({path})"
        )
    if lens.source_layers != expected_layers:
        # Spell the layers out when there are few: "0..30 (5 layers)" reads like a
        # contiguous range and hides that this is a sparse control fit.
        got = (
            str(lens.source_layers)
            if len(lens.source_layers) <= 8
            else f"{lens.source_layers[0]}..{lens.source_layers[-1]} "
            f"({len(lens.source_layers)} layers)"
        )
        raise ValueError(
            f"{path} has source_layers {got}, not the "
            f"0..{expected_layers[-1]} (all {len(expected_layers)}) of a {what} for "
            f"{spec.key}"
        )
    return lens


def load_lens(spec: LensSpec):
    """The block-anchored Jacobian lens ``J_l``, verified against ``spec``.

    Its prompt count is pinned too: :attr:`~src.config.LensSpec.n_prompts_fitted` is what
    every ``meta`` block cites for the reference lens, and Neuronpedia's early-stopped
    fits make that number a property of the file rather than of the request.
    """
    from jlens import JacobianLens

    path = lens_path(spec)
    lens = _check_lens(
        JacobianLens.load(path), spec, spec.source_layers, "block-anchored lens", path
    )
    if lens.n_prompts != spec.n_prompts_fitted:
        raise ValueError(
            f"{path} averaged {lens.n_prompts} prompts, but {spec.key} records "
            f"n_prompts_fitted={spec.n_prompts_fitted}"
        )
    logger.info(
        "loaded Jacobian lens %s (%d layers, n_prompts=%d)",
        path,
        len(lens.source_layers),
        lens.n_prompts,
    )
    return lens


def recorded_jz_digests(spec: LensSpec, paths: Paths | None = None) -> dict[int, str]:
    """``{n_prompts: sha256}`` of the ``J^z`` artifacts this checkout vouches for.

    :attr:`~src.config.LensSpec.jz_digests` when it is populated. Before a fresh fit's
    digest has been copied there, the merge sidecar ``scripts.merge_jz`` wrote stands in:
    it recorded the merged file's digest after verifying every shard's, so a lens read
    through it is tied to the bytes the merge produced exactly as a recorded one is.
    The fallback is logged as a warning because it is meant to be temporary -- the
    registry is what a fresh clone reads.
    """
    if spec.jz_digests:
        return dict(spec.jz_digests)
    import json

    sidecar = (paths or Paths.default()).run_dir(JZ_FIT_STAGE) / f"{spec.key}_merge.json"
    if not sidecar.exists():
        return {}
    record = json.loads(sidecar.read_text())["result"]
    logger.warning(
        "no J^z digest recorded in LensSpec for %s; taking %s from %s -- record it in "
        "LensSpec.jz_digests before publishing",
        spec.key,
        record["lens_sha256"][:12],
        sidecar,
    )
    return {int(record["prompts_fitted"]): record["lens_sha256"]}


def load_jz(
    spec: LensSpec, path: str, *, verify_digest: bool = True, paths: Paths | None = None
):
    """``J^z``, fitted by ``scripts.fit_lens`` at ``o_proj``'s input.

    Covers exactly :attr:`~src.config.LensSpec.jz_source_layers` -- every layer with a
    head anchor, which a block-anchored file never matches -- and its columns are indexed
    by ``o_proj``'s *input*, so slices are keyed to
    :attr:`~src.config.LensSpec.d_source`, not ``d_model``.

    Args:
        verify_digest: Check the file's SHA-256 against
            :func:`recorded_jz_digests`. On by default: every downstream driver cites a
            corpus size in its ``meta`` block, and without this the only thing tying that
            number to the bytes actually read is a filename.
        paths: Where the merge sidecar lives, for the fallback in
            :func:`recorded_jz_digests`.
    """
    from jlens import JacobianLens

    from src.provenance import file_digest

    if verify_digest:
        digests = recorded_jz_digests(spec, paths)
        if not digests:
            raise ValueError(
                f"no J^z digests recorded for {spec.key}; run scripts.fit_lens and "
                "scripts.merge_jz, copy the sidecar's lens_sha256 into "
                "LensSpec.jz_digests, or pass verify_digest=False for a lens that is not "
                "the shipped one"
            )
        digest = file_digest(path)
        known = {v: k for k, v in digests.items()}
        if digest not in known:
            raise ValueError(
                f"{path} has sha256 {digest}, which matches no recorded J^z artifact "
                f"for {spec.key} (have n_prompts={sorted(digests)})"
            )
        logger.info("J^z digest verified: n_prompts=%d", known[digest])

    lens = _check_lens(
        JacobianLens.load(path), spec, spec.jz_source_layers, "J^z lens", path
    )
    for layer, matrix in lens.jacobians.items():
        if tuple(matrix.shape) != (spec.d_model, spec.d_source):
            raise ValueError(
                f"{path} layer {layer}: J^z is {tuple(matrix.shape)}, expected "
                f"(d_model={spec.d_model}, d_source={spec.d_source})"
            )
    logger.info(
        "loaded J^z %s (%d layers, n_prompts=%d)",
        path,
        len(lens.source_layers),
        lens.n_prompts,
    )
    return lens
