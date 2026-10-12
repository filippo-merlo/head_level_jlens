"""The two block families the replication adds, checked on real (tiny) models in fp32.

Gemma-3 puts an RMSNorm between ``o_proj`` and the residual add; Qwen3.5 has an
``o_proj`` in one block of two (four, on the real model). Every function whose answer
depends on either fact is pinned here against the block itself -- recomputed from the
module's own forward, not from a formula this project wrote -- and the Llama fixture
runs through the same checks so the general code path is shown to reduce to the old one.
"""

from __future__ import annotations

import json

import pytest
import torch

from src.config import JZ_FIT_STAGE, LensSpec, Paths
from src.heads import (
    COMPUTE_DTYPE,
    effective_unembed,
    head_contribution,
    head_residual_writes,
    norm_gain,
    o_proj_weight,
    post_attention_gain,
    residual_o_proj_weight,
)
from src.hooks import (
    AnchorRecorder,
    has_o_proj,
    o_proj,
    post_sublayer_norm,
    pre_mlp_norm,
)
from src.jacobian import jacobian_for_prompt
from src.magnitude import ablation_kl, relative_write_norms
from src.model import check_model_facts, load_jz
from src.validation import additivity_error, wo_gate
from tests import tiny_gemma3, tiny_llama, tiny_qwen3_5

PROMPT = "the quick brown fox jumps over the lazy dog and keeps running for a while"


def _spec(fixture) -> LensSpec:
    return LensSpec(
        key=fixture.__name__,
        hf_model_id="tiny",
        lens_file="",
        n_layers=fixture.N_LAYERS,
        d_model=fixture.D_MODEL,
        n_heads=fixture.N_HEADS,
        head_dim=getattr(fixture, "HEAD_DIM", fixture.D_MODEL // fixture.N_HEADS),
        n_prompts_fitted=0,
        attention_layers=getattr(fixture, "ATTENTION_LAYERS", None),
        post_attention_norm=fixture is tiny_gemma3,
    )


@pytest.fixture(scope="module")
def llama():
    return tiny_llama.tiny_llama(), _spec(tiny_llama)


@pytest.fixture(scope="module")
def gemma():
    return tiny_gemma3.tiny_gemma3(), _spec(tiny_gemma3)


@pytest.fixture(scope="module")
def qwen():
    return tiny_qwen3_5.tiny_qwen3_5(), _spec(tiny_qwen3_5)


@pytest.fixture(params=["llama", "gemma", "qwen"])
def family(request):
    return request.getfixturevalue(request.param)


# ------------------------------------------------------------------ fixtures & facts


def test_fixtures_have_the_structure_they_stand_for(gemma, qwen, llama):
    g, _ = gemma
    assert all(hasattr(b, "pre_feedforward_layernorm") for b in g.layers)
    assert all(post_sublayer_norm(b) is not None for b in g.layers)
    q, _ = qwen
    assert [has_o_proj(b) for b in q.layers] == [
        l in tiny_qwen3_5.ATTENTION_LAYERS for l in range(tiny_qwen3_5.N_LAYERS)
    ]
    assert all(post_sublayer_norm(b) is None for b in q.layers)
    m, _ = llama
    assert all(post_sublayer_norm(b) is None for b in m.layers)
    with pytest.raises(ValueError, match="no self_attn.o_proj"):
        o_proj(q.layers[0])


def test_check_model_facts_reads_the_block_not_the_spec(family):
    model, spec = family
    found = check_model_facts(model, spec)
    assert found["attention_layers"] == spec.jz_source_layers
    assert found["num_key_value_heads"] == 2

    wrong_layers = LensSpec(**{**spec.__dict__, "attention_layers": (0, 1)})
    with pytest.raises(ValueError, match="head anchor"):
        check_model_facts(model, wrong_layers)
    wrong_norm = LensSpec(
        **{**spec.__dict__, "post_attention_norm": not spec.post_attention_norm}
    )
    with pytest.raises(ValueError, match="post_attention_norm"):
        check_model_facts(model, wrong_norm)


# ------------------------------------------------------------------ anchors


def test_attn_anchor_is_x_mid_on_every_family(family):
    """``x_mid`` = the residual after the attention add, whatever the norm order.

    Recomputed from the block's own modules: on Gemma the write is normalised first.
    """
    model, spec = family
    ids = model.encode(PROMPT)
    layer = spec.jz_source_layers[-1]
    block = model.layers[layer]
    with torch.no_grad(), AnchorRecorder(
        model.layers, zin_at=[layer], attn_at=[layer], out_at=[layer]
    ) as rec:
        model.forward(ids)
    z = rec.activations[("zin", layer)]
    x_mid = rec.activations[("attn", layer)]
    write = o_proj(block)(z)
    norm = post_sublayer_norm(block)
    if norm is not None:
        write = norm(write)
    # The block input is x_mid minus the write; check the identity the other way round
    # through the MLP branch: out = x_mid + mlp_branch(x_mid).
    mlp_in = pre_mlp_norm(block)(x_mid)
    branch = block.mlp(mlp_in)
    post_ff = getattr(block, "post_feedforward_layernorm", None)
    if post_ff is not None:
        branch = post_ff(branch)
    assert torch.allclose(x_mid + branch, rec.activations[("out", layer)], atol=1e-4)
    # And the write really is what separates x_mid from the block input.
    with torch.no_grad(), AnchorRecorder(model.layers, out_at=[layer - 1]) as prev:
        model.forward(ids)
    assert torch.allclose(prev.activations[("out", layer - 1)] + write, x_mid, atol=1e-4)


def test_additivity_holds_at_o_proj_on_every_family(family):
    model, spec = family
    errors = additivity_error(model, spec, PROMPT)
    assert set(errors) == set(spec.jz_source_layers)
    assert max(errors.values()) < 1e-5


# ------------------------------------------------------------------ gains


def test_norm_gain_reads_both_conventions(llama, gemma, qwen):
    m, _ = llama
    assert torch.allclose(norm_gain(m._final_norm), m._final_norm.weight.float(), atol=1e-5)
    for model, _ in (gemma, qwen):
        expected = 1 + model._final_norm.weight.float()
        got = norm_gain(model._final_norm)
        # ``rsqrt(1 + eps)`` on the ones vector is a single positive scalar.
        ratio = got / expected
        assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-6)
        assert abs(float(ratio[0]) - 1.0) < 1e-4


def test_effective_unembed_reproduces_model_logits(family):
    model, spec = family
    x = torch.randn(4, spec.d_model)
    reference = model.unembed(x)
    rows = effective_unembed(model)
    eps = getattr(model._final_norm, "variance_epsilon", getattr(model._final_norm, "eps", 0))
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    predicted = (x.float() / rms) @ rows.float().T
    assert torch.allclose(predicted, reference.float(), atol=1e-4)


def test_residual_o_proj_weight_is_w_o_without_a_post_norm(llama, qwen, gemma):
    for model, spec in (llama, qwen):
        layer = spec.jz_source_layers[0]
        assert post_attention_gain(model, layer) is None
        assert residual_o_proj_weight(model, layer) is o_proj_weight(model, layer)
    model, spec = gemma
    gain = post_attention_gain(model, 0)
    assert gain.shape == (spec.d_model,)
    scaled = residual_o_proj_weight(model, 0)
    assert torch.allclose(scaled, o_proj_weight(model, 0) * gain[:, None])


# ------------------------------------------------------------------ the head write


def test_head_residual_writes_reduce_to_head_contribution_without_a_norm(llama, qwen):
    for model, spec in (llama, qwen):
        layer = spec.jz_source_layers[-1]
        ids = model.encode(PROMPT)
        with torch.no_grad(), AnchorRecorder(model.layers, zin_at=[layer]) as rec:
            model.forward(ids)
        z = rec.activations[("zin", layer)][0]
        w_o = o_proj_weight(model, layer)
        writes = head_residual_writes(z, w_o, spec, norm=post_sublayer_norm(model.layers[layer]))
        for h in range(spec.n_heads):
            assert torch.equal(writes[:, h], head_contribution(z, w_o, h, spec))


def test_head_residual_writes_are_the_removal_difference_on_gemma(gemma):
    """Each head's write is what the normalised sub-layer output loses without it."""
    model, spec = gemma
    layer = 1
    block = model.layers[layer]
    ids = model.encode(PROMPT)
    with torch.no_grad(), AnchorRecorder(model.layers, zin_at=[layer]) as rec:
        model.forward(ids)
    z = rec.activations[("zin", layer)][0]
    w_o = o_proj_weight(model, layer)
    writes = head_residual_writes(z, w_o, spec, norm=post_sublayer_norm(block))
    assert writes.shape == (z.shape[0], spec.n_heads, spec.d_model)
    full = block.post_attention_layernorm(o_proj(block)(z).to(COMPUTE_DTYPE))
    for h in range(spec.n_heads):
        z_without = z.clone()
        z_without[:, spec.head_slice(h)] = 0.0
        without = block.post_attention_layernorm(o_proj(block)(z_without).to(COMPUTE_DTYPE))
        assert torch.allclose(writes[:, h], full - without, atol=1e-5)
    # Not additive any more, and visibly so: the norm is what makes Gemma different.
    assert not torch.allclose(writes.sum(1), full, atol=1e-3)


def test_relative_write_norm_is_the_in_context_write_over_the_stream(family):
    """On a pre-norm block the numerator is ``c_{l,h}``; on Gemma it is the removal
    difference of the normalised sub-layer output, not the raw ``W_O^h z_h`` that norm
    consumes (which lives on another scale entirely)."""
    model, spec = family
    layer = spec.jz_source_layers[-1]
    block = model.layers[layer]
    ids = model.encode(PROMPT)
    with torch.no_grad(), AnchorRecorder(model.layers, zin_at=[layer], attn_at=[layer]) as rec:
        model.forward(ids)
    skip = 16  # the driver's default; the fixtures' position 0 has a zero residual
    z = rec.activations[("zin", layer)][0][skip:]
    x_mid = rec.activations[("attn", layer)][0][skip:].to(COMPUTE_DTYPE)
    w_o = o_proj_weight(model, layer)
    rho = relative_write_norms(model, spec, ids, [layer], skip_first=skip)[layer]
    writes = head_residual_writes(z, w_o, spec, norm=post_sublayer_norm(block))
    expected = (writes.norm(dim=-1) / x_mid.norm(dim=-1)[:, None]).mean(0)
    assert torch.allclose(rho, expected.cpu(), rtol=1e-5, atol=1e-7)
    if post_sublayer_norm(block) is None:
        raw = torch.stack(
            [head_contribution(z, w_o, h, spec).norm(dim=-1) for h in range(spec.n_heads)], -1
        )
        assert torch.allclose(rho, (raw / x_mid.norm(dim=-1)[:, None]).mean(0).cpu(), rtol=1e-5)


def test_magnitude_statistics_on_every_family(family):
    model, spec = family
    ids = model.encode(PROMPT)
    layers = spec.jz_source_layers
    rho = relative_write_norms(model, spec, ids, layers)
    kl = ablation_kl(model, spec, ids, layers)
    for layer in layers:
        assert rho[layer].shape == (spec.n_heads,)
        assert torch.isfinite(rho[layer]).all() and (rho[layer] > 0).all()
        assert kl[layer].shape == (spec.n_heads,)
        assert torch.isfinite(kl[layer]).all() and (kl[layer] >= -1e-6).all()


# ------------------------------------------------------------------ the estimator and the gate


def test_jacobian_runs_through_deltanet_blocks(qwen):
    model, spec = qwen
    anchors = [("zin", l) for l in spec.jz_source_layers]
    jacobians, _, n_valid = jacobian_for_prompt(
        model, PROMPT, anchors=anchors, d_source=spec.d_source, dim_batch=4, skip_first=2
    )
    assert n_valid > 0
    for anchor in anchors:
        j = jacobians[anchor]
        assert j.shape == (spec.d_model, spec.d_source)
        assert torch.isfinite(j).all() and j.abs().sum() > 0


def test_wo_gate_uses_the_supplied_reference():
    """Row cosine ignores a positive per-row scale, so the gain reference acts only
    through its sign -- which is exactly what a ``1 + w`` gain can flip."""
    g = torch.tensor([-1.0, -1.0, -1.0, -1.0, 0.5, 2.0])
    w = torch.randn(6, 9)
    scaled = g[:, None] * w
    gate = wo_gate(scaled, scaled, reference="diag(g_post) W_O")
    assert gate["reference"] == "diag(g_post) W_O"
    assert abs(gate["median_row_cosine_vs_W_O"] - 1.0) < 1e-6
    assert gate["median_row_cosine_vs_identity"] is None and gate["PASS"]
    assert wo_gate(scaled, w)["median_row_cosine_vs_W_O"] < 0


# ------------------------------------------------------------------ loading the lens


def test_load_jz_falls_back_to_the_merge_sidecar(qwen, tmp_path, monkeypatch):
    from jlens import JacobianLens

    from src.provenance import file_digest

    model, spec = qwen
    matrices = {l: torch.zeros(spec.d_model, spec.d_source) for l in spec.jz_source_layers}
    path = tmp_path / "lens.pt"
    JacobianLens(matrices, n_prompts=3, d_model=spec.d_model).save(str(path))
    paths = Paths(results_root=tmp_path)

    with pytest.raises(ValueError, match="no J\\^z digests"):
        load_jz(spec, str(path), paths=paths)

    sidecar = paths.run_dir(JZ_FIT_STAGE) / f"{spec.key}_merge.json"
    sidecar.write_text(json.dumps({"result": {"prompts_fitted": 3, "lens_sha256": "0" * 64}}))
    with pytest.raises(ValueError, match="matches no recorded"):
        load_jz(spec, str(path), paths=paths)

    sidecar.write_text(
        json.dumps({"result": {"prompts_fitted": 3, "lens_sha256": file_digest(path)}})
    )
    lens = load_jz(spec, str(path), paths=paths)
    assert lens.source_layers == spec.jz_source_layers

    # A dense-layer file is not a J^z for a hybrid, whatever its digest says.
    dense = {l: torch.zeros(spec.d_model, spec.d_source) for l in range(spec.n_layers)}
    JacobianLens(dense, n_prompts=3, d_model=spec.d_model).save(str(path))
    sidecar.write_text(
        json.dumps({"result": {"prompts_fitted": 3, "lens_sha256": file_digest(path)}})
    )
    with pytest.raises(ValueError, match="source_layers"):
        load_jz(spec, str(path), paths=paths)


# ------------------------------------------------------------------ the model's own logits


def test_final_residual_unembeds_to_the_model_logits(family):
    """``unembed(final_residual(...))`` is the HF model's own ``logits``.

    ``last_hidden_state`` is post-norm, so unembedding it applies the final norm twice;
    this pins the pre-norm block output as the only thing ``unembed`` may take.
    """
    from src.model import final_residual

    model, spec = family
    ids = model.encode(PROMPT)
    with torch.no_grad():
        reference = model.hf_model(input_ids=ids).logits[0].float()
        ours = model.unembed(final_residual(model, ids)[0]).float()
        doubled = model.unembed(model.forward(ids).last_hidden_state[0]).float()
    assert torch.allclose(ours, reference, atol=1e-3)
    gain = norm_gain(model._final_norm)
    if not torch.allclose(gain, gain[0].expand_as(gain)):
        # A uniform gain makes the double norm invisible; the fixtures with a
        # per-dimension gain show what it does.
        assert not torch.allclose(doubled, reference, atol=1e-3)
