"""The readout arms and their ranking machinery, on CPU.

The arms are cheap to get subtly wrong -- a head slice taken on
the wrong axis, a rank that counts ties as wins -- and every such error yields plausible
vocabulary output rather than a crash. These pin the semantics.
"""

from __future__ import annotations

import pytest
import torch

from src.config import LensSpec
from src.heads import head_contribution, o_proj_weight
from src.hooks import AnchorRecorder
from src.readout import (
    HEAD_ARMS,
    all_heads_transported,
    best_rank,
    head_logits,
    head_transported,
    token_ranks,
    top_tokens,
)
from tests.tiny_llama import D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE, tiny_llama

PROMPT = "the quick brown fox jumps over the lazy dog and keeps running for a while"


def _spec() -> LensSpec:
    return LensSpec(
        key="tiny",
        hf_model_id="tiny",
        lens_file="",
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim=D_MODEL // N_HEADS,
        n_prompts_fitted=0,
    )


@pytest.fixture(scope="module")
def model():
    return tiny_llama()


@pytest.fixture(scope="module")
def activations(model):
    spec = _spec()
    input_ids = model.encode(PROMPT)
    with torch.no_grad(), AnchorRecorder(model.layers, zin_at=[1]) as recorder:
        model.forward(input_ids)
    z = recorder.activations[("zin", 1)][0, 5].float()
    torch.manual_seed(0)
    jz = torch.randn(spec.d_model, spec.d_source) / spec.d_model**0.5
    return z, jz






# ---------------------------------------------------------------------------- arms


def test_head_arms_are_transport_and_identity():
    assert HEAD_ARMS == ("jhead", "logithead")


def test_arms_read_the_same_head(model, activations):
    """``jhead`` and ``logithead`` differ by a transport, not by which vector they read.

    The central identity in code: ``J_{l,h} z_{l,h}`` and ``c_{l,h}`` are the same
    write, one transported and one not. If ``jz`` is built as ``J_attn @ W_O`` then
    transporting ``c`` by ``J_attn`` must reproduce the ``jhead`` arm exactly.
    """
    spec = _spec()
    z, _ = activations
    torch.manual_seed(1)
    j_attn = torch.randn(spec.d_model, spec.d_model) / spec.d_model**0.5
    w_o = o_proj_weight(model, 1).float()
    jz = j_attn @ w_o

    for head in range(spec.n_heads):
        via_jhead = head_transported(jz, z, head, spec, arm="jhead")
        c = head_contribution(z, w_o, head, spec)
        via_logithead = head_transported(jz, z, head, spec, arm="logithead", w_o=w_o)
        assert torch.allclose(via_logithead, c, atol=1e-6)
        assert torch.allclose(via_jhead, c @ j_attn.T, atol=1e-4), f"head {head}"


def test_arms_produce_distinct_readouts(model, activations):
    spec = _spec()
    z, jz = activations
    logits = {
        arm: head_logits(
            model, jz, z, 1, spec, arm=arm, layer=1
        )
        for arm in HEAD_ARMS
    }
    for arm, value in logits.items():
        assert value.shape == (VOCAB_SIZE,), arm
        assert torch.isfinite(value).all(), arm
    assert not torch.allclose(logits["jhead"], logits["logithead"])


def test_arms_require_their_own_inputs(model, activations):
    spec = _spec()
    z, jz = activations
    with pytest.raises(ValueError, match="w_o"):
        head_transported(jz, z, 0, spec, arm="logithead")
    with pytest.raises(ValueError, match="unknown head arm"):
        head_transported(jz, z, 0, spec, arm="nonsense")


def test_head_readout_only_sees_its_own_head(model, activations):
    """Zeroing head ``h``'s slice of ``z`` must change only head ``h``'s readout."""
    spec = _spec()
    z, jz = activations
    target = 2
    perturbed = z.clone()
    perturbed[spec.head_slice(target)] = 0.0

    for head in range(spec.n_heads):
        before = head_logits(model, jz, z, head, spec, arm="jhead")
        after = head_logits(model, jz, perturbed, head, spec, arm="jhead")
        if head == target:
            assert not torch.allclose(before, after)
        else:
            assert torch.allclose(before, after, atol=1e-5), f"head {head} leaked"


def test_all_heads_matches_per_head(model, activations):
    """The batched path must equal the per-head path, arm for arm and head for head.

    The batched form reshapes the transport into ``[d_model, n_heads, head_dim]``, which
    silently assumes head ``h`` owns columns ``h*head_dim : (h+1)*head_dim`` in exactly
    the order ``spec.head_slice`` uses. Getting that wrong permutes heads without
    changing any shape.
    """
    spec = _spec()
    z, jz = activations
    w_o = o_proj_weight(model, 1).float()

    for arm in HEAD_ARMS:
        batched = all_heads_transported(
            jz, z, spec, arm=arm, w_o=w_o
        )
        assert batched.shape == (spec.n_heads, spec.d_model)
        for head in range(spec.n_heads):
            single = head_transported(
                jz, z, head, spec, arm=arm, w_o=w_o
            )
            assert torch.allclose(batched[head], single, atol=1e-5), f"{arm} head {head}"


def test_all_heads_rejects_batched_positions(model, activations):
    z, jz = activations
    with pytest.raises(ValueError, match="one position"):
        all_heads_transported(jz, z.unsqueeze(0), _spec(), arm="jhead")


# -------------------------------------------------------------------------- ranks


def test_token_ranks_counts_strictly_greater():
    logits = torch.tensor([3.0, 1.0, 2.0, 0.0])
    assert token_ranks(logits, [0]).tolist() == [0]
    assert token_ranks(logits, [2]).tolist() == [1]
    assert token_ranks(logits, [1]).tolist() == [2]
    assert token_ranks(logits, [3]).tolist() == [3]
    assert best_rank(logits, [3, 2]) == 1


def test_token_ranks_ties_do_not_inflate():
    """Tied logits must not each count as beating the other.

    Counting ``>=`` instead of ``>`` would silently demote every target that ties with
    anything, which on a lens readout is common at low-signal layers.
    """
    logits = torch.tensor([1.0, 1.0, 1.0, 0.0])
    assert token_ranks(logits, [0, 1, 2]).tolist() == [0, 0, 0]


def test_token_ranks_rejects_batched_logits():
    with pytest.raises(ValueError, match="single position"):
        token_ranks(torch.zeros(2, 4), [0])


def test_top_tokens_is_ordered(model, activations):
    z, jz = activations
    logits = head_logits(model, jz, z, 0, _spec(), arm="jhead")
    tokens = top_tokens(logits, model.tokenizer, k=5)
    assert len(tokens) == 5
    scores = [score for _, score in tokens]
    assert scores == sorted(scores, reverse=True)
