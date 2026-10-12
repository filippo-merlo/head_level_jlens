"""A tiny but *real* Qwen3.5 for CPU tests: the hybrid stack with attention every other block.

Qwen3.5 interleaves Gated DeltaNet blocks, which have ``linear_attn.out_proj`` and no
``self_attn``, with softmax-attention blocks whose ``o_proj`` input is gated. Two things
only a real hybrid exercises: that every driver anchors at
:attr:`~src.config.LensSpec.attention_layers` and nowhere else, and that the estimator's
backward pass runs through a DeltaNet block (the pure-torch fallback, since no fused
kernel is installed here or on the GPU nodes). RMSNorm is the ``1 + w`` convention.

The real model puts attention at every fourth block; two of four keeps the fixture small
while still having a DeltaNet block on both sides of an attention block.
"""

from __future__ import annotations

import torch

from tests.tiny_llama import ByteTokenizer

D_MODEL = 32
N_LAYERS = 4
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 8
VOCAB_SIZE = 64
LAYER_TYPES = ("linear_attention", "full_attention", "linear_attention", "full_attention")
ATTENTION_LAYERS = (1, 3)


def tiny_qwen3_5(*, seed: int = 0, sublayer_gain: float = 3.0):
    """A seeded ``Qwen3_5ForCausalLM`` on CPU in fp32, wrapped by :func:`jlens.from_hf`."""
    import jlens
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(
        hidden_size=D_MODEL,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=2 * D_MODEL,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=128,
        layer_types=list(LAYER_TYPES),
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        attention_bias=False,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    hf_model = Qwen3_5ForCausalLM(config).to(torch.float32).eval()

    with torch.no_grad():
        for layer in hf_model.model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.o_proj.weight.mul_(sublayer_gain)
            layer.mlp.down_proj.weight.mul_(sublayer_gain)
        hf_model.model.norm.weight.copy_(torch.linspace(-0.3, 0.9, D_MODEL))

    model = jlens.from_hf(hf_model, ByteTokenizer(), force_bos=False)
    model.hf_model = hf_model
    return model
