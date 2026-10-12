"""A tiny but *real* Gemma-3 for CPU tests: the block with a norm on the attention write.

``Gemma3DecoderLayer`` differs from Llama's in the one place this project hooks: the
attention output passes through ``post_attention_layernorm`` *before* the residual add,
and the pre-MLP norm is ``pre_feedforward_layernorm``. Everything that reads ``x_mid`` or
calls a head's write "additive" has to be exercised against that order, and only a real
Gemma block exercises it. RMSNorm here is ``x * (1 + w)`` with ``w`` initialised to zero,
which is the other convention :func:`src.heads.norm_gain` has to read correctly.

Same shape and seeding conventions as :mod:`tests.tiny_llama`; ``n_heads * head_dim``
is deliberately not ``d_model``, as on Gemma-3-4B.
"""

from __future__ import annotations

import torch

from tests.tiny_llama import ByteTokenizer

D_MODEL = 32
N_LAYERS = 3
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 6
VOCAB_SIZE = 64
LAYER_TYPES = ("sliding_attention", "sliding_attention", "full_attention")


def tiny_gemma3(*, seed: int = 0, sublayer_gain: float = 3.0):
    """A seeded ``Gemma3ForCausalLM`` on CPU in fp32, wrapped by :func:`jlens.from_hf`."""
    import jlens
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(seed)
    config = Gemma3TextConfig(
        hidden_size=D_MODEL,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=2 * D_MODEL,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=128,
        sliding_window=8,
        layer_types=list(LAYER_TYPES),
        attention_bias=False,
        tie_word_embeddings=True,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        attn_implementation="eager",
    )
    hf_model = Gemma3ForCausalLM(config).to(torch.float32).eval()

    with torch.no_grad():
        for layer in hf_model.model.layers:
            layer.self_attn.o_proj.weight.mul_(sublayer_gain)
            layer.mlp.down_proj.weight.mul_(sublayer_gain)
            # Zero-initialised ``1 + w`` gains make every norm the identity up to
            # scale, which is exactly the case a gain-convention bug would survive.
            # Give the post-attention and final norms visible per-dimension gains.
            layer.post_attention_layernorm.weight.copy_(
                torch.linspace(-0.5, 1.5, D_MODEL)
            )
        hf_model.model.norm.weight.copy_(torch.linspace(-0.3, 0.9, D_MODEL))

    model = jlens.from_hf(hf_model, ByteTokenizer(), force_bos=False)
    model.hf_model = hf_model
    return model
