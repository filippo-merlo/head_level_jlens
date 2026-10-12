"""A tiny but *real* Llama for CPU tests.

Upstream's ``jlens.tests.tiny.TinyDecoder`` is a plain residual stack: it has no
attention sub-layer, so it cannot exercise either of this project's hook points
(``post_attention_layernorm``'s input, ``self_attn.o_proj``'s input). We instead build a
genuine :class:`~transformers.LlamaForCausalLM` from a small config, so every test runs
the production code path -- real RMSNorm, real RoPE, real GQA (4 query heads over 2 KV
heads, matching Llama-3.1-8B's 4:1 ratio), real ``o_proj``.

Weights are random but seeded, and deliberately *not* tiny: default init gives an
attention output so small that ``h_attn ~= h_in`` and a bug in the anchor would still
pass. :func:`tiny_llama` scales ``o_proj`` and the MLP up so each sub-layer visibly
moves the residual stream.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

#: Small enough to be fast, big enough that ``n_heads * head_dim == d_model`` is a real
#: constraint and RMSNorm averages over more than a handful of dimensions.
D_MODEL = 32
N_LAYERS = 3
N_HEADS = 4
N_KV_HEADS = 2
VOCAB_SIZE = 64


class ByteTokenizer:
    """Deterministic byte-level tokenizer -- just enough surface for ``LensModel``.

    A real tokenizer would mean a download; the fitter and the readouts never look at
    token *identity*, only at counts and positions.
    """

    bos_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 128,
        add_special_tokens: bool = True,
    ):
        """``add_special_tokens=False`` returns a bare id list, as HF tokenizers do.

        That form is what ``src.readout``'s target/direction helpers consume, and with it
        the fixture can exercise :func:`src.tasks.build_trials` -- which turns words into
        vocabulary ids -- rather than only the tensor path the fitter uses.
        """
        ids = [
            1 + (byte % (VOCAB_SIZE - 1)) for byte in text.encode()
        ][: max_length - 1]
        if not add_special_tokens:
            return SimpleNamespace(input_ids=ids)
        return SimpleNamespace(input_ids=torch.tensor([[self.bos_token_id, *ids]]))

    def decode(self, ids, **_kwargs) -> str:
        return "".join(chr(96 + int(i)) for i in ids)


def tiny_llama(*, seed: int = 0, sublayer_gain: float = 3.0, head_dim: int | None = None):
    """A seeded ``LlamaForCausalLM`` on CPU in fp32, wrapped by :func:`jlens.from_hf`.

    Args:
        seed: Torch seed for weight init.
        sublayer_gain: Multiplier on ``o_proj`` and ``mlp.down_proj``. The point is to
            make each sub-layer's contribution to the residual stream large enough that
            an anchor or decomposition bug cannot hide inside numerical noise.
        head_dim: Per-head width. ``None`` gives the default
            ``D_MODEL // N_HEADS``, for which ``n_heads * head_dim == d_model`` -- the
            same coincidence Llama-3.1-8B has. Pass something else (e.g. ``6``) to build
            a model where ``d_source != d_model``, which is the only way to catch code
            that conflates the two. See ``tests/test_heads.py``.

    Returns:
        A :class:`jlens.protocol.LensModel`; the raw HF model is at ``.hf_model``.
    """
    import jlens
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        hidden_size=D_MODEL,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        intermediate_size=2 * D_MODEL,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=128,
        attention_bias=False,
        tie_word_embeddings=False,
        # eager so tests can also ask for attention weights; the fitter itself is
        # implementation-agnostic because it uses reverse-mode AD only.
        attn_implementation="eager",
        **({} if head_dim is None else {"head_dim": head_dim}),
    )
    hf_model = LlamaForCausalLM(config).to(torch.float32).eval()

    with torch.no_grad():
        for layer in hf_model.model.layers:
            layer.self_attn.o_proj.weight.mul_(sublayer_gain)
            layer.mlp.down_proj.weight.mul_(sublayer_gain)

    model = jlens.from_hf(hf_model, ByteTokenizer(), force_bos=False)
    model.hf_model = hf_model
    return model
