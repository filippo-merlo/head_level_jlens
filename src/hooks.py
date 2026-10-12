"""Recording residual-stream and head-space activations at three anchors.

``jlens.hooks.ActivationRecorder`` records *block outputs* only. This project needs two
finer points inside a block, both of which exist as a module's **input** rather than its
output (``LlamaDecoderLayer.forward``)::

    hidden_states = self.self_attn(...)          #  <- o_proj's input is `zin`
    hidden_states = residual + hidden_states      #  `attn`: post-attention, pre-MLP
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)   # <- pre-hook here

Gemma-3 orders its norms differently (``Gemma3DecoderLayer.forward``)::

    hidden_states = self.self_attn(...)                            #  <- `zin` as above
    hidden_states = self.post_attention_layernorm(hidden_states)   #  normalises the write
    hidden_states = residual + hidden_states                       #  `attn`: x_mid
    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)  # <- pre-hook here

so ``x_mid`` is the input of whichever norm feeds the MLP -- :func:`pre_mlp_norm` picks it
structurally -- and on Gemma the norm between ``o_proj`` and the residual add,
:func:`post_sublayer_norm`, is what makes a head's write more than ``W_O^h z_h``.
Both anchors use ``register_forward_pre_hook``. The three anchors, in execution order
within a block, and what each is for:

===========  ==================================  =========================================
Anchor       Point                               Used for
===========  ==================================  =========================================
``zin``      ``o_proj``'s input, ``z_l``          per-head readout; ``J^z``'s anchor
``attn``     post-attention/pre-MLP, ``x_mid,l``  ``||x_mid||``; the consistency reference
``out``      block output, ``x_{l+1}``            the Jacobian lens's anchor; fit target
===========  ==================================  =========================================

``zin`` is the only anchor whose width is ``d_source = n_heads * head_dim`` rather than
``d_model``. Those coincide on Llama-3.1-8B and the code never assumes it.

Restricted to blocks with ``self_attn.o_proj`` and a norm feeding the MLP, and it says so
loudly rather than guessing: a wrong anchor produces a plausible-looking lens, not an
error. A Qwen3.5 Gated DeltaNet block has neither an ``o_proj`` nor a head, and asking
for its ``zin`` raises.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import torch
from torch import nn

#: ``("zin", l)`` is ``o_proj``'s input in block ``l``; ``("attn", l)`` the
#: post-attention/pre-MLP point; ``("out", l)`` the block's output.
AnchorKind = Literal["zin", "attn", "out"]
Anchor = tuple[AnchorKind, int]

#: Execution order of the kinds within one block.
_KIND_ORDER: dict[str, int] = {"zin": 0, "attn": 1, "out": 2}


def _order(anchor: Anchor) -> tuple[int, int]:
    """Execution order of an anchor in the forward pass: block, then sub-layer point."""
    kind, layer = anchor
    return (layer, _KIND_ORDER[kind])


def has_o_proj(block: nn.Module) -> bool:
    """Whether the block has a softmax-attention sub-layer with an output projection."""
    attn = getattr(block, "self_attn", None)
    return attn is not None and getattr(attn, "o_proj", None) is not None


def o_proj(block: nn.Module) -> nn.Module:
    """The attention output projection of one block.

    Isolated in one function because the name is family-specific -- ``self_attn.o_proj``
    on Llama/Qwen/Gemma/Mistral, ``attn.c_proj`` on GPT-2 -- so retargeting the project
    at another architecture touches exactly here.
    """
    if not has_o_proj(block):
        raise ValueError(
            f"block {type(block).__name__} has no self_attn.o_proj; the head-space "
            "anchor is defined for softmax-attention blocks only"
        )
    return block.self_attn.o_proj


def pre_mlp_norm(block: nn.Module) -> nn.Module:
    """The norm whose input is ``x_mid``: the residual after the attention add.

    ``post_attention_layernorm`` on Llama-style blocks; on Gemma-3 that name belongs to
    the norm *inside* the attention branch, and the pre-MLP one is
    ``pre_feedforward_layernorm``. Chosen by structure, not by family name, so a block
    that has the Gemma norm is read the Gemma way whatever it is called.
    """
    norm = getattr(block, "pre_feedforward_layernorm", None) or getattr(
        block, "post_attention_layernorm", None
    )
    if norm is None:
        raise ValueError(
            f"block {type(block).__name__} has neither pre_feedforward_layernorm nor "
            "post_attention_layernorm; the x_mid anchor is undefined for it"
        )
    return norm


def post_sublayer_norm(block: nn.Module) -> nn.Module | None:
    """The norm between ``o_proj``'s output and the residual add, if the block has one.

    ``None`` on plain pre-norm blocks (Llama, Qwen), where ``o_proj``'s output *is* the
    write. On Gemma-3 it is ``post_attention_layernorm``, identified by the presence of
    ``pre_feedforward_layernorm`` beside it -- the same structural fact
    :func:`pre_mlp_norm` keys on, so the two can never disagree about a block.
    """
    if getattr(block, "pre_feedforward_layernorm", None) is None:
        return None
    return block.post_attention_layernorm


class AnchorRecorder:
    """Capture activations at ``zin`` / ``attn`` / ``out`` anchors.

    Mirrors :class:`jlens.hooks.ActivationRecorder`: hooks are installed on
    ``__enter__`` and removed on ``__exit__``, captured tensors are **not** detached so
    they can be handed to :func:`torch.autograd.grad`, and ``root_graph`` reproduces
    upstream's trick of making the earliest captured tensor the autograd graph's leaf.

    That trick relies on every model parameter having ``requires_grad=False`` (which
    :func:`jlens.from_hf` enforces): the anchor tensor is then a leaf, and marking it
    bounds the retained graph to the part of the model actually differentiated.

    For **readout** -- which is all the downstream per-head pipeline needs -- call this
    under :func:`torch.no_grad` with ``root_graph=False``. Nothing then requires grad and
    the capture is an ordinary detached one; no graph is built and no rooting subtlety
    applies. Rooting is only needed when fitting, in :mod:`src.jacobian`.

    Args:
        blocks: The residual blocks, e.g. ``model.layers``.
        zin_at: Layers to capture ``o_proj``'s input at.
        attn_at: Layers to capture the post-attention/pre-MLP anchor at.
        out_at: Layers to capture the block output at.
        root_graph: Mark the *earliest* requested anchor as the graph root. Automatic
            rather than caller-specified so it cannot be pointed at an anchor that some
            other requested anchor precedes, which would silently exclude the latter
            from the graph.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        *,
        zin_at: Iterable[int] = (),
        attn_at: Iterable[int] = (),
        out_at: Iterable[int] = (),
        root_graph: bool = False,
    ) -> None:
        self._blocks = blocks
        anchors: list[Anchor] = [("zin", l) for l in sorted(set(zin_at))]
        anchors += [("attn", l) for l in sorted(set(attn_at))]
        anchors += [("out", l) for l in sorted(set(out_at))]
        if not anchors:
            raise ValueError(
                "AnchorRecorder needs at least one of zin_at / attn_at / out_at"
            )
        out_of_range = sorted(l for _, l in anchors if not 0 <= l < len(blocks))
        if out_of_range:
            raise ValueError(
                f"layers {out_of_range} out of range for {len(blocks)} blocks"
            )
        self.anchors = sorted(anchors, key=_order)
        self._root = self.anchors[0] if root_graph else None
        self.activations: dict[Anchor, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _store(self, anchor: Anchor, tensor: torch.Tensor) -> None:
        if anchor == self._root:
            if not tensor.is_leaf:
                raise RuntimeError(
                    f"anchor {anchor} is not a leaf tensor, so it cannot root the "
                    "autograd graph -- something upstream of it requires grad "
                    "(are the model parameters frozen? is grad enabled?)"
                )
            tensor.requires_grad_(True)
        self.activations[anchor] = tensor

    def __enter__(self) -> AnchorRecorder:
        try:
            for anchor in self.anchors:
                kind, layer = anchor
                block = self._blocks[layer]
                if kind == "out":
                    handle = block.register_forward_hook(
                        lambda _module, _args, output, _a=anchor: self._store(
                            _a, output if torch.is_tensor(output) else output[0]
                        )
                    )
                else:
                    # Both sub-layer anchors are module *inputs*: o_proj's input is the
                    # concatenated head results; the pre-MLP norm's input is x_mid (the
                    # residual add has happened, the MLP has not).
                    target = o_proj(block) if kind == "zin" else pre_mlp_norm(block)
                    handle = target.register_forward_pre_hook(
                        lambda _module, args, _a=anchor: self._store(_a, args[0])
                    )
                self._handles.append(handle)
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class ResidualPatcher:
    """Rewrite a block's **output** at chosen layers, mid-forward.

    The residual-stream counterpart of :class:`ZinPatcher`, and the one ``layer_wide``
    needs: the Jacobian lens is anchored at block outputs, so its lens coordinates live at
    ``x_{l+1}`` and an intervention in those coordinates has to be applied there.

    A block may return a bare tensor or a tuple whose first element is the hidden state;
    both are handled, and the tuple's remaining elements are passed through untouched.

    Args:
        edits: ``{layer: fn}``, where ``fn(x) -> x`` receives ``[batch, seq, d_model]``.
    """

    def __init__(self, blocks: Sequence[nn.Module], edits: dict[int, object]) -> None:
        out_of_range = sorted(l for l in edits if not 0 <= l < len(blocks))
        if out_of_range:
            raise ValueError(f"layers {out_of_range} out of range for {len(blocks)} blocks")
        self._blocks = blocks
        self._edits = edits
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, edit):
        def hook(_module, _args, output):
            is_tuple = not torch.is_tensor(output)
            hidden = output[0] if is_tuple else output
            edited = edit(hidden)
            if edited.shape != hidden.shape:
                raise ValueError(
                    f"edit returned shape {tuple(edited.shape)}, expected "
                    f"{tuple(hidden.shape)}"
                )
            return (edited, *output[1:]) if is_tuple else edited

        return hook

    def __enter__(self) -> ResidualPatcher:
        try:
            for layer, edit in self._edits.items():
                self._handles.append(
                    self._blocks[layer].register_forward_hook(self._make_hook(edit))
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class ZinPatcher:
    """Rewrite ``o_proj``'s input at chosen layers, mid-forward.

    The write side of the ``zin`` anchor. A forward pre-hook that *returns* a value
    replaces the module's arguments, so the layer's attention write becomes
    ``W_O @ edit(z_l)`` and everything downstream sees the edited stream -- which is what
    makes an intervention causal rather than a readout.

    Used by two things: the finite-difference gradient check (perturb one entry of
    ``z_{l,h}`` and watch the final residual move), and the lens-coordinate swap.

    Args:
        blocks: The residual blocks, e.g. ``model.layers``.
        edits: ``{layer: fn}``, where ``fn(z) -> z`` receives ``[batch, seq, d_source]``
            and must return the same shape. Return a *new* tensor rather than mutating
            in place; the input is an autograd-visible activation.
    """

    def __init__(self, blocks: Sequence[nn.Module], edits: dict[int, object]) -> None:
        out_of_range = sorted(l for l in edits if not 0 <= l < len(blocks))
        if out_of_range:
            raise ValueError(f"layers {out_of_range} out of range for {len(blocks)} blocks")
        self._blocks = blocks
        self._edits = edits
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, edit):
        def pre_hook(_module, args):
            z = args[0]
            edited = edit(z)
            if edited.shape != z.shape:
                raise ValueError(
                    f"edit returned shape {tuple(edited.shape)}, expected "
                    f"{tuple(z.shape)}"
                )
            return (edited, *args[1:])

        return pre_hook

    def __enter__(self) -> ZinPatcher:
        try:
            for layer, edit in self._edits.items():
                self._handles.append(
                    o_proj(self._blocks[layer]).register_forward_pre_hook(
                        self._make_hook(edit)
                    )
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
