"""How much a head's write actually matters -- the statistics that make legibility mean something.

RMSNorm is scale-invariant: ``norm(a x) == norm(x)`` for ``a > 0``. So reading a head's
contribution alone tells you the *direction* it pushes the residual stream and nothing
about how hard. A head whose write is negligible next to the stream it joins still
produces a confident-looking top-1 readout, and will happily score on pass@k. That is
the single largest interpretive trap in the design, and it is why **every** legibility
number is reported with a magnitude statistic beside it.

Two of them:

* **Relative write norm** ``rho_{l,h} = E[ ||c_{l,h}|| / ||x_mid,l|| ]`` -- how large the
  head's write is against the residual stream it is being added into. Cheap: forward
  passes only.
* **Ablation KL** -- the KL between the model's next-token distribution with and without
  head ``h`` zeroed at layer ``l``. Measures what the *rest of the network* does with the
  write, which a norm ratio cannot: a small write into a direction later components read
  carefully matters more than a large one nothing attends to.

Neither uses a lens readout, which is why they can sit in the same table as one: a
magnitude statistic derived from readout quality would be a restatement of the number it
is supposed to qualify.
"""

from __future__ import annotations

import logging

import torch

from src.config import SKIP_FIRST_N_POSITIONS
from src.heads import COMPUTE_DTYPE, head_residual_writes, o_proj_weight
from src.hooks import AnchorRecorder, ZinPatcher, post_sublayer_norm
from src.model import final_residual

logger = logging.getLogger(__name__)


def relative_write_norms(
    model,
    spec,
    input_ids: torch.Tensor,
    layers: list[int],
    *,
    position: int | None = None,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> dict[int, torch.Tensor]:
    """``||write_{l,h}|| / ||x_mid,l||`` for every head at every requested layer.

    The numerator is what the head adds to the residual stream in context
    (:func:`src.heads.head_residual_writes`): ``c_{l,h}`` on a pre-norm block, and on a
    block that normalises the sub-layer output before the add (Gemma-3) the change in
    that normalised output when the head is removed. The raw ``c_{l,h}`` would be the
    *input* of that norm there, on a scale the residual stream never sees.

    Args:
        position: Score this single position; ``None`` averages over the valid ones.
        skip_first: Positions below this index are dropped when averaging, mirroring
            ``jlens.fitting.SKIP_FIRST_N_POSITIONS`` and every other readout in this
            project. They act as attention sinks and were excluded when the lenses were
            estimated. It matters specifically here because Llama's sink positions carry
            anomalously large ``||x_mid,l||``, which is this ratio's *denominator*, so
            including them deflates rho by an unknown amount. Ignored when ``position``
            is given, since that names an already-chosen readout position.

    Returns:
        ``{layer: [n_heads]}`` -- the ratio, not yet averaged over prompts.
    """
    with torch.no_grad(), AnchorRecorder(
        model.layers, zin_at=layers, attn_at=layers
    ) as recorder:
        model.forward(input_ids)
        out: dict[int, torch.Tensor] = {}
        for layer in layers:
            z = recorder.activations[("zin", layer)][0]
            x_mid = recorder.activations[("attn", layer)][0].to(COMPUTE_DTYPE)
            if position is not None:
                z, x_mid = z[position : position + 1], x_mid[position : position + 1]
            elif skip_first and z.shape[0] > skip_first:
                z, x_mid = z[skip_first:], x_mid[skip_first:]
            w_o = o_proj_weight(model, layer)
            stream = x_mid.norm(dim=-1)
            writes = head_residual_writes(
                z, w_o, spec, norm=post_sublayer_norm(model.layers[layer])
            )
            out[layer] = (writes.norm(dim=-1) / stream[:, None]).mean(dim=0).cpu()
    return out


def ablation_kl(
    model, spec, input_ids: torch.Tensor, layers: list[int], *, position: int = -1
) -> dict[int, torch.Tensor]:
    """KL(clean || head-ablated) of the next-token distribution, per head.

    All ``n_heads`` ablations at a layer run in **one** forward pass: the prompt is
    replicated across the batch and batch element ``h`` has head ``h``'s slice of ``z_l``
    zeroed. Ablations are independent given the layer, so this is exact, and it turns
    ``n_heads`` forward passes per layer into one.

    Zeroing ``z_{l,h}`` rather than ``c_{l,h}`` is the same intervention -- ``c`` is
    ``W_O^h z``, so a zero slice contributes nothing to the sum -- and it is the one that
    can be expressed as a single edit to ``o_proj``'s input.

    Returns:
        ``{layer: [n_heads]}`` KL in nats.
    """
    with torch.no_grad():
        clean = model.unembed(final_residual(model, input_ids)[0, position])
        clean_logprobs = torch.log_softmax(clean.float(), dim=-1)

        out: dict[int, torch.Tensor] = {}
        batched = input_ids.expand(spec.n_heads, -1)
        for layer in layers:

            def zero_one_head_per_batch_element(z):
                edited = z.clone()
                for head in range(spec.n_heads):
                    edited[head, :, spec.head_slice(head)] = 0.0
                return edited

            with ZinPatcher(model.layers, {layer: zero_one_head_per_batch_element}):
                ablated = final_residual(model, batched)[:, position]
            ablated_logprobs = torch.log_softmax(
                model.unembed(ablated).float(), dim=-1
            )
            out[layer] = (
                (clean_logprobs.exp() * (clean_logprobs - ablated_logprobs))
                .sum(dim=-1)
                .cpu()
            )
    return out
