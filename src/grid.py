"""Readout grids: the same object read across two axes.

Every driver reads one cell at a time and reduces immediately -- a rank, a pass@k, a
verdict. Looking at the lens instead means keeping the whole grid: what did *every*
(layer, head) say at this position, and what did the Jacobian lens say at every
(layer, position)? That is what :mod:`notebooks.head_lens_explorer` draws, and this module
is the numbers behind it, with no plotting.

Two axes, one per lens family:

===================  =========================  ====================================
Function             Grid                       Transport
===================  =========================  ====================================
:func:`head_grid`    layer x **head**           ``J_{l,h}``, at one position
:func:`layer_grid`   layer x **position**       the Jacobian lens ``J_l``, whole residual
===================  =========================  ====================================

Neither introduces algebra. :func:`head_grid` is the arm sweep the readout endpoint runs
inline, lifted out so the notebook reads through the same code path;
:func:`layer_grid` is :func:`src.readout.layer_logits` swept over positions, which is the paper's own slice view (``jlens.vis.compute_slice``) restricted to
the parts a static figure can show.

A grid is one forward pass. Both functions take ``input_ids`` and run the model
themselves rather than accepting activations, because the two families need *different*
anchors (``zin`` vs ``out``) and threading a recorder through would make the caller
responsible for a correspondence that is easy to get silently wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from src.config import LensSpec
from src.heads import o_proj_weight
from src.hooks import AnchorRecorder
from src.readout import Arm, all_heads_transported, layer_logits, token_ranks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Grid:
    """A ``[n_rows, n_cols]`` sweep of lens readouts.

    Attributes:
        rows: Layer index of each row.
        cols: Head index (:func:`head_grid`) or position index (:func:`layer_grid`) of
            each column. Positions are absolute indices into the prompt, so a windowed
            grid still labels itself with where in the prompt it came from.
        top_ids: ``[n_rows, n_cols, top_n]`` vocabulary ids, best first.
        top_values: ``[n_rows, n_cols, top_n]`` their logits. Kept because a readout's
            *margin* is the only thing on the page that distinguishes a confident cell
            from a coin flip -- the ids alone always look decisive.
        target_ranks: ``[n_rows, n_cols]`` full-vocabulary rank of the target, minimised
            over its accepted surface forms, or ``None`` if no target was given.
    """

    rows: list[int]
    cols: list[int]
    top_ids: torch.Tensor
    top_values: torch.Tensor
    target_ranks: torch.Tensor | None = None

    def top1_ids(self) -> torch.Tensor:
        """``[n_rows, n_cols]`` -- the top-1 id per cell, the thing a grid figure prints."""
        return self.top_ids[..., 0]

    def decode(self, tokenizer, rank: int = 0) -> list[list[str]]:
        """Cell strings at ``rank`` (0 = top-1), as ``[n_rows][n_cols]``."""
        return [
            [
                tokenizer.decode([int(i)], clean_up_tokenization_spaces=False)
                for i in row
            ]
            for row in self.top_ids[..., rank]
        ]


def _summarise(
    logits: torch.Tensor, top_n: int, target_ids: list[int] | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Top-k and target rank for a ``[n_cells, vocab]`` block, moved to CPU.

    Logits are summarised one row-block at a time and dropped: a 32-layer grid at
    ``[32 heads, 128k vocab]`` in fp32 is 16 MB per layer, which is fine, but retaining
    all 32 of them alongside an 8B model is not the way to spend GPU memory.
    """
    values, ids = logits.topk(top_n, dim=-1)
    ranks = None
    if target_ids is not None:
        ranks = torch.stack(
            [token_ranks(logits[i], target_ids).min() for i in range(logits.shape[0])]
        ).cpu()
    return ids.cpu(), values.cpu(), ranks


@torch.no_grad()
def head_grid(
    model,
    jz_by_layer: dict[int, torch.Tensor],
    spec: LensSpec,
    input_ids: torch.Tensor,
    position: int,
    *,
    arm: Arm = "jhead",
    layers: list[int] | None = None,
    top_n: int = 5,
    target_ids: list[int] | None = None,
) -> Grid:
    """Every head at every layer, read at one position.

    The head-level analogue of the paper's slice view: there the second axis is position
    and the transport is the layer's, here the second axis is head and the transport is
    that head's column slice of ``J^z_l``.

    Args:
        jz_by_layer: ``{layer: J^z_l}``, each ``[d_model, d_source]`` and already on the
            model's device. A plain dict rather than a ``JacobianLens`` so the caller can
            pass ``W_O`` and get the exact identity check
            ``J_{l,h} z_{l,h} = c_{l,h}`` (see ``tests/test_grid.py``).
        position: Index into the prompt. Negative indices are *not* accepted: the eval
            sets locate their readout position absolutely
            (:func:`src.evals.readout_index`), and silently accepting both conventions is
            how a poetry item ends up read at its last token instead of its newline.
        arm: Which of the two head arms (see :mod:`src.readout`).
        layers: Defaults to every layer ``J^z`` covers.

    Returns:
        A :class:`Grid` whose ``cols`` are head indices ``0 .. n_heads-1``.
    """
    layers = list(jz_by_layer) if layers is None else list(layers)
    missing = [l for l in layers if l not in jz_by_layer]
    if missing:
        raise ValueError(f"no J^z for layers {missing}")
    seq_len = int(input_ids.shape[-1])
    if not 0 <= position < seq_len:
        raise ValueError(f"position {position} out of range for {seq_len} tokens")

    ids, values, ranks = [], [], []
    with AnchorRecorder(model.layers, zin_at=layers) as recorder:
        model.forward(input_ids)
        for layer in layers:
            z = recorder.activations[("zin", layer)][0, position]
            vectors = all_heads_transported(
                jz_by_layer[layer],
                z,
                spec,
                arm=arm,
                w_o=o_proj_weight(model, layer) if arm == "logithead" else None,
            )
            cell_ids, cell_values, cell_ranks = _summarise(
                model.unembed(vectors).float(), top_n, target_ids
            )
            ids.append(cell_ids)
            values.append(cell_values)
            ranks.append(cell_ranks)

    return Grid(
        rows=layers,
        cols=list(range(spec.n_heads)),
        top_ids=torch.stack(ids),
        top_values=torch.stack(values),
        target_ranks=None if target_ids is None else torch.stack(ranks),
    )


@torch.no_grad()
def layer_grid(
    model,
    lens,
    input_ids: torch.Tensor,
    *,
    positions: list[int] | None = None,
    layers: list[int] | None = None,
    top_n: int = 5,
    target_ids: list[int] | None = None,
) -> Grid:
    """The Jacobian lens on the whole residual stream, layer x position.

    The paper's object, drawn on the paper's axes, so the head grid beside it differs in
    exactly one thing: what is being read. ``jlens.vis.compute_slice`` computes the same
    numbers for its interactive page; this returns them as tensors.

    Args:
        positions: Defaults to every position. Windowing here rather than truncating the
            prompt keeps the forward pass identical -- the model still sees the whole
            context, only the grid is cropped.
        layers: Defaults to ``lens.source_layers``. The model's final layer is *not*
            appended (``jlens.vis`` does, rendering it with ``J = I`` as the model's own
            output); a grid that mixes a lens row with a non-lens row would need a caveat
            in every figure that used it.
    """
    layers = list(lens.source_layers) if layers is None else list(layers)
    unfitted = [l for l in layers if l not in lens.jacobians]
    if unfitted:
        raise ValueError(f"lens has no Jacobian at layers {unfitted}")
    seq_len = int(input_ids.shape[-1])
    positions = list(range(seq_len)) if positions is None else list(positions)
    out_of_range = [p for p in positions if not 0 <= p < seq_len]
    if out_of_range:
        raise ValueError(f"positions {out_of_range} out of range for {seq_len} tokens")

    index = torch.tensor(positions, device=model.input_device)
    ids, values, ranks = [], [], []
    with AnchorRecorder(model.layers, out_at=layers) as recorder:
        model.forward(input_ids)
        for layer in layers:
            residual = recorder.activations[("out", layer)][0].index_select(0, index)
            cell_ids, cell_values, cell_ranks = _summarise(
                layer_logits(model, lens, residual, layer), top_n, target_ids
            )
            ids.append(cell_ids)
            values.append(cell_values)
            ranks.append(cell_ranks)

    return Grid(
        rows=layers,
        cols=positions,
        top_ids=torch.stack(ids),
        top_values=torch.stack(values),
        target_ranks=None if target_ids is None else torch.stack(ranks),
    )
