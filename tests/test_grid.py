"""Readout grids, on CPU.

A grid is a sweep, and a sweep's failure mode is a silent permutation: heads read in the
wrong order, a position window labelled with the wrong indices, a layer row shifted by
one. None of those change a shape or raise. Each test below pins a grid against something
computed a different way -- a per-cell loop, or an algebraic identity that makes the
answer known in advance.
"""

from __future__ import annotations

import pytest
import torch

from src.config import LensSpec
from src.grid import head_grid, layer_grid
from src.heads import head_contribution, o_proj_weight
from src.hooks import AnchorRecorder
from src.readout import token_ranks
from tests.tiny_llama import D_MODEL, N_HEADS, N_LAYERS, tiny_llama

PROMPT = "the quick brown fox jumps over the lazy dog and keeps running for a while"
POSITION = 7


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
def input_ids(model):
    return model.encode(PROMPT)


@pytest.fixture(scope="module")
def jz_by_layer():
    """A random ``J^z`` per layer -- the shape of the real thing, none of its structure."""
    spec = _spec()
    torch.manual_seed(0)
    return {
        layer: torch.randn(spec.d_model, spec.d_source) / spec.d_model**0.5
        for layer in range(spec.n_layers)
    }


@pytest.fixture(scope="module")
def w_o_by_layer(model):
    return {layer: o_proj_weight(model, layer).float() for layer in range(N_LAYERS)}


@pytest.fixture(scope="module")
def identity_lens():
    """A lens with ``J_l = I``: transport is a no-op, so the readout is the logit lens."""
    from jlens import JacobianLens

    return JacobianLens(
        jacobians={layer: torch.eye(D_MODEL) for layer in range(N_LAYERS - 1)},
        n_prompts=1,
        d_model=D_MODEL,
    )


# ----------------------------------------------------------------------- head_grid


def test_head_grid_shape_and_axes(model, jz_by_layer, input_ids):
    spec = _spec()
    grid = head_grid(model, jz_by_layer, spec, input_ids, POSITION, top_n=3)
    assert grid.rows == list(range(N_LAYERS))
    assert grid.cols == list(range(N_HEADS)), "columns must be heads, in head order"
    assert grid.top_ids.shape == (N_LAYERS, N_HEADS, 3)
    assert grid.top_values.shape == (N_LAYERS, N_HEADS, 3)
    assert grid.target_ranks is None
    # topk is ordered, so a cell's own values must descend.
    assert (grid.top_values[..., :-1] >= grid.top_values[..., 1:]).all()
    assert grid.top1_ids().tolist() == grid.top_ids[..., 0].tolist()


def test_logithead_grid_matches_per_cell_contribution(
    model, jz_by_layer, w_o_by_layer, input_ids
):
    """Every cell of the ``logithead`` grid must equal ``unembed(c_{l,h})``, cell by cell.

    The grid's batched path reshapes the transport into ``[d_model, n_heads, head_dim]``;
    this is the check that head ``h``'s column ends up in column ``h``.
    """
    spec = _spec()
    grid = head_grid(
        model, jz_by_layer, spec, input_ids, POSITION, arm="logithead", top_n=1
    )
    with torch.no_grad(), AnchorRecorder(
        model.layers, zin_at=list(range(N_LAYERS))
    ) as recorder:
        model.forward(input_ids)
        for row, layer in enumerate(grid.rows):
            z = recorder.activations[("zin", layer)][0, POSITION]
            for head in range(spec.n_heads):
                c = head_contribution(z, w_o_by_layer[layer], head, spec)
                expected = model.unembed(c).float()
                assert int(grid.top_ids[row, head, 0]) == int(expected.argmax()), (
                    f"layer {layer} head {head}"
                )


def test_wo_as_transport_reproduces_the_logithead_grid(
    model, w_o_by_layer, input_ids
):
    """``J_{l,h} z_{l,h} = c_{l,h}`` when ``J^z = W_O``, so the two arms must coincide.

    The anchor identity used as a fixture: with the transport set to ``W_O`` the
    ``jhead`` arm *is* the ``logithead`` arm, so the whole grid is known in advance
    without a golden file. Any slicing or ordering error breaks it.
    """
    spec = _spec()
    kwargs = dict(top_n=4)
    jhead = head_grid(
        model, w_o_by_layer, spec, input_ids, POSITION, arm="jhead", **kwargs
    )
    logithead = head_grid(
        model, w_o_by_layer, spec, input_ids, POSITION, arm="logithead", **kwargs
    )
    assert torch.equal(jhead.top_ids, logithead.top_ids)
    assert torch.allclose(jhead.top_values, logithead.top_values, atol=1e-4)


def test_head_grid_target_ranks_match_direct_ranking(
    model, jz_by_layer, input_ids
):
    spec = _spec()
    target_ids = [3, 11, 29]
    grid = head_grid(
        model, jz_by_layer, spec, input_ids, POSITION, top_n=1, target_ids=target_ids
    )
    assert grid.target_ranks.shape == (N_LAYERS, N_HEADS)

    from src.readout import all_heads_transported

    with torch.no_grad(), AnchorRecorder(
        model.layers, zin_at=list(range(N_LAYERS))
    ) as recorder:
        model.forward(input_ids)
        for row, layer in enumerate(grid.rows):
            z = recorder.activations[("zin", layer)][0, POSITION]
            logits = model.unembed(
                all_heads_transported(jz_by_layer[layer], z, spec, arm="jhead")
            ).float()
            for head in range(spec.n_heads):
                expected = int(token_ranks(logits[head], target_ids).min())
                assert int(grid.target_ranks[row, head]) == expected


def test_head_grid_arms_disagree(model, jz_by_layer, input_ids):
    """A grid that looked the same under every arm would mean the arm argument is inert."""
    spec = _spec()
    grids = {
        arm: head_grid(
            model, jz_by_layer, spec, input_ids, POSITION, arm=arm
        )
        for arm in ("jhead", "logithead")
    }
    assert not torch.equal(grids["jhead"].top_ids, grids["logithead"].top_ids)


def test_head_grid_layer_subset_keeps_labels(model, jz_by_layer, input_ids):
    spec = _spec()
    subset = head_grid(model, jz_by_layer, spec, input_ids, POSITION, layers=[2])
    full = head_grid(model, jz_by_layer, spec, input_ids, POSITION)
    assert subset.rows == [2]
    assert torch.equal(subset.top_ids[0], full.top_ids[2]), "row 2 must be layer 2"


def test_head_grid_rejects_bad_arguments(model, jz_by_layer, input_ids):
    spec = _spec()
    seq_len = int(input_ids.shape[-1])
    with pytest.raises(ValueError, match="out of range"):
        head_grid(model, jz_by_layer, spec, input_ids, seq_len)
    # Negative indices are rejected rather than wrapped: the eval sets address positions
    # absolutely, and a silent -1 would read the wrong token for the newline-anchored set.
    with pytest.raises(ValueError, match="out of range"):
        head_grid(model, jz_by_layer, spec, input_ids, -1)
    with pytest.raises(ValueError, match="no J\\^z"):
        head_grid(model, {0: jz_by_layer[0]}, spec, input_ids, POSITION, layers=[0, 1])


# ---------------------------------------------------------------------- layer_grid


def test_layer_grid_with_identity_lens_is_the_logit_lens(
    model, identity_lens, input_ids
):
    """``J_l = I`` reduces the arm to ``unembed(x_{l+1})``, computable without the lens."""
    grid = layer_grid(model, identity_lens, input_ids, top_n=1)
    assert grid.rows == list(range(N_LAYERS - 1)), "top block is the target, not a source"
    assert grid.cols == list(range(int(input_ids.shape[-1])))

    with torch.no_grad(), AnchorRecorder(
        model.layers, out_at=list(range(N_LAYERS - 1))
    ) as recorder:
        model.forward(input_ids)
        for row, layer in enumerate(grid.rows):
            expected = model.unembed(
                recorder.activations[("out", layer)][0].float()
            ).float()
            assert grid.top_ids[row, :, 0].tolist() == expected.argmax(-1).tolist()


def test_layer_grid_window_uses_absolute_positions(model, identity_lens, input_ids):
    """A cropped grid must still say *where* in the prompt its columns came from."""
    seq_len = int(input_ids.shape[-1])
    window = list(range(seq_len - 4, seq_len))
    cropped = layer_grid(model, identity_lens, input_ids, positions=window)
    full = layer_grid(model, identity_lens, input_ids)
    assert cropped.cols == window
    assert torch.equal(cropped.top_ids, full.top_ids[:, -4:])


def test_layer_grid_target_ranks_match_direct_ranking(model, identity_lens, input_ids):
    target_ids = [5, 17]
    grid = layer_grid(model, identity_lens, input_ids, top_n=1, target_ids=target_ids)
    with torch.no_grad(), AnchorRecorder(
        model.layers, out_at=list(range(N_LAYERS - 1))
    ) as recorder:
        model.forward(input_ids)
        for row, layer in enumerate(grid.rows):
            logits = model.unembed(
                recorder.activations[("out", layer)][0].float()
            ).float()
            for col, position in enumerate(grid.cols):
                expected = int(token_ranks(logits[position], target_ids).min())
                assert int(grid.target_ranks[row, col]) == expected


def test_layer_grid_rejects_unfitted_layers(model, identity_lens, input_ids):
    with pytest.raises(ValueError, match="no Jacobian"):
        layer_grid(model, identity_lens, input_ids, layers=[N_LAYERS - 1])
    with pytest.raises(ValueError, match="out of range"):
        layer_grid(model, identity_lens, input_ids, positions=[999])


def test_decode_shapes_match_the_grid(model, jz_by_layer, input_ids):
    grid = head_grid(model, jz_by_layer, _spec(), input_ids, POSITION, top_n=2)
    strings = grid.decode(model.tokenizer)
    assert len(strings) == N_LAYERS
    assert all(len(row) == N_HEADS for row in strings)
    assert grid.decode(model.tokenizer, rank=1) != strings
