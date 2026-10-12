"""The production fit loop: accumulation, checkpointing, and resume.

The estimator itself is tested in ``test_heads.py``; ``src.jacobian.jacobian_for_prompt``
is reused byte-unchanged by :mod:`src.fit`, so everything new lives in the loop *around*
it -- accumulation, checkpointing and resume. That is where the tests go.

The load-bearing one is :func:`test_merge_of_disjoint_shards_matches_one_fit`. The shipped
lens is four disjoint shards merged, so ``merge([fit(a), fit(b)]) == fit(a + b)`` is what
makes the sharding exact rather than an approximation of a single 1000-prompt pass.
"""

from __future__ import annotations

import pytest
import torch

from src.config import LensSpec
from src.fit import CorpusSlice, fit
from tests.tiny_llama import D_MODEL, N_HEADS, N_LAYERS, tiny_llama

#: Long enough that ``skip_first=2`` still leaves valid positions.
PROMPTS = [f"the quick brown fox number {i} jumps over the lazy dog" for i in range(24)]
FIT_KWARGS = dict(dim_batch=4, skip_first=2, max_seq_len=64, checkpoint_every=None)


def _spec(head_dim: int) -> LensSpec:
    return LensSpec(
        key=f"tiny_hd{head_dim}",
        hf_model_id="(tiny fixture)",
        lens_file="",
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim=head_dim,
        n_prompts_fitted=0,
    )


@pytest.fixture(scope="module")
def square_model():
    return tiny_llama()


def _fit(model, prompts, **overrides):
    spec = overrides.pop("spec", _spec(D_MODEL // N_HEADS))
    kwargs = {
        "anchors": [("zin", l) for l in range(N_LAYERS)],
        "d_source": spec.d_source,
        **FIT_KWARGS,
        **overrides,
    }
    return fit(model, prompts, **kwargs)


# --------------------------------------------------------------------------
# The loop: accumulation, merge, resume
# --------------------------------------------------------------------------


def test_merge_of_disjoint_shards_matches_one_fit(square_model):
    """``merge([fit(a), fit(b)]) == fit(a + b)`` -- what makes sharding legitimate."""
    from jlens import JacobianLens

    whole = _fit(square_model, PROMPTS[:20]).lens(D_MODEL)
    shards = [_fit(square_model, PROMPTS[:10]), _fit(square_model, PROMPTS[10:20])]
    merged = JacobianLens.merge([s.lens(D_MODEL) for s in shards])

    assert merged.n_prompts == whole.n_prompts == 20
    for layer in whole.source_layers:
        error = (merged.jacobians[layer] - whole.jacobians[layer]).norm() / whole.jacobians[layer].norm()
        assert error < 1e-6, f"layer {layer}: merge differs by {error:.2e}"


def test_resume_reproduces_an_uninterrupted_fit(square_model, tmp_path):
    """Stopping at prompt 7 and resuming gives the same lens as never stopping."""
    checkpoint = tmp_path / "fit.ckpt.pt"
    uninterrupted = _fit(square_model, PROMPTS[:20])

    _fit(square_model, PROMPTS[:7], checkpoint_path=checkpoint, checkpoint_every=1)
    resumed = _fit(square_model, PROMPTS[:20], checkpoint_path=checkpoint, checkpoint_every=1)

    assert resumed.n_done == 20
    for anchor, matrix in uninterrupted.jacobians.items():
        error = (resumed.jacobians[anchor] - matrix).norm() / matrix.norm()
        assert error < 1e-6, f"{anchor}: resume differs by {error:.2e}"


def test_final_mean_rel_change_is_recorded_but_not_acted_on(square_model):
    """The convergence diagnostic reaches the sidecar; the loop still runs the budget.

    ``mean_rel_change`` compares a prompt against the running mean, so the first prompt
    has none. Nothing consumes it -- the fit stops when the prompts run out.
    """
    assert _fit(square_model, PROMPTS[:1]).final_mean_rel_change is None

    result = _fit(square_model, PROMPTS[:5])
    assert result.n_done == 5
    assert isinstance(result.final_mean_rel_change, float)
    assert result.final_mean_rel_change > 0


def test_short_prompt_advances_the_index_but_not_the_count(square_model, tmp_path):
    """A skipped prompt is neither counted in the divisor nor retried on resume."""
    checkpoint = tmp_path / "fit.ckpt.pt"
    prompts = [PROMPTS[0], "ab", PROMPTS[1]]

    result = _fit(square_model, prompts, checkpoint_path=checkpoint, checkpoint_every=1)
    assert (result.n_done, result.next_idx, result.n_skipped) == (2, 3, 1)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert (state["n_done"], state["next_idx"]) == (2, 3)

    resumed = _fit(square_model, prompts, checkpoint_path=checkpoint, checkpoint_every=1)
    assert (resumed.n_done, resumed.n_skipped) == (2, 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"anchors": [("zin", 0), ("zin", 2)]},
        {"d_source": 24},
        {"corpus": CorpusSlice(size=100, seed=7, offset=0, n_prompts=5)},
    ],
    ids=["layers", "d_source", "corpus_seed"],
)
def test_resume_rejects_a_checkpoint_from_a_different_fit(square_model, tmp_path, overrides):
    """Resuming across a changed corpus or geometry raises rather than averaging both."""
    checkpoint = tmp_path / "fit.ckpt.pt"
    corpus = CorpusSlice(size=100, seed=0, offset=0, n_prompts=5)
    _fit(square_model, PROMPTS[:3], checkpoint_path=checkpoint, checkpoint_every=1,
         corpus=corpus)

    kwargs = {
        **FIT_KWARGS,
        "anchors": [("zin", l) for l in range(N_LAYERS)],
        "d_source": D_MODEL,
        "corpus": corpus,
        "checkpoint_path": checkpoint,
        "checkpoint_every": 1,
    }
    with pytest.raises(ValueError, match="pass resume=False"):
        fit(square_model, PROMPTS[:5], **{**kwargs, **overrides})


def test_oblong_geometry_sizes_the_accumulator_by_d_source():
    """``d_source != d_model`` must survive the loop, which allocates the accumulator.

    Llama-3.1-8B has ``n_heads * head_dim == d_model``, so on the real model a width
    confusion is invisible. This fixture is the only place it can be caught -- and it
    matters more here than in ``src.jacobian``, which is handed its accumulator by
    ``jacobian_for_prompt``, whereas this loop allocates its own.
    """
    spec = _spec(head_dim=6)
    assert spec.d_source != spec.d_model
    result = _fit(tiny_llama(head_dim=6), PROMPTS[:2], spec=spec)
    for anchor, matrix in result.jacobians.items():
        assert tuple(matrix.shape) == (spec.d_model, spec.d_source), anchor



# --------------------------------------------------------------------------
# The corpus slice
# --------------------------------------------------------------------------


def test_corpus_slice_rejects_a_slice_past_the_draw():
    CorpusSlice(size=1400, seed=0, offset=1000, n_prompts=400)  # exactly fits
    with pytest.raises(ValueError, match="runs past"):
        CorpusSlice(size=1400, seed=0, offset=1200, n_prompts=400)


def test_corpus_slices_expose_disjointness():
    """What ``scripts/merge_jz.py`` checks before merging, since ``merge`` will not."""
    authoritative = CorpusSlice(size=1400, seed=0, offset=0, n_prompts=1000)
    shards = [CorpusSlice(size=1400, seed=0, offset=1000 + 100 * i, n_prompts=100)
              for i in range(4)]
    assert not set(authoritative.indices) & {i for s in shards for i in s.indices}
    assert sum(len(s.indices) for s in shards) == 400
