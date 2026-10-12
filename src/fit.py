"""The production ``J^z`` fit loop: accumulation, checkpointing, and resume.

:mod:`src.jacobian` holds the estimator and stays deliberately naive -- it exists so a
handful of prompts can be fitted at two anchors at once and compared. This module is the
part that runs for hours: it averages :func:`~src.jacobian.jacobian_for_prompt` over a
corpus slice and writes a resumable checkpoint.

Ported from :func:`jlens.fitting.fit`, with four changes.

* Accumulators are ``[d_model, d_source]``, keyed by :data:`~src.hooks.Anchor` rather
  than by layer, because a ``zin`` source is ``n_heads * head_dim`` wide -- equal to
  ``d_model`` on Llama-3.1-8B only by coincidence.
* ``checkpoint_every`` counts *work done* (``n_done``), not list position: with skipped
  prompts the two diverge, and a checkpoint should track the former.
* The running mean is materialised once per anchor per prompt. Upstream's expression
  spells ``jacobian_sum[l] / n_done`` twice inside a norm, which at this size builds
  three 67 MB temporaries per layer per prompt.
* The checkpoint records the corpus slice, so a resume cannot silently average a
  different draw.

**The fit runs to a fixed prompt budget**, exactly as :func:`jlens.fitting.fit` does.
``mean_rel_change`` is recorded once per prompt as a settling *diagnostic* of the running
mean and is never acted on. The estimator is an average rather than an optimisation, so
there is no optimisation to converge; no decision depends on the trace.

The pre-fitted lenses this project reads alongside its own were produced differently.
Neuronpedia's ``fit_lens.py`` -- not part of the released ``jlens`` package -- adds a
convergence criterion (``--min_prompts 100 --stop_window 10 --stop_at_delta 0.002``), and
every released lens stopped on it well short of the 1000 prompts its ``config.yaml``
requested: 461 for Llama-3.1-8B-Instruct, 546/844/278 for the three Gemma runs. So the
two families of lens are not comparable by prompt count, and a released lens's
``results.prompts_fitted`` is the number to cite for it.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from jlens.fitting import SKIP_FIRST_N_POSITIONS

from src.config import FIT_DIM_BATCH, MAX_SEQ_LEN
from src.hooks import Anchor
from src.jacobian import jacobian_for_prompt

logger = logging.getLogger(__name__)

#: Columns of the convergence trace, matching the ``*_convergence.csv`` shipped beside
#: each of Neuronpedia's released lenses where the quantities coincide.
#: ``identity_distance`` is
#: deliberately absent: measured at the ``zin`` anchor it is 1.44 at L0 and 1.49 at L31,
#: non-monotone and uninformative, because ``J^z`` approaches ``W_O`` at the top of the
#: model rather than the identity. The caller supplies its own health monitor instead.
CSV_COLUMNS = [
    "n_done",
    "prompt_idx",
    "seq_len",
    "n_valid_positions",
    "elapsed_s",
    "prompt_norm",
    "mean_rel_change",
]


@dataclass(frozen=True)
class CorpusSlice:
    """Which prompts a fit ran on: a slice of one deterministic draw.

    ``load_prompts(1000, seed=0)`` is **not** a prefix or a superset of
    ``load_prompts(100, seed=0)``. ``random.sample`` switches algorithm on ``k`` versus
    the pool size, and the two lists diverge from index 56 -- so a 1000-prompt fit is
    never "the 100 plus 900 more", and two fits described only by their prompt counts
    may overlap arbitrarily. Draw once at :data:`~src.config.FIT_CORPUS_SIZE` and slice:
    then ``offset`` and ``n_prompts`` say exactly which prompts were used, disjointness
    is decidable, and shards can be merged without double-counting.
    """

    size: int
    seed: int
    offset: int
    n_prompts: int

    def __post_init__(self) -> None:
        if self.offset < 0 or self.n_prompts <= 0:
            raise ValueError(f"bad slice: offset={self.offset} n_prompts={self.n_prompts}")
        if self.offset + self.n_prompts > self.size:
            raise ValueError(
                f"slice [{self.offset}:{self.offset + self.n_prompts}] runs past a "
                f"{self.size}-prompt corpus"
            )

    @property
    def indices(self) -> range:
        """Positions in the drawn corpus, for checking two shards are disjoint."""
        return range(self.offset, self.offset + self.n_prompts)

    def prompts(self) -> list[str]:
        from src.corpus import load_prompts

        return load_prompts(self.size, seed=self.seed)[
            self.offset : self.offset + self.n_prompts
        ]

    def as_meta(self) -> dict[str, int]:
        return {
            "corpus_size": self.size,
            "corpus_seed": self.seed,
            "prompt_offset": self.offset,
            "n_prompts_requested": self.n_prompts,
        }


@dataclass
class FitResult:
    """What a fit produced, and how it ended."""

    jacobians: dict[Anchor, torch.Tensor]
    n_done: int
    next_idx: int
    #: Last per-prompt ``mean_rel_change`` this run recorded; ``None`` if it fitted
    #: nothing new. A diagnostic only -- the full history is in the convergence trace.
    final_mean_rel_change: float | None = None
    n_skipped: int = 0

    def lens(self, d_model: int):
        """The ``zin`` anchors as a :class:`~jlens.lens.JacobianLens`, keyed by layer."""
        from jlens import JacobianLens

        matrices = {l: J for (kind, l), J in self.jacobians.items() if kind == "zin"}
        if not matrices:
            raise ValueError("no zin anchors were fitted; nothing to save as a J^z lens")
        return JacobianLens(matrices, n_prompts=self.n_done, d_model=d_model)


def _atomic_save(obj: object, path: str) -> None:
    """``torch.save`` to a temp file then ``os.replace`` so a crash never
    leaves a half-written checkpoint."""
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _identity(anchors: Sequence[Anchor], d_source: int, skip_first: int,
              corpus: CorpusSlice | None) -> dict[str, object]:
    """What a checkpoint must agree with to be resumable against this call.

    Resuming a 1000-prompt run against a 100-prompt draw would silently average a
    different corpus, so the slice is part of the identity, not just of the metadata.
    """
    fields: dict[str, object] = {
        "layers": sorted({l for _, l in anchors}),
        "anchor_kinds": sorted({k for k, _ in anchors}),
        "d_source": d_source,
        "skip_first": skip_first,
    }
    if corpus is not None:
        fields |= {
            "corpus_size": corpus.size,
            "corpus_seed": corpus.seed,
            "prompt_offset": corpus.offset,
        }
    return fields


def fit(
    model,
    prompts: Sequence[str],
    *,
    anchors: Sequence[Anchor],
    d_source: int,
    corpus: CorpusSlice | None = None,
    dim_batch: int = FIT_DIM_BATCH,
    max_seq_len: int = MAX_SEQ_LEN,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = 10,
    resume: bool = True,
    convergence_path: str | Path | None = None,
    monitor: Callable[[dict[Anchor, torch.Tensor], int], float] | None = None,
    monitor_name: str = "monitor",
) -> FitResult:
    """Average :func:`~src.jacobian.jacobian_for_prompt` over ``prompts``.

    Args:
        anchors: ``(kind, layer)`` pairs to fit. Fitting several costs no extra backward
            passes -- the count depends only on the *output* width -- just one
            ``[d_model, d_source]`` accumulator each.
        corpus: The slice ``prompts`` came from. Recorded in the checkpoint and checked
            on resume; ``None`` skips both, which is what the tests use.
        checkpoint_every: Write the running sum every N *completed* prompts. ``None``
            writes only at the end. At 32 layers each write is 2.1 GB, so upstream's
            default of 1 would move ~1 TB over a 500-prompt run.
        monitor: ``(running_sum, n_done) -> float``, called after each prompt and logged
            to the convergence trace under ``monitor_name``. Its job is to catch a
            wrongly-placed hook while the run is young; see ``scripts/fit_lens.py``.

    Returns:
        A :class:`FitResult` whose ``jacobians`` are means over the prompts used.
    """
    anchors = sorted(set(anchors))
    if not anchors:
        raise ValueError("no anchors requested")
    out_of_range = sorted(a for a in anchors if not 0 <= a[1] < model.n_layers)
    if out_of_range:
        raise ValueError(f"anchors {out_of_range} out of range for {model.n_layers} blocks")

    d_model = model.d_model
    identity = _identity(anchors, d_source, skip_first, corpus)
    checkpoint_path = None if checkpoint_path is None else str(checkpoint_path)

    jacobian_sum: dict[Anchor, torch.Tensor]
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in identity.items():
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint at {checkpoint_path} was fitted with {key}="
                    f"{state[key]!r}, not {expected!r}; pass resume=False to discard it"
                )
        jacobian_sum = state["jacobian_sum"]
        n_done, next_idx = state["n_done"], state["next_idx"]
        n_skipped = state.get("n_skipped", 0)
        logger.info(
            "  resuming from checkpoint: %d/%d prompts processed",
            next_idx,
            len(prompts),
        )
    else:
        jacobian_sum = {
            anchor: torch.zeros(
                d_model, d_source if anchor[0] == "zin" else d_model, dtype=torch.float32
            )
            for anchor in anchors
        }
        n_done, next_idx, n_skipped = 0, 0, 0

    logger.info(
        "fit: %d anchors (%s at %d layers), %d prompts, dim_batch=%d",
        len(anchors),
        ",".join(sorted({k for k, _ in anchors})),
        len({l for _, l in anchors}),
        len(prompts),
        dim_batch,
    )

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "n_skipped": n_skipped,
                    **identity,
                },
                checkpoint_path,
            )

    trace = _ConvergenceTrace(convergence_path, monitor_name)
    sqrt_d = math.sqrt(d_model)
    final_mean_rel_change: float | None = None

    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()
        try:
            per_prompt, seq_len, n_valid = jacobian_for_prompt(
                model,
                prompt,
                anchors=anchors,
                d_source=d_source,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError as exc:
            # Too short to have a valid position. ``next_idx`` advances but ``n_done``
            # does not, so the prompt is neither retried on resume nor counted in the
            # divisor.
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            n_skipped += 1
            continue
        if n_valid == 0:
            logger.warning("  skipping prompt %d: no valid positions", prompt_idx)
            next_idx = prompt_idx + 1
            n_skipped += 1
            continue

        # Per-prompt diagnostics, max over anchors: the prompt's own Jacobian norm flags
        # heavy-tailed outliers, and the relative shift in the running mean tracks
        # convergence (falls ~C/n once settled). Recorded to the trace and never acted
        # on -- the budget is fixed.
        prompt_norm = max(per_prompt[a].norm().item() for a in anchors) / sqrt_d
        mean_rel_change = float("nan")
        if n_done > 0:
            worst = 0.0
            for anchor in anchors:
                running_mean = jacobian_sum[anchor] / n_done
                shift = (per_prompt[anchor] - running_mean).norm() / (
                    (n_done + 1) * running_mean.norm()
                )
                worst = max(worst, shift.item())
            mean_rel_change = worst
            final_mean_rel_change = worst

        for anchor in anchors:
            jacobian_sum[anchor] += per_prompt[anchor]
        n_done += 1
        next_idx = prompt_idx + 1
        del per_prompt

        health = None if monitor is None else monitor(jacobian_sum, n_done)
        logger.info(
            "  prompt %d/%d  seq_len=%d n_valid=%d  %.0fs  max||J||/sqrt(d)=%.3f  "
            "d_mean=%.2e  %s=%s",
            prompt_idx + 1,
            len(prompts),
            seq_len,
            n_valid,
            time.perf_counter() - start_time,
            prompt_norm,
            mean_rel_change,
            monitor_name,
            "n/a" if health is None else f"{health:.4f}",
        )
        trace.append(
            n_done=n_done,
            prompt_idx=prompt_idx,
            seq_len=seq_len,
            n_valid_positions=n_valid,
            elapsed_s=round(time.perf_counter() - start_time, 3),
            prompt_norm=round(prompt_norm, 6),
            mean_rel_change=mean_rel_change,
            health=health,
        )

        if checkpoint_every is not None and n_done % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise RuntimeError("no prompt yielded a valid position; nothing was fitted")
    logger.info("fit: done, %d prompts (%d skipped)", n_done, n_skipped)
    return FitResult(
        jacobians={a: jacobian_sum[a] / n_done for a in anchors},
        n_done=n_done,
        next_idx=next_idx,
        final_mean_rel_change=final_mean_rel_change,
        n_skipped=n_skipped,
    )


class _ConvergenceTrace:
    """One row per fitted prompt, appended and flushed so a running fit can be watched.

    Appends rather than truncates: a resumed run continues the same trace, so a restarted
    fit still ends up with one unbroken convergence history.
    """

    def __init__(self, path: str | Path | None, monitor_name: str) -> None:
        self.path = None if path is None else Path(path)
        self.columns = [*CSV_COLUMNS[:-1], monitor_name, CSV_COLUMNS[-1]]

    def append(self, *, health: float | None, **row: object) -> None:
        if self.path is None:
            return
        row[self.columns[-2]] = "" if health is None else round(health, 6)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
