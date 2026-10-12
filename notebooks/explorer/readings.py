"""One forward pass' worth of readouts: J-head, Logit-head and J-layer at one position.

The computation is ``scripts/read.py``'s, line for line: one pass recording ``o_proj``'s
input at every layer and the block output at every reference layer, then per layer both
head arms through :func:`src.readout.all_heads_transported` and ``model.unembed``, and
J-layer through :func:`src.readout.layer_logits` on the one position's residual. That is
what lets :func:`check_against_record` demand the recorded ranks *exactly* for any
evaluation item read at its documented position.

Ranks are zero-based here, as in the code and the records; the charts and tables add one,
as the paper does ("rank 1 is the highest-scoring token", Section 3.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.grid import Grid, layer_grid
from src.hooks import AnchorRecorder
from src.magnitude import relative_write_norms
from src.passk import PAD
from src.readout import HEAD_ARMS, all_heads_transported, layer_logits

from .prompts import Selection, candidates
from .session import LensSession

#: Arms in display order; ``jlayer`` is the paper's whole-layer reference.
ARMS = (*HEAD_ARMS, "jlayer")
ARM_LABEL = {"jhead": "J-head", "logithead": "Logit-head", "jlayer": "J-layer"}
#: The thresholds the paper reports pass@k at.
K_VALUES = (1, 5, 25)


@dataclass(frozen=True)
class Concept:
    """A concept scored at every site: its candidate ids and their best rank per site.

    Attributes:
        ranks: ``{arm: array}``, zero-based. Head arms ``[n_layers, n_heads]``; ``jlayer``
            ``[n_layers]`` with :data:`src.passk.PAD` at the layer it has no readout at.
    """

    word: str
    ids: list[int]
    rule: str
    ranks: dict[str, np.ndarray]

    def best(self, arm: str) -> tuple[int, tuple[int, ...]]:
        """Best rank (zero-based) and the site reaching it: ``(layer, head)`` or ``(layer,)``.

        Ties go to the first site in (layer, head) order. Each arm takes its own best
        site, as in the paper: the annotation picks the winning head per item and arm.
        """
        ranks = self.ranks[arm]
        masked = np.where(ranks == PAD, np.iinfo(np.int64).max, ranks)
        flat = int(masked.argmin())
        return int(masked.flat[flat]), tuple(int(i) for i in np.unravel_index(flat, ranks.shape))

    def tied(self, arm: str) -> list[tuple[int, ...]]:
        """Every site reaching the best rank: an oracle best is often not unique."""
        rank = self.best(arm)[0]
        return [tuple(int(i) for i in site) for site in np.argwhere(self.ranks[arm] == rank)]

    def best_per_layer(self, arm: str) -> np.ndarray:
        """``[n_layers]`` best rank of the layer -- its best head for a head arm."""
        ranks = self.ranks[arm]
        return ranks.min(axis=1) if ranks.ndim == 2 else ranks


@dataclass(frozen=True)
class Readings:
    """Both head readouts and J-layer, read at one position of one prompt.

    Attributes:
        grids: ``{arm: Grid}``. Head arms are layer x head; ``jlayer`` is reference layer
            x 1 (this position only).
        concepts: One :class:`Concept` per typed concept that has a candidate.
        skipped: ``{word: reason}`` for concepts the rule could not score.
        rho: ``[n_layers, n_heads]``, ``||c_{l,h}|| / ||x_mid,l||`` at this position.
    """

    position: int
    layers: list[int]
    ref_layers: list[int]
    grids: dict[str, Grid]
    concepts: list[Concept]
    skipped: dict[str, str]
    rho: np.ndarray

    def concept(self, word: str | None) -> Concept | None:
        return next((c for c in self.concepts if c.word == word), None)


def parse_concepts(text: str) -> list[str]:
    """Comma-separated concepts, blanks dropped, order and spelling kept."""
    return [w.strip() for w in text.split(",") if w.strip()]


@torch.no_grad()
def read_position(
    session: LensSession,
    selection: Selection,
    *,
    position: int,
    words: list[str],
    top_n: int = 10,
) -> Readings:
    """Read J-head, Logit-head and J-layer at ``position`` and rank every concept.

    Args:
        words: Concepts to rank; each gets :func:`explorer.prompts.candidates`' ids.
        top_n: Tokens kept per cell, for tooltips and the cell inspector.
    """
    model, spec = session.model, session.spec
    layers, ref_layers = session.layers, session.ref_layers
    input_ids = selection.input_ids

    scored, skipped = [], {}
    for word in dict.fromkeys(words):
        try:
            ids, rule = candidates(session.tokenizer, selection, word)
        except ValueError as error:
            skipped[word] = str(error)
            continue
        scored.append((word, ids, rule))

    head_ranks = {
        arm: np.full((len(scored), len(layers), spec.n_heads), PAD, dtype=np.int64)
        for arm in HEAD_ARMS
    }
    layer_ranks = np.full((len(scored), len(layers)), PAD, dtype=np.int64)
    top = {arm: ([], []) for arm in ARMS}

    def keep(arm, logits):
        values, ids = logits.topk(top_n, dim=-1)
        top[arm][0].append(ids.cpu())
        top[arm][1].append(values.cpu())

    def rank_rows(logits, ids):
        # The best candidate's rank is the count of logits strictly above the highest-
        # scoring candidate: min over ids of src.readout.token_ranks, in one comparison.
        best = logits[..., ids].max(dim=-1, keepdim=True).values
        return (logits > best).sum(dim=-1).cpu().numpy()

    with AnchorRecorder(model.layers, zin_at=layers, out_at=ref_layers) as recorder:
        model.forward(input_ids)
        for li, layer in enumerate(layers):
            z = recorder.activations[("zin", layer)][0, position]
            for arm in HEAD_ARMS:
                logits = model.unembed(
                    all_heads_transported(
                        session.jz_by_layer[layer], z, spec, arm=arm,
                        w_o=session.w_o_by_layer[layer],
                    )
                ).float()
                keep(arm, logits)
                for ci, (_, ids, _) in enumerate(scored):
                    head_ranks[arm][ci, li] = rank_rows(logits, ids)
            if layer in ref_layers:
                residual = recorder.activations[("out", layer)][0, position]
                logits = layer_logits(model, session.published, residual, layer)
                keep("jlayer", logits[None])
                for ci, (_, ids, _) in enumerate(scored):
                    layer_ranks[ci, li] = int(rank_rows(logits, ids))

    def grid(arm, rows, cols):
        ids, values = top[arm]
        return Grid(rows=rows, cols=cols, top_ids=torch.stack(ids), top_values=torch.stack(values))

    rho = relative_write_norms(model, spec, input_ids, layers, position=position)
    return Readings(
        position=position,
        layers=layers,
        ref_layers=ref_layers,
        grids={
            "jhead": grid("jhead", layers, list(range(spec.n_heads))),
            "logithead": grid("logithead", layers, list(range(spec.n_heads))),
            "jlayer": grid("jlayer", ref_layers, [position]),
        },
        concepts=[
            Concept(
                word=word,
                ids=ids,
                rule=rule,
                ranks={
                    "jhead": head_ranks["jhead"][ci],
                    "logithead": head_ranks["logithead"][ci],
                    "jlayer": layer_ranks[ci],
                },
            )
            for ci, (word, ids, rule) in enumerate(scored)
        ],
        skipped=skipped,
        rho=torch.stack([rho[l] for l in layers]).numpy(),
    )


def window(
    session: LensSession, selection: Selection, ids: list[int] | None, *, n_positions: int = 14
) -> Grid:
    """J-layer over the last ``n_positions`` of the prompt: the Jacobian lens's slice view.

    :func:`src.grid.layer_grid`, ranking ``ids`` when given. Windowing the grid rather
    than the prompt keeps the forward pass identical.
    """
    n_tokens = len(selection.token_strs)
    return layer_grid(
        session.model,
        session.published,
        selection.input_ids,
        positions=list(range(max(0, n_tokens - n_positions), n_tokens)),
        top_n=1,
        target_ids=ids,
    )


def item_passk(concepts: list[Concept], arm: str) -> dict[int, float]:
    """pass@k of one item: the fraction of its concepts whose best rank is within ``k``.

    Section 3.3's definition before the mean over items: each concept takes its best
    rank over every site the arm searches (1,024 head sites, or 31 block outputs).
    """
    best = [c.best(arm)[0] for c in concepts]
    return {k: float(np.mean([r < k for r in best])) if best else float("nan") for k in K_VALUES}


def check_against_record(
    readings: Readings, selection: Selection, recorded: dict[str, np.ndarray] | None
) -> dict:
    """Do this item's live ranks equal ``results/read/``'s, at every site and arm?

    Only meaningful for an evaluation item read at its documented position with its own
    intermediates, which is the only case ``scripts/read.py`` recorded.

    Args:
        recorded: ``{arm: [n_intermediates, n_layers, n_heads]}`` from
            :func:`explorer.records.read_ranks`, or ``None`` if the prompt is not an item.

    Returns:
        ``{"status": "match" | "mismatch" | "n/a", "detail": str, ...}``.
    """
    if recorded is None or selection.item is None:
        return {"status": "n/a", "detail": "not an evaluation item, so nothing was recorded"}
    if readings.position != selection.default_position:
        return {
            "status": "n/a",
            "detail": f"read at position {readings.position}; the record reads at "
            f"{selection.default_position}",
        }
    compared, mismatched = 0, []
    for wi, word in enumerate(selection.item.intermediates):
        concept = readings.concept(word)
        if concept is None:
            continue
        for arm in ARMS:
            live = concept.ranks[arm]
            stored = recorded[arm][wi]
            stored = stored[..., 0] if arm == "jlayer" else stored
            compared += live.size
            if not np.array_equal(live, stored):
                mismatched.append(f"{word}/{ARM_LABEL[arm]}")
    if not compared:
        return {"status": "n/a", "detail": "none of the item's intermediates is being ranked"}
    if mismatched:
        return {
            "status": "mismatch",
            "detail": f"ranks differ from results/read/ for {', '.join(mismatched)}",
            "compared": compared,
        }
    return {
        "status": "match",
        "detail": f"all {compared:,} ranks equal results/read/'s",
        "compared": compared,
    }
