"""The two causal task sets, and how a swap on them is graded.

Both ship in the ``jlens`` repo under ``data/experiments/``, with protocols documented in
its ``README.md``.

**verbal-report** -- 14 categories x 14 candidate words. The prompt is
``Think of a {category}. Answer in one word:``; the model's greedy next token there is
taken as its answer, and each of the first 10 other candidates is swapped in. Success is
the swapped-in candidate reaching rank 1.

The prompt is used as raw completion-style text rather than through the chat template.
The documented protocol grades "at the final ``:``", and
Llama-3.1's chat template ends ``<|start_header_id|>assistant<|end_header_id|>\\n\\n``
with no colon anywhere -- that convention comes from the paper's Claude-format
``\\n\\nAssistant:`` prompts and does not transfer. Keeping the prompt raw preserves the
documented readout position exactly, and avoids the template's auto-injected system turn
("Cutting Knowledge Date...") which the paper's prompts also did not have.

**probe-swap** -- 90 two-hop factual prompts, already completion-style. The shipped
protocol swaps a *linear-probe* direction; substituting lens directions is a documented
deviation, not a reuse. Its 90 items are also not the 50-prompt set behind
the paper's Fig. 15 headline, so the numbers are not directly comparable to it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)

VERBAL_REPORT_PROMPT = "Think of a {category}. Answer in one word:"
#: Candidates tried per category, per the documented protocol.
N_CANDIDATES = 10


@dataclass(frozen=True)
class ProbeSwapItem:
    name: str
    category: str
    prompt: str
    intermediate: str
    answer: str
    swap_to: str
    swap_answer: str


def experiment_root() -> Path:
    import jlens

    root = Path(jlens.__file__).resolve().parent.parent / "data" / "experiments"
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; expected it in the jlens checkout")
    return root


@cache
def load_verbal_report() -> dict[str, tuple[str, ...]]:
    """``{category: (candidate words,)}`` -- 14 categories of 14."""
    payload = json.loads((experiment_root() / "verbal-report.json").read_text())
    candidates = {k: tuple(v) for k, v in payload["candidates"].items()}
    logger.info("loaded verbal-report: %d categories", len(candidates))
    return candidates


@cache
def load_probe_swap() -> tuple[ProbeSwapItem, ...]:
    payload = json.loads((experiment_root() / "probe-swap.json").read_text())
    items = payload["items"] if isinstance(payload, dict) else payload
    parsed = tuple(
        ProbeSwapItem(
            name=raw["name"],
            category=raw["category"],
            prompt=raw["prompt"],
            intermediate=raw["intermediate"],
            answer=raw["answer"],
            swap_to=raw["swap_to"],
            swap_answer=raw["swap_answer"],
        )
        for raw in items
    )
    logger.info("loaded probe-swap: %d items", len(parsed))
    return parsed


def target_rank(logits, ids):
    """0-based rank of the best-ranked of ``ids`` in each row: how many tokens score higher.

    Gurnee et al. grade verbal report by the target reaching the top 5; a target with
    several surface forms counts at its best one, and a tie counts in its favour, so rank 0
    does not guarantee greedy success when bfloat16 logits tie. A row with a non-finite
    logit gets NaN rather than the rank 0 that NaN comparisons would give.
    """
    import torch

    logits = logits.float()
    best = logits[..., list(ids)].max(-1, keepdim=True).values
    rank = (logits > best).sum(-1).float()
    return torch.where(torch.isfinite(logits).all(-1), rank, torch.full_like(rank, float("nan")))


@dataclass(frozen=True)
class Trial:
    """One swap trial: a prompt, the two directions, and what counts as success.

    Attributes:
        task: ``"verbal-report"`` or ``"probe-swap"``.
        prompt_key: Trials sharing a prompt share their activations, and therefore their
            site-loading map. Used as the cache key for that map.
        input_ids: The encoded prompt, ``[1, seq]``.
        source_id: Vocabulary id of the direction swapped **out**.
        target_id: Vocabulary id of the direction swapped **in**.
        success_ids: Every surface form that grades as a hit.
        answer: verbal-report only -- the model's own greedy answer, decoded.
        answer_in_class: verbal-report only. On some categories the model does not name a
            member of the concept class at all (``river -> ' calm'``), so the direction
            swapped out is not a concept direction and the trial does not test what the
            task says it tests. Recorded rather than corrected, so a reader can take the
            subset where the premise holds.
    """

    task: str
    prompt_key: str
    input_ids: object
    source_id: int
    target_id: int
    success_ids: tuple[int, ...]
    answer: str | None = None
    answer_in_class: bool | None = None


def build_trials(
    model,
    *,
    max_seq_len: int,
    max_categories: int | None = None,
    max_items: int | None = None,
    categories: dict[str, tuple[str, ...]] | None = None,
    items: tuple[ProbeSwapItem, ...] | None = None,
) -> list[Trial]:
    """The trial list both causal experiments run on, built once.

    This once lived duplicated verbatim in three separate drivers, which is why the
    site-selection run needed a bit-identity check against a separate sweep to prove
    its comparator was scoring the same trials. One list removes that check by
    construction: every condition is paired with every other by *being the same trial*.

    Two conventions the construction turns on, both of which are silent when wrong:

    * **The model's own answer is excluded by vocabulary id, not by decoded string.**
      ``answer_id`` is a single greedy token, so a multi-token answer decodes to a
      fragment (``' Spar'`` for Sparrow, ``' L'`` for Liver) and a string comparison
      leaves the word the model was already going to emit in the candidate list -- where
      "swapping it in" succeeds for every arm, controls included.
    * **The swap is driven along the space-prefixed first token**
      (:func:`src.readout.target_direction_id`), not the first element of the
      id-sorted grading set, whose leading entry is a BPE accident (``Soccer -> 'S'``,
      ``Lemon -> 'L'``, ``elephant -> 'ele'``). Grading still accepts any surface form.

    probe-swap contributes only items the model already answers correctly: otherwise
    "success" would include items it was going to answer differently anyway.

    Args:
        categories: verbal-report candidates, defaulting to :func:`load_verbal_report`.
        items: probe-swap items, defaulting to :func:`load_probe_swap`. Both exist so the
            construction can be exercised on a fixture without the ``jlens`` data files.
    """
    import torch

    from src.readout import target_direction_id, target_token_ids

    if categories is None:
        categories = load_verbal_report()
    if items is None:
        items = load_probe_swap()
    if max_categories:
        categories = dict(list(categories.items())[:max_categories])
    if max_items:
        items = items[:max_items]

    def greedy_answer(input_ids):
        from src.model import final_residual

        with torch.no_grad():
            logits = model.unembed(final_residual(model, input_ids)[0, -1]).float()
        return int(logits.argmax())

    trials: list[Trial] = []
    for category, words in categories.items():
        input_ids = model.encode(
            VERBAL_REPORT_PROMPT.format(category=category), max_length=max_seq_len
        )
        answer_id = greedy_answer(input_ids)
        answer = model.tokenizer.decode([answer_id])
        in_class = any(
            answer_id in target_token_ids(model.tokenizer, w) for w in words
        )
        candidates = [
            w for w in words if answer_id not in target_token_ids(model.tokenizer, w)
        ][:N_CANDIDATES]
        for word in candidates:
            trials.append(
                Trial(
                    task="verbal-report",
                    prompt_key=f"verbal-report:{category}",
                    input_ids=input_ids,
                    source_id=answer_id,
                    target_id=target_direction_id(model.tokenizer, word),
                    success_ids=tuple(target_token_ids(model.tokenizer, word)),
                    answer=answer,
                    answer_in_class=in_class,
                )
            )

    for item in items:
        input_ids = model.encode(item.prompt, max_length=max_seq_len)
        if greedy_answer(input_ids) not in target_token_ids(model.tokenizer, item.answer):
            continue
        trials.append(
            Trial(
                task="probe-swap",
                prompt_key=f"probe-swap:{item.name}",
                input_ids=input_ids,
                source_id=target_direction_id(model.tokenizer, item.intermediate),
                target_id=target_direction_id(model.tokenizer, item.swap_to),
                success_ids=tuple(target_token_ids(model.tokenizer, item.swap_answer)),
            )
        )

    logger.info(
        "%d trials (%d verbal-report, %d probe-swap of %d items)",
        len(trials),
        sum(1 for t in trials if t.task == "verbal-report"),
        sum(1 for t in trials if t.task == "probe-swap"),
        len(items),
    )
    return trials
