"""The prompt menu, where each prompt is read out, and which tokens count as a concept.

Everything the reading section draws is read at **one position** of one prompt, so the
menu, the position rule and the candidate rule belong together:

* An evaluation item is read where its set says (:func:`src.evals.readout_index`:
  poetry at the newline ending the couplet's first line, every other set at the final
  token), and its intermediates are scored with the set's own candidate rule
  (:func:`candidates`) -- exactly what ``scripts/read.py`` does, so the notebook can be
  checked against ``results/read/``.
* Everything else is read at the final token, and a typed concept is scored on the first
  token of its bare and space-prefixed spellings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from jlens.examples import EXAMPLES, resolve_prompt

from src.config import MAX_SEQ_LEN
from src.evals import (
    ALL_SETS,
    SINGLE_TOKEN_ONLY,
    EvalItem,
    load_eval,
    readout_index,
    target_forms,
)
from src.readout import target_token_ids
from src.tasks import load_probe_swap

#: The paper's running example (Section 1): Lyon -> France -> Paris.
DEFAULT_LABEL = "intermediate replacement — ex-city-capital-Lyon-Naples"


@dataclass(frozen=True)
class Source:
    """One menu entry: a prompt, the concepts it is about, and the eval item if any."""

    text: str
    targets: tuple[str, ...] = ()
    item: EvalItem | None = None


@dataclass(frozen=True)
class Selection:
    """One chosen prompt, tokenized, with the position its set says to read at."""

    label: str
    text: str
    item: EvalItem | None
    input_ids: torch.Tensor
    token_strs: list[str]
    default_position: int
    default_targets: tuple[str, ...]

    @property
    def set_name(self) -> str | None:
        return None if self.item is None else self.item.slug.removeprefix("lens-eval-")


def _strip_double_bos(text: str, tokenizer) -> str:
    """Drop a leading BOS from an already-templated prompt.

    The model is loaded with ``force_bos=True`` (upstream's default, correct for the raw
    completion-style prompts this project uses), which sets ``add_bos_token``. A chat
    template has already written a BOS, so drop it rather than encode two.
    """
    bos = getattr(tokenizer, "bos_token", None)
    return text.removeprefix(bos) if bos and text.startswith(bos) else text


def library(model: Any) -> dict[str, Source]:
    """Every prompt the paper uses, then Gurnee et al.'s worked examples.

    The six evaluation sets (Section 3), then the intermediate-replacement prompts
    (Section 4.1) with their annotated intermediate as the concept to read.
    """
    sources: dict[str, Source] = {}
    for item in load_probe_swap():
        sources[f"intermediate replacement — {item.name}"] = Source(
            item.prompt, (item.intermediate,)
        )
    for slug in ALL_SETS:
        for item in load_eval(slug):
            sources[f"{slug.removeprefix('lens-eval-')} — {item.name}"] = Source(
                item.prompt, item.intermediates, item
            )
    for example in EXAMPLES:
        sources[f"Gurnee et al. example — {example.slug}"] = Source(
            _strip_double_bos(resolve_prompt(example, model.tokenizer), model.tokenizer)
        )
    return sources


def select(model: Any, sources: dict[str, Source], label: str, custom: str = "") -> Selection:
    """Resolve the prompt widgets into one tokenized selection.

    Args:
        label: Key into ``sources``, from the dropdown.
        custom: Free text, which overrides the dropdown when non-blank. It carries no
            eval item, so it reads at the final token with no default concept.
    """
    source = Source(custom) if custom.strip() else sources[label]
    input_ids = model.encode(source.text, max_length=MAX_SEQ_LEN)
    token_strs = [
        model.tokenizer.decode([int(t)], clean_up_tokenization_spaces=False)
        for t in input_ids[0]
    ]
    return Selection(
        label="custom prompt" if custom.strip() else label,
        text=source.text,
        item=source.item,
        input_ids=input_ids,
        token_strs=token_strs,
        default_position=(
            readout_index(source.item, input_ids, model.tokenizer)
            if source.item is not None
            else len(token_strs) - 1
        ),
        default_targets=source.targets,
    )


def candidates(tokenizer, selection: Selection, word: str) -> tuple[list[int], str]:
    """Vocabulary ids that count as reading ``word``, and the rule that chose them.

    An annotated intermediate of an evaluation item gets its set's rule (Section 3.3):
    order-ops' synonym expansion, single-token spellings only on multihop and order-ops.
    Anything else gets the default rule.

    Raises:
        ValueError: If the rule leaves no candidate (a multi-token word on a
            single-token set) -- an unscoreable concept, not a lens failure.
    """
    item = selection.item
    if item is not None and word in item.intermediates:
        single = item.slug in SINGLE_TOKEN_ONLY
        ids = target_token_ids(tokenizer, target_forms(item, word), single_token_only=single)
        rule = f"the {selection.set_name} rule" + (" (single-token spellings)" if single else "")
        return ids, rule
    return target_token_ids(tokenizer, word), "first token, bare and space-prefixed"
