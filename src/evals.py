"""The six lens-quality eval sets, and where each one is read out.

Shipped in the ``jlens`` repo under ``data/evaluations/``, with the protocol documented
in its ``README.md``. Each item carries a ``prompt`` and one or more ``intermediates``:
concepts the model must be working with to answer, but which it never writes. The metric
is pass@k -- the mean fraction of intermediates whose min-over-layers lens rank is at
most ``k``, read at one documented position per set.

``target`` where present defines the readout position only and is **not** scored; the
whole point is to read what the model is thinking rather than what it is about to say.

The readout position is per-set and documented rather than inferred. Five of the six read
at the final prompt token; poetry reads at the newline ending the couplet's first line,
because that is where the rhyme target has to be decided.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)

#: ``data/evaluations`` inside the installed ``jlens`` package.
EVAL_DIR = Path("data/evaluations")

#: How each set locates its single readout position.
#:
#: ``"final"``   -- the last prompt token. Covers the sets whose prompt stops just before
#:                  the answer ("... is the "), those reading the closing period, and the
#:                  typo set reading the misspelling's last fragment.
#: ``"newline"`` -- the last newline token, ending line 1 of the couplet.
READOUT_POSITION: dict[str, str] = {
    "lens-eval-multihop": "final",
    "lens-eval-multilingual": "final",
    "lens-eval-poetry": "newline",
    "lens-eval-order-ops": "final",
    "lens-eval-association": "final",
    "lens-eval-typo": "final",
}

#: Sets whose intermediates are keys standing for a synonym set rather than literal words.
NEEDS_SYNONYM_EXPANSION = frozenset({"lens-eval-order-ops"})

#: Sets scored on **single-token** surface forms only, dropping spellings the tokenizer
#: splits rather than scoring them by their first fragment.
#:
#: Separate from :data:`NEEDS_SYNONYM_EXPANSION` because the two answer different
#: questions -- which forms count, versus which of those forms can be scored -- even
#: though order-ops needs both.
#:
#: * ``order-ops``: the README specifies it ("rank is the min over single-token
#:   synonyms").
#: * ``multihop``: added here. Its digit-form intermediates have a space-prefixed
#:   spelling that Llama-3.1 splits as ``[' ', '3']``, so the first-fragment proxy
#:   contributes the **bare space**, token 220 -- a candidate carrying no information
#:   about the concept, which an arbitrary direction hits far above chance. Dropping
#:   multi-token spellings removes it. Verified to leave every multihop intermediate
#:   scoreable: 103 of 103 slots keep at least one single-token candidate.
#:
#: Not applied to the remaining four sets, which the artefact does not touch (no slot
#: carries a whitespace candidate) and where the rule would *cost* coverage: it would
#: leave 14 multilingual intermediates ("Estonian", "português", ...) and 6 association
#: intermediates ("école", "père", ...) with no candidate at all. Pinned by
#: ``tests/test_passk.py``.
SINGLE_TOKEN_ONLY = frozenset({"lens-eval-order-ops", "lens-eval-multihop"})

#: The order-ops expansion, authored from the README's description ("numbers -> digit and
#: word forms; operations -> symbol and word forms") because the mapping is documented
#: but not shipped in the JSON. Scoring this set literally would understate it: the model
#: may well be holding "5" while the item names the intermediate "five", or holding "*"
#: while the item says "multiplication".
#:
#: The README also says rank is the min over **single-token** synonyms, so
#: :func:`src.readout.target_token_ids` filters these to forms the tokenizer represents
#: as one token. Multi-token forms like "multiplication" therefore drop out on Llama-3.1
#: rather than being scored by their first fragment, which would be a different and much
#: easier target.
_NUMBER_WORDS = {
    "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight",
    "9": "nine", "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "15": "fifteen", "16": "sixteen", "20": "twenty", "24": "twenty-four",
}
SYNONYMS: dict[str, tuple[str, ...]] = {
    **{digit: (digit, word) for digit, word in _NUMBER_WORDS.items()},
    "addition": ("+", "addition", "plus", "add", "sum"),
    "subtraction": ("-", "subtraction", "minus", "subtract"),
    "multiplication": ("*", "multiplication", "times", "multiply", "product"),
    "division": ("/", "division", "divide", "divided", "quotient"),
    "squared": ("squared", "square", "^", "**"),
    "mod": ("mod", "%", "modulo", "remainder"),
}


def target_forms(item: EvalItem, intermediate: str) -> tuple[str, ...]:
    """Surface forms that count as reading ``intermediate``.

    One form for most sets -- the intermediate itself -- and the documented synonym set
    for order-ops.

    Raises:
        ValueError: If an order-ops intermediate has no authored expansion. Silently
            falling back to the literal string would score that item on a harder target
            than its siblings and quietly depress the set's pass@k.
    """
    if item.slug not in NEEDS_SYNONYM_EXPANSION:
        return (intermediate,)
    if intermediate not in SYNONYMS:
        raise ValueError(
            f"{item.slug}: intermediate {intermediate!r} has no authored synonym set; "
            f"add it to src.evals.SYNONYMS"
        )
    return SYNONYMS[intermediate]

ALL_SETS = tuple(READOUT_POSITION)


@dataclass(frozen=True)
class EvalItem:
    """One scored prompt."""

    name: str
    prompt: str
    intermediates: tuple[str, ...]
    slug: str


def eval_root() -> Path:
    """Locate ``data/evaluations`` next to the installed ``jlens`` package."""
    import jlens

    root = Path(jlens.__file__).resolve().parent.parent / EVAL_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} not found; the jlens checkout should ship data/evaluations/"
        )
    return root


@cache
def load_eval(slug: str) -> tuple[EvalItem, ...]:
    """Load one eval set.

    Raises:
        ValueError: If ``slug`` is not one of the six documented sets, or an item is
            missing the fields the metric needs.
    """
    if slug not in READOUT_POSITION:
        raise ValueError(f"unknown eval set {slug!r}; expected one of {ALL_SETS}")
    payload = json.loads((eval_root() / f"{slug}.json").read_text())
    items = []
    for raw in payload["items"]:
        if not raw.get("intermediates"):
            raise ValueError(f"{slug}: item {raw.get('name')!r} has no intermediates")
        items.append(
            EvalItem(
                name=raw["name"],
                prompt=raw["prompt"],
                intermediates=tuple(raw["intermediates"]),
                slug=slug,
            )
        )
    logger.info("loaded %s: %d items", slug, len(items))
    return tuple(items)


def readout_index(item: EvalItem, input_ids, tokenizer) -> int:
    """Index of the position this item is read out at.

    Args:
        input_ids: ``[1, seq_len]`` as returned by ``model.encode``.

    Raises:
        ValueError: If a ``newline`` set's prompt has no newline token to read at --
            silently falling back to the final token would score the wrong position and
            look like a lens failure.
    """
    mode = READOUT_POSITION[item.slug]
    seq_len = input_ids.shape[-1]
    if mode == "final":
        return seq_len - 1
    if mode == "newline":
        for index in range(seq_len - 1, -1, -1):
            if "\n" in tokenizer.decode([int(input_ids[0, index])]):
                return index
        raise ValueError(f"{item.slug}: item {item.name!r} has no newline token")
    raise ValueError(f"unknown readout mode {mode!r} for {item.slug}")


def candidate_coverage(tokenizer, slug: str) -> dict[str, object]:
    """How many of a set's intermediates the candidate rule can score under ``tokenizer``.

    The rule is the metric's and does not change per model, but what it keeps does: a
    form that is one token under Llama's tokenizer may be two under another, and for
    the sets in :data:`SINGLE_TOKEN_ONLY` that leaves the slot with no candidate at all,
    which :func:`src.readout.target_token_ids` refuses to score. Recorded beside every
    read record so a cross-model comparison states what each tokenizer could score.

    Returns:
        ``{"n_intermediates", "n_scoreable", "unscoreable": [(item, word), ...],
        "items_without_candidates": [item, ...]}`` -- the last lists the items whose
        every intermediate is unscoreable, which :func:`src.passk.passk` leaves out.
    """
    from src.readout import target_token_ids

    n_total = 0
    unscoreable: list[tuple[str, str]] = []
    blank_items: list[str] = []
    for item in load_eval(slug):
        n_before = len(unscoreable)
        for word in item.intermediates:
            n_total += 1
            try:
                target_token_ids(
                    tokenizer, target_forms(item, word), single_token_only=slug in SINGLE_TOKEN_ONLY
                )
            except ValueError:
                unscoreable.append((item.name, word))
        if item.intermediates and len(unscoreable) - n_before == len(item.intermediates):
            blank_items.append(item.name)
    return {
        "n_intermediates": n_total,
        "n_scoreable": n_total - len(unscoreable),
        "unscoreable": unscoreable,
        "items_without_candidates": blank_items,
    }


#: Seed of the cycle :func:`control_items` walks; fixed so the control is one draw, recorded.
CONTROL_SEED = 0


def control_items(
    candidates: list[list[frozenset[int] | None]], seed: int = CONTROL_SEED
) -> list[list[int | None]]:
    """For each (item, intermediate), the item of the same set whose prompt reads it as a control.

    Counting the sites at which a concept ranks high says how many heads carry it only
    against the same count on a prompt that is not about it: a readout that ranks a
    token high wherever it is read would inflate both. The control prompt is the next
    item in one seeded random cycle over the set whose own intermediates share no
    candidate token with the concept -- the same tokens, scored where they are not
    annotated.

    Args:
        candidates: per item, per intermediate, its candidate token ids, or ``None``
            where the candidate rule could not score it.

    Returns:
        The same nesting, with the control item's index (``None`` stays ``None``).

    Raises:
        ValueError: If every other item shares a candidate with some concept.
    """
    import numpy as np

    n = len(candidates)
    order = np.random.default_rng(seed).permutation(n)
    position = np.empty(n, dtype=int)
    position[order] = np.arange(n)
    own = [frozenset().union(*(ids for ids in item if ids is not None)) for item in candidates]
    controls: list[list[int | None]] = []
    for i, item in enumerate(candidates):
        row: list[int | None] = []
        for ids in item:
            if ids is None:
                row.append(None)
                continue
            following = (int(order[(position[i] + step) % n]) for step in range(1, n))
            j = next((j for j in following if not own[j] & ids), None)
            if j is None:
                raise ValueError(f"item {i}: every other item shares a candidate with {set(ids)}")
            row.append(j)
        controls.append(row)
    return controls
