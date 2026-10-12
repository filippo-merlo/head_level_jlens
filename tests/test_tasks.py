"""Task-set loading and trial construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.config import MAX_SEQ_LEN
from src.readout import target_direction_id, target_token_ids
from src.tasks import (
    N_CANDIDATES,
    VERBAL_REPORT_PROMPT,
    ProbeSwapItem,
    build_trials,
    load_probe_swap,
    load_verbal_report,
    target_rank,
)
from tests.tiny_llama import ByteTokenizer, tiny_llama


def test_verbal_report_shape():
    categories = load_verbal_report()
    assert len(categories) == 14
    assert all(len(words) == 14 for words in categories.values())
    assert "country" in categories and "Japan" in categories["country"]


def test_verbal_report_prompt_ends_in_a_colon():
    """The documented grading position is the final ``:``; it has to exist.

    Llama-3.1's chat template has no colon, which is why this prompt stays raw.
    """
    rendered = VERBAL_REPORT_PROMPT.format(category="country")
    assert rendered.endswith(":")


def test_probe_swap_shape():
    items = load_probe_swap()
    assert len(items) == 90
    first = items[0]
    for field in ("prompt", "intermediate", "answer", "swap_to", "swap_answer", "category"):
        assert getattr(first, field)
    # The swap must actually change the expected answer, or the item scores nothing.
    assert all(i.answer != i.swap_answer for i in items)


# ------------------------------------------------------- the shared trial construction

class WordTokenizer(ByteTokenizer):
    """Byte tokenisation, plus a word table so candidates get *distinct* ids.

    Byte-level tokenisation gives every space-prefixed word the same first token -- the
    space -- which collapses every trial onto one swap direction and makes the
    conventions under test vacuously true. The table also reproduces the shape of the
    real BPE trap: for some words the bare spelling's first token carries the smaller
    vocabulary id, so ``sorted`` grading candidates lead with a fragment.
    """

    def __init__(self, words: dict[str, list[int]] | None = None):
        self.words = dict(words or {})

    def __call__(self, text: str, *, add_special_tokens: bool = True, **kwargs):
        if not add_special_tokens and text in self.words:
            return SimpleNamespace(input_ids=list(self.words[text]))
        return super().__call__(text, add_special_tokens=add_special_tokens, **kwargs)


#: ``{word: (space-prefixed ids, bare ids)}``. Every bare form leads with a smaller id
#: than its space-prefixed twin, which is the case ``target_direction_id`` exists for.
WORDS = {
    "cat": ([31], [3, 41]),
    "dog": ([32], [4, 42]),
    "owl": ([33], [5, 43]),
    "fox": ([34], [6, 44]),
    "red": ([35], [7, 45]),
    "blue": ([36], [8, 46]),
    "green": ([37], [9, 47]),
    "Brasilia": ([38], [10, 48]),
    "Lisbon": ([39], [11, 49]),
    "Brazil": ([40], [12, 50]),
    "France": ([51], [13, 52]),
    "Paris": ([53], [14, 54]),
}
CATEGORIES = {"animal": ("cat", "dog", "owl", "fox"), "colour": ("red", "blue", "green")}
ITEMS = tuple(
    ProbeSwapItem(
        name=f"item{i}",
        category="geography",
        prompt=f"The capital of country {i} is",
        intermediate="Brazil",
        answer=answer,
        swap_to="France",
        swap_answer="Paris",
    )
    for i, answer in enumerate(("Brasilia", "Lisbon"))
)


def _table(extra: dict[str, list[int]] | None = None) -> dict[str, list[int]]:
    table = {}
    for word, (prefixed, bare) in WORDS.items():
        table[f" {word}"] = prefixed
        table[word] = bare
    table.update(extra or {})
    return table


@pytest.fixture
def model():
    built = tiny_llama()
    built.tokenizer = WordTokenizer(_table())
    return built


def _greedy(model, prompt: str) -> int:
    input_ids = model.encode(prompt, max_length=MAX_SEQ_LEN)
    with torch.no_grad():
        return int(
            model.unembed(model.forward(input_ids).last_hidden_state[0, -1]).argmax()
        )


def _build(model, **kwargs):
    return build_trials(
        model, max_seq_len=MAX_SEQ_LEN, categories=CATEGORIES, items=ITEMS, **kwargs
    )


def test_build_trials_groups_trials_by_the_prompt_they_share(model):
    """The loading map is cached on ``prompt_key``, so trials sharing a prompt must share
    a key and trials on different prompts must not."""
    trials = _build(model)
    assert {t.prompt_key for t in trials if t.task == "verbal-report"} == {
        "verbal-report:animal",
        "verbal-report:colour",
    }
    by_key = {}
    for trial in trials:
        by_key.setdefault(trial.prompt_key, set()).add(tuple(trial.input_ids.flatten().tolist()))
    assert all(len(prompts) == 1 for prompts in by_key.values())


def test_the_models_own_answer_is_excluded_by_vocabulary_id(model):
    """A candidate the model was already going to emit 'succeeds' for every arm, controls
    included. The exclusion is by id: a multi-token answer decodes to a fragment, so a
    string comparison leaves it in the list."""
    answer = _greedy(model, VERBAL_REPORT_PROMPT.format(category="animal"))
    # Give ' dog' the answer's own id, so the exclusion has something to catch.
    model.tokenizer.words = _table({" dog": [answer], "dog": [answer]})
    trials = [t for t in _build(model) if t.prompt_key == "verbal-report:animal"]
    assert trials, "the other three candidates must survive"
    assert all(t.target_id != answer for t in trials)
    assert all(t.source_id == answer for t in trials)


def test_the_direction_is_the_space_prefixed_token_and_grades_as_a_hit(model):
    """Grading accepts any surface form; the swap is driven along the one that names the
    concept. ``target_token_ids`` is id-sorted and leads with the bare form's fragment."""
    assert target_token_ids(model.tokenizer, "cat")[0] == 3
    assert target_direction_id(model.tokenizer, "cat") == 31
    for trial in _build(model):
        assert trial.target_id in trial.success_ids
        assert trial.target_id == max(trial.success_ids)


def test_probe_swap_keeps_exactly_the_items_the_model_already_answers(model):
    """Otherwise a pre-existing miss would be scored as a successful swap."""
    # Rig one item's answer onto the token the model actually emits, so the filter has
    # something to keep as well as something to drop.
    kept_item = ITEMS[0]
    model.tokenizer.words = _table(
        {f" {kept_item.answer}": [_greedy(model, kept_item.prompt)]}
    )
    expected = {
        item.name
        for item in ITEMS
        if _greedy(model, item.prompt) in target_token_ids(model.tokenizer, item.answer)
    }
    assert expected == {kept_item.name}
    kept = {
        t.prompt_key.removeprefix("probe-swap:")
        for t in _build(model)
        if t.task == "probe-swap"
    }
    assert kept == expected


def test_probe_swap_drops_every_item_when_none_is_answered_correctly(model):
    """The fixture's random weights answer nothing; the construction must yield no
    probe-swap trials rather than scoring them all."""
    assert all(
        _greedy(model, item.prompt) not in target_token_ids(model.tokenizer, item.answer)
        for item in ITEMS
    )
    assert not [t for t in _build(model) if t.task == "probe-swap"]


def test_verbal_report_takes_at_most_ten_candidates_per_category(model):
    words = tuple(WORDS)[:12]
    trials = build_trials(
        model,
        max_seq_len=MAX_SEQ_LEN,
        categories={"thing": words},
        items=(),
    )
    assert len(words) > N_CANDIDATES
    assert len(trials) == N_CANDIDATES


def test_caps_bound_both_task_sets(model):
    trials = _build(model, max_categories=1)
    assert {t.prompt_key for t in trials if t.task == "verbal-report"} == {
        "verbal-report:animal"
    }


def test_answer_in_class_is_recorded_not_corrected(model):
    """On some categories the model does not name a member of the class at all, so the
    direction swapped out is not a concept direction. The trial still runs; the flag lets
    a reader take the subset where the task's premise holds."""
    for trial in _build(model):
        if trial.task != "verbal-report":
            assert trial.answer_in_class is None
            continue
        candidates = CATEGORIES[trial.prompt_key.removeprefix("verbal-report:")]
        assert trial.answer_in_class == any(
            trial.source_id in target_token_ids(model.tokenizer, w) for w in candidates
        )


def test_target_rank_counts_tokens_scoring_strictly_higher_than_the_best_form():
    logits = torch.tensor([[0.0, 5.0, 3.0, 4.0, 1.0],
                           [9.0, 5.0, 3.0, 4.0, 1.0],
                           [0.0, 1.0, float("nan"), 0.0, 0.0]])
    ranks = target_rank(logits, (2, 3))      # best form is id 3 (logit 4.0) in rows 0-1
    assert ranks[:2].tolist() == [1.0, 2.0]
    assert torch.isnan(ranks[2]), "a non-finite row must not read as rank 0"
    assert target_rank(logits[0], (1,)).item() == 0.0
