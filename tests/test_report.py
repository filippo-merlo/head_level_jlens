"""``scripts.report`` reads records and says what they contain; nothing is recomputed."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

SHIPPED = Path(__file__).resolve().parents[1] / "results" / "read" / "llama3.1-8b-it_read.json"


@pytest.mark.skipif(not SHIPPED.exists(), reason="needs the shipped Llama read record")
def test_report_says_which_items_a_tokenizer_could_not_score(tmp_path, monkeypatch, capsys):
    """A set whose rates average over fewer items than it has (src.passk.scoreable_items)
    is announced with the items and the words the candidate rule dropped; a set with
    every item scored gets no such line."""
    from scripts import report

    record = json.loads(SHIPPED.read_text())
    multihop = record["results"]["lens-eval-multihop"]
    multihop["n_items_scored"] = multihop["n_items"] - 2
    for slug, entry in record["results"].items():
        if slug != "lens-eval-multihop" and slug.startswith("lens-eval"):
            entry["n_items_scored"] = entry["n_items"]
    record["meta"]["candidate_coverage"] = {
        "lens-eval-multihop": {
            "n_intermediates": 103,
            "n_scoreable": 100,
            "unscoreable": [["nhop-deck-element", "52"], ["roman-states-us", "50"],
                            ["two-hop-city", "Paris"]],
            "items_without_candidates": ["nhop-deck-element", "roman-states-us"],
        }
    }
    root = tmp_path / "results"
    (root / "read").mkdir(parents=True)
    (root / "read" / "llama3.1-8b-it_read.json").write_text(json.dumps(record))
    monkeypatch.setenv("RESULTS_ROOT", str(root))
    monkeypatch.setattr(report, "best_site_agreement_table", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["report", "--lens", "llama3.1-8b-it"])
    assert report.main() == 0
    out = (root / "report" / "llama3.1-8b-it_report.md").read_text()
    n = multihop["n_items"]
    assert f"multihop {n - 2}/{n} (nhop-deck-element: 52, roman-states-us: 50)" in out
    assert "two-hop-city" not in out  # that item kept another intermediate
    assert out.count("Items scored") == 1
    assert out.count("| set | J-head | Logit-head | J-layer | ratio | paired diff ") == 3
    shutil.rmtree(root)


SHIPPED_WRITE = SHIPPED.parents[1] / "write" / "llama3.1-8b-it_write.json"


@pytest.mark.skipif(
    not (SHIPPED.exists() and SHIPPED_WRITE.exists()), reason="needs the shipped Llama records"
)
def test_report_says_what_the_model_answered_when_it_is_not_a_class_member(
    tmp_path, monkeypatch, capsys
):
    """The verbal-report premise -- the token swapped out names a class member -- is
    read off the record: the in-class count and, where it fails, the answers."""
    from scripts import report

    write = json.loads(SHIPPED_WRITE.read_text())
    selection = write["results"]["verbal-report"]["selection"]
    selection["n_trials_answer_in_class"] = 20
    selection["answers"] = {
        "verbal-report:fruit": {"answer": " Apple", "in_class": True},
        "verbal-report:river": {"answer": "\n\n", "in_class": False},
        "verbal-report:city": {"answer": " What", "in_class": False},
    }
    root = tmp_path / "results"
    (root / "read").mkdir(parents=True)
    (root / "write").mkdir()
    shutil.copy(SHIPPED, root / "read" / "llama3.1-8b-it_read.json")
    (root / "write" / "llama3.1-8b-it_write.json").write_text(json.dumps(write))
    monkeypatch.setenv("RESULTS_ROOT", str(root))
    monkeypatch.setattr(report, "best_site_agreement_table", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["report", "--lens", "llama3.1-8b-it"])
    assert report.main() == 0
    out = (root / "report" / "llama3.1-8b-it_report.md").read_text()
    n = write["results"]["verbal-report"]["n_trials"]
    assert f"member of its category on 20 of {n} trials; elsewhere it is ' What', '\\n\\n'." in out
    assert out.count("member of its category") == 1  # probe-swap has no such line
    shutil.rmtree(root)


@pytest.mark.skipif(not SHIPPED.exists(), reason="needs the shipped Llama read record")
@pytest.mark.parametrize("with_counts", [True, False])
def test_report_prints_site_counts_beside_their_control(tmp_path, monkeypatch, with_counts):
    """Each set's counts and controls are printed as recorded, with the paired difference
    of the two arms' excess over control; a record without counts gets no such table."""
    from scripts import report

    record = json.loads(SHIPPED.read_text())
    for entry in record["results"].values():
        entry.pop("sites_hit", None)
        if with_counts:
            entry["sites_hit"] = {
                str(k): {
                    "jhead": {"concept": 9.0, "control": 1.5},
                    "logithead": {"concept": 4.0, "control": 0.25},
                }
                for k in report.K_VALUES
            }
    root = tmp_path / "results"
    (root / "read").mkdir(parents=True)
    (root / "read" / "llama3.1-8b-it_read.json").write_text(json.dumps(record))
    monkeypatch.setenv("RESULTS_ROOT", str(root))
    monkeypatch.setattr(report, "best_site_agreement_table", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["report", "--lens", "llama3.1-8b-it"])
    assert report.main() == 0
    out = (root / "report" / "llama3.1-8b-it_report.md").read_text()
    n_tables = len(report.K_VALUES) if with_counts else 0
    assert out.count("paired diff of excess") == n_tables
    assert out.count("| multihop | 9.0 | 1.5 | 4.0 | 0.2 | +3.8 |") == n_tables
    shutil.rmtree(root)
