"""Write what is on screen out with enough provenance to trace it back.

A figure that leaves the notebook without the prompt, the position, the edit and the
``J^z`` digest it was drawn from is untraceable, and this notebook's figures are the ones
most likely to end up pasted somewhere. Everything goes under
``results/notebook_figures/`` (gitignored: exploratory by construction), with a ``meta``
block from :func:`src.provenance.meta_block`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.provenance import meta_block

from .editing import TrialInfo
from .prompts import Selection
from .readings import Readings
from .session import LensSession

#: Under ``results/``, but named so nothing mistakes it for a driver's output.
FIGURE_DIR = "notebook_figures"


def save(
    session: LensSession,
    *,
    selection: Selection,
    reading: Readings,
    trial: TrialInfo,
    edit: dict,
    charts: dict[str, object],
) -> str:
    """Save the reading, the edit and each chart (PNG, via ``vl-convert``) plus
    ``meta.json``; return what happened, as markdown."""
    out = Path(session.paths.results_root) / FIGURE_DIR
    out.mkdir(parents=True, exist_ok=True)

    written = []
    for name, chart in charts.items():
        # Charts draw on a transparent background in the notebook; a file needs its own.
        chart.properties(background="#fcfcfb").save(str(out / f"{name}.png"), scale_factor=2)
        written.append(f"{name}.png")

    reading_record = {
        "prompt": selection.text,
        "position": reading.position,
        "concepts": [
            {
                "word": c.word,
                "candidate_ids": c.ids,
                "rule": c.rule,
                "best": {
                    arm: {"rank": c.best(arm)[0] + 1, "site": list(c.best(arm)[1])}
                    for arm in ("jhead", "logithead", "jlayer")
                },
                "ranks_zero_based": {arm: r.tolist() for arm, r in c.ranks.items()},
            }
            for c in reading.concepts
        ],
        "rho": reading.rho.tolist(),
    }
    edit_record = {
        "trial": {
            "index": trial.index,
            "task": trial.trial.task,
            "prompt": trial.prompt,
            "source": trial.source,
            "target": trial.target,
            "answer": trial.answer,
        },
        **edit,
    }
    (out / "reading.json").write_text(json.dumps(reading_record, indent=1))
    (out / "edit.json").write_text(json.dumps(edit_record, indent=1))
    (out / "meta.json").write_text(
        json.dumps(
            meta_block(
                model=session.spec.hf_model_id,
                lens_key=session.spec.key,
                jz_path=str(session.jz_path),
                jz_sha256=session.jz_sha256,
                published_lens_n_prompts=session.published.n_prompts,
                prompt=selection.text,
                position=reading.position,
                trial=trial.label,
                gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                source="notebooks/head_lens_explorer.py (exploratory; not a result)",
                files=[*written, "reading.json", "edit.json"],
            ),
            indent=1,
        )
    )
    return (
        f"Wrote {', '.join(written + ['reading.json', 'edit.json', 'meta.json'])} to "
        f"`{out}`. The meta block carries the prompt, position, trial and `jz_sha256`, "
        "so anything taken out of this notebook can be traced to the bytes it came from."
    )
