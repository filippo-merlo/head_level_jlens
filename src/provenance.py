"""The ``meta`` block every results file carries.

A number is only reproducible if you can tell what produced it. Each JSON written under
``results/`` embeds :func:`meta_block`, so a figure can always be traced back to a
checkpoint, a lens file, a corpus seed and a commit -- including whether the working
tree was dirty at the time, which is the usual reason a result cannot be reproduced.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT


def git_revision() -> dict[str, Any]:
    """``{"sha": ..., "dirty": ...}``; ``sha`` is ``None`` outside a git checkout."""
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return None

    sha = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def file_digest(path: str | Path) -> str:
    """SHA-256 of a file, read in chunks so a multi-gigabyte lens does not land in RAM.

    Recorded for every lens a result depends on: filenames get reused, contents do not.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meta_block(**fields: Any) -> dict[str, Any]:
    """Provenance for one results file: the fixed part plus whatever the caller adds.

    Callers are expected to pass at least the model id, the lens file(s), the corpus
    seed and the dtype -- whatever it would take to run the thing again.
    """
    import torch
    import transformers

    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_revision(),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        **fields,
    }
