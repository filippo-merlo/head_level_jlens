"""Prompts drawn from the corpus the lenses were fitted on.

Using the fit corpus (wikitext-103-raw-v1, per each lens's ``config.yaml``) keeps the
activations in the range ``J_l`` was estimated over, which is free and avoids having to
argue the point later.

Selection is deterministic: a fixed seed over a fixed-size pool taken from the head of
the training split, so a re-run reproduces the same prompts.

The pool is read straight from the cached parquet with pyarrow rather than through
``datasets``' streaming loader. The streamed reader talks HTTP over ``fsspec``/
``aiohttp``, whose session is torn down during interpreter finalisation and aborts the
process (SIGABRT) *after* the work has finished -- turning a successful multi-hour fit
into an apparent crash. Reading the file directly removes the async machinery
altogether; it is also ~50x faster once cached, and yields byte-identical prompts
(verified against the streamed order).
"""

from __future__ import annotations

import logging
import random
from functools import cache

from src.config import FIT_DATASET, FIT_DATASET_FILE

logger = logging.getLogger(__name__)

#: Neuronpedia's fitting script truncated each document to this many characters before
#: tokenising; matching it keeps our prompts the same shape as the fit's.
MAX_CHARS = 2000

#: Documents pulled from the head of the split before sampling. Large enough that the
#: sample is not just "the first N documents", small enough to stay fast.
POOL_SIZE = 4000

#: wikitext has many near-empty lines and section headers; require real prose.
MIN_CHARS = 600


@cache
def _document_pool() -> tuple[str, ...]:
    """The first :data:`POOL_SIZE` usable documents of the training split.

    Cached: the pool is deterministic and every ``load_prompts`` call re-scanned the
    parquet to rebuild the identical list. Same idiom as ``src.evals`` and ``src.tasks``.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(FIT_DATASET, FIT_DATASET_FILE, repo_type="dataset")
    parquet = pq.ParquetFile(path)
    pool: list[str] = []
    for batch in parquet.iter_batches(batch_size=2000, columns=["text"]):
        for document in batch.column("text").to_pylist():
            text = document.strip()
            if len(text) >= MIN_CHARS and not text.startswith("="):
                pool.append(text[:MAX_CHARS])
        if len(pool) >= POOL_SIZE:
            break
    return tuple(pool[:POOL_SIZE])


def load_prompts(n_prompts: int, *, seed: int = 0) -> list[str]:
    """Return ``n_prompts`` deterministic wikitext documents.

    Raises:
        RuntimeError: If the split yields too few usable documents.
    """
    pool = _document_pool()
    if len(pool) < n_prompts:
        raise RuntimeError(
            f"only {len(pool)} usable documents in the first {POOL_SIZE} records of "
            f"{FIT_DATASET}, need {n_prompts}"
        )
    chosen = random.Random(seed).sample(list(pool), n_prompts)
    logger.info("sampled %d prompts from %s (seed=%d)", n_prompts, FIT_DATASET, seed)
    return chosen
