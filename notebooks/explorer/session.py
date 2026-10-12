"""Load the model and both lenses once, and prove they are the results' own bytes.

The load is the notebook's one expensive step (8B in bf16 plus a 2 GB fp32 ``J^z``,
~90 s) and depends on nothing a widget can change, so it is a single call returning a
single frozen object. :func:`self_check` is the reason the rest of the notebook can be
believed: it re-reads cells the pilot run already dumped and refuses to continue if the
two disagree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from jlens.examples import EXAMPLES, resolve_prompt

from src.config import DEFAULT_LENS, LENSES, LensSpec, Paths, jz_path
from src.grid import head_grid
from src.heads import residual_o_proj_weight
from src.model import load_jz, load_lens, load_model
from src.provenance import file_digest


@dataclass(frozen=True)
class LensSession:
    """Both lenses, staged on the device, with the provenance of the head-level one.

    Attributes:
        model: ``jlens`` ``LensModel`` wrapper around the HF checkpoint.
        published: The layer-output Jacobian lens ``J_l^out`` (J-layer), Neuronpedia's.
        jz_by_layer: ``{layer: J^z_l}`` on the model's input device -- the J-head map.
        w_o_by_layer: ``{layer: W_O}`` as ``scripts/read.py`` reads it
            (:func:`src.heads.residual_o_proj_weight`) -- the Logit-head map.
        layers: Head layers, every layer ``J^z`` covers (0-31 on Llama).
        ref_layers: Block outputs the Jacobian lens covers (0-30): its top block is its
            own target, so J-layer has no readout at the last layer.
        jz_sha256: Digest of the ``J^z`` file actually read -- the bytes, not the name.
    """

    spec: LensSpec
    paths: Paths
    model: Any
    published: Any
    jz_by_layer: dict[int, torch.Tensor]
    w_o_by_layer: dict[int, torch.Tensor]
    layers: list[int]
    ref_layers: list[int]
    jz_path: Path
    jz_sha256: str
    jz_n_prompts: int

    @classmethod
    def load(cls, lens_key: str = DEFAULT_LENS) -> LensSession:
        """Load the checkpoint, the fitted ``J^z`` and the Jacobian lens. ~90 s."""
        spec = LENSES[lens_key]
        paths = Paths.default()
        jz_file = Path(jz_path(spec, n_prompts=spec.jz_n_prompts, paths=paths))

        model = load_model(spec)
        jz = load_jz(spec, str(jz_file))  # digest-verified: the bytes, not the filename
        published = load_lens(spec)

        layers = list(spec.jz_source_layers)
        device = model.input_device
        return cls(
            spec=spec,
            paths=paths,
            model=model,
            published=published,
            jz_by_layer={l: jz.jacobians[l].to(device) for l in layers},
            w_o_by_layer={l: residual_o_proj_weight(model, l) for l in layers},
            layers=layers,
            ref_layers=list(spec.source_layers),
            jz_path=jz_file,
            jz_sha256=file_digest(jz_file),
            jz_n_prompts=jz.n_prompts,
        )

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def vocab_size(self) -> int:
        """Rows of ``W_U`` -- what a rank is out of, and every rank scale's top end.

        Read off ``_lm_head`` because that is where ``LensModel`` keeps the unembedding,
        the same reach-through ``src.heads.effective_unembed`` makes.
        """
        return int(self.model._lm_head.weight.shape[0])

    @property
    def n_sites(self) -> int:
        return len(self.layers) * self.spec.n_heads

    def decode(self, token_id: int) -> str:
        """One token as text, spaces kept: ``' France'`` and ``'France'`` differ."""
        return self.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def self_check(
    session: LensSession,
    *,
    example: str = "multihop",
    layers: tuple[int, ...] = (12, 16, 20, 24),
    heads: tuple[int, ...] = (0, 8, 16, 24),
    top_n: int = 6,
) -> dict[str, Any]:
    """Reproduce cells of ``results/pilot/`` exactly, or refuse to continue.

    ``results/pilot/`` dumped the J-head top-6 for the multihop example of Gurnee et al.
    at these layers and heads. Reproducing it says the notebook is reading the same lens
    bytes at the same anchor as the pipeline that produced every number in the paper.

    Raises:
        AssertionError: On any disagreement. Raising rather than rendering a warning is
            deliberate: if this disagrees, everything below is drawn from something
            other than the lens the results were computed on, and a red cell that can be
            scrolled past is not a strong enough signal.

    Returns:
        ``{"n_cells", "layers", "heads", "example", "top_n"}`` -- what was checked.
    """
    recorded = json.loads(
        (session.paths.results_root / "pilot" / f"{session.spec.key}_pilot.json").read_text()
    )["qualitative"][example]["top_tokens"]

    item = next(e for e in EXAMPLES if e.slug == example)
    input_ids = session.model.encode(resolve_prompt(item, session.tokenizer))
    grid = head_grid(
        session.model,
        session.jz_by_layer,
        session.spec,
        input_ids,
        int(input_ids.shape[-1]) - 1,
        layers=list(layers),
        top_n=top_n,
    )
    # Decoded exactly as src.readout.top_tokens does it, which is what the pilot recorded.
    mismatches = [
        f"L{layer} h{head}"
        for row, layer in enumerate(grid.rows)
        for head in heads
        if [session.tokenizer.decode([int(i)]) for i in grid.top_ids[row, head]]
        != recorded[str(layer)][str(head)]
    ]
    if mismatches:
        raise AssertionError(
            f"self-check failed at {', '.join(mismatches)}: this notebook disagrees "
            "with results/pilot/. The lens file, the anchor or the readout "
            "position is not what the pilot read."
        )
    return {
        "n_cells": len(layers) * len(heads),
        "layers": list(layers),
        "heads": list(heads),
        "example": example,
        "top_n": top_n,
    }
