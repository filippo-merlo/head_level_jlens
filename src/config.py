"""Filesystem layout, model/lens identifiers, and fit metadata.

Everything that knows *where things live* is here, so no other module hardcodes a path.

Three facts about the lenses are recorded here rather than rediscovered at run time,
because getting any of them wrong silently reads the lens in a basis it was not
estimated in:

* A lens must be applied to the **same HF checkpoint** it was fitted on
  (``LensSpec.hf_model_id``).
* The Jacobian lens's ``source_layers`` stop one short of the model's layer count --
  the top layer is the *target*, not a source.
* ``J^z`` covers **all** ``n_layers`` layers, because a layer's attention writes before
  that same layer's MLP. The layer *count* is the only thing distinguishing a ``J^z``
  file from a Jacobian-lens file on disk.

Two anchors, in the paper's notation:

===============  ==========================================  ====================
Notation         Hook point                                  Artifact
===============  ==========================================  ====================
``J_l``          block output (``x_{l+1}``)                   Neuronpedia's release
``J^z_l``        ``o_proj`` input (``z_l``)                   fitted here, see src.fit
===============  ==========================================  ====================

The ``attn`` anchor (post-attention/pre-MLP) is still hooked -- ``src.magnitude`` needs
``||x_mid||`` and ``src.validation`` fits it as the consistency reference -- but nothing
is stored at it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: ``.../<project>/src``
PACKAGE_ROOT = Path(__file__).resolve().parent
#: ``.../<project>``
PROJECT_ROOT = PACKAGE_ROOT.parent

#: Prompts come from the corpus each lens was fitted on.
FIT_DATASET = "Salesforce/wikitext"
FIT_DATASET_CONFIG = "wikitext-103-raw-v1"
FIT_DATASET_SPLIT = "train"
#: The first shard of the training split, read directly (see ``src.corpus`` for why).
FIT_DATASET_FILE = f"{FIT_DATASET_CONFIG}/train-00000-of-00002.parquet"

#: Positions below this index are excluded, mirroring
#: ``jlens.fitting.SKIP_FIRST_N_POSITIONS``: early positions act as attention sinks and
#: were excluded when the lenses were estimated, so reading out there compares against a
#: basis the lens never saw.
SKIP_FIRST_N_POSITIONS = 16

#: Sequence length the lenses were fitted at. Applying at a very different length is not
#: wrong, but keeps the comparison honest.
MAX_SEQ_LEN = 128

LENS_REPO = "neuronpedia/jacobian-lens"

#: Where this project publishes its own fitted ``J^z``, so the README's ~15 GPU-hour fit
#: are optional. Not a result inherited from elsewhere: the file is the one
#: ``scripts.fit_lens`` wrote here, and :func:`src.model.load_jz` digest-checks a
#: downloaded copy exactly as it checks a locally fitted one.
JZ_REPO = "fmerlo/head-level-jlens"

# ---------------------------------------------------------------------------
# ``J^z`` fit settings. The fit runs to a fixed prompt budget, as ``jlens.fitting.fit``
# does, with no stopping rule. See :mod:`src.fit`.
# ---------------------------------------------------------------------------

#: Output dims per backward pass. FLOPs are invariant in this, so it trades memory
#: (~62 GB peak at 32 layers of Llama-3.1-8B) against Python-loop iterations; 64 is the
#: setting the only measured throughput number on this hardware was taken at.
FIT_DIM_BATCH = 64
#: Prompts drawn from the corpus in one go; the fit takes the first 1000. The draw size
#: is part of what defines those prompts -- ``random.sample`` switches algorithm on ``k``
#: vs ``n``, so ``load_prompts(1000)`` is not a prefix of ``load_prompts(1400)`` -- and
#: it is kept at the value the shipped Llama lens was drawn with so every checkpoint is
#: fitted on the same prompts.
FIT_CORPUS_SIZE = 1400


@dataclass(frozen=True)
class LensSpec:
    """One checkpoint, its Jacobian lens, and the ``J^z`` fitted against it.

    Attributes:
        key: Short name used on the command line and in output filenames.
        hf_model_id: The exact HF checkpoint the lenses were fitted on.
        lens_file: Path inside :data:`LENS_REPO` for the block-anchored Jacobian lens.
        n_layers: Model depth. Published lens source layers are ``0 .. n_layers - 2``.
        d_model: Residual-stream width.
        n_heads: Query heads per layer.
        head_dim: Per-head width. ``n_heads * head_dim`` need not equal ``d_model`` in
            general, so head slices are always taken on the *input* side of ``o_proj``
            and keyed to :attr:`d_source`.
        n_prompts_fitted: Prompts the Jacobian lens actually averaged over -- 461,
            not the 1000 its ``config.yaml`` requested. Neuronpedia's fitting script
            applies a convergence criterion this project does not; see
            :mod:`src.fit` for what that means when reading the two lenses side by side.
            Recorded so results cite what was fitted, not what was requested.
        jz_digests: SHA-256 of each ``J^z`` artifact, keyed by the number of prompts it
            averaged over. Same convention as :attr:`n_prompts_fitted`: prompts too short
            to yield a valid position are skipped, so the ``n`` is what the loop reached
            rather than what it was handed. Filenames get reused; contents do not.
        attention_layers: Layers whose block has a softmax attention sub-layer with an
            ``o_proj``, i.e. the layers a head-level anchor exists at. ``None`` means
            every layer. Qwen3.5 interleaves Gated DeltaNet blocks, which have no
            ``o_proj`` and no head in this project's sense, with one attention block in
            four; ``J^z`` is fitted at the attention blocks only.
        post_attention_norm: The attention sub-layer's output passes through an RMSNorm
            *before* the residual add (Gemma-3). A head's write into the stream is then
            not ``W_O^h z_h`` but that vector's share of the normalised sum; see
            :func:`src.heads.head_residual_writes`. Verified against the block at load.
        fit_dim_batch: ``--dim-batch`` default for this checkpoint. Memory, not the
            estimate, depends on it.
    """

    key: str
    hf_model_id: str
    lens_file: str
    n_layers: int
    d_model: int
    n_heads: int
    head_dim: int
    n_prompts_fitted: int
    jz_digests: dict[int, str] = field(default_factory=dict)
    attention_layers: tuple[int, ...] | None = None
    post_attention_norm: bool = False
    fit_dim_batch: int = FIT_DIM_BATCH

    @property
    def jz_n_prompts(self) -> int:
        """Prompts the shipped ``J^z`` averaged over, from the digest it is pinned by.

        Downstream drivers select a *lens file* by prompt count, and skipped prompts make
        that count a result rather than a setting -- so a stale default would happily
        load a superseded lens while writing the new number into every ``meta`` block.
        Deriving it from :attr:`jz_digests` makes that impossible: the default is
        whichever lens this repository actually vouches for.
        """
        if len(self.jz_digests) != 1:
            raise ValueError(
                f"{self.key} has {len(self.jz_digests)} recorded J^z digests; pass "
                "--n-prompts explicitly to say which lens you mean"
            )
        return next(iter(self.jz_digests))

    @property
    def d_source(self) -> int:
        """Width of ``o_proj``'s *input*: the concatenated head results.

        Equals :attr:`d_model` for Llama-3.1-8B, which is a coincidence of this
        architecture and not a fact to rely on. Every head slice and every ``J^z``
        allocation is keyed to this, never to ``d_model``.
        """
        return self.n_heads * self.head_dim

    @property
    def source_layers(self) -> list[int]:
        """Layers the block-anchored Jacobian lens reads out at."""
        return list(range(self.n_layers - 1))

    @property
    def jz_source_layers(self) -> list[int]:
        """Layers ``J^z`` reads out at: every layer with a head anchor.

        A head writes at ``o_proj``'s input, which for the top block still precedes the
        target (that block's own residual add and MLP have yet to run). So on a model
        whose every block has attention, ``J^z`` covers ``0 .. n_layers-1`` where the
        Jacobian lens stops at ``n_layers-2``; on a hybrid it covers
        :attr:`attention_layers`. Either way the list differs from
        :attr:`source_layers`, which is the structural guard that keeps the two lens
        families apart on disk: :func:`src.model.load_jz` requires this exact list.
        """
        if self.attention_layers is not None:
            return list(self.attention_layers)
        return list(range(self.n_layers))

    def head_slice(self, head: int) -> slice:
        """Columns of ``o_proj`` (equivalently, of ``J^z``) belonging to ``head``."""
        if not 0 <= head < self.n_heads:
            raise ValueError(f"head {head} out of range for {self.n_heads} heads")
        return slice(self.head_dim * head, self.head_dim * (head + 1))


LENSES: dict[str, LensSpec] = {
    # Plain pre-norm blocks: a sublayer's output is added straight to the residual
    # stream, so head contributions decompose without any normalisation convention to
    # defend. ``n_heads * head_dim == d_model`` here, a coincidence nothing relies on.
    "llama3.1-8b-it": LensSpec(
        key="llama3.1-8b-it",
        hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
        lens_file="llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt",
        n_layers=32,
        d_model=4096,
        n_heads=32,
        head_dim=128,
        n_prompts_fitted=461,
        # Fitted 2026-08-13 as four disjoint 250-prompt shards over ``corpus[0:1000]``
        # (seed 0, size 1400) and merged. 1000 is a fixed budget, not a converged count.
        # Sampling details: src/corpus.py.
        jz_digests={
            1000: "07a9a25aead37574c16829c3569ca1b2be567e7246bd71b14040d102a2caa323",
        },
    ),
    # Hybrid: 24 Gated DeltaNet blocks and 8 softmax-attention blocks (every fourth,
    # 3, 7, ..., 31). The attention blocks are pre-norm like Llama's, and ``o_proj``'s
    # input is the attention result times a per-coordinate sigmoid gate -- still the
    # vector ``o_proj`` consumes, so the head slice and additivity hold exactly.
    # ``n_heads * head_dim = 4096 != d_model``; ``lm_head`` is tied to the embedding.
    "qwen3.5-4b": LensSpec(
        key="qwen3.5-4b",
        hf_model_id="Qwen/Qwen3.5-4B",
        lens_file="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt",
        n_layers=32,
        d_model=2560,
        n_heads=16,
        head_dim=256,
        n_prompts_fitted=417,
        attention_layers=tuple(range(3, 32, 4)),
        # The DeltaNet blocks run transformers' pure-torch fallback (no fused kernel
        # installed), whose retained graph is what runs an 80 GB card out of memory at
        # 64; 32 fits and costs 82 s/prompt against Gemma-3-4B's 22 s at 64.
        fit_dim_batch=32,
        # Fitted 2026-09-15 as four disjoint 250-prompt shards over ``corpus[0:1000]``
        # (seed 0, size 1400) and merged, exactly as Llama's.
        jz_digests={
            1000: "249929f24139c4f2722645cc0308ff5fd509455240db5890391c369094a74fa6",
        },
    ),
    # Every block has attention (29 sliding-window, 5 global; the window exceeds the
    # 128-token prompts so the distinction never acts). The attention output is
    # RMS-normalised before it joins the residual stream, hence ``post_attention_norm``.
    # ``n_heads * head_dim = 2048 != d_model``; ``lm_head`` is tied to the embedding.
    "gemma-3-4b-it": LensSpec(
        key="gemma-3-4b-it",
        hf_model_id="google/gemma-3-4b-it",
        lens_file="gemma-3-4b-it/jlens/Salesforce-wikitext/gemma-3-4b-it_jacobian_lens.pt",
        n_layers=34,
        d_model=2560,
        n_heads=8,
        head_dim=256,
        n_prompts_fitted=546,
        post_attention_norm=True,
        # Fitted 2026-09-16 as four disjoint 250-prompt shards over ``corpus[0:1000]``
        # (seed 0, size 1400) and merged, exactly as Llama's.
        jz_digests={
            1000: "5c9f6f1263f496a4569f7c46d0c3469fcefb61f1633e9ce549338544269dc6e9",
        },
    ),
}

DEFAULT_LENS = "llama3.1-8b-it"

#: Stage directory holding the ``J^z`` fit's checkpoint, convergence trace and sidecar.
JZ_FIT_STAGE = "fit"


def shard_suffix(shard: str | None) -> str:
    """``_name`` for a sharded artifact, empty for the authoritative one."""
    return f"_{shard}" if shard else ""


@dataclass(frozen=True)
class Paths:
    """Resolved output locations. Override the root with ``RESULTS_ROOT``."""

    results_root: Path

    @classmethod
    def default(cls) -> Paths:
        return cls(
            results_root=Path(
                os.environ.get("RESULTS_ROOT", PROJECT_ROOT / "results")
            ).resolve()
        )

    def run_dir(self, name: str) -> Path:
        """Directory for one driver's outputs, created on demand."""
        path = self.results_root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def lenses_dir(self) -> Path:
        return self.results_root / "lenses"

    def jz_fit_lens(self, spec: LensSpec, *, n_prompts: int, shard: str | None = None) -> Path:
        """``J^z``, fitted by ``scripts.fit_lens``.

        The filename carries the number of prompts *actually* averaged over, which
        skipped prompts make a result rather than a setting, and which a loaded
        ``JacobianLens`` does report -- but only once it has been read off disk.
        """
        return self.lenses_dir / f"{spec.key}_Jzfit_n{n_prompts}{shard_suffix(shard)}.pt"

    def jz_fit_checkpoint(self, spec: LensSpec, *, shard: str | None = None) -> Path:
        """Resumable running sum for an in-flight ``J^z`` fit. Gitignored (2.1 GB)."""
        return self.run_dir(JZ_FIT_STAGE) / f"{spec.key}{shard_suffix(shard)}.ckpt.pt"

    def jz_fit_convergence(self, spec: LensSpec, *, shard: str | None = None) -> Path:
        """Per-prompt convergence trace, appended to as the fit runs.

        Same shape as the ``*_convergence.csv`` shipped beside each of Neuronpedia's
        released lenses, so
        the two runs' diagnostics can be read side by side.
        """
        return self.run_dir(JZ_FIT_STAGE) / f"{spec.key}{shard_suffix(shard)}_convergence.csv"


def lens_path(spec: LensSpec) -> str:
    """Local path to the Jacobian lens, downloading it if the cache lacks it."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(LENS_REPO, spec.lens_file)


def jz_path(spec: LensSpec, *, n_prompts: int, paths: Paths | None = None) -> str:
    """``J^z`` fitted in this checkout, or :data:`JZ_REPO`'s copy if there is none.

    Local first, so a fresh fit is never shadowed by the published artifact. The Hub is a
    shortcut past the GPU-hours rather than a result taken on trust: the downloaded file
    reaches :func:`src.model.load_jz` by the same route and fails the same digest check.
    That is why this resolves a *path* -- ``JacobianLens.from_pretrained`` would fetch it
    in one call, but hands back a lens, and the digest is taken over the bytes on disk.
    """
    local = (paths or Paths.default()).jz_fit_lens(spec, n_prompts=n_prompts)
    if local.exists():
        return str(local)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(JZ_REPO, local.name)
