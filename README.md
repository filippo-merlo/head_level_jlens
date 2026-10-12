# Head-JLens

**Head-JLens** reads single attention heads through the Jacobian lens.

The Jacobian lens (Gurnee, Sofroniew, Pearce et al., 2026) reads a layer's residual stream.
It maps the stream to the final layer through an averaged, causally fitted linear map and
decodes the result with the model's own unembedding. Head-JLens fits that map one step
earlier, at the input of each layer's `o_proj`, where the heads' outputs are still
separate. Each head's map is its own block of columns, so one fit per layer gives every
head a vocabulary readout:

```
J_{l,h} = J^z_l[:, columns of head h]
```

The paper, [*Reading Attention Heads Through the Jacobian Lens*](head_jlens_paper.pdf), compares
this readout (**J-head**) with decoding each head's output directly through the
unembedding (**Logit-head**, the per-head logit lens) on Llama-3.1-8B-Instruct. It covers
the method, the experiments and the results. This README covers installing the code,
using the lens and re-running the experiments.

## Supported models

| `--lens` key | Checkpoint |
|---|---|
| `llama3.1-8b-it` (default, the paper's model) | `meta-llama/Llama-3.1-8B-Instruct` |
| `qwen3.5-4b` (softmax-attention layers only) | `Qwen/Qwen3.5-4B` |
| `gemma-3-4b-it` | `google/gemma-3-4b-it` |

The fitted lenses are on the Hub at
[`fmerlo/head-level-jlens`](https://huggingface.co/fmerlo/head-level-jlens) and download
on first use. [REPLICATION.md](REPLICATION.md) describes how the Qwen and Gemma lenses
were fitted and what the paper's two experiments give on them.

## Install

You need Python 3.11+, [`uv`](https://docs.astral.sh/uv/), and a Hugging Face account that
has accepted the licence of each gated checkpoint you use. Anything that loads a model
needs an NVIDIA GPU. `pyproject.toml` pins CUDA 13.0 PyTorch wheels; for an older driver,
point the `pytorch-cu130` index at the matching build. All runs were made on A100 80 GB
cards.

```bash
git clone https://github.com/filippo-merlo/head_level_jlens.git
cd head_level_jlens

# The jlens reference implementation. Not on PyPI, and it must be a checkout:
# its data/ directory holds the evaluation sets.
git clone https://github.com/anthropics/jacobian-lens.git vendor/jacobian-lens
git -C vendor/jacobian-lens checkout 581d398613e5602a5af361e1c34d3a92ea82ba8e

uv sync --all-extras
uv run pytest            # CPU only, no downloads, under a minute
uv run hf auth login     # before loading any model
```

## Use the lens

Read every head at the last position of a prompt:

```python
from src.config import LENSES, jz_path
from src.grid import head_grid
from src.model import load_jz, load_model
from src.readout import target_token_ids

spec = LENSES["llama3.1-8b-it"]
model = load_model(spec)                                   # bf16, on cuda
jz = load_jz(spec, jz_path(spec, n_prompts=spec.jz_n_prompts))
jz_by_layer = {l: jz.jacobians[l].to(model.input_device) for l in spec.jz_source_layers}

input_ids = model.encode("The capital of the country where Lyon is located is")
grid = head_grid(
    model, jz_by_layer, spec, input_ids,
    position=input_ids.shape[-1] - 1,
    target_ids=target_token_ids(model.tokenizer, "France"),
)

top1 = grid.decode(model.tokenizer)    # top1[layer][head]: each head's top token
grid.top_ids, grid.top_values          # [layers, heads, 5]: top-5 tokens and logits
grid.target_ranks                      # [layers, heads]: rank of "France" at each head
```

Pass `arm="logithead"` to `head_grid` for the Logit-head baseline. For single heads or
custom vectors, use the functions in [`src/readout.py`](src/readout.py) and
[`src/heads.py`](src/heads.py) directly.

`load_jz` uses a lens fitted in `results/lenses/` if one exists and the Hub copy
otherwise. Either way, it loads the file only if its sha256 matches one recorded in
`LensSpec.jz_digests` ([`src/config.py`](src/config.py)). To use a lens you fitted
yourself, add its digest there or pass `verify_digest=False`.

### Interactive explorer

```bash
HEAD_LENS_GPU=0 uv run marimo edit notebooks/head_lens_explorer.py --headless --port 2718
```

The explorer runs the paper's two experiments on one prompt at a time. It reads every
attention head through Head-JLens (J-head) and the per-head logit lens (Logit-head) side by
side, then edits the heads each readout selects with the paper's coordinate swap and shows
whether the model's answer changes. Prompts come from the paper's evaluation and
intervention sets, or you can type your own. The notebook checks its numbers against the
records: an evaluation item's ranks against `results/read/`, a swap trial against
`results/write_swap_sites_clamp/`. `marimo run` instead of `marimo edit` opens it as an app.

## Reproduce the paper

Every record the paper cites is already in `results/`, so the tables and figures can be
regenerated without a GPU:

```bash
uv run python -m scripts.report     # tables  -> results/report/
uv run python -m scripts.figures    # figures -> results/figures/
```

To re-run the experiments (times are for one A100 80 GB). `read`, `write` and
`loading_rank` overwrite their records in `results/`, so `git diff results/` shows any
difference. `write_swap_sites` never overwrites a record, so the re-run below writes to
`results/swap_rerun/`, to compare with the shipped `results/write_swap_sites_clamp/`:

```bash
# Experiment 1, reading (~25 min)
uv run python -m scripts.read --gpu 0                        # -> results/read/

# Experiment 2, editing: J-head vs Logit-head, 3 shards (~15 min each)
for i in 0 1 2; do
  uv run python -m scripts.write_swap_sites --gpu $i --shard $i --n-shards 3 \
      --out-name swap_rerun &
done; wait                                                   # -> results/swap_rerun/
uv run python -m scripts.report_swap_sites results/swap_rerun

# J-head-only edits, for Table S4 and Figure S4 (~4 min)
uv run python -m scripts.write --gpu 0                       # -> results/write/

# Source loading against readout rank, for Section S3.6 (~2 min)
uv run python -m scripts.loading_rank --gpu 0                # -> results/loading_rank/
```

For Section S3.1's recomputed coordinates, add `--coords recompute` and another new
`--out-name` to the swap shards; the shipped run is `results/write_swap_sites/`. In the
code and records, the paper's *intermediate replacement* task is keyed `probe-swap`.
Re-running on an A100 80 GB reproduces the stored records bit for bit.

Every GPU script takes `--gpu` and `--lens <key>`. For a one-minute end-to-end check, run
`read`, `write` or `write_swap_sites` with `--max-items 2` (and `--magnitude-items 2` for
`read`); those numbers mean nothing. Set `RESULTS_ROOT=/some/path` to write somewhere
other than `results/`.

### Fit the lens yourself

This is optional, because the experiments download the published lens. The fit takes
about 15 A100-hours, split here into four 250-prompt shards:

```bash
for i in 0 1 2 3; do
  uv run python -m scripts.fit_lens --gpu $i --n-prompts 250 \
      --prompt-offset $((i * 250)) --shard a$i &
done; wait
uv run python -m scripts.merge_jz --shards a0 a1 a2 a3      # -> results/lenses/
```

### Run everything on several GPUs

`scripts/pipeline.py` runs the whole protocol for one or more checkpoints over a pool of
GPUs, detached from the terminal. It skips finished jobs on relaunch. `--stages eval`
uses the Hub lenses, and `--stages all` also fits them. A job counts as finished when its
record exists, and every record is shipped in `results/`, so point `RESULTS_ROOT` at an
empty directory to run the pipeline from scratch:

```bash
mkdir -p logs/pipeline
RESULTS_ROOT=$HOME/head_jlens_rerun setsid nohup uv run python -m scripts.pipeline \
    --models qwen3.5-4b gemma-3-4b-it --gpus 0 1 2 3 --stages eval \
    > logs/pipeline/pipeline.out 2>&1 < /dev/null &
tail -f logs/pipeline/pipeline.log
```

## Repository layout

```
src/         the lens, the readouts and the edits (unit-tested)
scripts/     command-line entry points: fit, read, write, report, figures, pipeline
tests/       CPU tests on tiny Llama, Gemma-3 and Qwen3.5 models
notebooks/   marimo explorer
results/     the paper's records, each with a provenance block (lenses are gitignored)
head_jlens_paper.{pdf,tex,md}
REPLICATION.md   the lenses and results for Qwen3.5-4B and Gemma-3-4B-IT
LICENSE
```

## References

- Gurnee, Sofroniew, Pearce et al. (2026). *Verbalizable Representations Form a Global
  Workspace in Language Models.* Transformer Circuits Thread.
- Jacobian lens reference implementation: [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens)
- Layer-output Jacobian lenses: [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens)
- Fitted Head-JLens: [`fmerlo/head-level-jlens`](https://huggingface.co/fmerlo/head-level-jlens)

## Citation

```bibtex
@misc{merlo2026headjlens,
  author = {Merlo, Filippo},
  title  = {Reading Attention Heads Through the Jacobian Lens},
  year   = {2026},
  url    = {https://github.com/filippo-merlo/head_level_jlens}
}
```

## License

Apache-2.0; see [`LICENSE`](LICENSE). The `jlens` checkout under `vendor/` is upstream's
code under its own Apache-2.0 licence. Model checkpoints and datasets are governed by
their own licences on the Hub.
