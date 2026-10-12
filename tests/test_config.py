"""Where ``J^z`` is resolved from.

The Hub fallback exists so the fit is optional, which makes *which* file a driver opened
a thing a result depends on. Both branches are pinned here, with no network: the local
one must win, and the download must ask for the name the local layout would have used --
one naming convention rather than two that can drift.
"""

from __future__ import annotations

import pytest

from src.config import DEFAULT_LENS, JZ_REPO, LENSES, Paths, jz_path

SPEC = LENSES[DEFAULT_LENS]


def test_jz_path_prefers_a_local_fit(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("downloaded despite a local fit being present")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)
    paths = Paths(results_root=tmp_path)
    local = paths.jz_fit_lens(SPEC, n_prompts=1000)
    local.parent.mkdir(parents=True)
    local.touch()

    assert jz_path(SPEC, n_prompts=1000, paths=paths) == str(local)


@pytest.mark.parametrize("n_prompts", [1000, 400])
def test_jz_path_falls_back_to_the_hub(tmp_path, monkeypatch, n_prompts):
    asked: dict[str, str] = {}

    def record(repo_id, filename, **kwargs):
        asked.update(repo_id=repo_id, filename=filename)
        return "/cache/downloaded.pt"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", record)
    paths = Paths(results_root=tmp_path)

    assert jz_path(SPEC, n_prompts=n_prompts, paths=paths) == "/cache/downloaded.pt"
    assert asked == {
        "repo_id": JZ_REPO,
        "filename": f"{SPEC.key}_Jzfit_n{n_prompts}.pt",
    }


def test_registry_geometry_of_every_checkpoint():
    """The facts the drivers key on, per checkpoint: the head anchor's layers and width."""
    from src.config import LENSES

    llama, qwen, gemma = (LENSES[k] for k in ("llama3.1-8b-it", "qwen3.5-4b", "gemma-3-4b-it"))
    assert llama.jz_source_layers == list(range(32)) and llama.d_source == llama.d_model
    # One attention block in four, and J^z lives only there; the block-anchored lens
    # still stops one short of the top.
    assert qwen.jz_source_layers == [3, 7, 11, 15, 19, 23, 27, 31]
    assert qwen.source_layers == list(range(31))
    assert qwen.d_source == 16 * 256 != qwen.d_model
    assert qwen.head_slice(1) == slice(256, 512)
    assert gemma.jz_source_layers == list(range(34)) and gemma.source_layers == list(range(33))
    assert gemma.d_source == 8 * 256 != gemma.d_model
    assert gemma.post_attention_norm and not qwen.post_attention_norm
    for spec in (llama, qwen, gemma):
        assert spec.jz_source_layers != spec.source_layers, "the family guard needs this"
        assert spec.fit_dim_batch == (32 if spec.key == "qwen3.5-4b" else 64)
