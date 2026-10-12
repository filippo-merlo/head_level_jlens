"""The launcher's graph and scheduling, with fake drivers and a fake results root."""

from __future__ import annotations

import json
import sys

from scripts.pipeline import jobs_for, ready, resolve_argv, run


def test_graph_shape_and_done_files(tmp_path):
    jobs = jobs_for("k", tmp_path, "all")
    names = [j.name for j in jobs]
    assert names == [
        "k:fit_a0", "k:fit_a1", "k:fit_a2", "k:fit_a3", "k:merge", "k:read", "k:write",
        "k:swap_s0", "k:swap_s1", "k:swap_s2", "k:swap_report", "k:report", "k:figures",
    ]
    by = {j.name: j for j in jobs}
    assert by["k:merge"].deps == tuple(f"k:fit_a{i}" for i in range(4))
    assert by["k:read"].deps == ("k:merge",)
    # write copies the selected sites' write norms out of the read record.
    assert by["k:write"].deps == ("k:merge", "k:read")
    assert by["k:report"].deps == ("k:read", "k:write")
    assert by["k:fit_a2"].argv[-4:] == ("--prompt-offset", "500", "--shard", "a2")
    # The swap shards write where the swap report and the figures read.
    assert by["k:swap_s1"].deps == ("k:merge",)
    assert by["k:swap_s1"].argv[-6:] == (
        "--shard", "1", "--n-shards", "3", "--out-name", "write_swap_sites_clamp/k",
    )
    assert by["k:swap_s1"].done_file == tmp_path / "write_swap_sites_clamp/k/k_shard1of3.json"
    assert by["k:swap_report"].deps == ("k:swap_s0", "k:swap_s1", "k:swap_s2")
    assert by["k:swap_report"].argv[-1] == str(tmp_path / "write_swap_sites_clamp/k")
    assert by["k:figures"].deps == ("k:read", "k:write", "k:swap_report")
    assert by["k:figures"].argv[-4:] == (
        "--swap-summaries", str(by["k:swap_report"].done_file), "--out-name", "figures/k",
    )
    assert not by["k:merge"].gpu and not by["k:swap_report"].gpu and by["k:swap_s0"].gpu
    assert [j.name for j in jobs_for("k", tmp_path, "fit")] == names[:5]


def test_eval_stage_runs_on_the_pinned_lens_without_fitting(tmp_path):
    jobs = jobs_for("k", tmp_path, "eval")
    assert [j.name for j in jobs] == [j.name for j in jobs_for("k", tmp_path, "all")][5:]
    by = {j.name: j for j in jobs}
    assert by["k:read"].deps == () and by["k:swap_s0"].deps == ()
    assert by["k:write"].deps == ("k:read",)
    assert not any("--n-prompts" in j.argv for j in jobs)


def test_merge_prompt_count_is_read_at_launch(tmp_path):
    sidecar = tmp_path / "k_merge.json"
    sidecar.write_text(json.dumps({"result": {"prompts_fitted": 997}}))
    assert resolve_argv(("x", "--n-prompts", "@merge", str(sidecar))) == [
        "x", "--n-prompts", "997",
    ]


def test_run_respects_dependencies_pool_and_done_files(tmp_path, monkeypatch):
    """Fake drivers touch their done file; the merge must see all four shards first."""
    results = tmp_path / "results"
    jobs = jobs_for("k", results, "fit")
    (results / "fit").mkdir(parents=True)
    # Shard 1 is already done: it must be skipped, not rerun.
    (results / "fit" / "k_a1_fit.json").write_text("{}")
    script = tmp_path / "fake.py"
    script.write_text(
        "import sys, os, json, pathlib\n"
        "args = sys.argv[1:]\n"
        "root = pathlib.Path(os.environ['FAKE_ROOT'])\n"
        "if 'scripts.fit_lens' in args:\n"
        "    shard = args[args.index('--shard') + 1]\n"
        "    (root / 'fit' / f'k_{shard}_fit.json').write_text(json.dumps({'gpu': args[args.index('--gpu') + 1]}))\n"
        "    sys.exit(1)\n"  # the gate FAIL exit: tolerated because the sidecar exists
        "if 'scripts.merge_jz' in args:\n"
        "    n = len(list((root / 'fit').glob('k_a*_fit.json')))\n"
        "    (root / 'fit' / 'k_merge.json').write_text(json.dumps({'shards_seen': n}))\n"
    )
    launcher = [sys.executable, str(script)]
    monkeypatch.setenv("FAKE_ROOT", str(results))
    failed = run(jobs, ["7", "8"], log_dir=tmp_path / "logs", poll_seconds=0.05, launcher=launcher)
    assert failed == 0
    merge = json.loads((results / "fit" / "k_merge.json").read_text())
    assert merge["shards_seen"] == 4
    gpus = {json.loads((results / "fit" / f"k_a{i}_fit.json").read_text()).get("gpu") for i in (0, 2, 3)}
    assert gpus <= {"7", "8"}
    log = (tmp_path / "logs" / "pipeline.log").read_text()
    assert "START k:fit_a1" not in log and "gate FAILED; continuing" in log
    assert "START k:merge (gpu cpu)" in log


def test_strict_gate_blocks_the_merge(tmp_path, monkeypatch):
    results = tmp_path / "results"
    (results / "fit").mkdir(parents=True)
    jobs = jobs_for("k", results, "fit")
    script = tmp_path / "fake.py"
    script.write_text(
        "import sys, os, pathlib\n"
        "args = sys.argv[1:]\n"
        "root = pathlib.Path(os.environ['FAKE_ROOT'])\n"
        "shard = args[args.index('--shard') + 1]\n"
        "(root / 'fit' / f'k_{shard}_fit.json').write_text('{}')\n"
        "sys.exit(1)\n"
    )
    monkeypatch.setenv("FAKE_ROOT", str(results))
    failed = run(jobs, ["0"], log_dir=tmp_path / "logs", poll_seconds=0.05,
                 launcher=[sys.executable, str(script)], strict_gate=True)
    assert failed == 5  # four shards failed, the merge blocked
    assert not (results / "fit" / "k_merge.json").exists()


def test_ready_needs_every_dependency():
    from scripts.pipeline import State

    jobs = {j.name: j for j in jobs_for("k", __import__("pathlib").Path("/nonexistent"), "fit")}
    state = State()
    state.done.update(f"k:fit_a{i}" for i in range(3))
    assert not ready(jobs["k:merge"], state)
    state.done.add("k:fit_a3")
    assert ready(jobs["k:merge"], state)


def test_a_gpu_that_kills_jobs_at_startup_is_dropped_and_the_job_retried(tmp_path, monkeypatch):
    """GPU 9 cannot run anything; the shard must land on GPU 7 and GPU 9 leave the pool."""
    results = tmp_path / "results"
    (results / "fit").mkdir(parents=True)
    jobs = jobs_for("k", results, "fit")
    script = tmp_path / "fake.py"
    script.write_text(
        "import sys, os, json, pathlib\n"
        "args = sys.argv[1:]\n"
        "root = pathlib.Path(os.environ['FAKE_ROOT'])\n"
        "if 'scripts.merge_jz' in args:\n"
        "    (root / 'fit' / 'k_merge.json').write_text('{}'); sys.exit(0)\n"
        "gpu = args[args.index('--gpu') + 1]\n"
        "if gpu == '9':\n"
        "    sys.exit(3)\n"
        "shard = args[args.index('--shard') + 1]\n"
        "(root / 'fit' / f'k_{shard}_fit.json').write_text(json.dumps({'gpu': gpu}))\n"
    )
    monkeypatch.setenv("FAKE_ROOT", str(results))
    failed = run(jobs, ["9", "7"], log_dir=tmp_path / "logs", poll_seconds=0.05,
                 launcher=[sys.executable, str(script)])
    assert failed == 0
    gpus = {json.loads((results / "fit" / f"k_a{i}_fit.json").read_text())["gpu"] for i in range(4)}
    assert gpus == {"7"}
    log = (tmp_path / "logs" / "pipeline.log").read_text()
    assert "DROP  gpu 9" in log and "will retry" in log


def test_adopts_a_driver_that_is_already_running(tmp_path, monkeypatch):
    """A shard fitting from an earlier launch is waited for, not started twice."""
    import subprocess

    from scripts.pipeline import State, adopt

    results = tmp_path / "results"
    (results / "fit").mkdir(parents=True)
    jobs = jobs_for("k", results, "fit")
    state = State()
    drivers = {
        4242: "python -m scripts.fit_lens --lens k --n-prompts 250 --prompt-offset 250 --shard a1 --gpu 0",
        4243: "python -m scripts.fit_lens --lens other --n-prompts 250 --prompt-offset 0 --shard a0 --gpu 1",
    }
    adopt(jobs, state, drivers)
    assert state.adopted == {"k:fit_a1": 4242}
    assert state.held == {"k:fit_a1": "0"}
    assert not ready(jobs[1], state)

    # End to end with a real sleeper standing in for the running shard: the launcher must
    # neither start a1 nor finish before the sleeper's sidecar appears, and GPU 0 -- the
    # only card in the pool, and the one the sleeper "uses" -- stays out of the pool until
    # then, so no other shard may start before a1's sidecar exists.
    sleeper = subprocess.Popen(
        [sys.executable, "-c",
         "import time, pathlib, sys; time.sleep(0.4); "
         f"pathlib.Path({str(results / 'fit' / 'k_a1_fit.json')!r}).write_text('{{}}')"]
    )
    script = tmp_path / "fake.py"
    script.write_text(
        "import sys, os, json, pathlib\n"
        "args = sys.argv[1:]\n"
        "root = pathlib.Path(os.environ['FAKE_ROOT'])\n"
        "if 'scripts.merge_jz' in args:\n"
        "    (root / 'fit' / 'k_merge.json').write_text('{}'); sys.exit(0)\n"
        "shard = args[args.index('--shard') + 1]\n"
        "assert shard != 'a1', 'a1 was started twice'\n"
        "assert (root / 'fit' / 'k_a1_fit.json').exists(), 'started on the GPU a1 holds'\n"
        "(root / 'fit' / f'k_{shard}_fit.json').write_text('{}')\n"
    )
    monkeypatch.setenv("FAKE_ROOT", str(results))
    failed = run(jobs, ["0"], log_dir=tmp_path / "logs", poll_seconds=0.05,
                 launcher=[sys.executable, str(script)],
                 drivers={sleeper.pid: drivers[4242]})
    sleeper.wait()
    assert failed == 0
    log = (tmp_path / "logs" / "pipeline.log").read_text()
    assert "ADOPT k:fit_a1 pid=" in log and " on gpu 0," in log
    assert "START k:fit_a1" not in log and "START k:merge" in log
    assert "(gpu 0)" in log  # the held card went back into the pool once a1 resolved


def test_only_a_python_launcher_process_counts_as_another_launcher():
    """``uv run`` and shell wrappers mention the module on their command lines too."""
    from scripts.pipeline import is_launcher, other_launchers

    assert is_launcher("/x/.venv/bin/python3 -m scripts.pipeline --models k --gpus 0")
    assert is_launcher("python -m scripts.pipeline --models k --gpus 0")
    assert not is_launcher("uv run python -m scripts.pipeline --models k --gpus 0")
    assert not is_launcher("/bin/bash -c setsid nohup uv run python -m scripts.pipeline")
    assert not is_launcher("/x/.venv/bin/python3 -m scripts.fit_lens --lens k --shard a0")
    assert not is_launcher("python")
    # This test process is not a launcher, and no launcher is expected on the test host.
    for cmd in other_launchers().values():
        assert is_launcher(cmd)


def test_force_reruns_a_finished_job_and_llama_figures_stay_at_the_root(tmp_path, monkeypatch):
    results = tmp_path / "results"
    (results / "fit").mkdir(parents=True)
    for name in ("k_a0", "k_a1", "k_a2", "k_a3"):
        (results / "fit" / f"{name}_fit.json").write_text("{}")
    (results / "fit" / "k_merge.json").write_text("{}")
    jobs = jobs_for("k", results, "fit")
    script = tmp_path / "fake.py"
    script.write_text(
        "import sys, os, pathlib\n"
        "root = pathlib.Path(os.environ['FAKE_ROOT'])\n"
        "(root / 'fit' / 'k_merge.json').write_text('rerun')\n"
    )
    monkeypatch.setenv("FAKE_ROOT", str(results))
    run(jobs, ["0"], log_dir=tmp_path / "logs", poll_seconds=0.05,
        launcher=[sys.executable, str(script)], force=("k:merge",))
    assert (results / "fit" / "k_merge.json").read_text() == "rerun"
    from src.config import DEFAULT_LENS

    llama = {j.name: j for j in jobs_for(DEFAULT_LENS, results, "all")}
    assert llama[f"{DEFAULT_LENS}:figures"].argv[-1] == "figures"
    assert llama[f"{DEFAULT_LENS}:swap_report"].argv[-1] == str(results / "write_swap_sites_clamp")
    assert {j.name: j for j in jobs_for("k", results, "all")}["k:figures"].argv[-1] == "figures/k"
