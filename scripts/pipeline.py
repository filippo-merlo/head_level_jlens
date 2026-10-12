"""[0] Run the whole pipeline for one or more checkpoints, detached, over a pool of GPUs.

    setsid nohup uv run python -m scripts.pipeline \\
        --models gemma-3-4b-it qwen3.5-4b --gpus 0 2 3 5 --stages all \\
        > logs/pipeline/pipeline.out 2>&1 < /dev/null &

One job graph per model, the paper's protocol: four 250-prompt ``fit_lens`` shards over
``corpus[0:1000]`` -> ``merge_jz`` -> ``read`` -> ``write`` -> ``report``, and beside them
three ``write_swap_sites`` shards (J-head vs Logit-head, clamped coordinates) ->
``report_swap_sites``; ``figures`` waits for both branches. ``write`` waits for
``read`` because it copies the relative write norms of the selected sites out of the read
record, and runs without them if that record is missing -- a complete record is worth
the ten minutes it would have gained by running beside ``read``. A job is
*done* when the sidecar its driver writes exists, so relaunching the same command after
an interruption skips finished jobs, and an interrupted fit resumes from its own
checkpoint (``fit_lens`` does that by itself). A driver already running from an earlier
launch (its command line names the same ``--lens`` and ``--shard``) is *adopted* rather
than started again: the launcher waits for its sidecar and keeps the driver's ``--gpu``
out of the pool until then, so relaunching never runs two fits of one shard into one
checkpoint and never puts a second job on a card a fit is using. The earlier launcher
must be gone first -- two launchers would each start the jobs that are not running yet --
so a launch refuses while another ``scripts.pipeline`` process is alive. ``--stages fit``
stops after the merge, which is how to fit first and evaluate later with evaluation code
that was not final at launch; ``--stages eval`` skips the fit and the merge and runs the
rest on the pinned ``J^z`` (fitted here, or downloaded from the Hub); ``--force`` re-runs
named jobs whose sidecars exist.
Nothing here computes: every number comes from the drivers, and this only decides what
runs where and when.

Scheduling is the simplest thing that works: a fixed GPU pool, one job per GPU, jobs
started in list order as their dependencies finish and a GPU frees up. CPU-only jobs
(``merge``, ``report_swap_sites``, ``report``, ``figures``) run outside the pool. Each job's stdout goes to
``logs/pipeline/<model>/<job>.log``; the run's own log lists ``START``/``END rc=`` lines.

Exit codes are read, not interpreted: a ``fit_lens`` that exits 1 has written its sidecar
and failed the ``W_O`` gate, which is a number for the paper rather than a reason to stop
the merge, so it counts as done unless ``--strict-gate``. Any other failure is retried
once on another GPU -- a job that dies within :data:`STARTUP_SECONDS` takes its GPU out
of the pool, since a card that cannot open a CUDA context (GPU 4 on this node, at the
first launch) would otherwise fail every job in turn -- and then blocks the job's
dependants and is reported at the end.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SHARDS = 4
PROMPTS_PER_SHARD = 250
#: The J-head vs Logit-head swap runs as this many trial shards, one GPU each.
SWAP_SHARDS = 3
#: Where the swap shards go: the paper's clamped run, one subdirectory per other checkpoint.
SWAP_DIR = "write_swap_sites_clamp"
#: A job that dies this soon after starting never got going; blame the GPU, not the job.
STARTUP_SECONDS = 180.0
MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class Job:
    name: str
    argv: tuple[str, ...]
    done_file: Path
    deps: tuple[str, ...] = ()
    gpu: bool = True
    #: Exit codes that still leave a usable sidecar (fit_lens's gate FAIL).
    tolerated: tuple[int, ...] = ()


@dataclass
class State:
    running: dict[str, tuple[subprocess.Popen, str | None, float]] = field(default_factory=dict)
    #: Jobs found already running at launch, by pid; resolved by their sidecar appearing.
    adopted: dict[str, int] = field(default_factory=dict)
    #: The ``--gpu`` each adopted driver is using; out of the pool until the job resolves.
    held: dict[str, str] = field(default_factory=dict)
    done: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    attempts: dict[str, int] = field(default_factory=dict)


def _processes() -> dict[int, str]:
    """``{pid: command line}`` of every process on the node."""
    found: dict[int, str] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        found[int(entry.name)] = cmd.strip()
    return found


def running_drivers() -> dict[int, str]:
    """``{pid: command line}`` of every live ``python -m scripts.<driver>`` process."""
    return {
        pid: cmd for pid, cmd in _processes().items()
        if "python" in cmd and "-m scripts." in cmd and "scripts.pipeline" not in cmd
    }


def is_launcher(cmd: str) -> bool:
    """Whether ``cmd`` is a Python interpreter running this module (not a ``uv run`` or
    shell wrapper whose command line merely mentions it)."""
    tokens = cmd.split()
    return (
        len(tokens) >= 3
        and Path(tokens[0]).name.startswith("python")
        and tokens[1:3] == ["-m", "scripts.pipeline"]
    )


def other_launchers() -> dict[int, str]:
    """Launcher processes other than this one."""
    return {
        pid: cmd for pid, cmd in _processes().items()
        if pid != os.getpid() and is_launcher(cmd)
    }


def adopt(jobs: list[Job], state: State, drivers: dict[int, str]) -> None:
    """Mark jobs whose driver is already running, matched on the flags that name them."""
    for job in jobs:
        if job.name in state.done or "--lens" not in job.argv:
            continue
        module = job.argv[2]
        key = job.argv[job.argv.index("--lens") + 1]
        shard = job.argv[job.argv.index("--shard") + 1] if "--shard" in job.argv else None
        for pid, cmd in drivers.items():
            tokens = cmd.split()
            if module not in tokens or "--lens" not in tokens:
                continue
            if tokens[tokens.index("--lens") + 1] != key:
                continue
            if shard is not None and (
                "--shard" not in tokens or tokens[tokens.index("--shard") + 1] != shard
            ):
                continue
            state.adopted[job.name] = pid
            if "--gpu" in tokens:
                state.held[job.name] = tokens[tokens.index("--gpu") + 1]
            break


def jobs_for(key: str, results_root: Path, stages: str) -> list[Job]:
    """The job graph for one checkpoint; ``stages`` is ``fit``, ``all`` or ``eval``.

    ``eval`` is ``all`` without the fit and the merge: every driver loads the ``J^z`` its
    ``LensSpec`` pins, from ``results/lenses/`` or, if absent, from ``config.JZ_REPO``.
    """
    fit = results_root / "fit"
    py = ["python", "-m"]
    jobs = []
    if stages != "eval":
        jobs += [
            Job(
                name=f"{key}:fit_a{i}",
                argv=(
                    *py, "scripts.fit_lens", "--lens", key, "--n-prompts", str(PROMPTS_PER_SHARD),
                    "--prompt-offset", str(i * PROMPTS_PER_SHARD), "--shard", f"a{i}",
                ),
                done_file=fit / f"{key}_a{i}_fit.json",
                tolerated=(1,),
            )
            for i in range(SHARDS)
        ]
        jobs.append(
            Job(
                name=f"{key}:merge",
                argv=(*py, "scripts.merge_jz", "--lens", key, "--shards",
                      *(f"a{i}" for i in range(SHARDS))),
                done_file=fit / f"{key}_merge.json",
                deps=tuple(f"{key}:fit_a{i}" for i in range(SHARDS)),
                gpu=False,
            )
        )
    if stages == "fit":
        return jobs
    # The paper's figures and swap records sit at the root of their directories for the
    # default checkpoint; every other checkpoint gets a subdirectory.
    from src.config import DEFAULT_LENS

    figures_name = "figures" if key == DEFAULT_LENS else f"figures/{key}"
    swap_name = SWAP_DIR if key == DEFAULT_LENS else f"{SWAP_DIR}/{key}"
    swap_summary = results_root / swap_name / "summary.json"
    if stages == "eval":
        n_prompts, after_merge = (), ()
    else:
        # ``--n-prompts`` is what the merge actually averaged; read off its sidecar at
        # launch time (the merge has finished by the time the job is ready).
        n_prompts = ("--n-prompts", "@merge", str(fit / f"{key}_merge.json"))
        after_merge = (f"{key}:merge",)
    swap_shards = [
        Job(
            name=f"{key}:swap_s{i}",
            argv=(*py, "scripts.write_swap_sites", "--lens", key, "--shard", str(i),
                  "--n-shards", str(SWAP_SHARDS), "--out-name", swap_name),
            done_file=results_root / swap_name / f"{key}_shard{i}of{SWAP_SHARDS}.json",
            deps=after_merge,
        )
        for i in range(SWAP_SHARDS)
    ]
    jobs += [
        Job(
            name=f"{key}:read",
            argv=(*py, "scripts.read", "--lens", key, *n_prompts),
            done_file=results_root / "read" / f"{key}_read.json",
            deps=after_merge,
        ),
        Job(
            name=f"{key}:write",
            argv=(*py, "scripts.write", "--lens", key, *n_prompts),
            done_file=results_root / "write" / f"{key}_write.json",
            deps=(*after_merge, f"{key}:read"),
        ),
        *swap_shards,
        Job(
            name=f"{key}:swap_report",
            argv=(*py, "scripts.report_swap_sites", str(results_root / swap_name)),
            done_file=swap_summary,
            deps=tuple(j.name for j in swap_shards),
            gpu=False,
        ),
        Job(
            name=f"{key}:report",
            argv=(*py, "scripts.report", "--lens", key),
            done_file=results_root / "report" / f"{key}_report.md",
            deps=(f"{key}:read", f"{key}:write"),
            gpu=False,
        ),
        Job(
            name=f"{key}:figures",
            argv=(*py, "scripts.figures", "--lens", key, "--swap-summaries", str(swap_summary),
                  "--out-name", figures_name),
            done_file=results_root / figures_name / "meta.json",
            deps=(f"{key}:read", f"{key}:write", f"{key}:swap_report"),
            gpu=False,
        ),
    ]
    return jobs


def resolve_argv(argv: tuple[str, ...]) -> list[str]:
    """Substitute ``@merge <sidecar>`` with the prompt count the merge recorded."""
    out: list[str] = []
    it = iter(argv)
    for token in it:
        if token == "@merge":
            record = json.loads(Path(next(it)).read_text())
            out.append(str(record["result"]["prompts_fitted"]))
        else:
            out.append(token)
    return out


def ready(job: Job, state: State) -> bool:
    return (
        job.name not in state.done
        and job.name not in state.failed
        and job.name not in state.running
        and job.name not in state.adopted
        and all(d in state.done for d in job.deps)
    )


def blocked(job: Job, state: State) -> bool:
    return any(d in state.failed for d in job.deps)


def _stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _alive(pid: int) -> bool:
    """Whether ``pid`` is still running -- a zombie counts as gone, it holds no GPU."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def run(
    jobs: list[Job],
    gpus: list[str],
    *,
    log_dir: Path,
    strict_gate: bool = False,
    poll_seconds: float = 30.0,
    launcher: list[str] | None = None,
    force: tuple[str, ...] = (),
    drivers: dict[int, str] | None = None,
) -> int:
    """Drive the graph to completion; returns the number of failed jobs."""
    launcher = ["uv", "run"] if launcher is None else launcher
    state = State()
    for job in jobs:
        if job.done_file.exists() and job.name not in force:
            state.done.add(job.name)
    if state.done:
        logger.info("already done: %s", sorted(state.done))
    adopt(jobs, state, running_drivers() if drivers is None else drivers)
    free = [g for g in gpus if g not in state.held.values()]
    log_dir.mkdir(parents=True, exist_ok=True)
    pipeline_log = log_dir / "pipeline.log"

    def note(line: str) -> None:
        logger.info(line)
        with pipeline_log.open("a") as handle:
            handle.write(f"{_stamp()} {line}\n")

    for name, pid in state.adopted.items():
        where = f" on gpu {state.held[name]}" if name in state.held else ""
        note(f"ADOPT {name} pid={pid}{where}, already running; waiting for its sidecar")

    env = dict(os.environ)

    while True:
        # Adopted drivers: done when their sidecar appears; failed if they die without.
        for name, pid in list(state.adopted.items()):
            job = next(j for j in jobs if j.name == name)
            if _alive(pid):
                continue
            del state.adopted[name]
            gpu = state.held.pop(name, None)
            if gpu in gpus and gpu not in free:
                free.append(gpu)
            if job.done_file.exists():
                state.done.add(name)
                note(f"END   {name} (adopted pid {pid}) sidecar present")
            else:
                note(f"END   {name} (adopted pid {pid}) died without a sidecar -- will run it")
        # Reap finished processes.
        for name, (proc, gpu, started) in list(state.running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del state.running[name]
            job = next(j for j in jobs if j.name == name)
            if rc == 0 or (rc in job.tolerated and job.done_file.exists() and not strict_gate):
                state.done.add(name)
                if rc != 0:
                    note(f"END   {name} rc={rc} -- sidecar written, gate FAILED; continuing")
                else:
                    note(f"END   {name} rc={rc}")
                if gpu is not None:
                    free.append(gpu)
                continue
            died_at_startup = gpu is not None and time.monotonic() - started < STARTUP_SECONDS
            if died_at_startup and len(free) + len(state.running) > 0:
                note(f"DROP  gpu {gpu}: {name} died {time.monotonic() - started:.0f}s after start (rc={rc})")
            elif gpu is not None:
                free.append(gpu)
            if state.attempts[name] < MAX_ATTEMPTS:
                note(f"END   {name} rc={rc} FAILED -- will retry")
            else:
                state.failed.add(name)
                note(f"END   {name} rc={rc} FAILED")
        # Block dependants of failures.
        for job in jobs:
            if job.name not in state.done | state.failed and blocked(job, state):
                state.failed.add(job.name)
                note(f"BLOCKED {job.name} (a dependency failed)")
        # Launch what is ready.
        for job in jobs:
            if not ready(job, state):
                continue
            if job.gpu and not free:
                continue
            gpu = free.pop(0) if job.gpu else None
            argv = launcher + resolve_argv(job.argv) + (["--gpu", gpu] if gpu else [])
            model_dir = log_dir / job.name.split(":")[0]
            model_dir.mkdir(parents=True, exist_ok=True)
            out = (model_dir / f"{job.name.split(':')[1]}.log").open("a")
            proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, env=env)
            state.attempts[job.name] = state.attempts.get(job.name, 0) + 1
            state.running[job.name] = (proc, gpu, time.monotonic())
            note(f"START {job.name} (gpu {gpu if gpu else 'cpu'}) pid={proc.pid}: {' '.join(argv)}")
        pending = [
            j for j in jobs
            if j.name not in state.done | state.failed | set(state.running) | set(state.adopted)
        ]
        if not state.running and not state.adopted and not pending:
            break
        if not state.running and not state.adopted and not free and any(j.gpu for j in pending):
            for job in pending:
                if job.gpu:
                    state.failed.add(job.name)
                    note(f"FAILED {job.name}: no GPU left in the pool")
            continue
        time.sleep(poll_seconds)

    note(f"ALL DONE: {len(state.done)} done, {len(state.failed)} failed/blocked "
         f"{sorted(state.failed) if state.failed else ''}")
    return len(state.failed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", required=True, help="LensSpec keys")
    parser.add_argument("--gpus", nargs="+", required=True, help="nvidia-smi indices, in order of preference")
    parser.add_argument("--stages", choices=("fit", "all", "eval"), default="all",
                        help="fit: shards and merge only; eval: everything after the merge, "
                        "on the pinned J^z (local, or downloaded from the Hub)")
    parser.add_argument("--strict-gate", action="store_true", help="a W_O gate FAIL blocks the merge")
    parser.add_argument("--force", nargs="*", default=(), help="job names to run even if their sidecar exists, e.g. llama3.1-8b-it:read")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--log-dir", default=None, help="default: logs/pipeline")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    from src.config import LENSES, PROJECT_ROOT, Paths

    for key in args.models:
        if key not in LENSES:
            raise SystemExit(f"unknown lens {key!r}; have {sorted(LENSES)}")
    results_root = Paths.default().results_root
    log_dir = Path(args.log_dir) if args.log_dir else PROJECT_ROOT / "logs" / "pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [job for key in args.models for job in jobs_for(key, results_root, args.stages)]
    # Fit shards of every model first, so the pool is never idle while evaluations wait
    # on a merge; the graph's own dependencies order the rest.
    jobs.sort(key=lambda j: 0 if ":fit_" in j.name else 1)
    logger.info("%d jobs for %s on GPUs %s (results in %s)", len(jobs), args.models, args.gpus, results_root)
    os.chdir(PROJECT_ROOT)
    unknown = [name for name in args.force if name not in {j.name for j in jobs}]
    if unknown:
        raise SystemExit(f"--force names no job: {unknown}")
    others = other_launchers()
    if others:
        listing = "; ".join(f"pid {pid}: {cmd}" for pid, cmd in others.items())
        raise SystemExit(
            f"another launcher is running ({listing}); wait for it or kill it first -- two "
            "launchers would each start the jobs that are not running yet"
        )
    failed = run(jobs, list(args.gpus), log_dir=log_dir, strict_gate=args.strict_gate,
                 poll_seconds=args.poll_seconds, force=tuple(args.force))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
