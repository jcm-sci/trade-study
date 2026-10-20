"""Real frozen Git trees, Python workers, artifact guards and scheduler command I/O."""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] - real command tests
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from trade_study.slurm import (
    FrozenCheckout,
    SlurmResources,
    SlurmTask,
    freeze_checkout,
    prepare_slurm_plan,
)

if TYPE_CHECKING:
    from trade_study.slurm import SlurmPlan

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX Slurm worker")

_DRIVER = """
import json, os, sys
from pathlib import Path
from trade_study import Direction, Observable, run_grid

class World:
    def generate(self, config):
        return config['value'], config['value'] + 1

class Scorer:
    def score(self, truth, observations, config):
        return {'error': observations - truth}

def main():
    root, index = Path(sys.argv[1]), int(sys.argv[2])
    root.mkdir(parents=True, exist_ok=True)
    with (root / f'calls-{index}.txt').open('a') as handle:
        handle.write('call\\n')
    if (root / f'fail-{index}').exists():
        raise RuntimeError('synthetic task failure')
    result = run_grid(World(), Scorer(), [{'value': index}],
        [Observable('error', Direction.MINIMIZE)],
        checkpoint_path=root / f'grid-{index}.sqlite', checkpoint_key='fixture-v1')
    (root / f'result-{index}.json').write_text(json.dumps({
        'score': result.scores.tolist(), 'index': index,
        'cpus': os.environ['SLURM_CPUS_PER_TASK'], 'source': __file__,
    }))

if __name__ == '__main__':
    main()
"""


def _git(root: Path, *args: str) -> str:
    program = shutil.which("git")
    assert program is not None
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [program, "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init", "-q")
    (repository / "driver.py").write_text(_DRIVER)
    _git(repository, "add", "driver.py")
    _git(
        repository,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.org",
        "commit",
        "-qm",
        "initial",
    )
    return repository


@pytest.fixture
def plan(source: Path, tmp_path: Path) -> SlurmPlan:
    checkout = freeze_checkout(source, tmp_path / "frozen ' tree")
    output = tmp_path / "output ' directory"
    tasks = [
        SlurmTask(
            ("-m", "driver", str(output), str(index)),
            outputs=(str(output / f"result-{index}.json"),),
        )
        for index in range(3)
    ]
    return prepare_slurm_plan(
        checkout,
        tasks,
        directory=tmp_path / "plan ' directory",
        revision="fixture-data-v1",
        imports=("driver",),
        resources=SlurmResources(width=2, cpus_per_task=2),
    )


def _output(plan: SlurmPlan) -> Path:
    return Path(json.loads(plan.path.read_text())["tasks"][0]["outputs"][0]).parent


def test_frozen_tasks_reuse_real_grid_ledgers_and_completed_artifacts(
    plan: SlurmPlan, source: Path
) -> None:
    # Editing the submitter's tree cannot change queued source.
    (source / "driver.py").write_text("raise RuntimeError('mutable checkout')")
    plan.run_local()
    output = _output(plan)
    for index in range(3):
        result = json.loads((output / f"result-{index}.json").read_text())
        assert result["score"] == [[1.0]]
        assert result["cpus"] == "2"
        assert "frozen ' tree" in result["source"]
    plan.run_local()
    assert (output / "calls-0.txt").read_text() == "call\n"
    # Missing outputs rerun just that task, recovering its existing grid ledger.
    (output / "result-1.json").unlink()
    plan.run_local()
    assert (output / "calls-1.txt").read_text() == "call\ncall\n"
    assert (output / "calls-0.txt").read_text() == "call\n"


def test_failed_child_fails_pack_and_resumes_only_unfinished_task(
    plan: SlurmPlan,
) -> None:
    output = _output(plan)
    output.mkdir()
    (output / "fail-1").touch()
    with pytest.raises(subprocess.CalledProcessError):
        plan.run_local()
    assert not (plan.path.parent / "task-000001.json").exists()
    assert (plan.path.parent / "task-000000.json").exists()
    assert (plan.path.parent / "task-000002.json").exists()
    (output / "fail-1").unlink()
    plan.run_local()
    assert (output / "calls-0.txt").read_text() == "call\n"
    assert (output / "calls-1.txt").read_text() == "call\ncall\n"


def test_overlapping_submissions_do_not_duplicate_tasks(plan: SlurmPlan) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: plan.run_local(), range(2)))
    assert results == [None, None]
    output = _output(plan)
    assert all(
        (output / f"calls-{index}.txt").read_text() == "call\n" for index in range(3)
    )


def test_changed_source_import_and_artifact_are_refused(plan: SlurmPlan) -> None:
    plan.run_local()
    output = _output(plan)
    (output / "result-0.json").write_text("changed")
    with pytest.raises(subprocess.CalledProcessError, match="launcher"):
        plan.run_local()
    checkout = Path(json.loads(plan.path.read_text())["checkout"])
    (checkout / "driver.py").write_text("# changed source")
    with pytest.raises(subprocess.CalledProcessError):
        plan.run_local()


def test_plan_and_copied_launcher_integrity(plan: SlurmPlan) -> None:
    launcher = plan.path.parent / "launcher.py"
    launcher.write_text("# changed launcher")
    with pytest.raises(ValueError, match="launcher integrity"):
        plan.run_local()
    plan.path.write_text("{}")
    with pytest.raises(ValueError, match="plan integrity"):
        plan.run_local()


def test_wrong_editable_import_and_incompatible_plan_reuse_are_refused(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = freeze_checkout(source, tmp_path / "frozen")
    task = SlurmTask(("-m", "driver", str(tmp_path / "results"), "0"))
    # A stdlib/site-packages module cannot masquerade as this project's source.
    with pytest.raises(ValueError, match="outside the frozen"):
        prepare_slurm_plan(
            frozen, [task], directory=tmp_path / "bad", revision="v1", imports=("json",)
        )
    plan = prepare_slurm_plan(
        frozen, [task], directory=tmp_path / "good", revision="v1", imports=("driver",)
    )
    with pytest.raises(ValueError, match="Incompatible existing"):
        prepare_slurm_plan(
            frozen,
            [task],
            directory=plan.path.parent,
            revision="v2",
            imports=("driver",),
        )
    # Replace the module in sys.path with another checkout. -c puts cwd first,
    # so use a guarded package absent in the frozen root to model a stale editable.
    (source / "stale.py").write_text("value = 1")
    monkeypatch.setenv("PYTHONPATH", str(source))
    with pytest.raises(ValueError, match="outside the frozen"):
        prepare_slurm_plan(
            frozen,
            [task],
            directory=tmp_path / "stale",
            revision="v1",
            imports=("stale",),
        )


def _scheduler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "bin"
    directory.mkdir()
    calls = tmp_path / "scheduler-calls.json"
    program = f"""#!{sys.executable}
import json, os, subprocess, sys
from pathlib import Path
name = Path(sys.argv[0]).name
if name == 'squeue':
    job = sys.argv[sys.argv.index('--jobs') + 1]
    if job == '11': print('11|RUNNING')
    elif job == '14':
        print('slurm_load_jobs error: Invalid job id specified', file=sys.stderr)
        sys.exit(1)
    elif job == '15':
        print('Unable to contact controller', file=sys.stderr)
        sys.exit(1)
    elif job == '16': print('16|FAILED')
elif name == 'sacct':
    job = sys.argv[sys.argv.index('--jobs') + 1]
    if job in ('12','14'): print(job + '|COMPLETED|0:0')
    elif job == '13': print('13|FAILED|1:0')
elif name == 'sbatch':
    Path({str(calls)!r}).write_text(json.dumps(sys.argv[1:]))
    cwd = next(arg.split('=',1)[1] for arg in sys.argv if arg.startswith('--chdir='))
    subprocess.run(['/bin/bash', sys.argv[-1]], cwd=cwd, check=True)
    print('222;fixture')
"""
    for name in ("squeue", "sacct", "sbatch"):
        path = directory / name
        path.write_text(program)
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ.get("PATH", ""))
    return calls


def test_submission_checks_completed_dependencies_and_executes_real_batch(
    plan: SlurmPlan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _scheduler(tmp_path, monkeypatch)
    job = plan.submit(after=("11", "12", "14", "11"))
    assert job == "222;fixture"
    args = json.loads(calls.read_text())
    assert "--dependency=afterok:11" in args
    assert "--cpus-per-task=4" in args
    assert (plan.path.parent / "task-000002.json").exists()


@pytest.mark.parametrize("job", ["13", "99", "12;other"])
def test_failed_unknown_and_nonlocal_dependencies_prevent_submission(
    plan: SlurmPlan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, job: str
) -> None:
    calls = _scheduler(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=r"Dependency|Dependencies"):
        plan.submit(after=(job,))
    assert not calls.exists()


def test_queue_transport_errors_are_not_treated_as_completed_jobs(
    plan: SlurmPlan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _scheduler(tmp_path, monkeypatch)
    with pytest.raises(subprocess.CalledProcessError):
        plan.submit(after=("15",))
    assert not calls.exists()


def test_queued_failed_state_prevents_submission(
    plan: SlurmPlan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _scheduler(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="incompatible queue state"):
        plan.submit(after=("16",))
    assert not calls.exists()


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_invalid_cpu_packing_is_rejected(width: int) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        SlurmResources(width=width)


def test_frozen_checkout_reuse_checks_full_commit_and_dirty_index(
    source: Path, tmp_path: Path
) -> None:
    frozen = freeze_checkout(source, tmp_path / "frozen")
    assert freeze_checkout(source, frozen.path) == frozen
    with pytest.raises(ValueError, match="full commit"):
        FrozenCheckout(frozen.path, frozen.commit[:12]).verify()
    (frozen.path / "driver.py").write_text("# modified")
    _git(frozen.path, "add", "driver.py")
    with pytest.raises(ValueError, match="source/index mismatch"):
        frozen.verify()
