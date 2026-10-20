"""Frozen Python task submission and node packing for existing study runners.

Adapted from the FLU_USA, denv-ade-v2 and mcvf frozen-worktree/packing scripts.
Tasks are explicit Python argv lists; no live evaluators are pickled. Existing
grid ledgers and adaptive journals remain responsible for within-task recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] - reviewed argv
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "FrozenCheckout",
    "SlurmPlan",
    "SlurmResources",
    "SlurmTask",
    "freeze_checkout",
    "prepare_slurm_plan",
]


def _json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _program(name: str) -> str:
    location = shutil.which(name)
    if location is None:
        msg = f"Required executable is unavailable: {name}"
        raise FileNotFoundError(msg)
    return location


def _capture(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    # Commands are argv vectors supplied by this API, never shell interpolation.
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        argv, cwd=cwd, text=True, capture_output=True, check=True
    ).stdout.strip()


def _git(directory: Path, *args: str) -> str:
    return _capture([_program("git"), "-C", str(directory), *args])


@dataclass(frozen=True)
class FrozenCheckout:
    """A detached checkout's location and full commit identity.

    Construction alone does not verify files. ``verify`` checks the current
    commit and tracked worktree/index before submission and each task boundary.
    Keep data, logs, virtual environments and outputs outside the tracked tree.
    """

    path: Path
    commit: str

    def verify(self) -> None:
        """Refuse a moved commit or modified tracked source/index.

        Raises:
            ValueError: The current commit or tracked source differs.
        """
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.commit):
            msg = "A frozen checkout requires a full commit hash"
            raise ValueError(msg)
        if _git(self.path, "rev-parse", "--abbrev-ref", "HEAD") != "HEAD":
            msg = "Frozen execution requires a detached worktree"
            raise ValueError(msg)
        if _git(self.path, "rev-parse", "HEAD") != self.commit or _git(
            self.path, "status", "--porcelain", "--untracked-files=no"
        ):
            msg = "Frozen checkout commit or tracked source/index mismatch"
            raise ValueError(msg)


def freeze_checkout(
    repository: str | Path, destination: str | Path, *, ref: str = "HEAD"
) -> FrozenCheckout:
    """Create or reuse a detached worktree at an explicitly resolved commit.

    Args:
        repository: Existing local Git repository; no fetch or branch change occurs.
        destination: New worktree location, or the same previously frozen checkout.
        ref: Commit or ref resolved once, before any job is submitted.

    Returns:
        Verified full-commit checkout. Queued jobs do not read the submitter's tree.

    Git errors propagate as ``subprocess.CalledProcessError``; verification
    raises ``ValueError`` for a moved commit or changed tracked source.
    """
    repository = Path(repository).resolve()
    destination = Path(destination).resolve()
    commit = _git(
        repository, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"
    )
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _git(repository, "worktree", "add", "--detach", str(destination), commit)
    frozen = FrozenCheckout(destination, commit)
    frozen.verify()
    return frozen


@dataclass(frozen=True)
class SlurmResources:
    """One node's CPU packing and site-specific submission choices.

    ``width * cpus_per_task`` CPUs are requested in one allocation. Child tasks
    share that allocation's CPU set; no per-child CPU affinity is imposed.
    Memory and wall time apply to the whole allocation, rather than each child.

    Examples:
        >>> resource = SlurmResources(width=4, cpus_per_task=2)
        >>> resource.width * resource.cpus_per_task
        8
    """

    width: int = 1
    cpus_per_task: int = 1
    time_limit: str = "01:00:00"
    memory: str | None = None
    partition: str | None = None
    account: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid packing sizes and malformed resource tokens.

        Raises:
            ValueError: Counts are not positive integers or a token is invalid.
        """
        if any(
            type(value) is not int or value < 1
            for value in (self.width, self.cpus_per_task)
        ):
            msg = "Slurm width and CPUs per task must be positive integers"
            raise ValueError(msg)
        for value in (self.time_limit, self.memory, self.partition, self.account):
            if value is not None and not re.fullmatch(r"[A-Za-z0-9_.:+/-]+", value):
                msg = "Slurm resource values must be nonempty single tokens"
                raise ValueError(msg)


_RUNTIME_PROBE = """
import importlib, importlib.metadata, json, pathlib, platform, sys
origins = {}
for name in json.loads(sys.argv[1]):
    module = importlib.import_module(name)
    if module.__file__ is None:
        raise ValueError('Explicit file-backed import guards are required')
    origins[name] = str(pathlib.Path(module.__file__).resolve())
print(json.dumps({
    'python': sys.version,
    'executable': str(pathlib.Path(sys.executable).absolute()),
    'prefix': str(pathlib.Path(sys.prefix).resolve()),
    'platform': platform.system() + '-' + platform.machine(),
    'packages': sorted((dist.metadata['Name'], dist.version)
                       for dist in importlib.metadata.distributions()),
    'origins': origins,
}, sort_keys=True))
"""


@dataclass(frozen=True)
class SlurmTask:
    """One Python argv vector and optional declared output files.

    Output paths are absolute and outside the frozen checkout. Completion
    receipts hash these files. Missing files cause reevaluation; changed content
    is refused. Without declared outputs, a receipt attests successful exit only.
    """

    args: tuple[str, ...]
    outputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate immutable task arguments and explicit output locations.

        Raises:
            ValueError: Arguments or output paths do not meet the task contract.
        """
        if (
            not isinstance(self.args, tuple)
            or not self.args
            or not isinstance(self.outputs, tuple)
            or any(
                not isinstance(arg, str) or not arg or "\0" in arg for arg in self.args
            )
            or any(
                not isinstance(path, str) or not Path(path).is_absolute()
                for path in self.outputs
            )
        ):
            msg = "Use immutable Python argv and absolute declared output paths"
            raise ValueError(msg)


def _runtime(
    python: str, checkout: FrozenCheckout, imports: Sequence[str]
) -> dict[str, Any]:
    raw = _capture(
        [python, "-c", _RUNTIME_PROBE, json.dumps(list(imports))], cwd=checkout.path
    )
    identity: dict[str, Any] = json.loads(raw)
    for location in identity["origins"].values():
        origin = Path(location)
        if not origin.is_relative_to(checkout.path):
            msg = f"Import resolves outside the frozen checkout: {origin}"
            raise ValueError(msg)
        _git(
            checkout.path,
            "ls-files",
            "--error-unmatch",
            "--",
            str(origin.relative_to(checkout.path)),
        )
    return identity


def _write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            msg = "Incompatible existing Slurm plan; use a new output directory"
            raise ValueError(msg)
        return
    with path.open("xb") as handle:
        handle.write(content)


def _task_sources(
    checkout: FrozenCheckout, tasks: Sequence[SlurmTask], imports: Sequence[str]
) -> list[str]:
    names = set(imports)
    if isinstance(imports, str) or any(
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", name)
        for name in names
    ):
        msg = "Import guards require explicit module names"
        raise ValueError(msg)
    for task in tasks:
        entry = task.args[0]
        if entry in {"-m", "-c"}:
            if len(task.args) < 2:
                msg = "Python module/inline tasks require an entrypoint"
                raise ValueError(msg)
            if entry == "-m":
                names.add(task.args[1])
        else:
            # Check both the tracked link and its target, without resolving away
            # an untracked symlink used as the actual command entrypoint.
            source = (checkout.path / entry).absolute()
            if (
                entry.startswith("-")
                or not source.is_relative_to(checkout.path)
                or not source.resolve().is_relative_to(checkout.path)
            ):
                msg = "Task scripts must be tracked inside the frozen checkout"
                raise ValueError(msg)
            _git(
                checkout.path,
                "ls-files",
                "--error-unmatch",
                "--",
                str(source.relative_to(checkout.path)),
            )
    return sorted(names)


def prepare_slurm_plan(
    checkout: FrozenCheckout,
    tasks: Sequence[SlurmTask],
    *,
    directory: str | Path,
    revision: str,
    imports: Sequence[str],
    python: str | Path = sys.executable,
    resources: SlurmResources | None = None,
) -> SlurmPlan:
    """Freeze explicit Python task argv, runtime, code and external-data revision.

    Args:
        checkout: Detached worktree created by ``freeze_checkout``.
        tasks: Ordered tasks, e.g. ``SlurmTask(('-m', 'project.run', '0'))``.
            The selected interpreter is prepended; strings are not shell programs.
        directory: Plan/log/completion directory outside the frozen source tree.
        revision: Nonempty caller-managed identity for external data, configuration
            and environment behavior not captured by commits/package versions.
        imports: Project modules whose tracked import origins must be in the frozen
            checkout; at least one guard is required.
        python: Exact shared interpreter path, preserving virtualenv symlinks.
        resources: Single-node CPU packing and site-specific Slurm choices.

    Returns:
        Immutable plan that can be reviewed, run locally, or explicitly submitted.

    Raises:
        ValueError: Commands, revision, imports, source, runtime or reuse are invalid.
    """
    checkout = FrozenCheckout(checkout.path.resolve(), checkout.commit)
    checkout.verify()
    directory = Path(directory).resolve()
    python = str(Path(python).absolute())
    if directory.is_relative_to(checkout.path) or not revision.strip() or not imports:
        msg = "Use external outputs, a nonempty revision and project import guards"
        raise ValueError(msg)
    if not tasks or any(not isinstance(task, SlurmTask) for task in tasks):
        msg = "Tasks require explicit SlurmTask Python argv records"
        raise ValueError(msg)
    outputs = [str(Path(path).resolve()) for task in tasks for path in task.outputs]
    if len(set(outputs)) != len(outputs) or any(
        Path(path).is_relative_to(checkout.path) or Path(path).is_relative_to(directory)
        for path in outputs
    ):
        msg = "Declared outputs must be distinct files outside source and plan"
        raise ValueError(msg)
    imports = _task_sources(checkout, tasks, imports)
    runtime = _runtime(python, checkout, imports)
    launcher = Path(__file__).read_bytes()
    payload = {
        "schema": "trade-study-slurm-v1",
        "checkout": str(checkout.path),
        "commit": checkout.commit,
        "python": python,
        "tasks": [
            {
                "args": list(task.args),
                "outputs": [str(Path(p).resolve()) for p in task.outputs],
            }
            for task in tasks
        ],
        "imports": list(imports),
        "revision": revision,
        "resources": asdict(resources or SlurmResources()),
        "runtime": runtime,
        "launcher_digest": _hash(launcher),
    }
    directory.mkdir(parents=True, exist_ok=True)
    _write_once(directory / "launcher.py", launcher)
    _write_once(directory / "plan.json", _json(payload))
    return SlurmPlan(directory / "plan.json", _hash(_json(payload)))


def _read_plan(path: Path, identity: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if _hash(payload) != identity:
        msg = "Slurm plan integrity mismatch"
        raise ValueError(msg)
    plan: dict[str, Any] = json.loads(payload)
    if plan.get("schema") != "trade-study-slurm-v1":
        msg = "Unknown Slurm plan schema"
        raise ValueError(msg)
    return plan


def _verify_plan(plan: dict[str, Any]) -> FrozenCheckout:
    checkout = FrozenCheckout(Path(plan["checkout"]), plan["commit"])
    checkout.verify()
    if _runtime(plan["python"], checkout, plan["imports"]) != plan["runtime"]:
        msg = "Frozen Python runtime/package/import identity mismatch"
        raise ValueError(msg)
    return checkout


def _dependencies(jobs: Sequence[str]) -> list[str]:
    pending = []
    for job in dict.fromkeys(jobs):
        if not re.fullmatch(r"[0-9]+", job):
            msg = "Dependencies must be numeric job IDs on the connected cluster"
            raise ValueError(msg)
        try:
            queued = _capture([
                _program("squeue"),
                "--noheader",
                "--jobs",
                job,
                "--format=%i|%T",
            ])
        except subprocess.CalledProcessError as error:
            if "invalid job id" not in (error.stderr or "").lower():
                raise
            queued = ""
        if queued:
            rows = [row.strip().split("|") for row in queued.splitlines()]
            if len(rows) != 1 or rows[0][0] != job:
                msg = "Ambiguous dependency queue status"
                raise ValueError(msg)
            if rows[0][1] not in {
                "PENDING",
                "RUNNING",
                "CONFIGURING",
                "COMPLETING",
                "SUSPENDED",
                "RESIZING",
                "STAGE_OUT",
                "REQUEUED",
                "REQUEUE_FED",
                "REQUEUE_HOLD",
            }:
                msg = f"Dependency {job} has incompatible queue state {rows[0][1]}"
                raise ValueError(msg)
            pending.append(job)
            continue
        accounted = _capture([
            _program("sacct"),
            "--noheader",
            "--allocations",
            "--parsable2",
            "--jobs",
            job,
            "--format=JobIDRaw,State,ExitCode",
        ])
        rows = [row.strip().split("|") for row in accounted.splitlines() if row.strip()]
        matches = [row for row in rows if row[0] == job]
        if len(matches) != 1 or matches[0][1:3] != ["COMPLETED", "0:0"]:
            msg = f"Dependency {job} is failed, unavailable or not confirmed completed"
            raise ValueError(msg)
    return pending


@dataclass(frozen=True)
class SlurmPlan:
    """A reviewed launch manifest and its independently retained content identity.

    ``submit`` explicitly launches a batch job. ``run_local`` exercises packing
    without a scheduler. Both refuse changed source/runtime and reuse only task
    completions matching this exact plan. Keep the output directory unchanged.
    """

    path: Path
    identity: str

    def run_local(self) -> None:
        """Run the same guarded pack locally, propagating any child failure.

        Native child failures raise ``subprocess.CalledProcessError``.

        Raises:
            ValueError: Plan, launcher, source or runtime differs.
        """
        plan = _read_plan(self.path, self.identity)
        launcher = self.path.parent / "launcher.py"
        if _hash(launcher.read_bytes()) != plan["launcher_digest"]:
            msg = "Slurm launcher integrity mismatch"
            raise ValueError(msg)
        _capture([plan["python"], str(launcher), str(self.path), self.identity])

    def submit(self, *, after: Sequence[str] = ()) -> str:
        """Submit one packed job, checking finished dependencies before afterok.

        Scheduler errors propagate as ``subprocess.CalledProcessError``; failed
        submission commands are not silently retried.

        Args:
            after: Numeric job IDs on the connected Slurm cluster. Active jobs
                become afterok dependencies; confirmed successful jobs are omitted.

        Returns:
            Native parsable sbatch job reference, retaining any cluster suffix.

        Raises:
            ValueError: Plan/source/runtime or dependency status is incompatible.
        """
        plan = _read_plan(self.path, self.identity)
        checkout = _verify_plan(plan)
        launcher = self.path.parent / "launcher.py"
        if _hash(launcher.read_bytes()) != plan["launcher_digest"]:
            msg = "Slurm launcher integrity mismatch"
            raise ValueError(msg)
        # The fixed bootstrap checks copied code before executing it. Job args
        # are shell-quoted only here; all task commands themselves use argv.
        import shlex

        bootstrap = (
            "import hashlib,pathlib,runpy,sys; "
            "source=pathlib.Path(sys.argv.pop(1)); expected=sys.argv.pop(1); "
            "actual=hashlib.sha256(source.read_bytes()).hexdigest(); "
            "sys.exit('Slurm launcher integrity mismatch') "
            "if actual!=expected else None; "
            "sys.argv[0]=str(source); runpy.run_path(str(source),run_name='__main__')"
        )
        script = (
            "#!/bin/bash\nset -euo pipefail\nexec "
            + shlex.join([
                plan["python"],
                "-c",
                bootstrap,
                str(launcher),
                plan["launcher_digest"],
                str(self.path),
                self.identity,
            ])
            + "\n"
        )
        batch = self.path.parent / "batch.sh"
        _write_once(batch, script.encode())
        resource = SlurmResources(**plan["resources"])
        args = [
            _program("sbatch"),
            "--parsable",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={resource.width * resource.cpus_per_task}",
            f"--time={resource.time_limit}",
            f"--chdir={checkout.path}",
            f"--output={self.path.parent / 'slurm-%j.log'}",
        ]
        for flag, value in (
            ("mem", resource.memory),
            ("partition", resource.partition),
            ("account", resource.account),
        ):
            if value is not None:
                args.append(f"--{flag}={value}")
        pending = _dependencies(after)
        if pending:
            args.append("--dependency=afterok:" + ":".join(pending))
        submitted = _capture([*args, str(batch)])
        if not re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", submitted):
            msg = (
                "Unexpected sbatch job reference; inspect the scheduler before retrying"
            )
            raise ValueError(msg)
        return submitted


def _atomic_json(path: Path, value: object) -> None:
    descriptor, scratch = tempfile.mkstemp(dir=path.parent, prefix="completion-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json(value))
        Path(scratch).replace(path)
    finally:
        Path(scratch).unlink(missing_ok=True)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task(plan: dict[str, Any], path: Path, identity: str, index: int) -> int:
    # flock serializes duplicate submissions of a task on a shared filesystem.
    # Site-specific locking/durability semantics still need cluster qualification.
    locks = importlib.import_module("fcntl")
    stem = path.parent / f"task-{index:06d}"
    marker = stem.with_suffix(".json")
    with stem.with_suffix(".lock").open("a") as lock:
        locks.flock(lock, locks.LOCK_EX)
        _verify_plan(plan)
        task = plan["tasks"][index]
        expected = {"plan": identity, "task": index, "command": task["args"]}
        if marker.exists():
            stored = json.loads(marker.read_text())
            if {key: stored[key] for key in expected} != expected or set(
                stored["outputs"]
            ) != set(task["outputs"]):
                msg = "Task completion provenance mismatch"
                raise ValueError(msg)
            if all(Path(output).is_file() for output in task["outputs"]):
                if any(
                    _file_digest(Path(output)) != digest
                    for output, digest in stored["outputs"].items()
                ):
                    msg = "Task output integrity mismatch"
                    raise ValueError(msg)
                return 0
        environment = os.environ.copy()
        cpus = str(plan["resources"]["cpus_per_task"])
        environment.update(
            dict.fromkeys(
                (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "SLURM_CPUS_PER_TASK",
                ),
                cpus,
            )
        )
        with stem.with_suffix(".log").open("ab") as log:
            result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - explicit reviewed Python argv
                [plan["python"], *task["args"]],
                cwd=plan["checkout"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            return result.returncode
        _verify_plan(plan)
        _atomic_json(
            marker,
            expected
            | {
                "outputs": {
                    output: _file_digest(Path(output)) for output in task["outputs"]
                }
            },
        )
        return 0


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("identity")
    args = parser.parse_args()
    plan = _read_plan(args.manifest, args.identity)
    _verify_plan(plan)
    with ThreadPoolExecutor(max_workers=plan["resources"]["width"]) as pool:
        outcomes = list(
            pool.map(
                lambda index: _task(plan, args.manifest, args.identity, index),
                range(len(plan["tasks"])),
            )
        )
    if any(outcomes):
        sys.exit(1)


if __name__ == "__main__":
    _main()
