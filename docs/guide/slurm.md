# Frozen task submission with Slurm

`trade_study.slurm` prepares explicit Python tasks in a detached Git checkout,
records their runtime and data revision, and packs their processes into one
Slurm node allocation. The same plan can run locally before submission.
Git and a shared POSIX filesystem are required; Slurm commands are needed only
when submitting or checking job dependencies. No Python dependency is added.

This is a launch layer around existing runners. It does not add an `executor=`
argument to `run_grid` or `run_adaptive`, distribute live Python evaluators, merge
shard results, or choose an adaptive study's next trials. Each task calls the
existing runner or persistent session explicitly. Keep the runner's original
replicate IDs, checkpoint keys and journal contracts within each task.

## Prepare a reviewable plan

Resolve the source commit before submitting. Create the Python environment from
that checkout, using the project's own dependency lock or environment procedure.
For an editable install, install the project from the **frozen** checkout.
The package guard refuses an import resolving to the submitting checkout.
Use an explicit shared interpreter path, preserving its virtual-environment
symlink; resolving that symlink to a base Python binary would lose the environment.
The linked base interpreter and its libraries must also exist on the compute
nodes. A successful login-node import alone cannot establish that availability.

```python
from trade_study.slurm import (
    SlurmResources, SlurmTask, freeze_checkout, prepare_slurm_plan,
)

frozen = freeze_checkout("/shared/project", "/shared/runs/source-commit", ref="HEAD")
# Create/install /shared/envs/source-commit from frozen.path before preparing.
plan = prepare_slurm_plan(
    frozen,
    [
        SlurmTask(
            ("-m", "project.study", "--shard", "0", "--shards", "2"),
            outputs=("/shared/results/shard-0.json",),
        ),
        SlurmTask(
            ("-m", "project.study", "--shard", "1", "--shards", "2"),
            outputs=("/shared/results/shard-1.json",),
        ),
    ],
    directory="/shared/runs/launch-plan",
    python="/shared/envs/source-commit/bin/python",
    imports=("project",),
    revision="data-and-study-policy-v3",
    resources=SlurmResources(
        width=2, cpus_per_task=4, time_limit="04:00:00", memory="32G",
        partition="your-partition", account="your-account",
    ),
)
```

Inspect `plan.json` and retain `plan.identity` independently. It records the full
commit, ordered tasks, declared output paths, caller-managed revision, resources,
interpreter, installed package versions, guarded import locations and copied
launcher identity. A changed plan requires a new directory. The selected modules
must be safe to import without launching the study. Module entrypoints are added
to the import guards; script entrypoints must be tracked in the frozen checkout.

The source guard checks the full commit, a detached HEAD and the tracked index
and worktree. It checks each named module's tracked file location. Dependencies,
dynamic plugins, external data and environment behavior beyond those recorded
identities still need the caller's immutable environment/data procedure and
`revision`. Put declared output files outside both source and plan directories.
Package versions and hashes are compatibility checks, not publisher
authentication or scientific validation. Untracked data and generated files are
not added to Git by this API.

## Execute and resume

```python
plan.run_local()                 # same copied launcher and worker guards
job = plan.submit()              # explicit scheduler submission
# Both plans use the same connected cluster; after accepts numeric job IDs.
later_job = later_plan.submit(after=(job.split(";", 1)[0],))
```

The plan requests one node and `width * cpus_per_task` CPUs. It runs at most
`width` child Python processes at once. Each child gets the declared CPU count
in `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and
`SLURM_CPUS_PER_TASK`. These are batch children sharing the allocation's CPU set,
rather than separately bound `srun` steps. Set the whole-allocation memory and
time limits for that packing. Site accounting, CPU affinity, filesystem locking
and cancellation behavior require validation on the target cluster.

Each task appends to its own log. Any task failure fails the batch, while other
successful tasks retain atomic completion receipts. A shared file lock prevents
two submissions of the same plan from executing a task simultaneously. Resume
checks the original plan, source, environment and receipt before skipping work.
Declared output files are hashed: missing files rerun that task; changed content
is refused. Without declared outputs a receipt attests successful exit only.
Use the runner's ledger/session to recover work inside an unfinished task.

Dependencies use `afterok` only for active jobs on the connected cluster. Finished
jobs must have a matching `sacct` allocation record with `COMPLETED` and `0:0`;
failed or unavailable accounting records prevent submission. This accommodates
Slurm's finite dependency retention window. The returned parsable job reference
retains any cluster suffix; dependency IDs must be numeric and local to the
connected cluster. A scheduler command failure is reported, with no automatic
resubmission that could create duplicate jobs. Consult the native
[dependency](https://slurm.schedmd.com/sbatch.html) and
[exit-code](https://slurm.schedmd.com/job_exit_code.html) documentation for site behavior.

## Run the complete small study

```bash
python examples/frozen_slurm.py /tmp/frozen-normal-study
```

The example commits a small standalone normal-mean study, freezes its source,
runs two tasks locally and prints a table comparing empirical squared error
with the analytic expectation `1/n`. Sixteen paired replicates are a demonstration,
not a precision or calibration qualification. Calling it again reuses matching
outputs and task receipts. `--submit` explicitly submits the same demonstration
to the connected scheduler; review its resource/site defaults first.

The fixed seed gives this small table; the empirical values vary from the
expectations because only 16 replicates were run:

| n | Replicates | Empirical squared error | Analytic expected error |
| ---: | ---: | ---: | ---: |
| 8 | 16 | 0.11837 | 0.12500 |
| 32 | 16 | 0.04261 | 0.03125 |

Required tests execute real Git worktrees, Python subprocesses and grid ledgers,
including failed tasks, overlapping submissions, missing/corrupted output files,
wrong imports, source/manifest/launcher changes and a fixture that runs the actual
generated batch script through native Bash. The fixture checks scheduler command
arguments and dependency failures.

A small Rockfish pilot used Python 3.11.9, NumPy 2.0, two CPUs, 1 GB and a
two-minute limit. It reproduced the table above, then reused the same task
receipts in a second successful allocation. A dependency on a failed job was
refused before submission. An initial attempt failed before running tasks when
its virtual environment pointed to a login-only interpreter; selecting the
shared Python module resolved that site configuration. This single-site pilot
supports the launch example, rather than general cluster qualification or a
complete distributed adaptive executor.
