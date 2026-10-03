"""Batched ask/tell adaptive search with external evaluation (#132).

``run_adaptive`` evaluates each trial in-process. Expensive studies run their
evaluations elsewhere -- for example as cluster job arrays built from a
manifest of (config, replicate) tasks -- so the optimizer must propose a batch,
wait for results produced by other processes, and then update.
``AdaptiveSession`` exposes that loop on the same optuna NSGA-II search:

- ``ask(n)`` proposes ``n`` configs, each with a trial id;
- ``tell(trial_id, scores)`` records results, either one value or the
  per-replicate values of each observable (means are optimized; standard
  errors and counts are kept with the trial);
- ``results()`` returns completed trials as a ``ResultsTable``.

With ``path`` set, the session lives in an optuna journal file, so asking and
telling can happen in different processes. Constraints are passed to the
sampler so infeasible regions are learned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from ._warm_start import _fingerprint, _import_trials, _session_schema, _validate_config
from .design import FactorType
from .protocols import Direction, ResultsTable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import optuna

    from .design import Factor
    from .protocols import Constraint, Observable

_STUDY_NAME = "trade-study-adaptive"


def _constraint_value(constraint: Constraint, value: float) -> float:
    """Return optuna's constraint value (feasible when <= 0).

    Raises:
        ValueError: If the operator has no continuous violation measure.
    """
    if constraint.op in {"<=", "<"}:
        return _open_violation(
            value - constraint.threshold, strict=constraint.op == "<"
        )
    if constraint.op in {">=", ">"}:
        return _open_violation(
            constraint.threshold - value, strict=constraint.op == ">"
        )
    if constraint.op == "==":
        return abs(value - constraint.threshold)
    msg = f"Constraint operator {constraint.op!r} is not supported in adaptive search"
    raise ValueError(msg)


def _open_violation(violation: float, *, strict: bool) -> float:
    if strict and violation == 0:
        return float(np.nextafter(0.0, np.inf))
    return violation


def _summarize(values: float | Sequence[float]) -> tuple[float, float, int]:
    """Return the mean, Monte Carlo standard error and count of replicates."""
    array = np.atleast_1d(np.asarray(values, dtype=float))
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    error = (
        float(np.std(finite, ddof=1) / np.sqrt(finite.size))
        if finite.size > 1
        else float("nan")
    )
    return float(finite.mean()), error, int(finite.size)


@dataclass(frozen=True)
class SessionTrial:
    """Snapshot of an adaptive trial, including failure and retry metadata."""

    trial_id: int
    config: dict[str, Any]
    state: str
    metadata: dict[str, Any]


class AdaptiveSession:
    """Persistent ask/tell multi-objective search over design factors."""

    def __init__(
        self,
        factors: list[Factor],
        observables: list[Observable],
        *,
        constraints: list[Constraint] | None = None,
        seed: int = 42,
        path: str | Path | None = None,
        study_name: str = _STUDY_NAME,
        revision: str | None = None,
    ) -> None:
        """Create or reopen a session.

        Args:
            factors: Design factors to search.
            observables: Objectives, with directions and weights.
            constraints: Feasibility constraints on reported scores; the
                ``!=`` operator is rejected with ``ValueError`` because it has
                no continuous violation measure.
            seed: Sampler seed. A reopened session restarts the sampler's
                random stream; the population already in storage is kept.
            path: Journal file for a persistent session; in-memory when
                ``None``.
            study_name: Name of the study inside the journal.
            revision: Caller-managed simulator/scorer/data/fidelity revision.
                Required for importing completed observations between searches.

        Raises:
            ValueError: If the revision is empty or a journal's stored schema
                is incompatible or absent in an existing nonempty study.
        """
        import optuna as _optuna
        from optuna.storages.journal import JournalFileBackend

        self.factors = factors
        self.observables = observables
        self.constraints = list(constraints or [])
        if revision is not None and not revision.strip():
            msg = "revision must be a nonempty model/data revision"
            raise ValueError(msg)
        self.revision = revision
        self._schema = _session_schema(factors, observables, self.constraints, revision)
        for constraint in self.constraints:
            _constraint_value(constraint, 0.0)
        if path is None:
            self._storage: optuna.storages.BaseStorage = (
                _optuna.storages.InMemoryStorage()
            )
        else:
            self._storage = _optuna.storages.JournalStorage(
                JournalFileBackend(str(Path(path)))
            )
        sampler = _optuna.samplers.NSGAIISampler(
            seed=seed,
            constraints_func=self._constraint_values if self.constraints else None,
        )
        self._sampler = sampler
        self._study = _optuna.create_study(
            study_name=study_name,
            storage=self._storage,
            directions=[
                "minimize" if o.direction == Direction.MINIMIZE else "maximize"
                for o in observables
            ],
            sampler=sampler,
            load_if_exists=True,
        )
        self._study_id = self._storage.get_study_id_from_name(study_name)
        stored = self._study.user_attrs.get("trade_study_identity")
        if stored is None and not self._study.get_trials():
            stored = {"schema": self._schema, "session_id": str(uuid4())}
            self._study.set_user_attr("trade_study_identity", stored)
        elif not isinstance(stored, dict) or stored.get("schema") != self._schema:
            msg = "Incompatible or legacy session schema; use a new journal/study name"
            raise ValueError(msg)
        self._session_id = str(stored["session_id"])

    def _constraint_values(self, trial: optuna.trial.FrozenTrial) -> list[float]:
        scores = trial.user_attrs.get("scores", {})
        errors = trial.user_attrs.get("standard_error", {})
        values = []
        for c in self.constraints:
            mean = float(scores.get(c.observable, np.nan))
            error = float(errors.get(c.observable, np.nan))
            unknown = not np.isfinite(mean) or (
                c.confidence is not None and (not np.isfinite(error) or error < 0)
            )
            if unknown:
                values.append(float("inf"))
                continue
            bound = c.bound(mean, error)
            values.append(_constraint_value(c, bound))
        return values

    def _suggest(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for f in self.factors:
            if f.factor_type == FactorType.CONTINUOUS and f.bounds is not None:
                config[f.name] = trial.suggest_float(
                    f.name, f.bounds[0], f.bounds[1], log=f.log_scale
                )
            elif f.levels is not None:
                config[f.name] = trial.suggest_categorical(f.name, f.levels)
        return config

    def ask(self, n: int = 1) -> list[tuple[int, dict[str, Any]]]:
        """Propose ``n`` configs.

        Returns:
            ``(trial_id, config)`` pairs; pass the id back to :meth:`tell`.

        Raises:
            ValueError: If ``n`` is less than 1.
        """
        if n < 1:
            msg = f"n must be >= 1; got {n}"
            raise ValueError(msg)
        proposals = []
        for _ in range(n):
            trial = self._study.ask()
            config = self._suggest(trial)
            self._register_generation(trial.number)
            proposals.append((trial.number, config))
        return proposals

    def tell(
        self, trial_id: int, scores: Mapping[str, float | Sequence[float]]
    ) -> None:
        """Record a trial's scores.

        Args:
            trial_id: Id returned by :meth:`ask`.
            scores: Per observable, one value or the per-replicate values.
                Objectives are optimized on their (weighted) means; every
                reported score is available to constraints.

        Raises:
            ValueError: If the trial id is unknown, already told, or an
                objective or constraint score is missing, or a constraint
                has no finite mean/required standard error. Rejected tells
                leave the trial pending and may be corrected.
        """
        import optuna as _optuna

        internal = self._trial_id(trial_id)
        if self._storage.get_trial(internal).state != _optuna.trial.TrialState.RUNNING:
            msg = f"Trial {trial_id} was already told"
            raise ValueError(msg)
        missing = [o.name for o in self.observables if o.name not in scores]
        if missing:
            msg = f"Missing objective scores: {missing}"
            raise ValueError(msg)
        summary = {name: _summarize(values) for name, values in scores.items()}
        for constraint in self.constraints:
            if constraint.observable not in summary:
                msg = f"Missing constraint score: {constraint.observable!r}"
                raise ValueError(msg)
            mean, standard_error, _count = summary[constraint.observable]
            if not np.isfinite(mean):
                msg = f"Constraint {constraint.name!r} needs a finite reported mean"
                raise ValueError(msg)
            constraint.bound(mean, standard_error)
        self._storage.set_trial_user_attr(
            internal, "scores", {k: v[0] for k, v in summary.items()}
        )
        self._storage.set_trial_user_attr(
            internal, "standard_error", {k: v[1] for k, v in summary.items()}
        )
        self._storage.set_trial_user_attr(
            internal, "n_reps", {k: v[2] for k, v in summary.items()}
        )
        self._register_generation(trial_id)
        self._study.tell(
            trial_id, [summary[o.name][0] * o.weight for o in self.observables]
        )

    def _trial_id(self, trial_id: int) -> int:
        try:
            return self._storage.get_trial_id_from_study_id_trial_number(
                self._study_id, trial_id
            )
        except KeyError as error:
            msg = f"Unknown trial id {trial_id}"
            raise ValueError(msg) from error

    def _register_generation(self, trial_id: int) -> None:
        trial = self._storage.get_trial(self._trial_id(trial_id))
        self._sampler.get_trial_generation(self._study, trial)

    def trials(self, state: str | None = None) -> list[SessionTrial]:
        """Inspect trials in creation order without exposing storage internals.

        Args:
            state: Optional filter: ``pending``, ``complete``, ``failed``,
                ``waiting``, or ``pruned``. Pending trials are running trials
                whose external evaluations have not been reported.

        Returns:
            Independent snapshots, including raw scores, failure reasons,
            and retry lineage in metadata where available.

        Raises:
            ValueError: If the state filter is unknown.
        """
        states = {
            "RUNNING": "pending",
            "COMPLETE": "complete",
            "FAIL": "failed",
            "WAITING": "waiting",
            "PRUNED": "pruned",
        }
        if state is not None and state not in states.values():
            msg = f"Unknown trial state {state!r}"
            raise ValueError(msg)
        return [
            SessionTrial(
                t.number,
                dict(t.params or t.system_attrs.get("fixed_params", {})),
                states[t.state.name],
                t.user_attrs,
            )
            for t in self._study.get_trials(deepcopy=True)
            if state is None or states[t.state.name] == state
        ]

    def fail(self, trial_id: int, reason: str) -> None:
        """Mark a pending trial failed, preserving its configuration.

        Args:
            trial_id: Id returned by :meth:`ask` or :meth:`retry`.
            reason: Nonempty description of the evaluation failure.

        Raises:
            ValueError: If the id is unknown, the trial is not pending, or
                the reason is empty. Duplicate failure reports are rejected.
        """
        import optuna as _optuna

        if not reason.strip():
            msg = "A failure reason must be nonempty"
            raise ValueError(msg)
        internal = self._trial_id(trial_id)
        if self._storage.get_trial(internal).state != _optuna.trial.TrialState.RUNNING:
            msg = f"Trial {trial_id} is not pending"
            raise ValueError(msg)
        self._storage.set_trial_user_attr(internal, "failure_reason", reason)
        self._study.tell(trial_id, state=_optuna.trial.TrialState.FAIL)

    def retry(
        self, trial_id: int, *, max_retries: int = 1
    ) -> tuple[int, dict[str, Any]]:
        """Create a bounded retry of a failed trial with identical parameters.

        Args:
            trial_id: Failed trial to retry. To retry another failed attempt,
                pass that attempt's id.
            max_retries: Maximum additional attempts across the retry chain.

        Returns:
            New trial id and the original configuration. Repeating a request
            for the same failed id returns its existing child, including
            after reopening; it never creates a second child.

        Raises:
            ValueError: If the id is unknown, not failed, or its retry limit
                has been reached, or the limit is less than one.

        Notes:
            Serialize retry requests for a given id. Idempotency applies to
            repeated requests, not simultaneous requests from multiple writers.
        """
        import optuna as _optuna

        if max_retries < 1:
            msg = "max_retries must be >= 1"
            raise ValueError(msg)
        source = self._storage.get_trial(self._trial_id(trial_id))
        if source.state != _optuna.trial.TrialState.FAIL:
            msg = f"Trial {trial_id} is not failed"
            raise ValueError(msg)
        for trial in self._study.get_trials():
            if trial.user_attrs.get("retry_of") == trial_id:
                return trial.number, dict(trial.params)
        attempt = int(source.user_attrs.get("retry_attempt", 0)) + 1
        if attempt > max_retries:
            msg = f"Trial {trial_id} reached its retry limit ({max_retries})"
            raise ValueError(msg)
        template = _optuna.trial.create_trial(
            state=_optuna.trial.TrialState.RUNNING,
            params=source.params,
            distributions=source.distributions,
            user_attrs={"retry_of": trial_id, "retry_attempt": attempt},
        )
        internal = self._storage.create_new_trial(self._study_id, template)
        child = self._storage.get_trial(internal)
        self._register_generation(child.number)
        return child.number, dict(child.params)

    def enqueue(self, config: dict[str, Any]) -> None:
        """Queue a known configuration before sampling new configurations.

        Args:
            config: Complete factor configuration inside the declared domain.

        Raises:
            ValueError: If the configuration's factor names or values are invalid.

        Notes:
            Each call queues a distinct evaluation. Completed observations
            should instead be imported with :meth:`warm_start`.
        """
        if set(config) != {f.name for f in self.factors}:
            msg = "Configuration must contain every session factor exactly"
            raise ValueError(msg)
        self._study.enqueue_trial(_validate_config(config, self.factors))

    def warm_start(self, source: AdaptiveSession | ResultsTable) -> list[int]:
        """Import compatible completed evaluations, without re-evaluating them.

        Args:
            source: Session or saved/loaded table produced by session.results().
                Factor/objective/constraint definitions and explicit revision
                must match. Bare grid tables have no verifiable session schema.

        Returns:
            Newly imported trial ids. Repeated imports skip evaluations already
            present, including after reopening and through intermediate imports.

        Raises:
            ValueError: If revision/schema/provenance/observations are invalid,
                or an existing evaluation id has conflicting results.

        Notes:
            Serialize imports into a destination session. The revision is the
            caller's assertion of matching model, scorer, data and fidelity;
            the library cannot establish that assertion from scores alone.
        """
        import optuna as _optuna

        if self.revision is None:
            msg = "warm_start requires an explicit model/data revision"
            raise ValueError(msg)
        results = source.results() if isinstance(source, AdaptiveSession) else source
        templates = _import_trials(
            results, self.factors, self.observables, self.constraints, self._schema
        )
        trials = self._study.get_trials()
        unfinished = {
            t.user_attrs["evaluation_id"]: t.number
            for t in trials
            if t.state.name == "RUNNING" and "_import_values" in t.user_attrs
        }
        existing = {
            t.user_attrs.get(
                "evaluation_id", f"{self._session_id}:{t.number}"
            ): _fingerprint(t)
            for t in trials
            if t.state.name == "COMPLETE"
            or (t.state.name == "RUNNING" and "_import_values" in t.user_attrs)
        }
        pending = []
        for trial in templates:
            identity = trial.user_attrs["evaluation_id"]
            fingerprint = _fingerprint(trial)
            if identity in existing and existing[identity] != fingerprint:
                msg = f"Conflicting imported evaluation {identity!r}"
                raise ValueError(msg)
            if identity not in existing or identity in unfinished:
                trial.user_attrs["_resume_trial"] = unfinished.pop(identity, None)
                pending.append(trial)
                existing[identity] = fingerprint
        imported = []
        for trial in pending:
            number = trial.user_attrs.pop("_resume_trial")
            if number is None:
                running = _optuna.trial.create_trial(
                    state=_optuna.trial.TrialState.RUNNING,
                    params=trial.params,
                    distributions=trial.distributions,
                    user_attrs={**trial.user_attrs, "_import_values": trial.values},
                )
                internal = self._storage.create_new_trial(self._study_id, running)
                number = self._storage.get_trial(internal).number
            self._register_generation(number)
            self._study.tell(number, trial.values)
            imported.append(number)
        return imported

    def results(self) -> ResultsTable:
        """Return completed trials.

        Returns:
            ``ResultsTable`` with one row per told trial: the config, the
            weighted objective values (as ``run_adaptive`` reports them), and
            metadata holding the trial id, all reported means, their standard
            errors and replicate counts, and the constraint values (feasible
            when every value is at most zero).
        """
        import optuna as _optuna

        done = self._study.get_trials(states=(_optuna.trial.TrialState.COMPLETE,))
        return ResultsTable(
            configs=[dict(t.params) for t in done],
            scores=np.array([list(t.values) for t in done], dtype=float).reshape(
                len(done), len(self.observables)
            ),
            observable_names=[o.name for o in self.observables],
            metadata=[
                {
                    "trial": t.number,
                    "session_id": self._session_id,
                    "session_schema": self._schema,
                    "evaluation_id": t.user_attrs.get(
                        "evaluation_id", f"{self._session_id}:{t.number}"
                    ),
                    "scores": t.user_attrs.get("scores", {}),
                    "standard_error": t.user_attrs.get("standard_error", {}),
                    "n_reps": t.user_attrs.get("n_reps", {}),
                    "constraints": self._constraint_values(t),
                    **{
                        k: t.user_attrs[k]
                        for k in ("retry_of", "retry_attempt", "provenance")
                        if k in t.user_attrs
                    },
                }
                for t in done
            ],
        )
