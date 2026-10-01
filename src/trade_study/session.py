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

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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
        return value - constraint.threshold
    if constraint.op in {">=", ">"}:
        return constraint.threshold - value
    if constraint.op == "==":
        return abs(value - constraint.threshold)
    msg = f"Constraint operator {constraint.op!r} is not supported in adaptive search"
    raise ValueError(msg)


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
        """
        import optuna as _optuna
        from optuna.storages.journal import JournalFileBackend

        self.factors = factors
        self.observables = observables
        self.constraints = list(constraints or [])
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

    def _constraint_values(self, trial: optuna.trial.FrozenTrial) -> list[float]:
        scores = trial.user_attrs.get("scores", {})
        return [
            _constraint_value(c, float(scores.get(c.observable, np.inf)))
            for c in self.constraints
        ]

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
            proposals.append((trial.number, self._suggest(trial)))
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
                objective is missing.
        """
        import optuna as _optuna

        try:
            internal = self._storage.get_trial_id_from_study_id_trial_number(
                self._study_id, trial_id
            )
        except KeyError as error:
            msg = f"Unknown trial id {trial_id}"
            raise ValueError(msg) from error
        if self._storage.get_trial(internal).state != _optuna.trial.TrialState.RUNNING:
            msg = f"Trial {trial_id} was already told"
            raise ValueError(msg)
        missing = [o.name for o in self.observables if o.name not in scores]
        if missing:
            msg = f"Missing objective scores: {missing}"
            raise ValueError(msg)
        summary = {name: _summarize(values) for name, values in scores.items()}
        self._storage.set_trial_user_attr(
            internal, "scores", {k: v[0] for k, v in summary.items()}
        )
        self._storage.set_trial_user_attr(
            internal, "standard_error", {k: v[1] for k, v in summary.items()}
        )
        self._storage.set_trial_user_attr(
            internal, "n_reps", {k: v[2] for k, v in summary.items()}
        )
        self._study.tell(
            trial_id, [summary[o.name][0] * o.weight for o in self.observables]
        )

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
                    "scores": t.user_attrs.get("scores", {}),
                    "standard_error": t.user_attrs.get("standard_error", {}),
                    "n_reps": t.user_attrs.get("n_reps", {}),
                    "constraints": self._constraint_values(t),
                }
                for t in done
            ],
        )
