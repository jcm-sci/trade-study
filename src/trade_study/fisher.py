"""Bounded host-side feasibility screening of caller-qualified information."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal

import numpy as np

from .protocols import Constraint

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

ScreenStatus = Literal["feasible", "infeasible", "unreadable", "failed"]


def _label(value: str) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonnegative(value: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and isfinite(value)
        and value >= 0
    )


@dataclass(frozen=True)
class FisherContext:
    """Caller-owned meaning and qualification of an information matrix.

    All references are nonempty caller identifiers whose underlying evidence
    must be retained. They are declarations, not automatically verified hashes.
    The supplied matrix is already in ``parameter_coordinates`` with the
    recorded ``parameter_scales``; this adapter applies no further scaling.

    Attributes:
        parameter_names: Ordered matrix coordinates, distinct and nonempty.
        parameter_coordinates: Coordinate/transform definition and units.
        parameter_scales: Positive finite scales already used by the caller.
        observation_model: Measurement law, noise and observation definitions.
        design: Experimental design identity.
        reference_regime: Parameter point or declared sampled regime.
        nuisance_treatment: Explicit conditioning/profiling convention.
        sensitivity_reference: Measured-output sensitivity source.
        numerical_reference: Solver, derivative and numerical settings.
        qualification_reference: Independent qualification and its scope.
        information_reference: Identity of the caller's information artifact.
    """

    parameter_names: tuple[str, ...]
    parameter_coordinates: str
    parameter_scales: tuple[float, ...]
    observation_model: str
    design: str
    reference_regime: str
    nuisance_treatment: str
    sensitivity_reference: str
    numerical_reference: str
    qualification_reference: str
    information_reference: str

    def __post_init__(self) -> None:
        """Detach coordinate sequences and require explicit context.

        Raises:
            ValueError: If names, scales or context references are invalid.
        """
        names, scales = tuple(self.parameter_names), tuple(self.parameter_scales)
        references = (
            self.parameter_coordinates,
            self.observation_model,
            self.design,
            self.reference_regime,
            self.nuisance_treatment,
            self.sensitivity_reference,
            self.numerical_reference,
            self.qualification_reference,
            self.information_reference,
        )
        valid_names = (
            not isinstance(self.parameter_names, str)
            and bool(names)
            and all(_label(name) for name in names)
            and len(set(names)) == len(names)
        )
        valid_scales = len(scales) == len(names) and all(
            _nonnegative(scale) and scale > 0 for scale in scales
        )
        if (
            not valid_names
            or not valid_scales
            or not all(_label(r) for r in references)
        ):
            msg = (
                "Fisher context requires distinct names, positive scales and references"
            )
            raise ValueError(msg)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "parameter_scales", tuple(float(s) for s in scales))

    def to_metadata(self) -> dict[str, object]:
        """Return detached primitive JSON context.

        Returns:
            Explicit coordinates and caller-owned evidence references.
        """
        return {
            "parameter_names": list(self.parameter_names),
            "parameter_coordinates": self.parameter_coordinates,
            "parameter_scales": list(self.parameter_scales),
            "observation_model": self.observation_model,
            "design": self.design,
            "reference_regime": self.reference_regime,
            "nuisance_treatment": self.nuisance_treatment,
            "sensitivity_reference": self.sensitivity_reference,
            "numerical_reference": self.numerical_reference,
            "qualification_reference": self.qualification_reference,
            "information_reference": self.information_reference,
        }


@dataclass(frozen=True)
class FisherScreenPolicy:
    """Prespecified local minimum-eigenvalue screen.

    Attributes:
        threshold: Required minimum eigenvalue, finite and nonnegative.
        directions: Optional row-major p-by-k orthonormal direction basis in
            the matrix coordinates. The tested matrix is B.T @ F @ B.
            Its complement is fixed; this operation does not profile nuisance
            parameters. None tests all directions.
        atol: Absolute symmetry and positive-semidefiniteness tolerance.
        rtol: Relative symmetry and positive-semidefiniteness tolerance.
        max_parameters: Dense matrix dimension limit, default 128. This is
            not a whole-process memory bound.
    """

    threshold: float
    directions: tuple[tuple[float, ...], ...] | None = None
    atol: float = 1e-10
    rtol: float = 1e-8
    max_parameters: int = 128

    def __post_init__(self) -> None:
        """Validate the policy before evaluating any design.

        Raises:
            ValueError: If bounds or the orthonormal basis are invalid.
        """
        if (
            not all(_nonnegative(v) for v in (self.threshold, self.atol, self.rtol))
            or isinstance(self.max_parameters, bool)
            or not isinstance(self.max_parameters, Integral)
            or self.max_parameters < 1
        ):
            msg = (
                "Fisher policy requires nonnegative finite bounds and a positive limit"
            )
            raise ValueError(msg)
        if self.directions is not None:
            basis = np.asarray(self.directions)
            valid_shape = (
                basis.ndim == 2
                and 0 < basis.shape[1] <= basis.shape[0] <= self.max_parameters
                and basis.dtype.kind in "iuf"
            )
            if not valid_shape:
                msg = "directions must be a finite real p-by-k orthonormal basis"
                raise ValueError(msg)
            # Validate norms in float64, avoiding integer Gram-matrix overflow.
            with np.errstate(over="ignore", invalid="ignore"):
                basis = np.asarray(basis, dtype=np.float64)
                gram = basis.T @ basis
            if (
                not np.isfinite(basis).all()
                or not np.isfinite(gram).all()
                or not np.allclose(gram, np.eye(basis.shape[1]), rtol=1e-10, atol=1e-10)
            ):
                msg = "directions must be a finite real p-by-k orthonormal basis"
                raise ValueError(msg)
            object.__setattr__(
                self, "directions", tuple(tuple(float(v) for v in row) for row in basis)
            )

    def constraint(self, observable: str = "fisher_min_eigenvalue") -> Constraint:
        """Build the existing deterministic feasibility constraint.

        Args:
            observable: Scored column name.

        Returns:
            A >= constraint without an invented confidence level.

        Raises:
            ValueError: If the observable name is empty.
        """
        if not _label(observable):
            msg = "observable must be nonempty"
            raise ValueError(msg)
        return Constraint("local_information", observable, ">=", float(self.threshold))

    def to_metadata(self) -> dict[str, object]:
        """Return detached primitive JSON policy values.

        Returns:
            Threshold, basis, numerical tolerances and dense dimension limit.
        """
        return {
            "threshold": float(self.threshold),
            "directions": None
            if self.directions is None
            else [list(row) for row in self.directions],
            "atol": float(self.atol),
            "rtol": float(self.rtol),
            "max_parameters": int(self.max_parameters),
        }


@dataclass(frozen=True)
class FisherScreenResult:
    """A local screen outcome retaining its context and numerical evidence.

    Attributes:
        status: Feasible, infeasible, unreadable or failed.
        reason: Machine-readable outcome explanation or upstream failure text.
        eigenvalues: Actual tested eigenvalues; empty for unavailable results.
        matrix_digest: SHA256 of the supplied host float64 little-endian matrix
            bytes in C order, before symmetrization; None for missing inputs.
        context: Caller-owned scientific and numerical declarations.
        policy: The prespecified screen.
    """

    status: ScreenStatus
    reason: str
    eigenvalues: tuple[float, ...]
    matrix_digest: str | None
    context: FisherContext
    policy: FisherScreenPolicy

    def as_scores(self, observable: str = "fisher_min_eigenvalue") -> dict[str, float]:
        """Expose a readable result to an existing scorer/constraint.

        Args:
            observable: Scored column name.

        Returns:
            The actual minimum tested eigenvalue in a single scored column.

        Raises:
            ValueError: If the result is unavailable or the name is empty.
        """
        if self.status not in {"feasible", "infeasible"} or not _label(observable):
            msg = (
                "Only readable Fisher results with a nonempty observable can be scored"
            )
            raise ValueError(msg)
        return {observable: min(self.eigenvalues)}

    def to_metadata(self) -> dict[str, object]:
        """Return a result-bundle-compatible record without nonfinite sentinels.

        Returns:
            Versioned primitive JSON with actual values and preserved context.
        """
        return {
            "schema_version": 1,
            "scope": "local_directional_information_screen",
            "status": self.status,
            "reason": self.reason,
            "eigenvalues": list(self.eigenvalues),
            "matrix_digest": self.matrix_digest,
            "context": self.context.to_metadata(),
            "policy": self.policy.to_metadata(),
        }


def _prepare_information(
    information: NDArray[np.float64], context: FisherContext, policy: FisherScreenPolicy
) -> tuple[NDArray[np.float64] | None, str, str | None, float]:
    """Validate and snapshot a bounded host matrix before spectral arithmetic.

    Returns:
        Symmetrized matrix or None, reason, input digest and absolute tolerance.
    """
    digest = None
    p = len(context.parameter_names)
    if (
        not isinstance(information, np.ndarray)
        or information.ndim != 2
        or information.shape != (p, p)
        or information.dtype.kind not in "iuf"
        or p > policy.max_parameters
    ):
        return None, "unsupported_host_matrix_shape_dtype_or_limit", digest, 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = np.array(information, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(matrix).all():
        return None, "nonfinite_information", digest, 0.0
    digest = sha256(matrix.astype("<f8", copy=False).tobytes()).hexdigest()
    scale = float(np.max(np.abs(matrix)))
    tolerance = float(policy.atol + policy.rtol * scale)
    if not isfinite(tolerance):
        return None, "nonfinite_numerical_tolerance", digest, 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        symmetric = bool(np.all(np.abs(matrix - matrix.T) <= tolerance))
    if not symmetric:
        return None, "asymmetric_information", digest, tolerance
    return matrix / 2 + matrix.T / 2, "readable", digest, tolerance


def screen_fisher(  # ruff: ignore[too-many-return-statements]
    information: NDArray[np.float64] | None,
    *,
    context: FisherContext,
    policy: FisherScreenPolicy,
    failure: str | None = None,
) -> FisherScreenResult:
    """Screen a caller-qualified host information matrix in declared directions.

    No derivatives, Fisher model, scaling, nuisance profile, statistical
    uncertainty or global identifiability conclusion is constructed here.
    The caller explicitly transfers native backend artifacts to host NumPy.

    Args:
        information: Real host NumPy p-by-p matrix, or None when unavailable.
        context: Ordered coordinates and retained qualification references.
        policy: Prespecified threshold, directions and numerical limits.
        failure: Upstream failure reason, supplied only with information=None.

    Returns:
        A readable local result, an unreadable invalid/unsupported input, or
        a failed upstream/eigensolver record. No input matrix is retained.

    Raises:
        ValueError: If failure is empty or supplied alongside a matrix.
    """

    def result(
        status: ScreenStatus, reason: str, values: tuple[float, ...] = ()
    ) -> FisherScreenResult:
        return FisherScreenResult(status, reason, values, digest, context, policy)

    digest = None
    if failure is not None:
        if information is not None or not _label(failure):
            msg = "An upstream failure needs a reason and no information matrix"
            raise ValueError(msg)
        return result("failed", failure)
    if information is None:
        return result("unreadable", "missing_information")
    matrix, reason, digest, tolerance = _prepare_information(
        information, context, policy
    )
    if matrix is None:
        return result("unreadable", reason)
    p = len(context.parameter_names)
    basis = None if policy.directions is None else np.asarray(policy.directions)
    if basis is not None and basis.shape[0] != p:
        return result("unreadable", "direction_coordinate_mismatch")
    try:
        full_values = np.linalg.eigvalsh(matrix)
        with np.errstate(over="ignore", invalid="ignore"):
            values = (
                full_values
                if basis is None
                else np.linalg.eigvalsh(basis.T @ matrix @ basis)
            )
    except np.linalg.LinAlgError:
        return result("failed", "eigensolver_failed")
    if not np.isfinite(full_values).all() or not np.isfinite(values).all():
        return result("unreadable", "nonfinite_spectrum")
    if float(full_values[0]) < -tolerance:
        return result("unreadable", "indefinite_information")
    status: ScreenStatus = (
        "feasible" if float(values[0]) >= policy.threshold else "infeasible"
    )
    return result(status, "threshold_comparison", tuple(float(v) for v in values))


def fisher_screen_counts(
    records: Mapping[str, FisherScreenResult], *, planned_ids: Sequence[str]
) -> dict[str, int]:
    """Count outcomes against a fixed declared cohort without dropping failures.

    Args:
        records: At most one outcome per planned identifier.
        planned_ids: Complete, distinct, nonempty planned replicate/task IDs.

    Returns:
        Planned, feasible, infeasible, unreadable, failed and missing counts.
        These counts alone establish no probability, interval or gate verdict.

    Raises:
        ValueError: If identities or result statuses are invalid.
    """
    ids = tuple(planned_ids)
    if (
        isinstance(planned_ids, str)
        or not ids
        or not all(_label(identity) for identity in ids)
        or len(set(ids)) != len(ids)
        or set(records) - set(ids)
    ):
        msg = "Screen counts require a complete distinct planned-ID cohort"
        raise ValueError(msg)
    counts = dict.fromkeys(("feasible", "infeasible", "unreadable", "failed"), 0)
    for record in records.values():
        if not isinstance(record, FisherScreenResult) or record.status not in counts:
            msg = "Screen counts require valid FisherScreenResult statuses"
            raise ValueError(msg)
        counts[record.status] += 1
    return {"planned": len(ids), **counts, "missing": len(ids) - len(records)}
