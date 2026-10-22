"""Independent Gaussian design references and unavailable-outcome accounting."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import pytest

from trade_study import (
    Direction,
    FisherContext,
    FisherScreenPolicy,
    Observable,
    ResultsTable,
    fisher_screen_counts,
    load_result_bundle,
    run_grid,
    save_result_bundle,
    screen_fisher,
)

if TYPE_CHECKING:
    from pathlib import Path


def context(names: tuple[str, ...] = ("intercept", "slope")) -> FisherContext:
    return FisherContext(
        parameter_names=names,
        parameter_coordinates="physical coefficients; dimensionless",
        parameter_scales=(1.0,) * len(names),
        observation_model="independent Gaussian, fixed known variance",
        design="two independent observations at x=-1,+1",
        reference_regime="linear mean; all coefficient values",
        nuisance_treatment="none; coefficients both targeted",
        sensitivity_reference="analytic derivative of intercept + slope*x",
        numerical_reference="analytic mean, exact design, host float64",
        qualification_reference="independent Gaussian log-likelihood Hessian",
        information_reference="synthetic caller artifact",
    )


@pytest.mark.parametrize(
    ("variance", "expected_status"), [(0.25, "feasible"), (1.0, "infeasible")]
)
def test_gaussian_design_matches_independent_likelihood_hessian(
    variance: float, expected_status: str
):
    # -log L = ((y_minus-a+b)^2 + (y_plus-a-b)^2)/(2*variance)
    # gives Hessian diag(2/variance, 2/variance), independently of y.
    jacobian = np.array([[1.0, -1.0], [1.0, 1.0]])
    information = jacobian.T @ jacobian / variance
    policy = FisherScreenPolicy(4.0)
    result = screen_fisher(information, context=context(), policy=policy)
    assert result.status == expected_status
    np.testing.assert_allclose(result.eigenvalues, [2 / variance] * 2)
    assert policy.constraint().check(result.as_scores()["fisher_min_eigenvalue"]) == (
        expected_status == "feasible"
    )
    assert policy.constraint().confidence is None


def test_declared_identifiable_direction_survives_singular_nominal_matrix():
    # Three unit-variance observations of a+b identify that sum; a-b is a gauge.
    information = np.full((2, 2), 3.0)
    policy = FisherScreenPolicy(4.0)
    assert (
        screen_fisher(information, context=context(), policy=policy).status
        == "infeasible"
    )
    direction = ((1 / np.sqrt(2),), (1 / np.sqrt(2),))
    targeted = screen_fisher(
        information, context=context(), policy=replace(policy, directions=direction)
    )
    assert targeted.status == "feasible"
    np.testing.assert_allclose(targeted.eigenvalues, [6.0])
    gauge = replace(policy, directions=((1 / np.sqrt(2),), (-1 / np.sqrt(2),)))
    assert (
        screen_fisher(information, context=context(), policy=gauge).status
        == "infeasible"
    )


def test_projection_and_caller_profile_have_explicitly_different_meanings():
    information = np.array([[1.0, 1.0], [1.0, 2.0]])
    conditional = screen_fisher(
        information,
        context=replace(context(), nuisance_treatment="slope held fixed"),
        policy=FisherScreenPolicy(0.75, directions=((1.0,), (0.0,))),
    )
    # Analytically minimize the quadratic over slope: 1 - 1*(1/2)*1 = 1/2.
    profiled = screen_fisher(
        np.array([[0.5]]),
        context=replace(
            context(("intercept",)),
            nuisance_treatment="slope profiled by caller Schur complement",
        ),
        policy=FisherScreenPolicy(0.75),
    )
    assert conditional.status == "feasible"
    assert profiled.status == "infeasible"
    assert conditional.eigenvalues == (1.0,)
    assert profiled.eigenvalues == (0.5,)


def test_readonly_input_snapshot_and_declared_scales_are_preserved():
    information = np.diag([2.0, 8.0])
    information.flags.writeable = False
    ctx = replace(context(), parameter_scales=(100.0, 0.01))
    policy = FisherScreenPolicy(3.0)
    result = screen_fisher(information, context=ctx, policy=policy)
    assert result.eigenvalues == (2.0, 8.0)  # already in the declared coordinates
    digest = result.matrix_digest
    assert (
        screen_fisher(
            np.asfortranarray(information), context=ctx, policy=policy
        ).matrix_digest
        == digest
    )
    np.testing.assert_array_equal(information, np.diag([2.0, 8.0]))
    assert not information.flags.writeable
    metadata = result.to_metadata()
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
    metadata["eigenvalues"] = []
    assert result.eigenvalues == (2.0, 8.0)


@pytest.mark.parametrize(
    ("matrix", "reason"),
    [
        (None, "missing_information"),
        (np.ones((3, 3)), "unsupported_host_matrix_shape_dtype_or_limit"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "nonfinite_information"),
        (np.array([[1.0, np.inf], [np.inf, 1.0]]), "nonfinite_information"),
        (np.array([[1.0, 1.0], [0.0, 1.0]]), "asymmetric_information"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "indefinite_information"),
        (np.eye(2, dtype=complex), "unsupported_host_matrix_shape_dtype_or_limit"),
        (np.eye(2, dtype=bool), "unsupported_host_matrix_shape_dtype_or_limit"),
    ],
)
def test_unreadable_inputs_cannot_enter_generic_score_aggregation(
    matrix: Any, reason: str
):
    result = screen_fisher(matrix, context=context(), policy=FisherScreenPolicy(0.1))
    assert result.status == "unreadable"
    assert result.reason == reason
    with pytest.raises(ValueError, match="Only readable"):
        result.as_scores()
    json.dumps(result.to_metadata(), allow_nan=False)


def test_numerical_tolerances_are_recorded_without_softening_the_threshold():
    policy = FisherScreenPolicy(0.0, atol=1e-10, rtol=0.0)
    tiny_negative = screen_fisher(
        np.diag([-1e-12, 1.0]), context=context(), policy=policy
    )
    assert tiny_negative.status == "infeasible"
    assert tiny_negative.eigenvalues[0] == pytest.approx(
        -1e-12, abs=1e-20
    )  # retained, never clipped to pass
    invalid = screen_fisher(np.diag([-1e-8, 1.0]), context=context(), policy=policy)
    assert invalid.status == "unreadable"
    noisy = screen_fisher(
        np.array([[2.0, 1e-12], [0.0, 1.0]]), context=context(), policy=policy
    )
    assert noisy.status == "feasible"


def test_dimension_and_direction_limits_are_unreadable():
    too_large = screen_fisher(
        np.eye(2), context=context(), policy=FisherScreenPolicy(0.0, max_parameters=1)
    )
    assert too_large.status == "unreadable"
    wrong_basis = screen_fisher(
        np.eye(2),
        context=context(),
        policy=FisherScreenPolicy(0.0, directions=((1.0,),)),
    )
    assert wrong_basis.reason == "direction_coordinate_mismatch"


def test_failures_and_planned_denominators_remain_explicit(
    monkeypatch: pytest.MonkeyPatch,
):
    policy, ctx = FisherScreenPolicy(1.0), context()
    records = {
        "a": screen_fisher(np.eye(2), context=ctx, policy=policy),
        "b": screen_fisher(np.eye(2) / 2, context=ctx, policy=policy),
        "c": screen_fisher(None, context=ctx, policy=policy),
        "d": screen_fisher(
            None, context=ctx, policy=policy, failure="forward solve failed"
        ),
    }
    assert fisher_screen_counts(records, planned_ids=["a", "b", "c", "d", "e"]) == {
        "planned": 5,
        "feasible": 1,
        "infeasible": 1,
        "unreadable": 1,
        "failed": 1,
        "missing": 1,
    }

    def fail(_matrix: Any) -> NoReturn:
        raise np.linalg.LinAlgError

    monkeypatch.setattr(np.linalg, "eigvalsh", fail)
    result = screen_fisher(np.eye(2), context=ctx, policy=policy)
    assert result.status == "failed"
    assert result.reason == "eigensolver_failed"


@pytest.mark.parametrize("ids", [[], ["a", "a"], [""], "a", ["b"]])
def test_counts_reject_ambiguous_or_unplanned_identities(ids: Any):
    record = screen_fisher(None, context=context(), policy=FisherScreenPolicy(0.0))
    with pytest.raises(ValueError, match="planned-ID"):
        fisher_screen_counts({"a": record}, planned_ids=ids)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": -1.0},
        {"threshold": float("nan")},
        {"threshold": True},
        {"atol": -1.0},
        {"rtol": float("inf")},
        {"max_parameters": True},
        {"max_parameters": 1.5},
        {"max_parameters": 0},
        {"directions": ((0.0,), (0.0,))},
        {"directions": ((1.0, 0.0),)},
        {"directions": ((1j,),)},
        {"directions": ((2**63 - 1,),)},
    ],
)
def test_invalid_policy_fails_before_any_design_evaluation(kwargs: dict[str, Any]):
    values = {"threshold": 0.5, **kwargs}
    with pytest.raises(ValueError, match=r"Fisher policy|directions"):
        FisherScreenPolicy(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parameter_names": ("a", "a")},
        {"parameter_names": "ab"},
        {"parameter_scales": (0.0, 1.0)},
        {"parameter_scales": (1.0,)},
        {"nuisance_treatment": ""},
        {"qualification_reference": " "},
    ],
)
def test_missing_coordinate_or_qualification_context_is_rejected(
    kwargs: dict[str, Any],
):
    with pytest.raises(ValueError, match="context"):
        replace(context(), **kwargs)


class _GaussianWorld:
    def generate(self, config: dict[str, Any]) -> tuple[None, np.ndarray]:
        return None, np.diag([2.0, 2.0]) / config["variance"]


class _GaussianScorer:
    def score(
        self, truth: Any, observations: Any, config: dict[str, Any]
    ) -> dict[str, float]:
        return screen_fisher(
            observations, context=context(), policy=FisherScreenPolicy(4.0)
        ).as_scores()


def test_actual_grid_constraints_and_result_artifacts(tmp_path: Path):
    policy = FisherScreenPolicy(4.0)
    obs = [Observable("fisher_min_eigenvalue", Direction.MAXIMIZE)]
    table = run_grid(
        _GaussianWorld(),
        _GaussianScorer(),
        [{"variance": 0.25}, {"variance": 1.0}],
        obs,
    )
    np.testing.assert_array_equal(table.feasible([policy.constraint()]), [True, False])
    record = screen_fisher(np.eye(2) * 8, context=context(), policy=policy)
    table.metadata[0]["fisher_screen"] = record.to_metadata()
    table.metadata[1]["fisher_unavailable"] = screen_fisher(
        None, context=context(), policy=policy
    ).to_metadata()
    save_result_bundle(table, tmp_path / "fisher", observables=obs)
    restored = load_result_bundle(tmp_path / "fisher").results
    assert restored.metadata == table.metadata
    np.testing.assert_array_equal(restored.scores, table.scores)
    assert isinstance(restored, ResultsTable)
