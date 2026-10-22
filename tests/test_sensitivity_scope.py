"""Discrete-factor scope, conditional interactions and held-out-group evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trade_study import Factor, FactorType, ResultsTable, sensitivity_from_table


def _factorial() -> tuple[ResultsTable, list[Factor]]:
    """Build independent analytic q-by-log-prior and continuous-control effects.

    Returns:
        Full factorial table with source-aligned metadata and annotations.
    """
    configs = [
        {"q": q, "prior": float(prior), "z": float(z), "fixed": 0.5}
        for q in (2, 4)
        for prior in np.geomspace(1, 100, 9)
        for z in np.linspace(0, 1, 9)
    ]
    table = ResultsTable(
        configs,
        np.array([
            [(config["q"] - 2) * np.log10(config["prior"]) / 2 + config["z"]]
            for config in configs
        ]),
        ["loss"],
        annotations=np.arange(len(configs), dtype=float).reshape(-1, 1),
        annotation_names=["source_index"],
        metadata=[{"source_index": i} for i in range(len(configs))],
    )
    return table, [
        Factor("q", FactorType.DISCRETE, levels=[2, 4]),
        Factor("prior", FactorType.CONTINUOUS, bounds=(1, 100), log_scale=True),
        Factor("z", FactorType.CONTINUOUS, bounds=(0, 1)),
        Factor("fixed", FactorType.CONTINUOUS, bounds=(0, 1)),
    ]


@pytest.mark.parametrize("response", ["q-only", "interaction", "confounded"])
def test_pooled_discrete_effects_are_refused_before_surrogate_fitting(
    response: str,
) -> None:
    table, factors = _factorial()
    if response == "q-only":
        table.scores[:, 0] = [config["q"] for config in table.configs]
    elif response == "confounded":
        table.configs = [
            config
            for config in table.configs
            if (config["q"] == 2) == (config["prior"] <= 10)
        ]
        table.scores = np.array([[config["q"]] for config in table.configs])
        table.annotations, table.metadata = None, []
    with pytest.raises(ValueError, match=r"Varying.*q.*condition_on"):
        sensitivity_from_table(table, factors)


def test_conditional_surfaces_bind_all_constants_and_retain_source_rows() -> None:
    table, factors = _factorial()
    fitted = sensitivity_from_table(
        table,
        factors,
        condition_on={"q": 4},
        n_estimators=60,
        n_trajectories=256,
        seed=7,
        cv_group_by="design",
        cv_folds=3,
    )
    assert fitted.screened_factors == ("prior", "z")
    assert fitted.fixed_factors == {"q": 4, "fixed": 0.5}
    assert fitted.conditioning == {"q": 4}
    assert fitted.omitted_config_keys == ()
    assert fitted.source_rows == tuple(range(81, 162))
    assert fitted.cv_group_by == ("q", "prior", "z", "fixed")
    assert fitted.validation_scheme == "held-out-designs"
    assert fitted.cv_folds_requested == 3
    assert len(set(fitted.validation_groups)) == 81
    assert fitted.sampling_factors[0].log_scale
    assert fitted.sampling_factors[0].bounds == (1, 100)
    assert fitted.importance["loss"][0] == pytest.approx(0.8, abs=0.12)
    assert fitted.importance["loss"][1] == pytest.approx(0.2, abs=0.12)
    assert fitted.surrogate_cv_r2["loss"] > 0.8
    assert fitted.surrogate_cv_rmse["loss"] >= 0
    assert table.configs[0]["q"] == 2
    assert table.metadata[81] == {"source_index": 81}


def test_q_interaction_changes_the_conditional_surface_not_a_discrete_index() -> None:
    table, factors = _factorial()
    fitted = sensitivity_from_table(
        table,
        factors,
        condition_on={"q": 2},
        n_estimators=50,
        n_trajectories=256,
        seed=7,
    )
    assert fitted.screened_factors == ("prior", "z")
    assert fitted.importance["loss"][0] < 0.05
    assert fitted.importance["loss"][1] > 0.9
    assert "q" not in fitted.screened_factors


def test_conditioning_does_not_screen_an_unobserved_continuous_dimension() -> None:
    table = ResultsTable(
        [{"q": 2, "prior": 1.0}, {"q": 2, "prior": 1.0}, {"q": 4, "prior": 100.0}],
        np.array([[2], [2], [4]], dtype=float),
        ["loss"],
    )
    factors = [
        Factor("q", FactorType.DISCRETE, levels=[2, 4]),
        Factor("prior", FactorType.CONTINUOUS, bounds=(1, 100), log_scale=True),
    ]
    with pytest.raises(ValueError, match="observed variation"):
        sensitivity_from_table(table, factors, condition_on={"q": 2})


def test_grouped_cv_exposes_design_leakage_and_unavailable_groups() -> None:
    table = ResultsTable(
        [{"x": float(i)} for i in range(6) for _ in range(12)],
        np.column_stack((
            np.repeat([-3, 8, -5, 1, 9, -8], 12),
            [1.0] * 12 + [np.nan] * 60,
        )),
        ["loss", "single_group"],
    )
    fitted = sensitivity_from_table(
        table,
        [Factor("x", FactorType.CONTINUOUS, bounds=(0, 5))],
        n_estimators=30,
        n_trajectories=32,
        seed=4,
        cv_group_by="design",
        cv_folds=3,
        warn_below_r2=None,
    )
    assert fitted.surrogate_row_cv_r2["loss"] > 0.95
    assert fitted.surrogate_cv_r2["loss"] < fitted.surrogate_row_cv_r2["loss"] - 0.5
    assert fitted.surrogate_cv_rmse["loss"] > fitted.surrogate_row_cv_rmse["loss"]
    assert fitted.validation_groups == [i for i in range(6) for _ in range(12)]
    assert np.isnan(fitted.surrogate_cv_r2["single_group"])
    assert np.isnan(fitted.surrogate_cv_rmse["single_group"])
    assert fitted.source_rows == tuple(range(72))


@pytest.mark.parametrize("keys", [[], ["seed"]])
def test_unmodeled_varying_keys_require_explicit_bookkeeping(keys: list[str]) -> None:
    table, factors = _factorial()
    for i, config in enumerate(table.configs):
        config["seed"] = i
    if not keys:
        with pytest.raises(ValueError, match=r"unmodeled.*seed"):
            sensitivity_from_table(table, factors, condition_on={"q": 2})
    else:
        fitted = sensitivity_from_table(
            table,
            factors,
            condition_on={"q": 2},
            metadata_keys=keys,
            n_estimators=10,
            n_trajectories=16,
        )
        assert fitted.omitted_config_keys == ("seed",)
        assert fitted.metadata_keys == ("seed",)
        assert fitted.screened_factors == ("prior", "z")


@pytest.mark.parametrize(
    "case",
    [
        "missing-factor",
        "missing-key",
        "unknown-condition",
        "continuous-condition",
        "unknown-level",
        "empty-stratum",
        "duplicate-factor",
        "metadata-overlap",
        "metadata-unknown",
        "metadata-duplicate",
    ],
)
def test_scope_declarations_cannot_conceal_an_input(case: str) -> None:
    table, factors = _factorial()
    options: dict[str, Any] = {"condition_on": {"q": 2}}
    if case == "missing-factor":
        del table.configs[0]["fixed"]
    elif case == "missing-key":
        table.configs[0]["extra"] = 1
    elif case == "unknown-condition":
        options["condition_on"] = {"absent": 1}
    elif case == "continuous-condition":
        options["condition_on"] = {"prior": 1}
    elif case == "unknown-level":
        options["condition_on"] = {"q": 3}
    elif case == "empty-stratum":
        factors[0] = Factor("q", FactorType.DISCRETE, levels=[2, 4, 6])
        options["condition_on"] = {"q": 6}
    elif case == "duplicate-factor":
        factors.append(factors[0])
    elif case == "metadata-overlap":
        options["metadata_keys"] = ["q"]
    elif case == "metadata-unknown":
        options["metadata_keys"] = ["absent"]
    else:
        table.configs[0]["extra"] = 1
        options["metadata_keys"] = ["extra", "extra"]
    with pytest.raises(ValueError, match=r"factor|key|Condition|condition|observed"):
        sensitivity_from_table(table, factors, **options)


def test_declared_factor_grouping_is_distinct_from_metadata_replicate_grouping() -> (
    None
):
    table, factors = _factorial()
    with pytest.raises(ValueError, match="known factor names"):
        sensitivity_from_table(
            table,
            factors,
            condition_on={"q": 2},
            cv_group_by=["source_index"],
            n_estimators=10,
        )
    fitted = sensitivity_from_table(
        table,
        factors,
        condition_on={"q": 2},
        cv_group_by=["prior"],
        n_estimators=10,
        n_trajectories=16,
        warn_below_r2=None,
    )
    assert fitted.validation_scheme == "held-out-factor-groups"
    assert fitted.cv_group_by == ("prior",)
    assert len(set(fitted.validation_groups)) == 9


def test_discrete_only_inputs_get_a_direct_contrast_diagnostic() -> None:
    with pytest.raises(ValueError, match="direct factorial or paired contrasts"):
        sensitivity_from_table(
            ResultsTable([{"q": 2}] * 2, np.ones((2, 1)), ["loss"]),
            [Factor("q", FactorType.DISCRETE, levels=[2, 4])],
        )


def test_conditional_target_extrapolation_is_reported_once_per_observable() -> None:
    table = ResultsTable(
        [{"x": x} for x in (0.25, 0.5, 0.75)],
        np.array([[0.25], [0.5], [0.75]]),
        ["loss"],
    )
    with pytest.warns(UserWarning, match="outside observed marginal support") as caught:
        fitted = sensitivity_from_table(
            table,
            [Factor("x", FactorType.CONTINUOUS, bounds=(0, 1))],
            n_estimators=10,
            n_trajectories=16,
            warn_below_r2=None,
        )
    assert len(caught) == 1
    assert fitted.support_ranges == {"loss": {"x": (0.25, 0.75)}}


def test_gp_conditioning_uses_the_same_scope_and_validation_contract() -> None:
    table = ResultsTable(
        [{"q": q, "x": x} for q in (2, 4) for x in (0.0, 0.25, 0.5, 0.75, 1.0)],
        np.array([[q * x] for q in (2, 4) for x in (0.0, 0.25, 0.5, 0.75, 1.0)]),
        ["loss"],
    )
    fitted = sensitivity_from_table(
        table,
        [
            Factor("q", FactorType.DISCRETE, levels=[2, 4]),
            Factor("x", FactorType.CONTINUOUS, bounds=(0, 1)),
        ],
        condition_on={"q": 4.0},
        surrogate_method="gp",
        cv_group_by="design",
        cv_folds=2,
        n_trajectories=32,
        warn_below_r2=None,
    )
    assert fitted.fixed_factors == {"q": 4}
    assert fitted.source_rows == tuple(range(5, 10))
    assert fitted.importance["loss"][0] == pytest.approx(1, abs=0.12)
    assert fitted.validation_groups == list(range(5))


def test_numerically_constant_continuous_inputs_are_fixed_across_python_types() -> None:
    table = ResultsTable(
        [{"x": x} for x in (1, 1.0, np.float64(1))], np.ones((3, 1)), ["loss"]
    )
    with pytest.raises(ValueError, match="observed variation"):
        sensitivity_from_table(
            table, [Factor("x", FactorType.CONTINUOUS, bounds=(0, 2))]
        )


def test_complete_conditional_and_paired_contrast_guide() -> None:
    path = Path(__file__).resolve().parents[1] / "docs/guide/table-sensitivity.md"
    namespace: dict[str, Any] = {}
    for block in re.findall(r"```python\n(.*?)\n```", path.read_text(), re.DOTALL):
        exec(block, namespace)  # ruff: ignore[exec-builtin] - trusted complete guide
    np.testing.assert_allclose(list(namespace["effects"].values()), [0.0, 1.0, 2.0])
