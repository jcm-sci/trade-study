"""Tests for information curves and minimum adequate sizes (#240)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trade_study import (
    InformationCurve,
    ResultsTable,
    information_curve,
    minimum_adequate_information,
    model_information,
)

LABELS = ["A", "B"]
MU = 0.4


def _exact_information(mu: float) -> float:
    """Exact I(M; Y) of Y | A ~ N(-mu, 1), Y | B ~ N(+mu, 1) with equal priors.

    Returns:
        The mutual information in nats.
    """
    y = np.linspace(-12.0 - mu, 12.0 + mu, 400_001)
    density = (
        0.5
        * (np.exp(-0.5 * (y + mu) ** 2) + np.exp(-0.5 * (y - mu) ** 2))
        / math.sqrt(2 * math.pi)
    )
    p_b = 1.0 / (1.0 + np.exp(-2.0 * mu * y))
    entropy = -(
        p_b * np.log(np.clip(p_b, 1e-300, 1))
        + (1 - p_b) * np.log(np.clip(1 - p_b, 1e-300, 1))
    )
    f = density * entropy
    return math.log(2) - float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(y)))


def _study(
    sizes: tuple[int, ...], replicates: int, seed: int = 0, extra: object = None
) -> ResultsTable:
    """Cohorts of n iid N(+-mu, 1) draws; the exact reader sees their sum.

    The sufficient statistic is N(+-n mu, n), so I(M; Y) at size n equals the
    one-observation information at mu * sqrt(n).

    Returns:
        Replicate rows with factors model, n (and an optional site) and the
        observable log_score = log q(truth | y).
    """
    rng = np.random.default_rng(seed)
    configs: list[dict[str, object]] = []
    scores: list[list[float]] = []
    for n in sizes:
        for model, sign in (("A", -1.0), ("B", 1.0)):
            total = rng.normal(sign * n * MU, math.sqrt(n), replicates)
            p_b = 1.0 / (1.0 + np.exp(-2.0 * MU * total))
            p_true = p_b if model == "B" else 1.0 - p_b
            for value in np.log(np.clip(p_true, 1e-300, 1.0)):
                config: dict[str, object] = {"model": model, "n": n}
                if extra is not None:
                    config["site"] = extra
                configs.append(config)
                scores.append([float(value)])
    return ResultsTable(configs, np.array(scores), ["log_score"])


def test_curve_matches_the_exact_information_at_every_size() -> None:
    sizes = (1, 4, 16, 64)
    (curve,) = information_curve(
        _study(sizes, 40_000), "log_score", "model", "n", LABELS
    )
    assert isinstance(curve, InformationCurve)
    assert curve.values == sizes
    for point in curve.points:
        exact = _exact_information(MU * math.sqrt(point.value))
        assert abs(point.bound - exact) < 4 * point.standard_error
    bounds = [p.bound for p in curve.points]
    assert bounds == sorted(bounds)
    assert curve.prior_entropy == pytest.approx(math.log(2))


def test_a_point_equals_model_information_for_the_same_replicates() -> None:
    table = _study((8,), 500, seed=1)
    (curve,) = information_curve(table, "log_score", "model", "n", LABELS)
    scores = np.asarray(table.scores)[:, 0]
    truth = [c["model"] for c in table.configs]
    p_true = np.exp(scores)
    posterior = np.array([
        [p, 1 - p] if t == "A" else [1 - p, p]
        for p, t in zip(p_true, truth, strict=True)
    ])
    info = model_information(posterior, truth, LABELS)
    assert curve.points[0].bound == pytest.approx(info.bound)
    assert curve.points[0].standard_error == pytest.approx(info.standard_error)


def test_minimum_adequate_size_agrees_with_the_exact_curve() -> None:
    sizes = (1, 2, 4, 8, 16, 32, 64)
    (curve,) = information_curve(
        _study(sizes, 40_000, seed=2), "log_score", "model", "n", LABELS
    )
    target = 0.2
    exact_first = next(
        n for n in sizes if _exact_information(MU * math.sqrt(n)) >= target
    )
    by_estimate = minimum_adequate_information([curve], target, rule="estimate")[0]
    by_bound = minimum_adequate_information([curve], target)[0]
    assert by_estimate.value == exact_first
    assert by_bound.value >= by_estimate.value
    assert by_bound.point is not None
    assert by_bound.point.low >= target
    assert by_bound.monotone_above


def test_fraction_target_and_unreachable_target() -> None:
    sizes = (1, 4, 16)
    (curve,) = information_curve(
        _study(sizes, 5_000, seed=3), "log_score", "model", "n", LABELS
    )
    fraction = minimum_adequate_information([curve], 0.5, fraction=True)[0]
    nats = minimum_adequate_information([curve], 0.5 * math.log(2))[0]
    assert fraction.value == nats.value
    assert fraction.fraction
    none = minimum_adequate_information([curve], 0.69)[0]
    assert none.value is None
    assert none.point is None
    assert none.monotone_above is False


def test_strata_are_separate_curves_and_bonferroni_widens() -> None:
    a = _study((4, 16), 3_000, seed=4, extra="s1")
    b = _study((4, 16), 3_000, seed=5, extra="s2")
    both = ResultsTable(
        a.configs + b.configs,
        np.vstack([a.scores, b.scores]),
        a.observable_names,
    )
    curves = information_curve(both, "log_score", "model", "n", LABELS)
    assert [c.stratum for c in curves] == [{"site": "s1"}, {"site": "s2"}]
    wide = information_curve(
        both, "log_score", "model", "n", LABELS, adjustment="bonferroni"
    )
    assert wide[0].confidence == pytest.approx(1 - 0.05 / 2)
    assert wide[0].family_confidence == pytest.approx(0.95)
    p, q = curves[0].points[0], wide[0].points[0]
    assert q.high - q.low > p.high - p.low


def test_non_monotone_curve_is_reported() -> None:
    rows = []
    for n, value in ((1, 0.0), (2, math.log(0.5)), (3, 0.0)):
        for model in LABELS:
            rows += [({"model": model, "n": n}, value)] * 3
    table = ResultsTable(
        [c for c, _ in rows], np.array([[v] for _, v in rows]), ["log_score"]
    )
    (curve,) = information_curve(table, "log_score", "model", "n", LABELS)
    chosen = minimum_adequate_information([curve], 0.5)[0]
    assert chosen.value == 1
    assert chosen.monotone_above is False


def _tiny(
    scores: list[float], models: list[str], extra: dict[str, object] | None = None
) -> ResultsTable:
    configs = [{"model": m, "n": 1, **(extra or {})} for m in models]
    return ResultsTable(configs, np.array([[s] for s in scores]), ["log_score"])


@pytest.mark.parametrize(
    ("table", "kwargs", "message"),
    [
        (_tiny([float("nan")] * 4, list("AABB")), {}, "non-finite"),
        (_tiny([0.1] * 4, list("AABB")), {}, "must not exceed 0"),
        (_tiny([0.0] * 3, list("AABB")[:3]), {}, "at least two"),
        (
            _tiny([0.0] * 4, list("AABB")),
            {"log_score": "missing"},
            "not in the results",
        ),
        (_tiny([0.0] * 4, list("AABB")), {"truth": "missing"}, "absent from"),
        (_tiny([0.0] * 4, list("AABB")), {"along": "missing"}, "absent from"),
        (_tiny([0.0] * 4, list("AABB")), {"adjustment": "holm"}, "adjustment"),
        (_tiny([0.0] * 4, list("AABC")), {}, "not declared"),
        (_tiny([0.0] * 4, list("AABB")), {"prior": [0.9, 0.9]}, "sum to 1"),
        (_tiny([0.0] * 4, list("AABB")), {"confidence": 0.0}, "confidence"),
    ],
)
def test_invalid_inputs_are_refused(
    table: ResultsTable, kwargs: dict[str, object], message: str
) -> None:
    args: dict[str, object] = {
        "log_score": "log_score",
        "truth": "model",
        "along": "n",
        "labels": LABELS,
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        information_curve(table, **args)  # type: ignore[arg-type]


def test_unorderable_sizes_and_bad_minimum_arguments_are_refused() -> None:
    rows = [("A", 1), ("A", "x"), ("B", 1), ("B", "x")] * 2
    table = ResultsTable(
        [{"model": m, "n": n} for m, n in rows],
        np.zeros((len(rows), 1)),
        ["log_score"],
    )
    with pytest.raises(ValueError, match="orderable"):
        information_curve(table, "log_score", "model", "n", LABELS)
    good = information_curve(
        _tiny([0.0] * 4, list("AABB")), "log_score", "model", "n", LABELS
    )
    for kwargs in (
        {"rule": "median"},
        {"target": -1.0},
        {"target": 2.0, "fraction": True},
    ):
        args: dict[str, object] = {"curves": good, "target": 0.1}
        args.update(kwargs)
        with pytest.raises(ValueError, match=r"rule|target"):
            minimum_adequate_information(**args)  # type: ignore[arg-type]
