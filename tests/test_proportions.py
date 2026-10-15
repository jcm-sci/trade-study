"""Independent numerical and raw-indicator contracts for fixed-sample intervals."""

from __future__ import annotations

import builtins
from dataclasses import replace
from math import exp, log, sqrt
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy.special import betaincc  # type: ignore[import-untyped]
from scipy.stats import binom, binomtest  # type: ignore[import-untyped]

from trade_study import (
    Direction,
    Observable,
    ResultsTable,
    load_result_bundle,
    proportion_interval,
    replicates_for_half_width,
    save_result_bundle,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("method", ["wilson", "clopper-pearson"])
@pytest.mark.parametrize("confidence", [0.5, 0.9, 0.95, 0.999])
def test_matches_independent_binomial_test_intervals(
    method: str, confidence: float
) -> None:
    for n in [1, 2, 7, 30, 200, 10000]:
        for k in sorted({0, 1, n // 2, n - 1, n}):
            actual = proportion_interval(k, n, method=method, confidence=confidence)
            reference = binomtest(k, n).proportion_ci(
                confidence_level=confidence,
                method="exact" if method == "clopper-pearson" else method,
            )
            assert (actual.low, actual.high) == pytest.approx(
                (reference.low, reference.high), abs=2e-11
            )
            assert actual.estimate == k / n
            assert actual.k == k
            assert actual.n == n
            assert actual.method == method
            assert actual.confidence == confidence


def test_exact_interval_matches_published_nist_example() -> None:
    interval = proportion_interval(4, 20, confidence=0.9, method="clopper-pearson")
    assert interval.low == pytest.approx(0.071354, abs=5e-7)
    assert interval.high == pytest.approx(0.401029, abs=1e-6)


@pytest.mark.parametrize("n", [1, 30, 200, 10000])
def test_exact_boundary_cases_match_zero_event_probability(n: int) -> None:
    zero = proportion_interval(0, n, method="clopper-pearson")
    all_success = proportion_interval(n, n, method="clopper-pearson")
    assert zero.low == 0
    assert zero.high == pytest.approx(1 - exp(log(0.025) / n))
    assert all_success.low == pytest.approx(exp(log(0.025) / n))
    assert all_success.high == 1


@pytest.mark.parametrize("n", [1, 2, 5, 20, 50])
def test_exact_coverage_by_deterministic_binomial_enumeration(n: int) -> None:
    intervals = [
        proportion_interval(k, n, method="clopper-pearson") for k in range(n + 1)
    ]
    for probability in np.linspace(0, 1, 101):
        included = [ci.low <= probability <= ci.high for ci in intervals]
        coverage = binom.pmf(np.arange(n + 1), n, probability)[included].sum()
        assert coverage >= 0.95 - 1e-12


def test_extreme_confidence_preserves_small_tail() -> None:
    confidence = float(np.nextafter(1.0, 0.0))
    interval = proportion_interval(
        1, 100, confidence=confidence, method="clopper-pearson"
    )
    assert interval.high < 1
    assert betaincc(2, 99, interval.high) == pytest.approx(
        (1 - confidence) / 2, rel=1e-10, abs=0
    )
    wilson = proportion_interval(1, 100, confidence=confidence)
    assert 0 < wilson.low < wilson.estimate < wilson.high < 1


def test_large_integer_counts_are_supported_without_truncation() -> None:
    interval = proportion_interval(np.int64(10), np.int64(30))
    assert type(interval.k) is int
    assert type(interval.n) is int
    for k in [0, 1, 2**52, 2**53 - 1, 2**53]:
        interval = proportion_interval(k, 2**53)
        assert 0 <= interval.low <= interval.estimate <= interval.high <= 1


@pytest.mark.parametrize(
    ("k", "n"),
    [
        (-1, 2),
        (3, 2),
        (0, 0),
        (0, -1),
        (True, 2),
        (1, True),
        (np.bool_(1), 2),
        (1.0, 2),
        (1, 2.0),
        (None, 2),
        ("1", 2),
        (0, 2**53 + 1),
    ],
)
def test_invalid_counts_fail_without_truncation(k: Any, n: Any) -> None:
    with pytest.raises(ValueError, match=r"count|integer"):
        proportion_interval(k, n)


@pytest.mark.parametrize("confidence", [0, 1, -0.1, np.nan, np.inf, True, "0.95"])
def test_invalid_confidence_is_rejected(confidence: Any) -> None:
    with pytest.raises(ValueError, match="confidence"):
        proportion_interval(1, 2, confidence=confidence)
    with pytest.raises(ValueError, match="confidence"):
        replicates_for_half_width(0.1, confidence=confidence)


def test_confidence_too_close_to_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="floating-point boundary"):
        proportion_interval(1, 2, confidence=float(np.nextafter(0.0, 1.0)))


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="method"):
        proportion_interval(1, 2, method="wald")


@pytest.mark.parametrize(
    "half_width", [0, -0.1, 0.51, np.nan, np.inf, True, "0.1", 1e-300]
)
def test_invalid_or_unsupported_precision_is_rejected(half_width: Any) -> None:
    with pytest.raises(ValueError, match="half_width"):
        replicates_for_half_width(half_width)


@pytest.mark.parametrize("confidence", [0.5, 0.95, 0.999])
@pytest.mark.parametrize("half_width", [0.5, 0.17, 0.07, 0.05, 0.01])
def test_sizing_bounds_all_attainable_wilson_widths(
    confidence: float, half_width: float
) -> None:
    n = replicates_for_half_width(half_width, confidence=confidence)
    # Only the nearest outcomes to n/2 can maximize width; also check boundaries.
    for k in {0, n // 2, (n + 1) // 2, n}:
        ci = proportion_interval(k, n, confidence=confidence)
        assert (ci.high - ci.low) / 2 <= half_width + 1e-15
    if n > 1:
        # For even n-1 the continuous worst case is attainable.
        previous = n - 1 if (n - 1) % 2 == 0 else n - 2
        if previous > 0:
            ci = proportion_interval(previous // 2, previous, confidence=confidence)
            assert (ci.high - ci.low) / 2 > half_width - 1e-15


def test_sizing_rounding_at_an_exact_interval_width() -> None:
    ci = proportion_interval(50, 100)
    assert replicates_for_half_width((ci.high - ci.low) / 2) == 100
    assert replicates_for_half_width(0.05) == 381


@pytest.mark.parametrize(
    "confidence", [0.5, 0.95, 0.999, float(np.nextafter(1.0, 0.0))]
)
def test_sizing_at_supported_count_limit_stays_in_range(confidence: float) -> None:
    z = -NormalDist().inv_cdf((1 - confidence) / 2)
    target = z / (2 * sqrt(2**53 + z * z))
    count = replicates_for_half_width(target, confidence=confidence)
    assert 1 <= count <= 2**53
    assert z / (2 * sqrt(count + z * z)) <= target


def test_wilson_needs_no_optional_statistics_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def reject_scipy(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "scipy" or name.startswith("scipy."):
            msg = "Optional statistics dependency disabled"
            raise ImportError(msg)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_scipy)
    assert proportion_interval(1, 2).estimate == pytest.approx(0.5)
    assert replicates_for_half_width(0.05) == 381
    with pytest.raises(ImportError, match=r"trade-study\[stats\]"):
        proportion_interval(1, 2, method="clopper-pearson")


def _table() -> ResultsTable:
    return ResultsTable(
        configs=[{"design": "b"}] * 2 + [{"design": "a"}] * 3,
        scores=np.array([[1, 10], [1, 20], [0, 30], [1, 40], [0, 50]], dtype=float),
        observable_names=["recovered", "cost"],
        annotations=np.array([[2], [2], [1], [1], [1]], dtype=float),
        annotation_names=["size"],
        metadata=[
            {"design_point": dp, "rep": rep}
            for dp, rep in [(1, 0), (1, 1), (0, 0), (0, 1), (0, 2)]
        ],
    )


def test_opt_in_summary_retains_counts_and_exports_intervals(tmp_path: Path) -> None:
    raw = _table()
    summary = raw.aggregate_replicates(proportion_observables=["recovered"])
    assert summary.configs == [{"design": "a"}, {"design": "b"}]
    np.testing.assert_array_equal(summary.scores[:, 1], [40, 15])
    np.testing.assert_array_equal(summary.annotations, [[1], [2]])
    for meta, k, n in zip(summary.metadata, [1, 2], [3, 2], strict=True):
        stored = meta["proportion_intervals"]["recovered"]
        assert stored["k"] == k
        assert stored["n"] == n
        assert stored["estimate"] == k / n
        assert stored["scope"] == "fixed-sample-marginal"
    frame = summary.to_dataframe()
    assert frame["meta.proportion_intervals.recovered.n"].tolist() == [3, 2]
    assert (
        frame["meta.proportion_intervals.recovered.method"].tolist() == ["wilson"] * 2
    )
    # Interval metadata is primitive JSON and survives the versioned artifact API.
    save_result_bundle(
        summary,
        tmp_path / "bundle",
        observables=[
            Observable("recovered", Direction.MAXIMIZE),
            Observable("cost", Direction.MINIMIZE),
        ],
    )
    assert load_result_bundle(tmp_path / "bundle").results.metadata == summary.metadata
    assert "proportion_intervals" not in raw.metadata[0]
    assert "proportion_intervals" not in raw.aggregate_replicates().metadata[0]


def test_exact_summary_records_requested_method_and_level() -> None:
    summary = _table().aggregate_replicates(
        proportion_observables=["recovered"],
        proportion_method="clopper-pearson",
        confidence=0.9,
    )
    interval = summary.metadata[0]["proportion_intervals"]["recovered"]
    reference = proportion_interval(1, 3, method="clopper-pearson", confidence=0.9)
    assert interval["low"] == reference.low
    assert interval["high"] == reference.high
    assert interval["method"] == "clopper-pearson"
    assert interval["confidence"] == pytest.approx(0.9)


@pytest.mark.parametrize("value", [0.5, -1, 2, np.nan, np.inf])
def test_indicator_summary_rejects_invalid_outcomes_without_dropping(
    value: float,
) -> None:
    raw = _table()
    raw.scores[0, 0] = value
    with pytest.raises(ValueError, match="0/1 outcomes"):
        raw.aggregate_replicates(proportion_observables=["recovered"])


@pytest.mark.parametrize("rep", [None, -1, True, 0.5, 0])
def test_indicator_summary_rejects_missing_invalid_or_duplicate_rep_ids(
    rep: Any,
) -> None:
    raw = _table()
    raw.metadata[1]["rep"] = rep
    with pytest.raises(ValueError, match=r"integer count|unique replicate"):
        raw.aggregate_replicates(proportion_observables=["recovered"])


@pytest.mark.parametrize("names", [[], ["recovered", "recovered"], ["unknown"]])
def test_indicator_summary_rejects_ambiguous_or_unknown_names(names: list[str]) -> None:
    with pytest.raises(ValueError, match="proportion_observables"):
        _table().aggregate_replicates(proportion_observables=names)


def test_indicator_summary_rejects_incomplete_parallel_rows() -> None:
    raw = _table()
    for broken in [
        replace(raw, metadata=raw.metadata[:-1]),
        replace(raw, scores=raw.scores[:-1]),
    ]:
        with pytest.raises(ValueError, match=r"matching.*rows"):
            broken.aggregate_replicates(proportion_observables=["recovered"])


def test_indicator_summary_rejects_duplicate_observable_columns() -> None:
    raw = replace(_table(), observable_names=["recovered", "recovered"])
    with pytest.raises(ValueError, match="unique observable"):
        raw.aggregate_replicates(proportion_observables=["recovered"])


def test_zero_trial_indicator_summary_has_no_interval() -> None:
    raw = ResultsTable(
        configs=[], scores=np.empty((0, 1)), observable_names=["recovered"]
    )
    with pytest.raises(ValueError, match="nonempty"):
        raw.aggregate_replicates(proportion_observables=["recovered"])
