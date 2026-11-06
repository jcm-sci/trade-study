"""Tests for the held-out lower bound on model information (#238)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trade_study import ModelInformation, model_information

LABELS = ["A", "B"]


def _gaussian_study(
    n_a: int, n_b: int, mu: float, temper: float = 1.0, seed: int = 0
) -> tuple[np.ndarray, list[str]]:
    """Simulate Y | A ~ N(-mu, 1) and Y | B ~ N(+mu, 1) with a tempered reader.

    The reader gives p(B | y) = sigmoid(temper * 2 * mu * y); temper 1 is exact.

    Returns:
        The posterior matrix and the generating labels.
    """
    rng = np.random.default_rng(seed)
    y = np.concatenate([rng.normal(-mu, 1.0, n_a), rng.normal(mu, 1.0, n_b)])
    p_b = 1.0 / (1.0 + np.exp(-temper * 2.0 * mu * y))
    return np.column_stack([1.0 - p_b, p_b]), ["A"] * n_a + ["B"] * n_b


def _exact_information(mu: float) -> float:
    """Exact I(M; Y) for equal priors, by quadrature of H(M) - E H(M | Y).

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


def test_flat_reader_resolves_nothing_and_certain_right_reader_everything() -> None:
    truth = ["A", "A", "B", "B"]
    flat = model_information([[0.5, 0.5]] * 4, truth, LABELS)
    assert flat.bound == pytest.approx(0.0, abs=1e-12)
    assert flat.standard_error == pytest.approx(0.0, abs=1e-12)
    sure = model_information(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], truth, LABELS
    )
    assert sure.bound == pytest.approx(math.log(2))
    assert sure.fraction_resolved == pytest.approx(1.0)
    assert sure.bound_bits == pytest.approx(1.0)
    assert sure.prior_entropy == pytest.approx(math.log(2))


def test_exact_posterior_recovers_the_mutual_information() -> None:
    mu = 0.8
    posterior, truth = _gaussian_study(100_000, 100_000, mu)
    info = model_information(posterior, truth, LABELS)
    exact = _exact_information(mu)
    assert (
        info.low <= exact <= info.high
        or abs(info.bound - exact) < 4 * info.standard_error
    )
    assert 0.0 < info.bound < math.log(2)
    assert info.n_floored == 0
    assert info.n == 200_000


def test_unbalanced_replicates_target_the_same_quantity() -> None:
    mu = 0.8
    posterior, truth = _gaussian_study(150_000, 50_000, mu, seed=1)
    info = model_information(posterior, truth, LABELS)
    assert abs(info.bound - _exact_information(mu)) < 4 * info.standard_error
    assert [t.n for t in info.by_truth] == [150_000, 50_000]


def test_miscalibrated_reader_gives_a_lower_bound() -> None:
    mu = 0.8
    exact = _exact_information(mu)
    for temper in (0.3, 3.0):
        posterior, truth = _gaussian_study(100_000, 100_000, mu, temper, seed=2)
        info = model_information(posterior, truth, LABELS)
        assert info.high < exact


def test_failed_rows_are_refused_or_counted_as_prior() -> None:
    posterior, truth = _gaussian_study(200, 200, 1.0, seed=3)
    holed = posterior.copy()
    holed[::7] = np.nan
    with pytest.raises(ValueError, match="missing"):
        model_information(holed, truth, LABELS)
    kept = model_information(holed, truth, LABELS, failed="prior")
    full = model_information(posterior, truth, LABELS)
    assert kept.n_failed == int(np.isnan(holed[:, 0]).sum())
    assert kept.bound < full.bound
    assert kept.n == full.n


def test_per_truth_scores_and_prior_weighting() -> None:
    truth = ["A", "A", "A", "B", "B", "B"]
    posterior = [[0.9, 0.1]] * 3 + [[0.4, 0.6]] * 3
    info = model_information(posterior, truth, LABELS, prior=[0.25, 0.75])
    a, b = info.by_truth
    assert a.mean_log_probability == pytest.approx(math.log(0.9))
    assert b.mean_probability == pytest.approx(0.6)
    h = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    expected = h + 0.25 * math.log(0.9) + 0.75 * math.log(0.6)
    assert info.bound == pytest.approx(expected)
    assert info.prior == (0.25, 0.75)


def test_floor_is_reported() -> None:
    truth = ["A", "A", "B", "B"]
    posterior = [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    info = model_information(posterior, truth, LABELS)
    assert info.n_floored == 1
    assert math.isfinite(info.bound)
    assert info.bound < 0


def test_interval_widens_with_confidence() -> None:
    posterior, truth = _gaussian_study(300, 300, 0.7, seed=4)
    narrow = model_information(posterior, truth, LABELS, confidence=0.8)
    wide = model_information(posterior, truth, LABELS, confidence=0.99)
    assert wide.high - wide.low > narrow.high - narrow.low
    assert narrow.low < narrow.bound < narrow.high
    assert isinstance(narrow, ModelInformation)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"posterior": [[0.5, 0.5]] * 3}, "shape"),
        ({"posterior": [[0.6, 0.6]] * 4}, "sum to 1"),
        ({"posterior": [[1.2, -0.2]] * 4}, r"\[0, 1\]"),
        ({"posterior": [[np.inf, 0.0]] * 4}, "infinite"),
        ({"truth": ["A", "A", "B", "C"]}, "not declared"),
        ({"truth": ["A", "A", "A", "B"]}, "at least two"),
        ({"prior": [0.5]}, "one probability"),
        ({"prior": [0.7, 0.7]}, "sum to 1"),
        ({"prior": [1.0, 0.0]}, "prior probability of zero"),
        ({"labels": ["A", "A"]}, "repeated"),
        ({"failed": "drop"}, "failed must be"),
        ({"confidence": 1.0}, "confidence"),
        ({"floor": 0.0}, "floor"),
        ({"truth": [], "posterior": []}, "at least one replicate"),
    ],
)
def test_invalid_inputs_are_refused(kwargs: dict[str, object], message: str) -> None:
    args: dict[str, object] = {
        "posterior": [[0.5, 0.5]] * 4,
        "truth": ["A", "A", "B", "B"],
        "labels": LABELS,
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        model_information(**args)  # type: ignore[arg-type]


def test_partially_missing_row_is_refused() -> None:
    posterior = [[0.5, 0.5], [0.5, 0.5], [np.nan, 0.5], [0.5, 0.5]]
    with pytest.raises(ValueError, match="complete or entirely missing"):
        model_information(posterior, ["A", "A", "B", "B"], LABELS, failed="prior")
