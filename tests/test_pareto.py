"""Tests for pareto module (issues #11, #12, #13, #14)."""

from __future__ import annotations

import numpy as np
import pytest

from trade_study.pareto import extract_front, hypervolume, igd_plus, pareto_rank
from trade_study.protocols import Direction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_scores() -> np.ndarray:
    """Four points in 2-D objective space (both minimise).

    Point layout (obj1, obj2):
        A(1,4)  B(2,2)  C(4,1)  D(3,3)
    Front = {A, B, C}; D is dominated by B.

    Returns:
        Score matrix of shape (4, 2).
    """
    return np.array([
        [1.0, 4.0],  # A
        [2.0, 2.0],  # B
        [4.0, 1.0],  # C
        [3.0, 3.0],  # D — dominated by B
    ])


@pytest.fixture
def min_directions() -> list[Direction]:
    """Two minimize directions.

    Returns:
        List of two MINIMIZE directions.
    """
    return [Direction.MINIMIZE, Direction.MINIMIZE]


# ---------------------------------------------------------------------------
# extract_front (#11)
# ---------------------------------------------------------------------------


def test_extract_front_indices(
    simple_scores: np.ndarray,
    min_directions: list[Direction],
) -> None:
    front = extract_front(simple_scores, min_directions)
    assert set(front) == {0, 1, 2}


def test_extract_front_excludes_dominated(
    simple_scores: np.ndarray,
    min_directions: list[Direction],
) -> None:
    front = extract_front(simple_scores, min_directions)
    assert 3 not in front


def test_extract_front_returns_intp(
    simple_scores: np.ndarray,
    min_directions: list[Direction],
) -> None:
    front = extract_front(simple_scores, min_directions)
    assert front.dtype == np.intp


def test_extract_front_maximize() -> None:
    """Maximize both objectives: highest values are non-dominated."""
    scores = np.array([
        [1.0, 1.0],  # dominated
        [5.0, 3.0],  # front
        [3.0, 5.0],  # front
        [4.0, 4.0],  # dominated by convex combination, but not by single point
    ])
    dirs = [Direction.MAXIMIZE, Direction.MAXIMIZE]
    front = extract_front(scores, dirs)
    assert 1 in front
    assert 2 in front
    assert 0 not in front


def test_extract_front_mixed_directions() -> None:
    """Minimize obj1, maximize obj2."""
    scores = np.array([
        [1.0, 5.0],  # low cost, high quality — front
        [2.0, 3.0],  # dominated by A
        [1.0, 3.0],  # dominated by A
    ])
    dirs = [Direction.MINIMIZE, Direction.MAXIMIZE]
    front = extract_front(scores, dirs)
    assert set(front) == {0}


def test_extract_front_single_point() -> None:
    scores = np.array([[2.0, 3.0]])
    front = extract_front(scores, [Direction.MINIMIZE, Direction.MINIMIZE])
    assert set(front) == {0}


def test_extract_front_all_identical() -> None:
    scores = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    front = extract_front(scores, [Direction.MINIMIZE, Direction.MINIMIZE])
    # All equivalent → all on front
    assert set(front) == {0, 1, 2}


# ---------------------------------------------------------------------------
# pareto_rank (#12)
# ---------------------------------------------------------------------------


def test_pareto_rank_front_is_zero(
    simple_scores: np.ndarray,
    min_directions: list[Direction],
) -> None:
    ranks = pareto_rank(simple_scores, min_directions)
    assert ranks[0] == 0  # A
    assert ranks[1] == 0  # B
    assert ranks[2] == 0  # C


def test_pareto_rank_dominated_is_one(
    simple_scores: np.ndarray,
    min_directions: list[Direction],
) -> None:
    ranks = pareto_rank(simple_scores, min_directions)
    assert ranks[3] == 1  # D


def test_pareto_rank_shape(
    simple_scores: np.ndarray,
    min_directions: list[Direction],
) -> None:
    ranks = pareto_rank(simple_scores, min_directions)
    assert ranks.shape == (4,)
    assert ranks.dtype == np.intp


def test_pareto_rank_multiple_layers() -> None:
    """Three layers: front, second, third."""
    scores = np.array([
        [1.0, 3.0],  # front
        [3.0, 1.0],  # front
        [2.0, 2.5],  # rank 1 — dominated by (1,3)
        [3.0, 3.0],  # rank 1 — dominated by (1,3) and (3,1)
        [4.0, 4.0],  # rank 2 — dominated by rank-1 points
    ])
    dirs = [Direction.MINIMIZE, Direction.MINIMIZE]
    ranks = pareto_rank(scores, dirs)
    assert ranks[0] == 0
    assert ranks[1] == 0
    assert ranks[4] > ranks[2]  # deepest layer


def test_pareto_rank_maximize() -> None:
    scores = np.array([
        [5.0, 5.0],  # front
        [1.0, 1.0],  # worst
    ])
    dirs = [Direction.MAXIMIZE, Direction.MAXIMIZE]
    ranks = pareto_rank(scores, dirs)
    assert ranks[0] == 0
    assert ranks[1] == 1


# ---------------------------------------------------------------------------
# hypervolume (#13)
# ---------------------------------------------------------------------------


def test_hypervolume_positive(
    simple_scores: np.ndarray,
    min_directions: list[Direction],
) -> None:
    front_idx = extract_front(simple_scores, min_directions)
    front = simple_scores[front_idx]
    ref = np.array([5.0, 5.0])
    hv = hypervolume(front, ref, min_directions)
    assert hv > 0.0


def test_hypervolume_known_value() -> None:
    """Single front point (1,1) with ref (2,2) → area = 1.0."""
    front = np.array([[1.0, 1.0]])
    ref = np.array([2.0, 2.0])
    hv = hypervolume(front, ref, [Direction.MINIMIZE, Direction.MINIMIZE])
    assert hv == pytest.approx(1.0)


def test_hypervolume_two_points() -> None:
    """Two front points forming an L-shape.

    Front: (1,3), (3,1) with ref (4,4).
    Verifies HV with two points exceeds HV with a single point.
    """
    front = np.array([[1.0, 3.0], [3.0, 1.0]])
    ref = np.array([4.0, 4.0])
    hv = hypervolume(front, ref, [Direction.MINIMIZE, Direction.MINIMIZE])
    single_hv = hypervolume(
        np.array([[1.0, 3.0]]),
        ref,
        [Direction.MINIMIZE, Direction.MINIMIZE],
    )
    assert hv > single_hv


def test_hypervolume_maximize() -> None:
    """Maximize: front point (5,5) with ref (0,0) → area = 25."""
    front = np.array([[5.0, 5.0]])
    ref = np.array([0.0, 0.0])
    hv = hypervolume(front, ref, [Direction.MAXIMIZE, Direction.MAXIMIZE])
    assert hv == pytest.approx(25.0)


def test_hypervolume_no_directions() -> None:
    """Without directions arg, assumes raw minimisation."""
    front = np.array([[1.0, 1.0]])
    ref = np.array([2.0, 2.0])
    hv = hypervolume(front, ref)
    assert hv == pytest.approx(1.0)


def test_hypervolume_returns_float() -> None:
    front = np.array([[1.0, 1.0]])
    ref = np.array([3.0, 3.0])
    hv = hypervolume(front, ref)
    assert isinstance(hv, float)


# ---------------------------------------------------------------------------
# igd_plus (#14)
# ---------------------------------------------------------------------------


def test_igd_plus_identical_fronts() -> None:
    """IGD+ of a front against itself should be zero."""
    front = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    val = igd_plus(front, front, [Direction.MINIMIZE, Direction.MINIMIZE])
    assert val == pytest.approx(0.0)


def test_igd_plus_worse_front_positive() -> None:
    """A worse front should have positive IGD+ against the true front."""
    true_front = np.array([[1.0, 3.0], [3.0, 1.0]])
    worse_front = np.array([[2.0, 4.0], [4.0, 2.0]])
    val = igd_plus(worse_front, true_front, [Direction.MINIMIZE, Direction.MINIMIZE])
    assert val > 0.0


def test_igd_plus_returns_float() -> None:
    front = np.array([[1.0, 1.0]])
    ref = np.array([[0.0, 0.0]])
    val = igd_plus(front, ref, [Direction.MINIMIZE, Direction.MINIMIZE])
    assert isinstance(val, float)


def test_igd_plus_no_directions() -> None:
    """Without directions, assumes raw minimisation."""
    front = np.array([[1.0, 1.0]])
    val = igd_plus(front, front)
    assert val == pytest.approx(0.0)


def test_igd_plus_maximize() -> None:
    """Maximize: identical fronts still yield IGD+ = 0."""
    front = np.array([[5.0, 5.0], [3.0, 7.0]])
    val = igd_plus(front, front, [Direction.MAXIMIZE, Direction.MAXIMIZE])
    assert val == pytest.approx(0.0)
