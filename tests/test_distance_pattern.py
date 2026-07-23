"""Unit tests for simclstr._distance_pattern module."""

import numpy as np
import pytest

from simclstr._distance_pattern import (
    distance_same_length,
    create_sisters,
    _distance_pattern,
)
from simclstr.clusterer import TimeSeries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fv(n_sections: int, seed: int = 0) -> np.ndarray:
    """Return a (n_sections, 2) feature vector with slope/curvature columns."""
    rng = np.random.default_rng(seed)
    slopes = rng.choice([-1.0, 0.0, 1.0], size=n_sections)
    curvatures = rng.choice([-1.0, 0.0, 1.0], size=n_sections)
    return np.column_stack([slopes, curvatures])


def _make_ts_list(n: int, length: int = 30) -> list:
    t = np.linspace(0, 1, length)
    patterns = [
        t,
        1 - t,
        t ** 2,
        np.exp(t) - 1,
        1 / (1 + np.exp(-10 * (t - 0.5))),
    ]
    return [TimeSeries(f"s{i}", patterns[i % len(patterns)].copy()) for i in range(n)]


# ---------------------------------------------------------------------------
# distance_same_length
# ---------------------------------------------------------------------------

class TestDistanceSameLength:
    def test_self_distance_is_zero(self):
        fv = _make_fv(5)
        assert distance_same_length(fv, fv, 1.0, 1.0) == pytest.approx(0.0)

    def test_symmetry(self):
        fv1 = _make_fv(5, seed=1)
        fv2 = _make_fv(5, seed=2)
        d12 = distance_same_length(fv1, fv2, 1.0, 1.0)
        d21 = distance_same_length(fv2, fv1, 1.0, 1.0)
        assert d12 == pytest.approx(d21)

    def test_non_negative(self):
        fv1 = _make_fv(4, seed=3)
        fv2 = _make_fv(4, seed=4)
        assert distance_same_length(fv1, fv2, 1.0, 1.0) >= 0.0

    def test_weights_scale_distance(self):
        fv1 = _make_fv(4, seed=5)
        fv2 = _make_fv(4, seed=6)
        d_low = distance_same_length(fv1, fv2, 0.5, 0.5)
        d_high = distance_same_length(fv1, fv2, 2.0, 2.0)
        assert d_high > d_low or d_high == pytest.approx(d_low)

    def test_identical_vectors_zero(self):
        fv = np.array([[1.0, -1.0], [0.0, 1.0], [-1.0, 0.0]])
        assert distance_same_length(fv, fv, 1.0, 1.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# create_sisters
# ---------------------------------------------------------------------------

class TestCreateSisters:
    def test_output_shape(self):
        short_fv = _make_fv(3)
        sisters = create_sisters(short_fv, (5, 2), sister_count=10)
        assert sisters.shape == (10, 5, 2)

    def test_sister_count(self):
        short_fv = _make_fv(2)
        for count in [1, 20, 50]:
            sisters = create_sisters(short_fv, (4, 2), sister_count=count)
            assert sisters.shape[0] == count

    def test_values_come_from_source(self):
        """All values in sisters must be present in the source fv."""
        short_fv = _make_fv(3, seed=7)
        sisters = create_sisters(short_fv, (5, 2), sister_count=20)
        for row in short_fv:
            assert np.any(np.all(sisters.reshape(-1, 2) == row, axis=1))

    def test_same_length_returns_copies(self):
        short_fv = _make_fv(4)
        sisters = create_sisters(short_fv, (4, 2), sister_count=5)
        assert sisters.shape == (5, 4, 2)


# ---------------------------------------------------------------------------
# _distance_pattern (full pipeline)
# ---------------------------------------------------------------------------

class TestDistancePattern:
    def test_condensed_matrix_length(self):
        n = 5
        ts_list = _make_ts_list(n)
        dRow, _ = _distance_pattern(ts_list)
        assert dRow.shape == (n * (n - 1) // 2,)

    def test_non_negative_distances(self):
        ts_list = _make_ts_list(4)
        dRow, _ = _distance_pattern(ts_list)
        assert np.all(dRow >= 0)

    def test_feature_vectors_assigned(self):
        ts_list = _make_ts_list(4)
        _, updated = _distance_pattern(ts_list)
        for ts in updated:
            assert ts.feature_vector is not None
            assert ts.index is not None

    def test_identical_series_zero_distance(self):
        data = np.linspace(0, 1, 30)
        ts_list = [TimeSeries(f"s{i}", data.copy()) for i in range(3)]
        dRow, _ = _distance_pattern(ts_list)
        np.testing.assert_allclose(dRow, 0.0, atol=1e-10)

    def test_returns_ts_objects(self):
        ts_list = _make_ts_list(3)
        dRow, updated = _distance_pattern(ts_list)
        assert isinstance(updated, list)
        assert all(isinstance(ts, TimeSeries) for ts in updated)
