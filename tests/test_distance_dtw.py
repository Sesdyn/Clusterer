"""Unit tests for simclstr._distance_dtw module."""

import numpy as np
import pytest

from simclstr._distance_dtw import _distance_dtw, compute_dtw_distances
from simclstr.clusterer import TimeSeries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts_list(arrays: list) -> list:
    return [TimeSeries(f"s{i}", arr.astype(float)) for i, arr in enumerate(arrays)]


# ---------------------------------------------------------------------------
# compute_dtw_distances (numba kernel)
# ---------------------------------------------------------------------------

class TestComputeDtwDistances:
    def test_identical_series_zero(self):
        data = np.tile(np.linspace(0, 1, 10), (3, 1))
        dRow = np.zeros(3)
        dRow = compute_dtw_distances(data, dRow)
        np.testing.assert_allclose(dRow, 0.0)

    def test_condensed_length(self):
        n = 5
        data = np.random.rand(n, 10)
        dRow = np.zeros(n * (n - 1) // 2)
        dRow = compute_dtw_distances(data, dRow)
        assert dRow.shape == (n * (n - 1) // 2,)

    def test_non_negative(self):
        data = np.random.rand(4, 12)
        dRow = np.zeros(4 * 3 // 2)
        dRow = compute_dtw_distances(data, dRow)
        assert np.all(dRow >= 0)


# ---------------------------------------------------------------------------
# _distance_dtw (full pipeline)
# ---------------------------------------------------------------------------

class TestDistanceDtw:
    def test_condensed_matrix_length(self):
        n = 5
        ts_list = _make_ts_list([np.linspace(i, i + 1, 20) for i in range(n)])
        dRow, _ = _distance_dtw(ts_list)
        assert dRow.shape == (n * (n - 1) // 2,)

    def test_identical_series_zero_distance(self):
        data = np.sin(np.linspace(0, np.pi, 20))
        ts_list = _make_ts_list([data.copy() for _ in range(4)])
        dRow, _ = _distance_dtw(ts_list)
        np.testing.assert_allclose(dRow, 0.0)

    def test_offset_series_positive_distance(self):
        t = np.linspace(0, 1, 20)
        ts_list = _make_ts_list([t, t + 5.0])
        dRow, _ = _distance_dtw(ts_list)
        assert dRow[0] > 0

    def test_symmetry(self):
        t = np.linspace(0, 1, 20)
        ts_ab = _make_ts_list([t, t ** 2])
        ts_ba = _make_ts_list([t ** 2, t])
        dRow_ab, _ = _distance_dtw(ts_ab)
        dRow_ba, _ = _distance_dtw(ts_ba)
        assert dRow_ab[0] == pytest.approx(dRow_ba[0])

    def test_feature_vectors_set_to_data(self):
        data = np.linspace(0, 1, 15)
        ts_list = _make_ts_list([data.copy(), data[::-1].copy()])
        _, updated = _distance_dtw(ts_list)
        np.testing.assert_array_equal(updated[0].feature_vector, data)

    def test_index_assigned(self):
        ts_list = _make_ts_list([np.ones(10), np.zeros(10)])
        _, updated = _distance_dtw(ts_list)
        assert updated[0].index == 0
        assert updated[1].index == 1

    def test_non_negative_distances(self):
        ts_list = _make_ts_list([np.random.rand(15) for _ in range(5)])
        dRow, _ = _distance_dtw(ts_list)
        assert np.all(dRow >= 0)
