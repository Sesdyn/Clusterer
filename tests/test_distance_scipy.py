"""Unit tests for simclstr._distance_scipy module."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from simclstr._distance_scipy import _distance_scipy
from simclstr.clusterer import TimeSeries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts_list(arrays: list) -> list:
    return [TimeSeries(f"s{i}", arr.astype(float)) for i, arr in enumerate(arrays)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDistanceScipy:
    def test_condensed_matrix_length(self):
        n = 5
        ts_list = _make_ts_list([np.linspace(i, i + 1, 20) for i in range(n)])
        dRow, _ = _distance_scipy(ts_list, metric="euclidean")
        assert dRow.shape == (n * (n - 1) // 2,)

    def test_euclidean_matches_scipy_pdist(self):
        arrays = [np.linspace(i, i + 1, 15) for i in range(4)]
        ts_list = _make_ts_list(arrays)
        dRow, _ = _distance_scipy(ts_list, metric="euclidean")
        expected = pdist(np.array(arrays), metric="euclidean")
        np.testing.assert_allclose(dRow, expected)

    def test_cosine_distance(self):
        arrays = [np.linspace(1, 2, 10), np.linspace(2, 4, 10)]
        ts_list = _make_ts_list(arrays)
        dRow, _ = _distance_scipy(ts_list, metric="cosine")
        assert dRow.shape == (1,)
        assert dRow[0] >= 0

    def test_identical_series_zero_euclidean(self):
        data = np.arange(10, dtype=float)
        ts_list = _make_ts_list([data.copy() for _ in range(3)])
        dRow, _ = _distance_scipy(ts_list, metric="euclidean")
        np.testing.assert_allclose(dRow, 0.0)

    def test_feature_vectors_set_to_data(self):
        data = np.linspace(0, 1, 10)
        ts_list = _make_ts_list([data.copy(), data[::-1].copy()])
        _, updated = _distance_scipy(ts_list, metric="euclidean")
        np.testing.assert_array_equal(updated[0].feature_vector, data)

    def test_index_assigned(self):
        ts_list = _make_ts_list([np.ones(8), np.zeros(8)])
        _, updated = _distance_scipy(ts_list, metric="euclidean")
        assert updated[0].index == 0
        assert updated[1].index == 1

    def test_invalid_metric_raises(self):
        ts_list = _make_ts_list([np.ones(5), np.zeros(5)])
        with pytest.raises(ValueError):
            _distance_scipy(ts_list, metric="not_a_real_metric")

    def test_cityblock_metric(self):
        arrays = [np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])]
        ts_list = _make_ts_list(arrays)
        dRow, _ = _distance_scipy(ts_list, metric="cityblock")
        expected = pdist(np.array(arrays), metric="cityblock")
        np.testing.assert_allclose(dRow, expected)
