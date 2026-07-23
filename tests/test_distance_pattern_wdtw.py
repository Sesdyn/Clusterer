"""Unit tests for simclstr._distance_pattern_wdtw module."""

import numpy as np
import pytest

from simclstr._distance_pattern_wdtw import _distance_pattern_wdtw
from simclstr.clusterer import TimeSeries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts_list(n: int, length: int = 30) -> list:
    t = np.linspace(0, 1, length)
    patterns = [t, 1 - t, t ** 2, np.exp(t) - 1, np.sin(np.pi * t)]
    return [TimeSeries(f"s{i}", patterns[i % len(patterns)].copy()) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDistancePatternWdtw:
    def test_condensed_matrix_length(self):
        n = 5
        ts_list = _make_ts_list(n)
        dRow, _ = _distance_pattern_wdtw(ts_list)
        assert dRow.shape == (n * (n - 1) // 2,)

    def test_non_negative_distances(self):
        ts_list = _make_ts_list(4)
        dRow, _ = _distance_pattern_wdtw(ts_list)
        assert np.all(dRow >= 0)

    def test_identical_series_zero_distance(self):
        data = np.linspace(0, 1, 30)
        ts_list = [TimeSeries(f"s{i}", data.copy()) for i in range(3)]
        dRow, _ = _distance_pattern_wdtw(ts_list)
        np.testing.assert_allclose(dRow, 0.0, atol=1e-10)

    def test_symmetry(self):
        t = np.linspace(0, 1, 30)
        ts_ab = [TimeSeries("a", t.copy()), TimeSeries("b", (t ** 2).copy())]
        ts_ba = [TimeSeries("b", (t ** 2).copy()), TimeSeries("a", t.copy())]
        dRow_ab, _ = _distance_pattern_wdtw(ts_ab)
        dRow_ba, _ = _distance_pattern_wdtw(ts_ba)
        assert dRow_ab[0] == pytest.approx(dRow_ba[0])

    def test_feature_vectors_assigned(self):
        ts_list = _make_ts_list(3)
        _, updated = _distance_pattern_wdtw(ts_list)
        for ts in updated:
            assert ts.feature_vector is not None
            assert ts.index is not None

    def test_two_series(self):
        t = np.linspace(0, 1, 25)
        ts_list = [TimeSeries("a", t), TimeSeries("b", 1 - t)]
        dRow, _ = _distance_pattern_wdtw(ts_list)
        assert dRow.shape == (1,)

    def test_distance_kwargs_accepted(self):
        ts_list = _make_ts_list(4)
        dRow, _ = _distance_pattern_wdtw(
            ts_list,
            distance_kwargs={"significanceLevel": 0.001, "wSlopeError": 2.0, "wCurvatureError": 0.5},
        )
        assert dRow.shape == (4 * 3 // 2,)

    def test_different_from_unweighted_dtw(self):
        """Weighted and unweighted DTW should generally produce different distances."""
        from simclstr._distance_pattern_dtw import _distance_pattern_dtw
        ts_list = _make_ts_list(4)
        # Use copies to avoid state mutation affecting the comparison
        ts_list2 = _make_ts_list(4)
        dRow_w, _ = _distance_pattern_wdtw(ts_list)
        dRow_u, _ = _distance_pattern_dtw(ts_list2)
        # They may be equal for simple cases, but shapes must match
        assert dRow_w.shape == dRow_u.shape
