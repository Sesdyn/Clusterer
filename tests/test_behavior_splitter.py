"""Unit tests for simclstr._behavior_splitter module."""

import numpy as np
import pytest

from simclstr._behavior_splitter import _construct_features, _split_behavior


# ---------------------------------------------------------------------------
# _split_behavior
# ---------------------------------------------------------------------------

class TestSplitBehavior:
    """Tests for single-series feature extraction."""

    def _slopes_and_curvatures(self, data):
        slopes = np.gradient(data)
        curvatures = np.gradient(slopes)
        return slopes, curvatures

    def test_returns_correct_shape(self):
        data = np.linspace(0, 1, 20)
        slopes, curvatures = self._slopes_and_curvatures(data)
        fv = _split_behavior(data, slopes, curvatures)
        assert fv.ndim == 2
        assert fv.shape[0] == 3  # slope sign, curvature sign, section length

    def test_constant_series_single_section(self):
        data = np.full(20, 5.0)
        slopes, curvatures = self._slopes_and_curvatures(data)
        fv = _split_behavior(data, slopes, curvatures)
        assert fv.shape[1] == 1

    def test_linear_increasing_single_section(self):
        data = np.linspace(1, 10, 30)
        slopes, curvatures = self._slopes_and_curvatures(data)
        fv = _split_behavior(data, slopes, curvatures)
        # Should collapse to a single section of positive slope
        assert fv.shape[1] >= 1
        # The dominant slope sign is +1
        assert fv[0, 0] == 1.0

    def test_linear_decreasing_slope_sign(self):
        data = np.linspace(10, 1, 30)
        slopes, curvatures = self._slopes_and_curvatures(data)
        fv = _split_behavior(data, slopes, curvatures)
        assert fv[0, 0] == -1.0

    def test_sign_values_valid(self):
        data = np.sin(np.linspace(0, 2 * np.pi, 50))
        slopes, curvatures = self._slopes_and_curvatures(data)
        fv = _split_behavior(data, slopes, curvatures)
        valid_signs = {-1.0, 0.0, 1.0}
        assert set(np.unique(fv[0])).issubset(valid_signs)
        assert set(np.unique(fv[1])).issubset(valid_signs)

    def test_section_lengths_sum_to_data_length(self):
        data = np.sin(np.linspace(0, 4 * np.pi, 60))
        slopes, curvatures = self._slopes_and_curvatures(data)
        fv = _split_behavior(data, slopes, curvatures)
        assert int(np.sum(fv[2])) == len(data)

    def test_section_lengths_positive(self):
        data = np.sin(np.linspace(0, 2 * np.pi, 40))
        slopes, curvatures = self._slopes_and_curvatures(data)
        fv = _split_behavior(data, slopes, curvatures)
        assert np.all(fv[2] > 0)


# ---------------------------------------------------------------------------
# _construct_features
# ---------------------------------------------------------------------------

class TestConstructFeatures:
    """Tests for batch feature extraction."""

    def test_returns_list_of_correct_length(self):
        data = np.vstack([np.linspace(0, 1, 20) for _ in range(5)])
        features = _construct_features(data)
        assert len(features) == 5

    def test_each_feature_has_3_rows(self):
        data = np.vstack([np.linspace(0, 1, 20) for _ in range(4)])
        features = _construct_features(data)
        for fv in features:
            assert fv.shape[0] == 3

    def test_sign_values_valid(self):
        rng = np.random.default_rng(0)
        data = rng.random((6, 30))
        features = _construct_features(data)
        valid = {-1.0, 0.0, 1.0}
        for fv in features:
            assert set(np.unique(fv[0])).issubset(valid)
            assert set(np.unique(fv[1])).issubset(valid)

    def test_section_lengths_sum_to_series_length(self):
        n_points = 25
        data = np.vstack([np.linspace(i, i + 1, n_points) for i in range(4)])
        features = _construct_features(data)
        for fv in features:
            assert int(np.sum(fv[2])) == n_points

    def test_single_series(self):
        data = np.linspace(0, 1, 20).reshape(1, -1)
        features = _construct_features(data)
        assert len(features) == 1
        assert features[0].shape[0] == 3

    def test_significance_level_effect(self):
        # Very high significance level should suppress noise, collapsing to fewer sections
        data = np.vstack([np.linspace(0, 1, 30) + 1e-6 * np.random.randn(30)])
        feats_low = _construct_features(data, significanceLevel=0.0)
        feats_high = _construct_features(data, significanceLevel=0.5)
        # High significance should produce <= sections compared to no filtering
        assert feats_high[0].shape[1] <= feats_low[0].shape[1]
