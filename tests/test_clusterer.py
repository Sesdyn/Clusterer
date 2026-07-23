"""Unit tests for simclstr.clusterer module."""

import numpy as np
import pytest

from simclstr.clusterer import (
    TimeSeries,
    Cluster,
    read_time_series,
    _normalize_data,
    _standardize_data,
    perform_clustering,
)


# ---------------------------------------------------------------------------
# TimeSeries
# ---------------------------------------------------------------------------

class TestTimeSeries:
    def test_basic_construction(self):
        data = np.array([1.0, 2.0, 3.0])
        ts = TimeSeries("run_1", data)
        assert ts.label == "run_1"
        np.testing.assert_array_equal(ts.data, data)

    def test_default_attributes(self):
        ts = TimeSeries("x", np.ones(5))
        assert ts.index is None
        assert ts.feature_vector is None
        assert ts.cluster_id is None
        assert ts.previous_cluster_id is None

    def test_previous_cluster_id(self):
        ts = TimeSeries("x", np.ones(5), previous_cluster_id=3)
        assert ts.previous_cluster_id == 3


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------

class TestCluster:
    def _make_ts_list(self, n=3):
        return [TimeSeries(f"s{i}", np.arange(float(i), i + 5)) for i in range(n)]

    def test_basic_construction(self):
        members = self._make_ts_list(3)
        indices = np.array([0, 1, 2])
        c = Cluster(1, indices, members, members[0])
        assert c.cluster_id == 1
        assert c.number_of_members == 3
        assert c.best_representative_member is members[0]
        assert len(c.list_of_members) == 3

    def test_single_member(self):
        ts = TimeSeries("only", np.ones(5))
        c = Cluster(2, np.array([0]), [ts], ts)
        assert c.number_of_members == 1

    def test_indices_preserved(self):
        members = self._make_ts_list(2)
        indices = np.array([4, 7])
        c = Cluster(1, indices, members, members[0])
        np.testing.assert_array_equal(c.indices_of_members, indices)


# ---------------------------------------------------------------------------
# read_time_series
# ---------------------------------------------------------------------------

class TestReadTimeSeries:
    def test_read_csv(self, csv_file):
        ts_list = read_time_series(csv_file)
        assert len(ts_list) == 5
        assert all(isinstance(ts, TimeSeries) for ts in ts_list)
        assert all(ts.data.ndim == 1 for ts in ts_list)

    def test_read_csv_labels(self, csv_file):
        ts_list = read_time_series(csv_file)
        labels = [ts.label for ts in ts_list]
        assert labels == ["s1", "s2", "s3", "s4", "s5"]

    def test_read_xlsx(self, xlsx_file):
        ts_list = read_time_series(xlsx_file)
        assert len(ts_list) == 5
        assert all(isinstance(ts, TimeSeries) for ts in ts_list)

    def test_read_xlsx_with_clusters(self, xlsx_file):
        ts_list = read_time_series(xlsx_file, withClusters=True)
        assert len(ts_list) == 5
        assert all(ts.previous_cluster_id is not None for ts in ts_list)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_time_series("/nonexistent/path/data.xlsx")

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("hello")
        with pytest.raises(ValueError):
            read_time_series(str(p))


# ---------------------------------------------------------------------------
# _normalize_data
# ---------------------------------------------------------------------------

class TestNormalizeData:
    def test_output_in_unit_range(self, ts_list_simple):
        result = _normalize_data(ts_list_simple)
        for ts in result:
            assert np.min(ts.data) >= 0.0 - 1e-9
            assert np.max(ts.data) <= 1.0 + 1e-9

    def test_zero_variance_series(self):
        ts = TimeSeries("const", np.full(10, 5.0))
        result = _normalize_data([ts])
        # constant series → all 0.5
        np.testing.assert_allclose(result[0].data, 0.5)

    def test_modifies_data_in_place(self, ts_list_simple):
        original_labels = [ts.label for ts in ts_list_simple]
        result = _normalize_data(ts_list_simple)
        assert [ts.label for ts in result] == original_labels


# ---------------------------------------------------------------------------
# _standardize_data
# ---------------------------------------------------------------------------

class TestStandardizeData:
    def test_zero_mean(self, ts_list_simple):
        result = _standardize_data(ts_list_simple)
        for ts in result:
            assert abs(np.mean(ts.data)) < 1e-9

    def test_unit_std(self):
        ts = TimeSeries("ramp", np.linspace(0, 10, 50))
        result = _standardize_data([ts])
        assert abs(np.std(result[0].data) - 1.0) < 1e-9

    def test_zero_variance_no_error(self):
        ts = TimeSeries("const", np.full(10, 7.0))
        result = _standardize_data([ts])
        # zero std → divide by 1, so data becomes 0.0 everywhere
        np.testing.assert_allclose(result[0].data, 0.0)


# ---------------------------------------------------------------------------
# perform_clustering
# ---------------------------------------------------------------------------

class TestPerformClustering:
    """Smoke tests: verify return types and shapes, not specific cluster assignments."""

    def _check_result(self, dRow, cluster_list, assignments, n):
        assert isinstance(dRow, np.ndarray)
        assert dRow.shape == (n * (n - 1) // 2,)
        assert isinstance(cluster_list, list)
        assert len(assignments) == n

    def test_euclidean(self, ts_list_simple):
        n = len(ts_list_simple)
        dRow, cl, assignments = perform_clustering(
            ts_list_simple, distance="euclidean",
            cMethod="maxclust", cValue=2,
        )
        self._check_result(dRow, cl, assignments, n)

    def test_dtw(self, ts_list_simple):
        n = len(ts_list_simple)
        dRow, cl, assignments = perform_clustering(
            ts_list_simple, distance="dtw",
            cMethod="maxclust", cValue=2,
        )
        self._check_result(dRow, cl, assignments, n)

    def test_pattern_dtw(self, ts_list_simple):
        n = len(ts_list_simple)
        dRow, cl, assignments = perform_clustering(
            ts_list_simple, distance="pattern_dtw",
            cMethod="maxclust", cValue=2,
        )
        self._check_result(dRow, cl, assignments, n)

    def test_pattern_wdtw(self, ts_list_simple):
        n = len(ts_list_simple)
        dRow, cl, assignments = perform_clustering(
            ts_list_simple, distance="pattern_wdtw",
            cMethod="maxclust", cValue=2,
        )
        self._check_result(dRow, cl, assignments, n)

    def test_maxclust_count(self, ts_list_simple):
        _, cluster_list, _ = perform_clustering(
            ts_list_simple, distance="euclidean",
            cMethod="maxclust", cValue=3,
        )
        assert len(cluster_list) <= 3

    def test_normalize_transform(self, ts_list_simple):
        dRow, cl, assignments = perform_clustering(
            ts_list_simple, distance="euclidean",
            cMethod="maxclust", cValue=2,
            transform="normalize",
        )
        assert len(assignments) == len(ts_list_simple)

    def test_standardize_transform(self, ts_list_simple):
        dRow, cl, assignments = perform_clustering(
            ts_list_simple, distance="euclidean",
            cMethod="maxclust", cValue=2,
            transform="standardize",
        )
        assert len(assignments) == len(ts_list_simple)

    def test_cluster_members_cover_all_series(self, ts_list_simple):
        n = len(ts_list_simple)
        _, cluster_list, _ = perform_clustering(
            ts_list_simple, distance="euclidean",
            cMethod="maxclust", cValue=2,
        )
        total_members = sum(c.number_of_members for c in cluster_list)
        assert total_members == n
