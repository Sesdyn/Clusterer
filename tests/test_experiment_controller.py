"""Unit tests for simclstr.experiment_controller module."""

import numpy as np
import pytest
import pandas as pd

from simclstr.experiment_controller import _compare_clusterings, experiment_controller
from simclstr.clusterer import TimeSeries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts(label, data, cluster_id, previous_cluster_id):
    ts = TimeSeries(label, data, previous_cluster_id=previous_cluster_id)
    ts.cluster_id = cluster_id
    ts.index = 0  # index is not used by _compare_clusterings
    return ts


def _make_xlsx(tmp_path, n: int = 6, n_points: int = 20):
    """Create a minimal .xlsx fixture for experiment_controller."""
    t = np.linspace(0, 1, n_points)
    patterns = [t, 1 - t, t ** 2, np.exp(t) - 1, np.sin(np.pi * t), t ** 0.5]
    data_rows = [[f"run_{i}"] + list(patterns[i % len(patterns)]) for i in range(n)]
    df_data = pd.DataFrame(data_rows)
    df_clusters = pd.DataFrame({"cluster": [i % 2 + 1 for i in range(n)]})

    path = tmp_path / "exp_data.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_data.to_excel(writer, sheet_name="data", index=False, header=False)
        df_clusters.to_excel(writer, sheet_name="clusters", index=False, header=False)
    return str(path)


# ---------------------------------------------------------------------------
# _compare_clusterings
# ---------------------------------------------------------------------------

class TestCompareClusterings:
    def test_identical_clusterings_rand_one(self):
        ts_list = [
            _make_ts("a", np.ones(5), cluster_id="1", previous_cluster_id="1"),
            _make_ts("b", np.ones(5), cluster_id="1", previous_cluster_id="1"),
            _make_ts("c", np.ones(5), cluster_id="2", previous_cluster_id="2"),
            _make_ts("d", np.ones(5), cluster_id="2", previous_cluster_id="2"),
        ]
        rand, jaccard = _compare_clusterings(ts_list)
        assert rand == pytest.approx(1.0)
        assert jaccard == pytest.approx(1.0)

    def test_values_in_unit_range(self):
        ts_list = [
            _make_ts("a", np.ones(5), cluster_id="1", previous_cluster_id="2"),
            _make_ts("b", np.ones(5), cluster_id="2", previous_cluster_id="1"),
            _make_ts("c", np.ones(5), cluster_id="1", previous_cluster_id="1"),
            _make_ts("d", np.ones(5), cluster_id="2", previous_cluster_id="2"),
        ]
        rand, jaccard = _compare_clusterings(ts_list)
        assert 0.0 <= rand <= 1.0
        assert 0.0 <= jaccard <= 1.0

    def test_returns_two_floats(self):
        ts_list = [
            _make_ts("a", np.ones(5), cluster_id="1", previous_cluster_id="1"),
            _make_ts("b", np.ones(5), cluster_id="2", previous_cluster_id="1"),
        ]
        result = _compare_clusterings(ts_list)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_completely_different_clusterings(self):
        # Every pair that was together is now split
        ts_list = [
            _make_ts("a", np.ones(5), cluster_id="1", previous_cluster_id="1"),
            _make_ts("b", np.ones(5), cluster_id="2", previous_cluster_id="1"),
            _make_ts("c", np.ones(5), cluster_id="1", previous_cluster_id="2"),
            _make_ts("d", np.ones(5), cluster_id="2", previous_cluster_id="2"),
        ]
        rand, jaccard = _compare_clusterings(ts_list)
        # Rand should be 0 (all pairs disagree)
        assert rand == pytest.approx(0.0)

    def test_single_cluster_per_side(self):
        ts_list = [
            _make_ts("a", np.ones(5), cluster_id="1", previous_cluster_id="1"),
            _make_ts("b", np.ones(5), cluster_id="1", previous_cluster_id="1"),
            _make_ts("c", np.ones(5), cluster_id="1", previous_cluster_id="1"),
        ]
        rand, jaccard = _compare_clusterings(ts_list)
        assert rand == pytest.approx(1.0)
        assert jaccard == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# experiment_controller (smoke test)
# ---------------------------------------------------------------------------

class TestExperimentController:
    def test_returns_dict_with_expected_keys(self, tmp_path):
        xlsx = _make_xlsx(tmp_path)
        result = experiment_controller(
            file_path=xlsx,
            distance="euclidean",
            cMethod="maxclust",
            cValue=2,
            output_dir=str(tmp_path / "output"),
        )
        expected_keys = {
            "cluster_list", "list_of_ts_objects", "distance_matrix",
            "rand_index", "jaccard_index", "run_time", "total_time",
            "num_clusters", "output_file", "plot_file",
        }
        assert expected_keys.issubset(result.keys())

    def test_output_file_created(self, tmp_path):
        xlsx = _make_xlsx(tmp_path)
        output_dir = str(tmp_path / "output")
        result = experiment_controller(
            file_path=xlsx,
            distance="euclidean",
            cMethod="maxclust",
            cValue=2,
            output_dir=output_dir,
        )
        import os
        assert result["output_file"] is not None
        assert os.path.exists(result["output_file"])

    def test_cluster_list_non_empty(self, tmp_path):
        xlsx = _make_xlsx(tmp_path)
        result = experiment_controller(
            file_path=xlsx,
            distance="euclidean",
            cMethod="maxclust",
            cValue=2,
            output_dir=str(tmp_path / "output"),
        )
        assert len(result["cluster_list"]) > 0

    def test_num_clusters_matches_maxclust(self, tmp_path):
        xlsx = _make_xlsx(tmp_path, n=6)
        result = experiment_controller(
            file_path=xlsx,
            distance="euclidean",
            cMethod="maxclust",
            cValue=3,
            output_dir=str(tmp_path / "output"),
        )
        assert result["num_clusters"] <= 3

    def test_run_time_positive(self, tmp_path):
        xlsx = _make_xlsx(tmp_path)
        result = experiment_controller(
            file_path=xlsx,
            distance="euclidean",
            cMethod="maxclust",
            cValue=2,
            output_dir=str(tmp_path / "output"),
        )
        assert result["run_time"] >= 0
        assert result["total_time"] >= result["run_time"]

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            experiment_controller(
                file_path="/no/such/file.xlsx",
                output_dir=str(tmp_path / "output"),
            )
