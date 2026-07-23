"""Shared fixtures for simclstr unit tests."""

import numpy as np
import pytest
import pandas as pd

from simclstr.clusterer import TimeSeries


# ---------------------------------------------------------------------------
# Synthetic time-series helpers
# ---------------------------------------------------------------------------

def make_ts(label: str, data: np.ndarray, previous_cluster_id=None) -> TimeSeries:
    return TimeSeries(label, data.astype(float), previous_cluster_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ts_list_simple():
    """5 synthetic time series with distinct shapes."""
    t = np.linspace(0, 1, 30)
    return [
        make_ts("linear_up",    t),
        make_ts("linear_down",  1 - t),
        make_ts("exponential",  np.exp(t) - 1),
        make_ts("s_curve",      1 / (1 + np.exp(-10 * (t - 0.5)))),
        make_ts("quadratic",    t ** 2),
    ]


@pytest.fixture
def ts_list_identical():
    """3 identical time series (distance should be 0 between any pair)."""
    data = np.sin(np.linspace(0, 2 * np.pi, 30))
    return [make_ts(f"identical_{i}", data.copy()) for i in range(3)]


@pytest.fixture
def ts_list_with_clusters():
    """6 time series that carry pre-assigned cluster labels for comparison tests."""
    t = np.linspace(0, 1, 20)
    ts = [
        make_ts("a", t,         previous_cluster_id="A"),
        make_ts("b", t * 1.1,   previous_cluster_id="A"),
        make_ts("c", 1 - t,     previous_cluster_id="B"),
        make_ts("d", 1 - t * 0.9, previous_cluster_id="B"),
        make_ts("e", t ** 2,    previous_cluster_id="C"),
        make_ts("f", t ** 1.9,  previous_cluster_id="C"),
    ]
    # Assign matching cluster IDs so Rand = 1 when compared to previous
    for ts_obj in ts:
        ts_obj.cluster_id = ts_obj.previous_cluster_id
    return ts


@pytest.fixture
def csv_file(tmp_path):
    """A temporary CSV file with 5 short time series."""
    t = np.linspace(0, 1, 10)
    rows = {
        "label": ["s1", "s2", "s3", "s4", "s5"],
    }
    for i, col in enumerate(range(10)):
        rows[f"t{i}"] = [
            t[i], 1 - t[i], t[i] ** 2,
            np.exp(t[i]) - 1,
            1 / (1 + np.exp(-10 * (t[i] - 0.5))),
        ]
    df = pd.DataFrame(rows)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def xlsx_file(tmp_path):
    """A temporary XLSX file with a 'data' sheet and a 'clusters' sheet."""
    t = np.linspace(0, 1, 10)
    data_rows = []
    for i in range(5):
        row = [f"run_{i}"] + list(t + i * 0.1)
        data_rows.append(row)

    df_data = pd.DataFrame(data_rows)
    df_clusters = pd.DataFrame({"cluster": [1, 1, 2, 2, 3]})

    path = tmp_path / "data.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_data.to_excel(writer, sheet_name="data", index=False, header=False)
        df_clusters.to_excel(writer, sheet_name="clusters", index=False, header=False)

    return str(path)
