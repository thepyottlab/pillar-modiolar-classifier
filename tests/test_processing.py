from __future__ import annotations

import pandas as pd

from pmc_app.models import FinderConfig
from pmc_app.processing import merge_dfs, process_position_df, process_volume_df


def test_process_volume_df_handles_missing_set_columns():
    df = pd.DataFrame({"ID": [1, 2], "Volume": [10.0, 20.0]})
    out = process_volume_df(df)
    assert set(out.columns) >= {"object_id", "volume"}


def test_process_position_df_derives_apical_basal():
    raw = pd.DataFrame(
        {
            "id": ["G1"] * 4,
            "Surpass Object": ["Spots 1", "Spots 1", "pillar", "modiolar"],
            "ID": [5, 9, 1, 2],
            "Position X": [0.0, 10.0, 1.0, 2.0],
            "Position Y": [0.0, 0.0, 1.0, 2.0],
            "Position Z": [0.0, 0.0, 1.0, 2.0],
        }
    )
    cfg = FinderConfig(folder=__file__)  # dummy
    out, _ = process_position_df(raw, cfg)
    assert set(out["object"]) >= {"apical", "basal", "pillar", "modiolar"}


def test_merge_roundtrip_columns():
    r = pd.DataFrame({"id": ["G1"], "ihc_label": ["1"], "object": ["ribbons"], "object_id": [5], "volume": [12.0]})
    p = pd.DataFrame({"id": ["G1"], "ihc_label": ["1"], "object": ["PSDs"], "object_id": [9], "volume": [7.0]})
    pos = pd.DataFrame(
        {
            "id": ["G1", "G1"],
            "object": ["ribbons", "PSDs"],
            "object_id": [5, 9],
            "pos_x": [0.0, 1.0],
            "pos_y": [0.0, 1.0],
            "pos_z": [0.0, 1.0],
        }
    )
    out = merge_dfs(r, p, pos)
    expected_cols = ["id", "object", "ihc_label", "object_id", "pos_x", "pos_y", "pos_z", "volume"]
    assert list(out.columns) == expected_cols
