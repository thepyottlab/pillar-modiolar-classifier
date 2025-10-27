import pandas as pd
import pytest

from pmc_app.models import FinderConfig, Group
from pmc_app import parser as parser_mod
from pmc_app.processing import process_position_df, process_volume_df, merge_dfs


def _vol_sheet():
    # Mimic parser._read_excel_safe() output: headers already normalized.
    # Two rows -> one labeled to Set 1, one to Set 2
    return pd.DataFrame(
        {
            "ID": [1, 2],
            "Volume": [10.0, 20.0],
            "Set 1": ["x", ""],
            "Set 2": ["", "x"],
        }
    )


def _pos_sheet():
    # Minimal positions: three synapse points for IHC-1 (IDs 1..3)
    # plus pillar/modiolar anchors.
    return pd.DataFrame(
        {
            "Position X": [0, 1, 2, 0, 0],
            "Position Y": [0, 0, 0, -1, 1],
            "Position Z": [0, 0, 0, 0, 0],
            "ID": [1, 2, 3, 100, 101],
            # 'Spots 1' for synapses so ihc_label can be derived;
            # pillar/modiolar use their object names.
            "Surpass Object": ["Spots 1", "Spots 1", "Spots 1", "pillar", "modiolar"],
        }
    )


@pytest.fixture
def cfg():
    return FinderConfig(
        folder=pytest.ensuretemp("unused"),
        ribbons="rib",
        psds="psd",
        positions="pos",
        extensions=".xlsx",
        ribbons_obj="ribbons",
        psds_obj="PSDs",
        pillar_obj="pillar",
        modiolar_obj="modiolar",
    )


def test_parse_and_process_monkeypatched(monkeypatch, cfg):
    # Patch the safe excel reader to return our synthetic sheets.
    def fake_read_excel_safe(path, sheet: str):
        if sheet == "Volume":
            return _vol_sheet()
        if sheet == "Position":
            return _pos_sheet()
        raise AssertionError("unexpected sheet")

    monkeypatch.setattr(parser_mod, "_read_excel_safe", fake_read_excel_safe)

    # Provide both role and token keys so the test works with either implementation.
    g = Group(
        id="G1",
        file_paths={
            "ribbons": "G1rib.xlsx",
            "psds": "G1psd.xlsx",
            "positions": "G1pos.xlsx",
            # token aliases (in case parse_group looks up by tokens)
            cfg.ribbons: "G1rib.xlsx",
            cfg.psds: "G1psd.xlsx",
            cfg.positions: "G1pos.xlsx",
        },
    )

    ribbons_df, psds_df, positions_df = parser_mod.parse_group(g, cfg)

    # Process volume sheets down to compact schema (adds ihc_label from Set N)
    rib_p = process_volume_df(ribbons_df)
    psd_p = process_volume_df(psds_df)

    assert set(rib_p.columns) == {"id", "ihc_label", "object", "object_id", "volume"}
    assert set(psd_p.columns) == set(rib_p.columns)
    # Expect two rows with ihc_label extracted as "1" and "2"
    assert set(rib_p["ihc_label"].astype(str)) == {"1", "2"}

    pos_p = process_position_df(positions_df, cfg)
    # apical/basal tags should be assigned to min/max object_id within Spots 1
    ap_rows = pos_p[pos_p["object"] == "apical"]
    ba_rows = pos_p[pos_p["object"] == "basal"]
    assert len(ap_rows) == 1 and ap_rows.iloc[0]["object_id"] == 1
    assert len(ba_rows) == 1 and ba_rows.iloc[0]["object_id"] == 3

    # Merge keeps synapses and positions together; unassigned synapses get "Unclassified ..."
    merged = merge_dfs(rib_p, psd_p, pos_p, cfg)
    assert {"pillar", "modiolar"}.issubset(set(merged["object"]))
    assert "Unclassified ribbons" in set(merged["object"]) or "Unclassified PSDs" in set(merged["object"])
