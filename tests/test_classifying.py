"""Tests for parsing and minimal processing used by classification paths.

These tests monkeypatch Excel reading to provide small in-memory DataFrames that
mimic the expected parser outputs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pmc_app import parser as parser_mod
from pmc_app.models import FinderConfig, Group
from pmc_app.processing import merge_dfs, process_position_df, process_volume_df


def _vol_sheet() -> pd.DataFrame:
    """Return a tiny 'Volume' sheet fixture."""
    return pd.DataFrame(
        {"ID": [1, 2], "Volume": [10.0, 20.0], "Set 1": ["x", ""], "Set 2": ["", "x"]}
    )


def _pos_sheet() -> pd.DataFrame:
    """Return a tiny 'Position' sheet with three synapses and two anchors."""
    return pd.DataFrame(
        {
            "Position X": [0, 1, 2, 0, 0, 0, 0],
            "Position Y": [0, 0, 0, -1, 1, 0, 0],
            "Position Z": [0, 0, 0, 0, 0, 0, 1],
            "ID": [1, 2, 3, 100, 101, 1, 3],
            "Surpass Object": [
                "ribbons",
                "PSDs",
                "PSDs",
                "pillar",
                "modiolar",
                "Spots 1",
                "Spots 1",
            ],
        }
    )


@pytest.fixture
def cfg(tmp_path: Path) -> FinderConfig:
    """Create a minimal :class:`FinderConfig` rooted at a temporary folder."""
    return FinderConfig(
        folder=tmp_path,
        ribbons="rib",
        psds="psd",
        positions="pos",
        extensions=".xlsx",
        ribbons_obj="ribbons",
        psds_obj="PSDs",
        pillar_obj="pillar",
        modiolar_obj="modiolar",
    )


def test_parse_and_process_monkeypatched(
    monkeypatch: pytest.MonkeyPatch,
    cfg: FinderConfig,
    tmp_path: Path,  # noqa: ARG001
) -> None:
    """Patched Excel reader yields expected processed/merged DataFrames."""

    def fake_read_excel_safe(_path: Path, sheet: str) -> pd.DataFrame:
        if sheet == "Volume":
            return _vol_sheet()
        if sheet == "Position":
            return _pos_sheet()
        raise AssertionError("unexpected sheet")

    monkeypatch.setattr(parser_mod, "_read_excel_safe", fake_read_excel_safe)

    g = Group(
        id="G1",
        file_paths={
            "ribbons": Path("G1rib.xlsx"),
            "psds": Path("G1psd.xlsx"),
            "positions": Path("G1pos.xlsx"),
            cfg.ribbons: Path("G1rib.xlsx"),
            cfg.psds: Path("G1psd.xlsx"),
            cfg.positions: Path("G1pos.xlsx"),
        },
    )

    ribbons_df, psds_df, positions_df = parser_mod.parse_group(g, cfg)

    rib_p = process_volume_df(ribbons_df)
    psd_p = process_volume_df(psds_df)
    assert set(rib_p.columns) == {"id", "ihc_label", "object", "object_id", "volume"}
    assert set(psd_p.columns) == set(rib_p.columns)
    assert set(rib_p["ihc_label"].astype(str)) == {"1", "2"}

    pos_p = process_position_df(positions_df)
    ap_rows = pos_p[pos_p["object"] == "apical"]
    ba_rows = pos_p[pos_p["object"] == "basal"]
    assert len(ap_rows) == 1 and ap_rows.iloc[0]["object_id"] == 1
    assert len(ba_rows) == 1 and ba_rows.iloc[0]["object_id"] == 3

    merged = merge_dfs(rib_p, psd_p, pos_p, cfg)
    objs = set(merged["object"])
    assert {"pillar", "modiolar"}.issubset(objs)
    assert ("Unclassified ribbons" in objs) or ("Unclassified PSDs" in objs)
