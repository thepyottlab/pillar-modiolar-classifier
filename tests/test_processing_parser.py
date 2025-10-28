"""Integration-style tests for parsing and processing helpers.

Uses monkeypatched Excel reads to validate downstream processing and merge logic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pmc_app import parser as parser_mod
from pmc_app.models import FinderConfig, Group
from pmc_app.processing import merge_dfs, process_position_df, process_volume_df


def _build_volume_df() -> pd.DataFrame:
    """Return a tiny 'Volume' sheet with two labeled objects across Set 1/2."""
    return pd.DataFrame(
        {"ID": [1, 2], "Volume": [10.0, 20.0], "Set 1": ["x", ""], "Set 2": ["", "x"]}
    )


def _build_position_df() -> pd.DataFrame:
    """Return a tiny 'Position' sheet with synapses and pillar/modiolar anchors."""
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
    monkeypatch: pytest.MonkeyPatch, cfg: FinderConfig
) -> None:
    """Patched parser + processing derive labels and merge correctly."""

    def fake_read_excel_safe(_path: Path, sheet: str) -> pd.DataFrame:
        if sheet == "Volume":
            return _build_volume_df()
        if sheet == "Position":
            return _build_position_df()
        raise AssertionError(f"unexpected sheet: {sheet!r}")

    monkeypatch.setattr(parser_mod, "_read_excel_safe", fake_read_excel_safe)

    group = Group(
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

    rib_raw, psd_raw, pos_raw = parser_mod.parse_group(group, cfg)

    rib = process_volume_df(rib_raw)
    psd = process_volume_df(psd_raw)
    assert set(rib.columns) == {"id", "ihc_label", "object", "object_id", "volume"}
    assert set(psd.columns) == set(rib.columns)
    assert set(rib["ihc_label"].astype(str)) == {"1", "2"}

    pos = process_position_df(pos_raw)
    ap = pos[pos["object"] == "apical"]
    ba = pos[pos["object"] == "basal"]
    assert len(ap) == 1 and ap.iloc[0]["object_id"] == 1
    assert len(ba) == 1 and ba.iloc[0]["object_id"] == 3

    merged = merge_dfs(rib, psd, pos, cfg)
    objs = set(merged["object"])
    assert {"pillar", "modiolar"} <= objs
    assert ("Unclassified ribbons" in objs) or ("Unclassified PSDs" in objs)
