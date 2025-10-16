from __future__ import annotations

from pathlib import Path

import pandas as pd

from pmc_app.finder import find_groups
from pmc_app.models import FinderConfig


def test_find_groups(tmp_path: Path):
    # Create files: "A ribbon.xls", "A sum.xls", "B psd.xls", "B sum.xls"
    (tmp_path / "A ribbon.xls").write_text("dummy")
    (tmp_path / "A sum.xls").write_text("dummy")
    (tmp_path / "B psd.xls").write_text("dummy")
    (tmp_path / "B sum.xls").write_text("dummy")
    (tmp_path / "ignored.txt").write_text("nope")

    cfg = FinderConfig(folder=tmp_path, ribbons="ribbon", psds="psd", positions="sum", extensions=".xls")
    groups = find_groups(cfg)

    assert "A" in groups and "B" in groups
    assert cfg.ribbons in groups["A"].file_paths
    assert cfg.positions in groups["A"].file_paths
    assert cfg.psds in groups["B"].file_paths
    assert cfg.positions in groups["B"].file_paths
