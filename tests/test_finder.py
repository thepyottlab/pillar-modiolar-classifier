from pathlib import Path

from pmc_app.finder import find_groups
from pmc_app.models import FinderConfig


def touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_find_groups_complete_only(tmp_path: Path):
    # Create a complete group "A01" and an incomplete group "B01"
    for name in ["A01rib.xlsx", "A01psd.xlsx", "A01pos.xlsx", "B01pos.xlsx"]:
        touch(tmp_path / name)

    cfg = FinderConfig(
        folder=tmp_path,
        ribbons="rib",
        psds="psd",
        positions="pos",
        extensions=".xlsx",
        case_insensitive=True,
        ribbons_only=False,
        psds_only=False,
    )

    groups = find_groups(cfg)
    assert set(groups.keys()) == {"A01"}
    g = groups["A01"]
    # Finder should key by roles
    assert set(g.file_paths.keys()) == {"ribbons", "psds", "positions"}
    assert g.file_paths["ribbons"].name.endswith("A01rib.xlsx")
    assert g.file_paths["psds"].name.endswith("A01psd.xlsx")
    assert g.file_paths["positions"].name.endswith("A01pos.xlsx")


def test_find_groups_ribbons_only_allows_missing_psd(tmp_path: Path):
    for name in ["C01rib.xls", "C01pos.xls"]:
        touch(tmp_path / name)

    cfg = FinderConfig(
        folder=tmp_path,
        ribbons="rib",
        psds="psd",
        positions="pos",
        extensions=".xls",
        ribbons_only=True,
        psds_only=False,
    )
    groups = find_groups(cfg)
    assert set(groups.keys()) == {"C01"}
    g = groups["C01"]
    assert "ribbons" in g.file_paths and "positions" in g.file_paths
    # In ribbons-only mode, PSDs are not required
    assert "psds" not in g.file_paths
