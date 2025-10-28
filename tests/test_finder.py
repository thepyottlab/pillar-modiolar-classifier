"""Tests for file-group discovery in the finder module.

These tests rely on example files located under the repository's ``examples/`` folder.
If that folder is missing, the suite is skipped.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pmc_app.finder import find_groups
from pmc_app.models import FinderConfig, Group

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
pytestmark = pytest.mark.skipif(
    not EXAMPLES_DIR.exists(), reason="examples/ folder missing"
)


def _copy(dst: Path, names: list[str]) -> None:
    """Copy example files into ``dst``, creating parents if needed.

    Args:
      dst: Destination directory.
      names: Filenames (relative to ``EXAMPLES_DIR``) to copy.

    Raises:
      FileNotFoundError: A source file does not exist.
      OSError: Copying failed due to an I/O error.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copy(EXAMPLES_DIR / n, dst / n)


def _find_groups_allow_partial(cfg: FinderConfig) -> dict[str, Group]:
    """Return groups from :func:`find_groups`, accepting partial results.

    If :func:`find_groups` raises and the exception has a ``groups`` attribute,
    return that partial mapping; otherwise re-raise the exception.

    Args:
      cfg: Finder configuration.

    Returns:
      Mapping of group ID to :class:`Group` (possibly partial).

    Raises:
      Exception: Re-raised if no partial ``groups`` are available.
    """
    try:
        return find_groups(cfg)
    except Exception as e:  # noqa: BLE001 - broad for test helper
        groups = getattr(e, "groups", None)
        if groups is not None:
            return groups
        raise


def test_find_groups_ribbons_only_from_examples(tmp_path: Path) -> None:
    """Ribbons-only mode accepts groups with no PSD file."""
    ribonly_rib = next(
        (p for p in EXAMPLES_DIR.glob("*rib.xls") if "ribonly" in p.stem.lower()), None
    )
    ribonly_pos = next(
        (p for p in EXAMPLES_DIR.glob("*pos.xls") if "ribonly" in p.stem.lower()), None
    )

    if ribonly_rib is None or ribonly_pos is None:
        pytest.skip("No ribonly example pair found in examples/")
        return  # appease type checkers

    _copy(tmp_path, [ribonly_rib.name, ribonly_pos.name])

    cfg = FinderConfig(
        folder=tmp_path,
        ribbons="_ribonly_rib",
        psds="psd",
        positions="_ribonly_pos",
        extensions=".xls",
        ribbons_only=True,
        psds_only=False,
    )

    groups = _find_groups_allow_partial(cfg)
    assert groups, "Expected a ribbons-only group to be detected"

    _, grp = next(iter(groups.items()))
    assert "ribbons" in grp.file_paths and "positions" in grp.file_paths
    assert "psds" not in grp.file_paths
