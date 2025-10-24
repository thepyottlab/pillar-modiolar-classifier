"""CSV export and safe filename helpers."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from qtpy.QtWidgets import QFileDialog, QWidget


def prompt_export_dir(parent: QWidget | None = None) -> Path | None:
    """Open a native select-directory dialog.

    Returns:
        Path | None: The selected directory, or None if the dialog was canceled.
    """
    opts = QFileDialog.Options()
    opts |= QFileDialog.ShowDirsOnly
    opts |= QFileDialog.DontResolveSymlinks

    dlg = QFileDialog(parent)
    dirname = dlg.getExistingDirectory(parent, "Select export folder", "", options=opts)
    return Path(dirname) if dirname else None


def _safe_grouplabel(gid: str) -> str:
    """Return a filesystem-safe group label for filenames.

    Args:
        gid: Original group id.

    Returns:
        str: Sanitized label up to a maximum of 200 characters.
    """
    s = gid.strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:200] if len(s) > 200 else s


def export_df_csv(df: pd.DataFrame, gid: str, out_dir: Path) -> Path:
    """Export a classified dataframe to CSV and replace literal 'nan' strings.

    Args:
        df: Classified dataframe.
        gid: Group identifier for naming the CSV.
        out_dir: Output directory.

    Returns:
        Path: Path to the written CSV file.
    """
    df = df.replace("nan", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{_safe_grouplabel(gid)}_classified.csv"
    path = out_dir / fname
    df.to_csv(path, index=False)
    return path
