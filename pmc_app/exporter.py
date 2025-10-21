from __future__ import annotations
from pathlib import Path

import pandas as pd
import re
from qtpy.QtWidgets import QFileDialog


def prompt_export_dir() -> Path | None:
    """Open native select-directory dialog; return None if canceled."""
    dirname = QFileDialog.getExistingDirectory(
        None, "Select export folder", "", QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
    )
    return Path(dirname) if dirname else None


def _safe_grouplabel(gid: str) -> str:
    """Filesystem-safe group id for filenames."""
    s = gid.strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:200] if len(s) > 200 else s


def export_df_csv(df: pd.DataFrame, gid: str, out_dir: Path) -> Path:
    """Write CSV for a group as '<gid>_classified.csv' and replace 'nan' by
    empty spaces."""
    df = df.replace('nan', '')
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{_safe_grouplabel(gid)}_classified.csv"
    path = out_dir / fname
    df.to_csv(path, index=False)
    return path
