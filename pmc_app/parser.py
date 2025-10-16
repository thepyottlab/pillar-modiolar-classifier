from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from .exceptions import ParseError
from .models import FinderConfig, Group, InputMode

_EMPTY_VOL_COLS = ["id", "ihc_label", "object", "object_id", "volume", "source_file"]


def _empty_volume_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_VOL_COLS)


def _read_excel_safe(path, sheet: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except Exception as e:
        raise ParseError(f"Failed reading '{path}' sheet='{sheet}': {e}") from e

    # Promote first row to header (keeps existing behavior)
    try:
        df.columns = df.iloc[0].astype(str).str.strip()
        df = df.iloc[1:].reset_index(drop=True)
    except Exception as e:
        raise ParseError(f"Unexpected sheet format in '{path}' sheet='{sheet}': {e}") from e
    return df


def parse_group(group: Group, cfg: FinderConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read ribbons, psds, and positions for a group.

    Returns:
        (ribbons_df, psds_df, positions_df) — never None; ribbons/psds may be empty.
    """
    # Positions (required)
    pos_path = group.file_paths.get(cfg.positions)
    if pos_path is None:
        raise ParseError(f"Positions file (token '{cfg.positions}') missing for '{group.id}'.")
    positions_df = _read_excel_safe(pos_path, "Position")
    positions_df["id"] = group.id
    positions_df["source_file"] = pos_path.name

    # Ribbons
    ribbons_df: Optional[pd.DataFrame]
    if cfg.mode in (InputMode.BOTH, InputMode.RIBBONS_ONLY):
        rib_path = group.file_paths.get(cfg.ribbons)
        if rib_path is not None:
            ribbons_df = _read_excel_safe(rib_path, "Volume")
            ribbons_df["id"] = group.id
            ribbons_df["source_file"] = rib_path.name
            ribbons_df["object"] = cfg.ribbons_obj
        else:
            ribbons_df = _empty_volume_df()
    else:
        ribbons_df = _empty_volume_df()

    # PSDs
    psds_df: Optional[pd.DataFrame]
    if cfg.mode in (InputMode.BOTH, InputMode.PSDS_ONLY):
        psd_path = group.file_paths.get(cfg.psds)
        if psd_path is not None:
            psds_df = _read_excel_safe(psd_path, "Volume")
            psds_df["id"] = group.id
            psds_df["source_file"] = psd_path.name
            psds_df["object"] = cfg.psds_obj
        else:
            psds_df = _empty_volume_df()
    else:
        psds_df = _empty_volume_df()

    return ribbons_df, psds_df, positions_df
