"""Parsing of Imaris-exported Excel sheets into normalized DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .exceptions import ParseError
from .models import FinderConfig, Group, InputMode

_EMPTY_VOL_COLS = ["id", "ihc_label", "object", "object_id", "volume", "source_file"]


def _empty_volume_df() -> pd.DataFrame:
    """Return an empty volume table with the expected columns."""
    return pd.DataFrame(columns=_EMPTY_VOL_COLS)


def _read_excel_safe(path: Path, sheet: str) -> pd.DataFrame:
    """Read an Excel sheet and normalize the header row placement.

    Promotes the first row to the header to preserve legacy behavior.

    Args:
        path: File system path to the Excel file.
        sheet: Sheet name to read.

    Returns:
        pd.DataFrame: Parsed table with the first row promoted to header.

    Raises:
        ParseError: On I/O or unexpected sheet format.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except Exception as e:
        raise ParseError(f"Failed reading '{path}' sheet='{sheet}': {e}") from e

    try:
        df.columns = df.iloc[0].astype(str).str.strip()
        df = df.iloc[1:].reset_index(drop=True)
    except Exception as e:
        raise ParseError(
            f"Unexpected sheet format in '{path}' sheet='{sheet}': {e}"
        ) from e

    return df


def parse_group(
    group: Group, cfg: FinderConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read ribbons, PSDs, and positions for a single group.

    Note:
        `group.file_paths` stores **role** keys ("ribbons"|"psds"|"positions"),
        not the suffix tokens. That is why we index by the role strings here.

    Args:
        group: Group container with role→path mapping.
        cfg: Configuration; used for object labels and mode.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (ribbons_df, psds_df, positions_df).
        Ribbons/PSDs frames may be empty depending on mode.

    Raises:
        ParseError: If the required Position sheet is missing.
    """
    pos_path = group.file_paths.get("positions")
    if pos_path is None:
        raise ParseError(
            f"Positions file (token '{cfg.positions}') missing for '{group.id}'."
        )

    positions_df = _read_excel_safe(pos_path, "Position")
    positions_df["id"] = group.id
    positions_df["source_file"] = pos_path.name

    ribbons_df: pd.DataFrame | None
    if cfg.mode in (InputMode.BOTH, InputMode.RIBBONS_ONLY):
        rib_path = group.file_paths.get("ribbons")
        if rib_path is not None:
            ribbons_df = _read_excel_safe(rib_path, "Volume")
            ribbons_df["id"] = group.id
            ribbons_df["source_file"] = rib_path.name
            ribbons_df["object"] = cfg.ribbons_obj
        else:
            ribbons_df = _empty_volume_df()
    else:
        ribbons_df = _empty_volume_df()

    psds_df: pd.DataFrame | None
    if cfg.mode in (InputMode.BOTH, InputMode.PSDS_ONLY):
        psd_path = group.file_paths.get("psds")
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
