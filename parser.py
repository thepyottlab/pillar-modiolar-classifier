"""Utilities for parsing Excel workbooks into tidy pandas DataFrames."""

from typing import Dict, Tuple

import pandas as pd

from models import FinderConfig, Group


def parse_group(group: Group, cfg: FinderConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and normalize the ribbon, PSD and position sheets for a group."""

    ribbons_df = pd.read_excel(group.file_paths[cfg.ribbons], sheet_name="Volume")
    psds_df = pd.read_excel(group.file_paths[cfg.psds], sheet_name="Volume")
    positions_df = pd.read_excel(group.file_paths[cfg.positions], sheet_name="Position")

    inputs: Dict[str, pd.DataFrame] = {
        cfg.ribbons: ribbons_df,
        cfg.psds: psds_df,
        cfg.positions: positions_df,
    }

    outputs: Dict[str, pd.DataFrame] = {}
    for suffix, df in inputs.items():
        df = df.copy()

        df.columns = df.iloc[0].astype(str).str.strip()
        df = df.iloc[1:].reset_index(drop=True)

        df["id"] = group.id
        df["source_file"] = group.file_paths[suffix].name

        if suffix == cfg.ribbons:
            df["object"] = cfg.ribbons_obj
        elif suffix == cfg.psds:
            df["object"] = cfg.psds_obj

        outputs[suffix] = df

    return outputs[cfg.ribbons], outputs[cfg.psds], outputs[cfg.positions]
