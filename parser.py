import pandas as pd
from models import FinderConfig, Group
from typing import Tuple

def parse_group(group: Group, cfg: FinderConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read group's files based on suffix names from cfg and normalize headers.
    Returns (ribbons_df, psds_df, positions_df).
    Assumes files exist and sheets are named 'Volume' (ribbons/psds) and 'Position' (positions).
    """
    # read by config suffix names
    ribbons_df = pd.read_excel(group.file_paths[cfg.ribbons], sheet_name="Volume")
    psds_df = pd.read_excel(group.file_paths[cfg.psds], sheet_name="Volume")
    positions_df = pd.read_excel(group.file_paths[cfg.positions], sheet_name="Position")

    inputs = {
        cfg.ribbons: ribbons_df,
        cfg.psds: psds_df,
        cfg.positions: positions_df,
    }

    outputs = {}
    for key, df in inputs.items():
        # promote first row to header
        df.columns = df.iloc[0].astype(str).str.strip()
        df = df.iloc[1:].reset_index(drop=True)

        # always add id and source_file
        df["id"] = group.id
        df["source_file"] = group.file_paths[key].name

        # add 'object' for ribbons and psds only (not for positions)
        if key in (cfg.ribbons):
            df["object"] = cfg.ribbons_obj
        if key in (cfg.psds):
            df["object"] = cfg.psds_obj

        outputs[key] = df

    return outputs[cfg.ribbons], outputs[cfg.psds], outputs[cfg.positions]
