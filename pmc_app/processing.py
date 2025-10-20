from __future__ import annotations

import pandas as pd

from .models import FinderConfig


def process_volume_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a volume table to:

    Columns: ['id', 'ihc_label', 'object', 'object_id', 'volume'].

    Robust to empty input or missing 'Set N' columns.

    Args:
        df: Raw 'Volume' sheet.

    Returns:
        Normalized DataFrame subset.
    """
    out_cols = ["id", "ihc_label", "object", "object_id", "volume"]

    set_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Set ")]

    if len(set_cols) == 0 or len(df) == 0:
        existing = [c for c in out_cols if c in df.columns]
        return df[existing].copy()

    mask = df[set_cols].notna() & (df[set_cols].astype(str).apply(lambda s: s.str.strip()) != "")
    has_any = mask.fillna(False).filter(like="Set").any(axis=1)
    set_col_series = mask.fillna(False).idxmax(axis=1).where(has_any)
    ihc_label = set_col_series.str.extract(r"Set\s*(\d+)", expand=False).astype(object)
    df = df.copy()
    df["ihc_label"] = ihc_label

    df = df.rename(columns={"ID": "object_id", "Volume": "volume"})

    existing = [c for c in out_cols if c in df.columns]
    return df[existing].copy()


def process_position_df(df: pd.DataFrame, cfg: FinderConfig) -> pd.DataFrame:
    """Normalize 'Position' sheet columns and derive apical/basal marks."""
    df = df.rename(
        columns={
            "Position X": "pos_x",
            "Position Y": "pos_y",
            "Position Z": "pos_z",
            "ID": "object_id",
            "Surpass Object": "object",
        }
    )

    out_cols = ["id", "object", "object_id", "pos_x", "pos_y", "pos_z"]
    df = df[[c for c in out_cols if c in df.columns]].copy()

    df["ihc_label"] = df["object"].astype(str).str.extract(r"^Spots\s*(\d+)", expand=False)

    mask = df["ihc_label"].notna()
    min_id_masked = df.loc[mask].groupby("ihc_label")["object_id"].transform("min")
    max_id_masked = df.loc[mask].groupby("ihc_label")["object_id"].transform("max")

    min_id = min_id_masked.reindex(df.index)
    max_id = max_id_masked.reindex(df.index)

    df.loc[df["object_id"] == min_id, "object"] = "apical"
    df.loc[df["object_id"] == max_id, "object"] = "basal"

    return df


def merge_dfs(ribbons_df: pd.DataFrame, psds_df: pd.DataFrame, positions_df:
pd.DataFrame, cfg: FinderConfig) -> pd.DataFrame:
    """Merge normalized volume and position frames into a unified table."""
    df_synapses = pd.concat([psds_df, ribbons_df], ignore_index=True)
    df = df_synapses.merge(
        positions_df, on=["id", "object", "object_id"], how="outer", suffixes=("", "_temp")
    )

    if "ihc_label_temp" in df.columns:
        df["ihc_label"] = df["ihc_label"].fillna(df["ihc_label_temp"])
        df = df.drop(columns="ihc_label_temp")

    mask = df["ihc_label"].isna() & df["object"].isin([cfg.ribbons_obj, cfg.psds_obj])
    df.loc[mask, "object"] = "Unclassified " + df.loc[mask, "object"]

    df = df[["id", "object", "ihc_label", "object_id", "pos_x", "pos_y", "pos_z", "volume"]]

    df = df.astype(
        {
            "id": "string",
            "object": "string",
            "ihc_label": "str",
            "object_id": "int64",
            "pos_x": "float64",
            "pos_y": "float64",
            "pos_z": "float64",
            "volume": "float64",
        },
        errors="ignore",
    )
    return df
