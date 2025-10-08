"""DataFrame transformations for the pillar-modiolar processing pipeline."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from models import FinderConfig


def _normalize_set_columns(df: pd.DataFrame, set_columns: Iterable[str]) -> pd.Series:
    """Return a series containing the ``Set N`` label associated with each row."""

    if not set_columns:
        return pd.Series([None] * len(df), index=df.index, dtype=object)

    trimmed = df[set_columns].fillna("")
    trimmed = trimmed.applymap(lambda value: str(value).strip())
    mask = trimmed != ""

    # idxmax returns the first True column; guaranteed by data that only one per row.
    selected_column = mask.idxmax(axis=1)
    return selected_column.str.extract(r"Set\s*(\d+)", expand=False)


def process_volume_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ribbon/PSD volume worksheets into a tidy DataFrame."""

    set_columns = [column for column in df.columns if isinstance(column, str) and column.startswith("Set ")]
    df = df.copy()

    df["ihc_label"] = _normalize_set_columns(df, set_columns)

    df = df.rename(columns={"ID": "object_id", "Volume": "volume"})
    ordered_columns = ["id", "object", "object_id", "volume", "ihc_label"]
    available_columns = [column for column in ordered_columns if column in df.columns]
    return df[available_columns]


def process_position_df(df: pd.DataFrame, cfg: FinderConfig) -> tuple[pd.DataFrame, FinderConfig]:
    """Normalize the position worksheet and enrich it with apical/basal labels."""

    df = df.rename(
        columns={
            "Position X": "pos_x",
            "Position Y": "pos_y",
            "Position Z": "pos_z",
            "ID": "object_id",
            "Surpass Object": "object",
        }
    )

    ordered_columns = ["id", "object", "object_id", "pos_x", "pos_y", "pos_z"]
    available_columns = [column for column in ordered_columns if column in df.columns]
    df = df[available_columns].copy()

    df["ihc_label"] = df["object"].str.extract(r"^Spots\s*(\d+)", expand=False)
    has_spot_label = df["ihc_label"].notna()

    group_by_label = df.loc[has_spot_label].groupby("ihc_label")
    min_object_id = group_by_label["object_id"].transform("min").reindex(df.index)
    max_object_id = group_by_label["object_id"].transform("max").reindex(df.index)

    df.loc[df["object_id"] == min_object_id, "object"] = "apical"
    df.loc[df["object_id"] == max_object_id, "object"] = "basal"
    df.loc[df["object"] == cfg.ihc_obj, "ihc_label"] = df.loc[df["object"] == cfg.ihc_obj, "object_id"]

    allowed_objects = {
        cfg.ribbons_obj,
        cfg.psds_obj,
        cfg.ihc_obj,
        cfg.pillar_obj,
        cfg.modiolar_obj,
        "apical",
        "basal",
    }
    df = df[df["object"].isin(allowed_objects)].reset_index(drop=True)

    return df, cfg


def merge_dfs(ribbons_df: pd.DataFrame, psds_df: pd.DataFrame, positions_df: pd.DataFrame) -> pd.DataFrame:
    """Merge the individual worksheets into a unified synapse DataFrame."""

    df_synapses = pd.concat([psds_df, ribbons_df], ignore_index=True)
    df = df_synapses.merge(
        positions_df,
        on=["id", "object", "object_id"],
        how="outer",
        suffixes=("", "_temp"),
    )

    df["ihc_label"] = df["ihc_label"].fillna(df["ihc_label_temp"])
    df = df.drop(columns="ihc_label_temp")

    ordered_columns = ["id", "object", "ihc_label", "object_id", "pos_x", "pos_y", "pos_z", "volume"]
    df = df[ordered_columns]

    return df.astype(
        {
            "id": "string",
            "object": "string",
            "ihc_label": "string",
            "object_id": "int64",
            "pos_x": "float64",
            "pos_y": "float64",
            "pos_z": "float64",
            "volume": "float64",
        }
    )
