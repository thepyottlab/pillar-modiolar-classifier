# processing.py  (or place in the same file where you do transforms)
import pandas as pd
from models import FinderConfig

def process_volume_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the first row into header, collapse 'Set N' columns into ihc_label (int),
    rename 'ID' -> 'object_id', and return columns:
      ['id', 'object', 'object_id', 'volume', 'ihc_label']
    Assumes exactly one Set column per row has a value.
    """
    # find Set columns flexibly
    set_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Set ")]

    # boolean mask of non-empty values
    mask = df[set_cols].notna() & (df[set_cols].astype(str).apply(lambda s: s.str.strip()) != "")

    # get the column name that is True for each row
    set_col_series = mask.idxmax(axis=1)

    # extract the number after 'Set ' and convert to int
    ihc_label = set_col_series.str.extract(r"Set\s*(\d+)",
                                           expand=False).astype(object)

    df["ihc_label"] = ihc_label

    # rename columns
    df = df.rename(columns={"ID": "object_id",
                            "Volume": "volume"})

    # keep only requested columns in order, plus ihc_label
    out_cols = ["id", "ihc_label", "object", "object_id", "volume"]
    df = df[[c for c in out_cols if c in df.columns]]

    return df

def process_position_df(df: pd.DataFrame, cfg: FinderConfig) -> tuple[
    pd.DataFrame, FinderConfig]:
    df = df.rename(columns={"Position X": "pos_x",
                            "Position Y": "pos_y",
                            "Position Z": "pos_z",
                            "ID" : "object_id",
                            "Surpass Object" : "object"})

    out_cols = ["id", "object", "object_id", "pos_x", "pos_y", "pos_z"]
    df = df[[c for c in out_cols if c in df.columns]]



    df['ihc_label'] = df['object'].str.extract(r'^Spots\s*(\d+)',
                                                 expand=False)

    mask = df['ihc_label'].notna()

    # 3) compute per-spot min/max object_id for the masked rows (aligned to the masked index)
    min_id_masked = df.loc[mask].groupby('ihc_label')['object_id'].transform(
        'min')
    max_id_masked = df.loc[mask].groupby('ihc_label')['object_id'].transform(
        'max')

    # 4) reindex to full DataFrame index so comparisons align correctly
    min_id = min_id_masked.reindex(df.index)
    max_id = max_id_masked.reindex(df.index)

    # 5) set object to apical/basal where object_id equals the per-spot min/max
    df.loc[df['object_id'] == min_id, 'object'] = 'apical'
    df.loc[df['object_id'] == max_id, 'object'] = 'basal'

    df.loc[df['object'] == 'IHC', 'ihc_label'] = df.loc[
        df['object'] == 'IHC', 'object_id']

    df = df[df['object'].isin([cfg.ribbons_obj, cfg.psds_obj, cfg.ihc_obj,
                               cfg.pillar_obj,
                               cfg.modiolar_obj, "apical",
                               "basal"])].reset_index(
        drop=True)
    return df, cfg

def merge_dfs(ribbons_df: pd.DataFrame, psds_df: pd.DataFrame,
              positions_df: pd.DataFrame) -> pd.DataFrame:
    df_synapses = pd.concat([psds_df, ribbons_df], ignore_index=True)
    df = df_synapses.merge(positions_df, on=['id', 'object', 'object_id'],
                              how='outer', suffixes=('', '_temp'))

    # keep vertical's ihc_label when present; otherwise use positions' ihc_label
    df['ihc_label'] = df['ihc_label'].fillna(
        df['ihc_label_temp'])
    df = df.drop(columns='ihc_label_temp')

    df = df[['id', 'object', 'ihc_label', 'object_id', 'pos_x', 'pos_y',
             'pos_z', 'volume']]

    df = df.astype({
        'id': 'string',
        'object': 'string',
        'ihc_label': 'str',
        'object_id': 'int64',
        'pos_x': 'float64',
        'pos_y': 'float64',
        'pos_z': 'float64',
        'volume': 'float64'
    })
    return df