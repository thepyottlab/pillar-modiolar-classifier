from pathlib import Path

import numpy as np
import pandas as pd

from pmc_app.classifier import (
    build_hc_planes,
    build_pm_planes,
    classify_synapses,
    identify_poles,
)
from pmc_app.exporter import export_df_csv
from pmc_app.models import FinderConfig


def _minimal_processed_df(cfg):
    """Return a tiny, already-processed dataframe in the schema expected by the classifier.

    One IHC (label "1"):
      - apical at (0, 0, 0), basal at (2, 0, 0)
      - pillar at (1, -1, 0), modiolar at (1, 1, 0)
      - 2 ribbons: one near pillar side (y=-0.5), one near modiolar side (y=+0.5)
    All coordinates are in XYZ; classifier expects ZYX ordering internally, which it handles.
    """
    rows = [
        # anchors
        dict(id="G", object="apical", ihc_label="1", object_id=1, pos_x=0.0, pos_y=0.0, pos_z=0.0, volume=np.nan),
        dict(id="G", object="basal",  ihc_label="1", object_id=2, pos_x=2.0, pos_y=0.0, pos_z=0.0, volume=np.nan),
        dict(id="G", object=cfg.pillar_obj,   ihc_label=np.nan, object_id=900, pos_x=1.0, pos_y=-1.0, pos_z=0.0, volume=np.nan),
        dict(id="G", object=cfg.modiolar_obj, ihc_label=np.nan, object_id=901, pos_x=1.0, pos_y= 1.0, pos_z=0.0, volume=np.nan),
        # synapses (ribbons) with volumes
        dict(id="G", object=cfg.ribbons_obj, ihc_label="1", object_id=10, pos_x=1.0, pos_y=-0.5, pos_z=0.0, volume=5.0),
        dict(id="G", object=cfg.ribbons_obj, ihc_label="1", object_id=11, pos_x=1.0, pos_y= 0.5, pos_z=0.0, volume=6.0),
    ]
    return pd.DataFrame(rows)


def test_classify_and_export(tmp_path):
    cfg = FinderConfig(
        folder=tmp_path,
        ribbons_obj="ribbons",
        psds_obj="PSDs",
        pillar_obj="pillar",
        modiolar_obj="modiolar",
        identify_poles=True,
    )
    df = _minimal_processed_df(cfg)

    # Optional relabel (should be stable here)
    df = identify_poles(df, cfg)

    pm = build_pm_planes(df, cfg)
    hc = build_hc_planes(df, cfg)

    out = classify_synapses(df, cfg, planes=pm, hc_planes=hc)

    # We expect both ribbons to be localized, with opposite signs along PM axis
    syn = out[out["object"] == cfg.ribbons_obj].copy()
    assert syn["localization"].notna().all()
    assert {"pillar", "modiolar"} == set(syn["localization"])
    # Pillar side should be negative, modiolar positive (per implementation)
    pm_vals = syn.set_index("localization")["pillar_modiolar_axis"]
    assert pm_vals["pillar"] < 0 and pm_vals["modiolar"] > 0

    # Export to CSV
    csv_path = export_df_csv(out, gid="G", out_dir=tmp_path)
    assert Path(csv_path).exists()
    text = Path(csv_path).read_text(encoding="utf-8")
    assert "pillar_modiolar_axis" in text and "habenular_cuticular_axis" in text
