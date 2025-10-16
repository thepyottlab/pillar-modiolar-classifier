from __future__ import annotations

import numpy as np
import pandas as pd

from pmc_app.classifier import build_planes, classify_synapses
from pmc_app.models import FinderConfig


def _df_simple():
    # Two synapses on opposite sides; apical/basal along z, pillar/modiolar anchors
    data = {
        "id": ["G1"] * 8,
        "object": ["apical", "basal", "pillar", "modiolar", "ribbons", "PSDs", "ribbons", "PSDs"],
        "ihc_label": ["1", "1", None, None, "1", "1", "1", "1"],
        "object_id": [1, 2, 99, 98, 10, 11, 12, 13],
        "pos_x": [0, 0, -1, 1, -0.5, -0.4, 0.5, 0.4],
        "pos_y": [0, 0, 0, 0, 0, 0, 0, 0],
        "pos_z": [0.0, 1.0, 0, 0, 0.5, 0.6, 0.5, 0.6],
        "volume": [np.nan] * 8,
    }
    return pd.DataFrame(data)


def test_build_planes_and_classify_localization():
    df = _df_simple()
    cfg = FinderConfig(folder=".", ribbons_obj="ribbons", psds_obj="PSDs", ihc_obj="IHC", pillar_obj="pillar", modiolar_obj="modiolar")
    planes = build_planes(df, cfg)
    assert len(planes) == 1
    out = classify_synapses(df, cfg, planes)
    loc = out.loc[out["object"].isin(["ribbons", "PSDs"]), "localization"].astype(str).tolist()
    assert set(loc) <= {"pillar", "modiolar", "nan"}  # ensure labels resolved
    assert np.all(np.isfinite(out.loc[out["object"].isin(["ribbons", "PSDs"]), "dist_to_plane"]))
