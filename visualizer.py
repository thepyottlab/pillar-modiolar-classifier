"""Visualization helpers for the pillar-modiolar classifier demo."""

from __future__ import annotations

import napari
import numpy as np
import pandas as pd

from models import FinderConfig


def _compute_polygon(
    df: pd.DataFrame,
    label: str,
    apical_row: pd.Series,
    basal_row: pd.Series,
    ribbon_label: str,
    psd_label: str,
) -> np.ndarray:
    """Compute the rectangle polygon for the pillar-modiolar plane."""

    x1, y1, z1 = float(apical_row["pos_x"]), float(apical_row["pos_y"]), float(apical_row["pos_z"])
    x2, y2, z2 = float(basal_row["pos_x"]), float(basal_row["pos_y"]), float(basal_row["pos_z"])

    vx, vy = x2 - x1, y2 - y1
    norm = float(np.hypot(vx, vy))
    px, py = (-vy / norm, vx / norm) if norm > 1e-9 else (0.0, 1.0)

    group = df[(df["ihc_label"] == label) & (df["object"].isin([ribbon_label, psd_label]))]
    gxy = group[["pos_x", "pos_y"]].to_numpy(dtype=float)

    if norm <= 1e-9 and len(gxy):
        dx = float(gxy[:, 0].ptp())
        dy = float(gxy[:, 1].ptp())
        if dx >= dy and dx > 1e-9:
            px, py = 0.0, 1.0
        elif dy > 1e-9:
            px, py = 1.0, 0.0

    apxy = np.array([x1, y1], float)
    dots = (gxy - apxy).dot(np.array([px, py], float))
    if len(dots):
        min_dot = float(dots.min())
        max_dot = float(dots.max())
    else:
        min_dot = max_dot = 0.0

    c1x, c1y = x1 + px * max_dot, y1 + py * max_dot
    c2x, c2y = x1 + px * min_dot, y1 + py * min_dot
    c3x, c3y = x2 + px * min_dot, y2 + py * min_dot
    c4x, c4y = x2 + px * max_dot, y2 + py * max_dot

    return np.array([[z1, c1y, c1x], [z1, c2y, c2x], [z2, c3y, c3x], [z2, c4y, c4x]], float)


def draw_objects(df: pd.DataFrame, cfg: FinderConfig) -> napari.Viewer:
    """Render the dataset into a 3D Napari viewer."""

    viewer = napari.Viewer(ndisplay=3)

    apical_basal = df[df["object"].isin(["apical", "basal"])][["ihc_label", "object"]]
    matched_labels = set(apical_basal.groupby("ihc_label")["object"].nunique().loc[lambda series: series >= 2].index)

    polygons: list[np.ndarray] = []
    for label in matched_labels:
        apical_row = df[(df["ihc_label"] == label) & (df["object"] == "apical")].iloc[0]
        basal_row = df[(df["ihc_label"] == label) & (df["object"] == "basal")].iloc[0]
        polygons.append(
            _compute_polygon(df, label, apical_row, basal_row, cfg.ribbons_obj, cfg.psds_obj)
        )

    if polygons:
        viewer.add_shapes(
            polygons,
            shape_type="rectangle",
            edge_width=0.1,
            edge_color="white",
            face_color="white",
            opacity=0.3,
            name="Pillar-modiolar plane",
        )

    palette = {
        cfg.ribbons_obj: "#CB2027",
        cfg.psds_obj: "#059748",
        cfg.ihc_obj: "#8C8C8C",
        cfg.pillar_obj: "#9D722A",
        cfg.modiolar_obj: "#7B3A96",
        "apical": "#256DAB",
        "basal": "#265DAB",
    }

    for obj in df["object"].unique():
        subset = df[df["object"] == obj]
        coords = subset[["pos_z", "pos_y", "pos_x"]].to_numpy(dtype=float)

        if obj == cfg.ihc_obj:
            size = 3
            visible = False
        elif obj in (cfg.pillar_obj, cfg.modiolar_obj, "apical", "basal"):
            size = 2
            visible = True
        else:
            size = 1
            visible = True

        viewer.add_points(
            coords,
            size=size,
            face_color=palette.get(obj, "white"),
            name=str(obj),
            blending="additive",
            visible=visible,
        )

    return viewer
