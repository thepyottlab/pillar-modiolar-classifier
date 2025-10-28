"""Visualization of PM and HC planes and related point layers in napari."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any, cast
from weakref import WeakKeyDictionary

import napari
import numpy as np
import pandas as pd
from magicgui.widgets import CheckBox, Container, PushButton
from napari.layers import Layer, Points, Shapes, Vectors
from numpy.typing import NDArray

_VIEWER_STATE: WeakKeyDictionary[napari.Viewer, dict[str, Any]] = WeakKeyDictionary()


def _state_for(viewer: napari.Viewer) -> dict[str, Any]:
    """Return (and lazily create) a per-viewer state dict."""
    st = _VIEWER_STATE.get(viewer)
    if st is None:
        st = {"docks": []}
        _VIEWER_STATE[viewer] = st
    return st


def _cleanup_viewer(viewer: napari.Viewer) -> None:
    """Remove previously added docks and clear layers if possible."""
    st = _state_for(viewer)
    for w in list(st.get("docks", [])):
        with suppress(Exception):
            viewer.window.remove_dock_widget(w)
    st["docks"].clear()
    with suppress(Exception):
        viewer.layers.clear()


def draw_objects(
    df: pd.DataFrame,
    cfg: Any,
    pm_bundle: tuple[list[np.ndarray], list[str]] | None,
    hc_bundle: tuple[list[np.ndarray], list[str]] | None,
    viewer: napari.Viewer | None,
) -> napari.Viewer:
    """Render the dataset and PM/HC planes into a napari viewer.

    Args:
        df: Unified dataframe with points, volumes, and classification columns.
        cfg: Finder configuration (object names and flags).
        pm_bundle: Tuple (pm_planes, pm_labels) with 4×3 ZYX rectangles.
        hc_bundle: Tuple (hc_planes, hc_labels) with 4×3 ZYX rectangles.
        viewer: Optional viewer to reuse (cleared before redraw).

    Returns:
        napari.Viewer: The viewer containing the rendered scene.
    """
    if viewer is None:
        viewer = napari.Viewer(ndisplay=3)
    else:
        with suppress(Exception):
            viewer.dims.ndisplay = 3
        _cleanup_viewer(viewer)

    pm_planes: list[np.ndarray]
    pm_labels: list[str]
    if pm_bundle is None:
        pm_planes, pm_labels = [], []
    else:
        pm_planes, pm_labels = pm_bundle

    hc_planes: list[np.ndarray]
    hc_labels: list[str]
    if hc_bundle is None:
        hc_planes, hc_labels = [], []
    else:
        hc_planes, hc_labels = hc_bundle

    pm_label_to_idx: dict[str, int] = {}
    for i, label in enumerate(pm_labels):
        k = str(label).strip()
        if k and k not in pm_label_to_idx:
            pm_label_to_idx[k] = i

    hc_label_to_idx: dict[str, int] = {}
    for i, label in enumerate(hc_labels):
        k = str(label).strip()
        if k and k not in hc_label_to_idx:
            hc_label_to_idx[k] = i

    ribbons, psds = cfg.ribbons_obj, cfg.psds_obj

    coords = df[["pos_z", "pos_y", "pos_x"]].to_numpy(float)
    (zmin, ymin, xmin), (zmax, ymax, xmax) = coords.min(0), coords.max(0)
    xc, yc = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    side = 1.10 * max(xmax - xmin, ymax - ymin)
    half = side / 2.0
    xmin_sq, xmax_sq, ymin_sq, ymax_sq = xc - half, xc + half, yc - half, yc + half
    dz = zmax - zmin
    zmin_exp, zmax_exp = zmin - 0.05 * dz, zmax + 0.05 * dz
    box_edges = [
        [[zmin_exp, ymin_sq, xmin_sq], [zmin_exp, ymin_sq, xmax_sq]],
        [[zmin_exp, ymax_sq, xmin_sq], [zmin_exp, ymax_sq, xmax_sq]],
        [[zmax_exp, ymin_sq, xmin_sq], [zmax_exp, ymin_sq, xmax_sq]],
        [[zmax_exp, ymax_sq, xmin_sq], [zmax_exp, ymax_sq, xmax_sq]],
        [[zmin_exp, ymin_sq, xmin_sq], [zmin_exp, ymax_sq, xmin_sq]],
        [[zmin_exp, ymin_sq, xmax_sq], [zmin_exp, ymax_sq, xmax_sq]],
        [[zmax_exp, ymin_sq, xmin_sq], [zmax_exp, ymax_sq, xmin_sq]],
        [[zmax_exp, ymin_sq, xmax_sq], [zmax_exp, ymax_sq, xmax_sq]],
        [[zmin_exp, ymin_sq, xmin_sq], [zmax_exp, ymin_sq, xmin_sq]],
        [[zmin_exp, ymin_sq, xmax_sq], [zmax_exp, ymin_sq, xmax_sq]],
        [[zmin_exp, ymax_sq, xmin_sq], [zmax_exp, ymax_sq, xmin_sq]],
        [[zmin_exp, ymax_sq, xmax_sq], [zmax_exp, ymax_sq, xmax_sq]],
    ]
    viewer.add_shapes(
        box_edges,
        shape_type="path",
        edge_width=0.1,
        opacity=0.15,
        edge_color="white",
        name="Bounding box",
    )

    def hex_to_rgba(h: str) -> NDArray[np.float32]:
        """Convert #RRGGBB[AA] hex to RGBA float array."""
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
        a = int(h[6:8], 16) / 255 if len(h) == 8 else 1.0
        return np.array([r, g, b, a], np.float32)

    def solid(n: int, hex_color: str) -> NDArray[np.float32]:
        """Repeat the same RGBA color n times."""
        return np.tile(hex_to_rgba(hex_color), (n, 1))

    col_ribbon, col_psd = "#CB2027", "#059748"
    col_pillar, col_modiolar, col_gray, col_default = (
        "#9D722A",
        "#7B3A96",
        "#8C8C8C",
        "#FFFFFF",
    )
    col_white, col_yellow = "#FFFFFF", "#FFFF00"

    labels = sorted(
        [
            s
            for s in df["ihc_label"].dropna().astype(str).str.strip().unique()
            if s.lower() != "nan"
        ],
        key=lambda x: int(x),
    )

    max_vol_ribbon = df.loc[df["object"] == ribbons, "volume"].astype(float).max()
    max_vol_psd = df.loc[df["object"] == psds, "volume"].astype(float).max()
    if not np.isfinite(max_vol_ribbon):
        max_vol_ribbon = 1.0
    if not np.isfinite(max_vol_psd):
        max_vol_psd = 1.0

    def plane_normal(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (unit normal, anchor) for a rectangle polygon in ZYX space."""
        a0 = poly[0]
        u = poly[2] - poly[0]
        v = poly[1] - poly[0]
        n = np.cross(u, v)
        nn = np.linalg.norm(n)
        if nn == 0:
            return np.array([0.0, 0.0, 1.0], float), a0
        return n / nn, a0

    def pm_normal_for_label(lab_str: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Lookup PM plane normal by IHC label."""
        i = pm_label_to_idx.get(lab_str)
        if i is None or i >= len(pm_planes):
            return None
        return plane_normal(pm_planes[i])

    def hc_normal_for_label(lab_str: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Lookup HC plane normal by IHC label."""
        i = hc_label_to_idx.get(lab_str)
        if i is None or i >= len(hc_planes):
            return None
        return plane_normal(hc_planes[i])

    cb_pil_rib = CheckBox(text="Highlight pillar ribbons", value=False)
    cb_mod_rib = CheckBox(text="Highlight modiolar ribbons", value=False)
    cb_pil_psd = CheckBox(text="Highlight pillar PSDs", value=False)
    cb_mod_psd = CheckBox(text="Highlight modiolar PSDs", value=False)

    show_rib = not getattr(cfg, "psds_only", False)
    show_psd = not getattr(cfg, "ribbons_only", False)

    toggle_widgets = []
    if show_rib:
        toggle_widgets += [cb_pil_rib, cb_mod_rib]
    if show_psd:
        toggle_widgets += [cb_pil_psd, cb_mod_psd]

    def colors_for_rows(rows: pd.DataFrame, obj: str) -> NDArray[np.float32]:
        """Return RGBA colors for a subset of rows for a given object name."""
        n = len(rows)
        if obj == ribbons:
            c = solid(n, col_ribbon)
            loc = rows["localization"]
            if cb_pil_rib.value:
                c[loc.eq("pillar").to_numpy()] = hex_to_rgba(col_white)
            if cb_mod_rib.value:
                c[loc.eq("modiolar").to_numpy()] = hex_to_rgba(col_white)
            return c
        if obj == psds:
            c = solid(n, col_psd)
            loc = rows["localization"]
            if cb_pil_psd.value:
                c[loc.eq("pillar").to_numpy()] = hex_to_rgba(col_yellow)
            if cb_mod_psd.value:
                c[loc.eq("modiolar").to_numpy()] = hex_to_rgba(col_yellow)
            return c
        if obj == cfg.pillar_obj:
            return solid(n, col_pillar)
        if obj == cfg.modiolar_obj:
            return solid(n, col_modiolar)
        if obj in ("apical", "basal"):
            return solid(n, col_gray)
        return solid(n, col_default)

    def base_size_for(obj: str) -> float:
        """Base glyph size for a category (scaled later for volumes)."""
        if obj == "apical":
            return 3.0
        if obj in (cfg.pillar_obj, cfg.modiolar_obj, "basal"):
            return 2.0
        return 1.0

    def _coerce_pts(pts: Any) -> np.ndarray:
        pts = np.asarray(pts, float)
        if pts.ndim == 1:
            return pts.reshape((0, 3)) if pts.size == 0 else pts.reshape((1, 3))
        if pts.ndim != 2 or (pts.size > 0 and pts.shape[1] != 3):
            return np.zeros((0, 3), float)
        return pts

    def _coerce_sizes(sizes: Any, n: int) -> np.ndarray:
        if np.ndim(sizes) == 0:
            sz = float(sizes) if n > 0 else 0.0
            return np.full((n,), sz, float)
        arr = np.asarray(sizes, float)
        return arr.reshape((n,)) if n > 0 else np.array([], float)

    def _coerce_colors(colors: Any, n: int) -> np.ndarray:
        col = np.asarray(colors, np.float32)
        if col.ndim == 1 and col.size == 4:
            return np.tile(col, (n, 1))
        out = np.ones((n, 4), np.float32)
        if col.ndim == 2 and col.size > 0:
            m = min(n, col.shape[0])
            k = min(4, col.shape[1])
            if m:
                out[:m, :k] = col[:m, :k]
        return out

    def update_points(
        layer: Points,
        pts: NDArray[np.float64],
        sizes: Any,
        colors: NDArray[np.float32],
    ) -> None:
        """Safely update geometry first, then colors, avoiding stale view indices."""
        pts = _coerce_pts(pts)
        n = int(pts.shape[0])
        sizes = _coerce_sizes(sizes, n)
        colors = _coerce_colors(colors, n)

        layer.data = pts
        layer.size = sizes

        with suppress(Exception):
            layer._set_view_slice()
        layer.refresh()

        layer.face_color = colors
        layer.border_color = colors
        layer.refresh()

    def update_shapes(layer: Shapes, data_list: list[np.ndarray]) -> None:
        """Batch-update a shapes layer."""
        with layer.events.blocker_all():
            layer.data = data_list
        layer.refresh()

    def update_vectors(layer: Vectors | None, data: np.ndarray) -> None:
        """Batch-update a vectors layer."""
        if layer is None:
            return
        with layer.events.blocker_all():
            layer.data = data
        layer.refresh()

    def update_label_points(
        layer: Points | None,
        pos3: NDArray[np.float64] | np.ndarray,
        labels_txt: list[str] | None,
    ) -> None:
        """Batch-update a points layer used as stand-ins for text labels."""
        if layer is None:
            return
        pos3 = _coerce_pts(pos3)
        n = int(pos3.shape[0])
        if labels_txt is None:
            lab: list[str] = [""] * n
        else:

            def _to_label(x: object) -> str:
                if x is None:
                    return ""
                if isinstance(x, float | np.floating) and not np.isfinite(x):
                    return ""
                s = str(x)
                return "" if s.lower() in {"nan", "none"} else s

            lab = [_to_label(x) for x in labels_txt]
            if len(lab) != n:
                lab = lab[:n] if len(lab) > n else lab + [""] * (n - len(lab))
        prev_vis = bool(getattr(layer.text, "visible", True))
        with layer.events.blocker_all():
            with suppress(Exception):
                layer.text.visible = False
            layer.data = pos3
            cast(Any, layer.text).string = {"array": lab, "default": ""}
            with suppress(Exception):
                layer.text.visible = prev_vis
        layer.refresh()

    point_layers: dict[str, Any] = {}
    user_vis: dict[str, bool] = {}
    vis_prog = False

    def bind_user_vis(
        layer: Layer | None, key: str, seed_from_layer: bool = True
    ) -> None:
        """Track user-driven visibility changes for a layer under `key`."""
        if layer is None:
            return
        if seed_from_layer:
            user_vis[key] = bool(layer.visible)

        def _on_visible_change(
            _event: Any = None, _k: str = key, _layer: Layer = layer
        ) -> None:
            nonlocal vis_prog
            if vis_prog:
                return
            user_vis[_k] = bool(_layer.visible)

        layer.events.visible.connect(_on_visible_change)

    def track(layer: Layer | None, key: str) -> None:
        """Bind visibility and seed `user_vis` from the layer."""
        bind_user_vis(layer, key, seed_from_layer=True)

    pm_layer = viewer.add_shapes(
        pm_planes,
        shape_type="rectangle",
        edge_width=1,
        edge_color="transparent",
        face_color="#7a7a7aff",
        blending="translucent_no_depth",
        opacity=0.25,
        name="Habenular-cuticular axes",
        visible=True,
    )
    track(pm_layer, "__pm_planes__")

    hc_layer = viewer.add_shapes(
        hc_planes,
        shape_type="rectangle",
        edge_width=1,
        edge_color="transparent",
        face_color="#7a7a7aff",
        blending="translucent_no_depth",
        opacity=0.20,
        name="Pillar-modiolar axes",
        visible=True,
    )
    track(hc_layer, "__hc_planes__")

    distance_paths_ribbons_pm = distance_labels_ribbons_pm = None
    distance_paths_psds_pm = distance_labels_psds_pm = None
    distance_paths_ribbons_hc = distance_labels_ribbons_hc = None
    distance_paths_psds_hc = distance_labels_psds_hc = None

    dz_text = -0.015 * max(1e-9, (zmax - zmin))

    def make_label_cfg() -> dict[str, Any]:
        """Return a text config dict reused for distance label layers."""
        return {
            "string": "{label}",
            "size": 0.3,
            "color": "white",
            "anchor": "center",
            "translation": np.array([0.0, dz_text, 0.0], float),
            "scaling": True,
        }

    if show_rib:
        distance_paths_ribbons_pm = viewer.add_vectors(
            np.zeros((0, 2, 3), float),
            name="PM distance vectors (ribbons)",
            edge_width=0.2,
            edge_color="white",
            opacity=0.4,
            vector_style="triangle",
            blending="translucent_no_depth",
            visible=False,
        )
        track(distance_paths_ribbons_pm, "__pm_vec_r__")

        distance_labels_ribbons_pm = viewer.add_points(
            np.zeros((0, 3), float),
            name="PM distance labels (ribbons)",
            size=0.001,
            face_color="transparent",
            border_color="transparent",
            opacity=1.0,
            visible=False,
            features=pd.DataFrame({"label": []}),
            text=make_label_cfg(),
        )
        track(distance_labels_ribbons_pm, "__pm_lbl_r__")

        distance_paths_ribbons_hc = viewer.add_vectors(
            np.zeros((0, 2, 3), float),
            name="HC distance vectors (ribbons)",
            edge_width=0.2,
            edge_color="white",
            opacity=0.4,
            vector_style="triangle",
            blending="translucent_no_depth",
            visible=False,
        )
        track(distance_paths_ribbons_hc, "__hc_vec_r__")

        distance_labels_ribbons_hc = viewer.add_points(
            np.zeros((0, 3), float),
            name="HC distance labels (ribbons)",
            size=0.001,
            face_color="transparent",
            border_color="transparent",
            opacity=1.0,
            visible=False,
            features=pd.DataFrame({"label": []}),
            text=make_label_cfg(),
        )
        track(distance_labels_ribbons_hc, "__hc_lbl_r__")

    if show_psd:
        distance_paths_psds_pm = viewer.add_vectors(
            np.zeros((0, 2, 3), float),
            name="PM distance vectors (PSDs)",
            edge_width=0.2,
            edge_color="white",
            opacity=0.4,
            vector_style="triangle",
            blending="translucent_no_depth",
            visible=False,
        )
        track(distance_paths_psds_pm, "__pm_vec_p__")

        distance_labels_psds_pm = viewer.add_points(
            np.zeros((0, 3), float),
            name="PM distance labels (PSDs)",
            size=0.001,
            face_color="transparent",
            border_color="transparent",
            opacity=1.0,
            visible=False,
            features=pd.DataFrame({"label": []}),
            text=make_label_cfg(),
        )
        track(distance_labels_psds_pm, "__pm_lbl_p__")

        distance_paths_psds_hc = viewer.add_vectors(
            np.zeros((0, 2, 3), float),
            name="HC distance vectors (PSDs)",
            edge_width=0.2,
            edge_color="white",
            opacity=0.4,
            vector_style="triangle",
            blending="translucent_no_depth",
            visible=False,
        )
        track(distance_paths_psds_hc, "__hc_vec_p__")

        distance_labels_psds_hc = viewer.add_points(
            np.zeros((0, 3), float),
            name="HC distance labels (PSDs)",
            size=0.001,
            face_color="transparent",
            border_color="transparent",
            opacity=1.0,
            visible=False,
            features=pd.DataFrame({"label": []}),
            text=make_label_cfg(),
        )
        track(distance_labels_psds_hc, "__hc_lbl_p__")

    def set_vis(layer: Layer | None, key: str, has: bool) -> None:
        """Programmatically set layer visibility without clobbering user choice."""
        nonlocal vis_prog
        if layer is not None:
            vis_prog = True
            try:
                layer.visible = bool(user_vis.get(key, False) and has)
            finally:
                vis_prog = False

    for obj in list(df["object"].unique()):
        mask_obj = df["object"] == obj
        rows: pd.DataFrame = df.loc[mask_obj]
        pts = rows[["pos_z", "pos_y", "pos_x"]].to_numpy(float)
        base = base_size_for(str(obj))
        if obj == ribbons:
            sizes = base * (rows["volume"].astype(float).to_numpy() / max_vol_ribbon)
        elif obj == psds:
            sizes = base * (rows["volume"].astype(float).to_numpy() / max_vol_psd)
        else:
            sizes = base
        colors = colors_for_rows(rows, str(obj))

        _allowed_visible = {
            cfg.ribbons_obj,
            cfg.psds_obj,
            cfg.pillar_obj,
            cfg.modiolar_obj,
            "apical",
            "basal",
        }
        _initial_visible = obj in _allowed_visible

        layer = viewer.add_points(
            pts,
            size=sizes,
            border_color=colors,
            face_color=colors,
            name=str(obj),
            blending="translucent",
            visible=_initial_visible,
        )
        layer.metadata["object"] = obj
        point_layers[str(obj)] = layer
        track(layer, str(obj))

    checks = {lab: CheckBox(text=lab, value=True) for lab in labels}
    btn_all_on, btn_all_off = PushButton(text="Show all"), PushButton(text="Hide all")

    dock = Container(
        widgets=[*toggle_widgets, btn_all_on, btn_all_off, *checks.values()],
        layout="vertical",
        labels=False,
    ).native
    viewer.window.add_dock_widget(dock, name="Classification controls", area="right")
    _state_for(viewer)["docks"] = [dock]

    s = df["ihc_label"].astype(str).str.strip()
    unlabeled_mask = df["ihc_label"].isna() | s.isin(["", "nan", "None"])

    max_range = float(max(zmax - zmin, ymax - ymin, xmax - xmin))
    d_eps = 1e-6 * max_range if np.isfinite(max_range) and max_range > 0 else 1e-6

    def build_distance_for(
        obj_name: str,
        sel_labels: list[str],
        dist_col: str,
        normal_getter: Callable[[str], tuple[np.ndarray, np.ndarray] | None],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build vectors and midpoints for distances along a plane normal."""
        mask = (df["object"] == obj_name) & (
            df["ihc_label"].astype(str).isin(sel_labels)
        )
        rows = df.loc[mask]
        if rows.empty:
            return np.zeros((0, 2, 3), float), np.zeros((0, 3), float), []

        normals: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        for lab in rows["ihc_label"].astype(str).unique():
            normals[lab] = normal_getter(lab)

        vec_list: list[NDArray[np.float64]] = []
        mids: list[NDArray[np.float64]] = []
        txt: list[str] = []
        for _, r in rows.iterrows():
            key = str(r["ihc_label"]).strip()
            pn = normals.get(key)
            if pn is None:
                continue
            n_hat, _ = pn
            if not np.all(np.isfinite(n_hat)):
                continue

            p = np.array([r["pos_z"], r["pos_y"], r["pos_x"]], float)
            d = float(r.get(dist_col, np.nan))
            if not np.isfinite(d) or abs(d) < d_eps:
                continue

            v = -d * n_hat
            if not np.all(np.isfinite(v)) or np.linalg.norm(v) < d_eps:
                continue

            vec_list.append(np.stack([p, v], 0))
            mids.append(p + 0.5 * v)
            txt.append(f"{d:.2f}")

        if len(vec_list) == 0:
            return np.zeros((0, 2, 3), float), np.zeros((0, 3), float), []
        return np.stack(vec_list, 0), np.vstack(mids), txt

    point_layers_order = list(point_layers.items())
    batching = False

    current_df_points: pd.DataFrame | None = None

    def update_view(selected: list[str]) -> None:
        """Rebuild layers according to the selected labels."""
        nonlocal vis_prog, current_df_points

        empty_vec = np.zeros((0, 2, 3), float)
        empty_pts = np.zeros((0, 3), float)
        empty_txt: list[str] = []

        if not selected:
            df_points = df.loc[unlabeled_mask]
            idxs_pm: list[int] = []
            idxs_hc: list[int] = []
            vec_r_pm, mid_r_pm, txt_r_pm = empty_vec, empty_pts, empty_txt
            vec_p_pm, mid_p_pm, txt_p_pm = empty_vec, empty_pts, empty_txt
            vec_r_hc, mid_r_hc, txt_r_hc = empty_vec, empty_pts, empty_txt
            vec_p_hc, mid_p_hc, txt_p_hc = empty_vec, empty_pts, empty_txt
        else:
            labeled = df.loc[
                ~unlabeled_mask & df["ihc_label"].astype(str).str.strip().isin(selected)
            ]
            df_points = pd.concat([labeled, df.loc[unlabeled_mask]], ignore_index=True)

            idxs_pm = [
                pm_label_to_idx[label] for label in selected if label in pm_label_to_idx
            ]
            idxs_hc = [
                hc_label_to_idx[label] for label in selected if label in hc_label_to_idx
            ]

            if idxs_pm:
                update_shapes(pm_layer, [pm_planes[i] for i in idxs_pm])
            else:
                update_shapes(pm_layer, [])
            if idxs_hc:
                update_shapes(hc_layer, [hc_planes[i] for i in idxs_hc])
            else:
                update_shapes(hc_layer, [])

            vec_r_pm, mid_r_pm, txt_r_pm = (
                build_distance_for(
                    ribbons, selected, "pillar_modiolar_axis", pm_normal_for_label
                )
                if show_rib
                else (empty_vec, empty_pts, empty_txt)
            )
            vec_p_pm, mid_p_pm, txt_p_pm = (
                build_distance_for(
                    psds, selected, "pillar_modiolar_axis", pm_normal_for_label
                )
                if show_psd
                else (empty_vec, empty_pts, empty_txt)
            )
            vec_r_hc, mid_r_hc, txt_r_hc = (
                build_distance_for(
                    ribbons, selected, "habenular_cuticular_axis", hc_normal_for_label
                )
                if show_rib
                else (empty_vec, empty_pts, empty_txt)
            )
            vec_p_hc, mid_p_hc, txt_p_hc = (
                build_distance_for(
                    psds, selected, "habenular_cuticular_axis", hc_normal_for_label
                )
                if show_psd
                else (empty_vec, empty_pts, empty_txt)
            )

        current_df_points = df_points

        vis_prog = True
        try:
            pm_layer.visible = bool(
                user_vis.get("__pm_planes__", True) and bool(idxs_pm)
            )
            hc_layer.visible = bool(
                user_vis.get("__hc_planes__", True) and bool(idxs_hc)
            )
        finally:
            vis_prog = False

        for obj, layer in point_layers_order:
            mask_sel = df_points["object"] == obj
            sel: pd.DataFrame = df_points.loc[mask_sel]
            n = len(sel)
            if n > 0:
                pts = sel[["pos_z", "pos_y", "pos_x"]].to_numpy(float)
                base = base_size_for(str(obj))
                if obj == ribbons:
                    sizes = base * (
                        sel["volume"].astype(float).to_numpy() / max_vol_ribbon
                    )
                    colors = colors_for_rows(sel, str(obj))
                elif obj == psds:
                    sizes = base * (
                        sel["volume"].astype(float).to_numpy() / max_vol_psd
                    )
                    colors = colors_for_rows(sel, str(obj))
                else:
                    sizes = base
                    if obj == cfg.pillar_obj:
                        colors = solid(n, col_pillar)
                    elif obj == cfg.modiolar_obj:
                        colors = solid(n, col_modiolar)
                    elif obj in ("apical", "basal"):
                        colors = solid(n, col_gray)
                    else:
                        colors = solid(n, col_default)
                update_points(layer, pts, sizes, colors)
            vis_prog = True
            try:
                layer.visible = bool(user_vis.get(str(obj), True) and (n > 0))
            finally:
                vis_prog = False

        set_vis(distance_paths_ribbons_pm, "__pm_vec_r__", len(vec_r_pm) > 0)
        set_vis(distance_labels_ribbons_pm, "__pm_lbl_r__", len(vec_r_pm) > 0)
        set_vis(distance_paths_psds_pm, "__pm_vec_p__", len(vec_p_pm) > 0)
        set_vis(distance_labels_psds_pm, "__pm_lbl_p__", len(vec_p_pm) > 0)
        set_vis(distance_paths_ribbons_hc, "__hc_vec_r__", len(vec_r_hc) > 0)
        set_vis(distance_labels_ribbons_hc, "__hc_lbl_r__", len(vec_r_hc) > 0)
        set_vis(distance_paths_psds_hc, "__hc_vec_p__", len(vec_p_hc) > 0)
        set_vis(distance_labels_psds_hc, "__hc_lbl_p__", len(vec_p_hc) > 0)

        update_vectors(distance_paths_ribbons_pm, vec_r_pm)
        update_vectors(distance_paths_psds_pm, vec_p_pm)
        update_vectors(distance_paths_ribbons_hc, vec_r_hc)
        update_vectors(distance_paths_psds_hc, vec_p_hc)

        update_label_points(distance_labels_ribbons_pm, mid_r_pm, txt_r_pm)
        update_label_points(distance_labels_psds_pm, mid_p_pm, txt_p_pm)
        update_label_points(distance_labels_ribbons_hc, mid_r_hc, txt_r_hc)
        update_label_points(distance_labels_psds_hc, mid_p_hc, txt_p_hc)

    def recolor_layer_for_rows(layer: Points, rows: pd.DataFrame, obj: str) -> None:
        """Only recolor an existing points layer; do not change geometry."""
        if not hasattr(layer, "data"):
            return
        n = len(getattr(layer, "data", []))
        if n == 0:
            return
        cols = colors_for_rows(rows, obj)
        cols = _coerce_colors(cols, n)
        if cols.shape != (n, 4):
            return
        layer.face_color = cols
        layer.border_color = cols
        layer.refresh()

    def recolor_only() -> None:
        """Recompute colors for current selection without touching geometry."""
        if current_df_points is None:
            return
        dfp = current_df_points
        for obj, layer in point_layers_order:
            rows = dfp.loc[dfp["object"] == obj]
            if len(rows) == len(getattr(layer, "data", [])):
                recolor_layer_for_rows(layer, rows, str(obj))

    def apply() -> None:
        """Apply current label selection (rebuild geometry and distances)."""
        if not batching:
            update_view([lab for lab, cb in checks.items() if cb.value])

    for cb in checks.values():
        cb.changed.connect(apply)

    if show_rib:
        cb_pil_rib.changed.connect(recolor_only)
        cb_mod_rib.changed.connect(recolor_only)
    if show_psd:
        cb_pil_psd.changed.connect(recolor_only)
        cb_mod_psd.changed.connect(recolor_only)

    def set_all(val: bool) -> None:
        """Set all label checkboxes to the same value and refresh."""
        nonlocal batching
        batching = True
        for cb in checks.values():
            cb.value = val
        batching = False
        apply()

    btn_all_on.clicked.connect(lambda: set_all(True))
    btn_all_off.clicked.connect(lambda: set_all(False))

    apply()
    viewer.fit_to_view()
    return viewer
