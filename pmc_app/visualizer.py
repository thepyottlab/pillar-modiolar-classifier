"""Visualization of PM and HC planes and related point layers in napari."""

from __future__ import annotations

from weakref import WeakKeyDictionary
from typing import Any

import napari
import numpy as np
import pandas as pd
from magicgui.widgets import CheckBox, Container, PushButton
from numpy.typing import NDArray

_VIEWER_STATE = WeakKeyDictionary()


def _state_for(viewer: napari.Viewer):
    """Return (and lazily create) a per-viewer state dict."""
    st = _VIEWER_STATE.get(viewer)
    if st is None:
        st = {"docks": []}
        _VIEWER_STATE[viewer] = st
    return st


def _cleanup_viewer(viewer: napari.Viewer):
    """Remove previously added docks and clear layers if possible."""
    st = _state_for(viewer)
    for w in list(st.get("docks", [])):
        try:
            viewer.window.remove_dock_widget(w)
        except Exception:
            pass
    st["docks"].clear()
    try:
        viewer.layers.clear()
    except Exception:
        pass


def draw_objects(
    df: pd.DataFrame,
    cfg,
    pm_bundle: tuple[list[np.ndarray], list[str]] | None,
    hc_bundle: tuple[list[np.ndarray], list[str]] | None = None,
    viewer: napari.Viewer | None = None,
):
    """Render the dataset and PM/HC planes into a napari viewer.

    Args:
        df: Unified dataframe with points, volumes, and classification columns.
        cfg: Finder configuration (object names and flags).
        pm_bundle: Tuple ``(pm_planes, pm_labels)`` with 4×3 ZYX rectangles.
        hc_bundle: Tuple ``(hc_planes, hc_labels)`` with 4×3 ZYX rectangles.
        viewer: Optional viewer to reuse (cleared before redraw).

    Returns:
        napari.Viewer: The viewer containing the rendered scene.
    """
    if viewer is None:
        viewer = napari.Viewer(ndisplay=3)
    else:
        try:
            viewer.dims.ndisplay = 3
        except Exception:
            pass
        _cleanup_viewer(viewer)

    if pm_bundle is None:
        pm_planes, pm_labels = [], []
    else:
        pm_planes, pm_labels = pm_bundle

    if hc_bundle is None:
        hc_planes, hc_labels = [], []
    else:
        hc_planes, hc_labels = hc_bundle

    pm_label_to_idx: dict[str, int] = {}
    for i, l in enumerate(pm_labels):
        k = str(l).strip()
        if k and k not in pm_label_to_idx:
            pm_label_to_idx[k] = i

    hc_label_to_idx: dict[str, int] = {}
    for i, l in enumerate(hc_labels):
        k = str(l).strip()
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

    COL_RIBBON, COL_PSD = "#CB2027", "#059748"
    COL_PILLAR, COL_MODIOLAR, COL_GRAY, COL_DEFAULT = "#9D722A", "#7B3A96", "#8C8C8C", "#FFFFFF"
    COL_WHITE, COL_YELLOW = "#FFFFFF", "#FFFF00"

    labels = sorted(
        [
            s
            for s in df["ihc_label"].dropna().astype(str).str.strip().unique()
            if s.lower() not in {"nan", "none", ""}
        ]
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
        U = poly[2] - poly[0]
        V = poly[1] - poly[0]
        n = np.cross(U, V)
        nn = np.linalg.norm(n)
        if nn == 0:
            return np.array([0.0, 0.0, 1.0], float), a0
        return n / nn, a0

    def pm_normal_for_label(lab_str: str):
        """Lookup PM plane normal by IHC label."""
        i = pm_label_to_idx.get(lab_str)
        if i is None or i >= len(pm_planes):
            return None
        return plane_normal(pm_planes[i])

    def hc_normal_for_label(lab_str: str):
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
        n = int(len(rows))
        if obj == ribbons:
            c = solid(n, COL_RIBBON)
            loc = rows["localization"]
            if cb_pil_rib.value:
                c[loc.eq("pillar").to_numpy()] = hex_to_rgba(COL_WHITE)
            if cb_mod_rib.value:
                c[loc.eq("modiolar").to_numpy()] = hex_to_rgba(COL_WHITE)
            return c
        if obj == psds:
            c = solid(n, COL_PSD)
            loc = rows["localization"]
            if cb_pil_psd.value:
                c[loc.eq("pillar").to_numpy()] = hex_to_rgba(COL_YELLOW)
            if cb_mod_psd.value:
                c[loc.eq("modiolar").to_numpy()] = hex_to_rgba(COL_YELLOW)
            return c
        if obj == cfg.pillar_obj:
            return solid(n, COL_PILLAR)
        if obj == cfg.modiolar_obj:
            return solid(n, COL_MODIOLAR)
        if obj in ("apical", "basal"):
            return solid(n, COL_GRAY)
        return solid(n, COL_DEFAULT)

    def base_size_for(obj: str) -> float:
        """Base glyph size for a category (scaled later for volumes)."""
        if obj == "apical":
            return 3.0
        if obj in (cfg.pillar_obj, cfg.modiolar_obj, "basal"):
            return 2.0
        return 1.0

    def update_points(layer, pts: NDArray[np.float64], sizes, colors: NDArray[np.float32]):
        """Batch-update a points layer without intermediate repaints."""
        with layer.events.blocker_all():
            layer.data = pts
            layer.size = sizes
            layer.face_color = colors
            layer.border_color = colors
        layer.refresh()

    def update_shapes(layer, data_list):
        """Batch-update a shapes layer."""
        with layer.events.blocker_all():
            layer.data = data_list
        layer.refresh()

    def update_vectors(layer, data):
        """Batch-update a vectors layer."""
        if layer is None:
            return
        with layer.events.blocker_all():
            layer.data = data
        layer.refresh()

    def update_label_points(layer, pos3, labels_txt):
        """Batch-update a points layer used as stand-ins for text labels."""
        if layer is None:
            return
        pos3 = np.asarray(pos3, float)
        if pos3.ndim == 1:
            pos3 = pos3.reshape((0, 3)) if pos3.size == 0 else pos3.reshape((1, 3))
        elif pos3.ndim != 2 or (pos3.size > 0 and pos3.shape[1] != 3):
            pos3 = np.zeros((0, 3), float)
        n = int(pos3.shape[0])
        if labels_txt is None:
            lab = [""] * n
        else:
            lab = [
                "" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else str(x)
                for x in labels_txt
            ]
            if len(lab) != n:
                lab = lab[:n] if len(lab) > n else lab + [""] * (n - len(lab))
        prev_vis = bool(getattr(layer.text, "visible", True))
        with layer.events.blocker_all():
            try:
                layer.text.visible = False
            except Exception:
                pass
            layer.data = pos3
            layer.text.string = {'array':lab}
            try:
                layer.text.visible = prev_vis
            except Exception:
                pass
        layer.refresh()

    point_layers: dict[str, Any] = {}
    user_vis: dict[str, bool] = {}
    vis_prog = False

    def bind_user_vis(layer, key: str):
        """Track user-driven visibility changes for a layer under `key`."""
        if layer is None:
            return

        def _on_visible_change(event=None, _k=key, _layer=layer):
            nonlocal vis_prog
            if vis_prog:
                return
            user_vis[_k] = bool(_layer.visible)

        layer.events.visible.connect(_on_visible_change)

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
    bind_user_vis(pm_layer, "__pm_planes__")
    user_vis["__pm_planes__"] = True

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
    bind_user_vis(hc_layer, "__hc_planes__")
    user_vis["__hc_planes__"] = True

    distance_paths_ribbons_pm = distance_labels_ribbons_pm = None
    distance_paths_psds_pm = distance_labels_psds_pm = None
    distance_paths_ribbons_hc = distance_labels_ribbons_hc = None
    distance_paths_psds_hc = distance_labels_psds_hc = None

    dz_text = -0.015 * max(1e-9, (zmax - zmin))

    def make_label_cfg():
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

    for key in [
        "__pm_vec_r__",
        "__pm_lbl_r__",
        "__pm_vec_p__",
        "__pm_lbl_p__",
        "__hc_vec_r__",
        "__hc_lbl_r__",
        "__hc_vec_p__",
        "__hc_lbl_p__",
    ]:
        user_vis[key] = False

    def bind_all_vis():
        """Bind visibility trackers for all dynamic layers."""
        bind_user_vis(distance_paths_ribbons_pm, "__pm_vec_r__")
        bind_user_vis(distance_labels_ribbons_pm, "__pm_lbl_r__")
        bind_user_vis(distance_paths_psds_pm, "__pm_vec_p__")
        bind_user_vis(distance_labels_psds_pm, "__pm_lbl_p__")
        bind_user_vis(distance_paths_ribbons_hc, "__hc_vec_r__")
        bind_user_vis(distance_labels_ribbons_hc, "__hc_lbl_r__")
        bind_user_vis(distance_paths_psds_hc, "__hc_vec_p__")
        bind_user_vis(distance_labels_psds_hc, "__hc_lbl_p__")

    bind_all_vis()

    for obj in [o for o in df["object"].unique()]:
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

        _allowed_visible = {cfg.ribbons_obj, cfg.psds_obj, cfg.pillar_obj, cfg.modiolar_obj, "apical", "basal"}
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
        user_vis[str(obj)] = _initial_visible
        bind_user_vis(layer, str(obj))

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

    def build_distance_for(obj_name: str, sel_labels, dist_col: str, normal_getter):
        """Build vectors and midpoints for distances along a plane normal."""
        mask = (df["object"] == obj_name) & (df["ihc_label"].astype(str).isin(sel_labels))
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

            P = np.array([r["pos_z"], r["pos_y"], r["pos_x"]], float)
            d = float(r.get(dist_col, np.nan))
            if not np.isfinite(d) or abs(d) < d_eps:
                continue

            V = -d * n_hat
            if not np.all(np.isfinite(V)) or np.linalg.norm(V) < d_eps:
                continue

            vec_list.append(np.stack([P, V], 0))
            mids.append(P + 0.5 * V)
            txt.append(f"{d:.2f}")

        if len(vec_list) == 0:
            return np.zeros((0, 2, 3), float), np.zeros((0, 3), float), []
        return np.stack(vec_list, 0), np.vstack(mids), txt

    point_layers_order = list(point_layers.items())
    batching = False

    def selected_labels_now():
        """Return current set of selected IHC labels in the dock."""
        return [lab for lab, cb in checks.items() if cb.value]

    def update_view(selected):
        """Rebuild layers according to the selected labels."""
        nonlocal vis_prog

        EMPTY_VEC = np.zeros((0, 2, 3), float)
        EMPTY_PTS = np.zeros((0, 3), float)
        EMPTY_TXT: list[str] = []

        if not selected:
            df_points = df.loc[unlabeled_mask]
            idxs_pm: list[int] = []
            idxs_hc: list[int] = []
            vec_r_pm, mid_r_pm, txt_r_pm = EMPTY_VEC, EMPTY_PTS, EMPTY_TXT
            vec_p_pm, mid_p_pm, txt_p_pm = EMPTY_VEC, EMPTY_PTS, EMPTY_TXT
            vec_r_hc, mid_r_hc, txt_r_hc = EMPTY_VEC, EMPTY_PTS, EMPTY_TXT
            vec_p_hc, mid_p_hc, txt_p_hc = EMPTY_VEC, EMPTY_PTS, EMPTY_TXT
        else:
            labeled = df.loc[~unlabeled_mask & df["ihc_label"].astype(str).str.strip().isin(selected)]
            df_points = pd.concat([labeled, df.loc[unlabeled_mask]], ignore_index=True)

            idxs_pm = [pm_label_to_idx[l] for l in selected if l in pm_label_to_idx]
            idxs_hc = [hc_label_to_idx[l] for l in selected if l in hc_label_to_idx]

            if idxs_pm:
                update_shapes(pm_layer, [pm_planes[i] for i in idxs_pm])
            if idxs_hc:
                update_shapes(hc_layer, [hc_planes[i] for i in idxs_hc])

            vec_r_pm, mid_r_pm, txt_r_pm = (
                build_distance_for(ribbons, selected, "pillar_modiolar_axis", pm_normal_for_label)
                if show_rib
                else (EMPTY_VEC, EMPTY_PTS, EMPTY_TXT)
            )
            vec_p_pm, mid_p_pm, txt_p_pm = (
                build_distance_for(psds, selected, "pillar_modiolar_axis", pm_normal_for_label)
                if show_psd
                else (EMPTY_VEC, EMPTY_PTS, EMPTY_TXT)
            )
            vec_r_hc, mid_r_hc, txt_r_hc = (
                build_distance_for(ribbons, selected, "habenular_cuticular_axis", hc_normal_for_label)
                if show_rib
                else (EMPTY_VEC, EMPTY_PTS, EMPTY_TXT)
            )
            vec_p_hc, mid_p_hc, txt_p_hc = (
                build_distance_for(psds, selected, "habenular_cuticular_axis", hc_normal_for_label)
                if show_psd
                else (EMPTY_VEC, EMPTY_PTS, EMPTY_TXT)
            )

        vis_prog = True
        pm_layer.visible = bool(user_vis.get("__pm_planes__", True) and bool(idxs_pm))
        hc_layer.visible = bool(user_vis.get("__hc_planes__", True) and bool(idxs_hc))
        vis_prog = False

        for obj, layer in point_layers_order:
            mask_sel = df_points["object"] == obj
            sel: pd.DataFrame = df_points.loc[mask_sel]
            n = len(sel)
            if n > 0:
                pts = sel[["pos_z", "pos_y", "pos_x"]].to_numpy(float)
                base = base_size_for(str(obj))
                if obj == ribbons:
                    sizes = base * (sel["volume"].astype(float).to_numpy() / max_vol_ribbon)
                    colors = colors_for_rows(sel, str(obj))
                elif obj == psds:
                    sizes = base * (sel["volume"].astype(float).to_numpy() / max_vol_psd)
                    colors = colors_for_rows(sel, str(obj))
                else:
                    sizes = base
                    if obj == cfg.pillar_obj:
                        colors = solid(n, COL_PILLAR)
                    elif obj == cfg.modiolar_obj:
                        colors = solid(n, COL_MODIOLAR)
                    elif obj in ("apical", "basal"):
                        colors = solid(n, COL_GRAY)
                    else:
                        colors = solid(n, COL_DEFAULT)
                update_points(layer, pts, sizes, colors)
            vis_prog = True
            layer.visible = bool(user_vis.get(str(obj), True) and (n > 0))
            vis_prog = False

        def set_vis(layer, key, has):
            if layer is not None:
                layer.visible = bool(user_vis.get(key, False) and has)

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

    def apply():
        """Apply current toggles if not in a batch operation."""
        if not batching:
            update_view([lab for lab, cb in checks.items() if cb.value])

    for cb in checks.values():
        cb.changed.connect(apply)
    if show_rib:
        cb_pil_rib.changed.connect(apply)
        cb_mod_rib.changed.connect(apply)
    if show_psd:
        cb_pil_psd.changed.connect(apply)
        cb_mod_psd.changed.connect(apply)

    def set_all(val: bool):
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
