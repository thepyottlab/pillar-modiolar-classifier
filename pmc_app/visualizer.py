from __future__ import annotations

"""
Visualization module for Pillar–Modiolar Classifier.

Builds the Napari scene: points, planes, distance vectors, labels, and a dock
panel for interactive recoloring and per-IHC selection.
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Applying the encoding failed\. Using the safe fallback value instead\.",
    module="napari.layers.utils.style_encoding",
)

import napari
import numpy as np
import pandas as pd
from magicgui.widgets import Container, CheckBox, PushButton
from weakref import WeakKeyDictionary


_VIEWER_STATE = WeakKeyDictionary()


def _state_for(viewer: napari.Viewer):
    st = _VIEWER_STATE.get(viewer)
    if st is None:
        st = {"docks": []}
        _VIEWER_STATE[viewer] = st
    return st


def _cleanup_viewer(viewer: napari.Viewer):
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


def draw_objects(df: pd.DataFrame, cfg, planes, viewer: napari.Viewer | None = None):
    """
    Render the dataset into a Napari viewer with interactive controls.
    If an existing viewer is provided, it will be cleared and redrawn in-place.

    Args:
        df: Unified dataframe containing positions, volumes, classifications.
        cfg: FinderConfig-like object with ribbons/psds/anchor object names.
        planes: List of per-IHC finite rectangles in ZYX coords.
        viewer: Optional existing napari.Viewer to reuse.

    Returns:
        A configured Napari viewer instance.
    """
    if viewer is None:
        viewer = napari.Viewer(ndisplay=3)
    else:
        try:
            viewer.dims.ndisplay = 3
        except Exception:
            pass
        _cleanup_viewer(viewer)

    ribbons, psds = cfg.ribbons_obj, cfg.psds_obj

    max_vol_ribbon = df.loc[df["object"] == ribbons, "volume"].astype(float).max()
    max_vol_psd = df.loc[df["object"] == psds, "volume"].astype(float).max()
    if not np.isfinite(max_vol_ribbon) or max_vol_ribbon <= 0:
        max_vol_ribbon = 1.0
    if not np.isfinite(max_vol_psd) or max_vol_psd <= 0:
        max_vol_psd = 1.0

    def hex_to_rgba(h: str) -> np.ndarray:
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
        a = int(h[6:8], 16) / 255 if len(h) == 8 else 1.0
        return np.array([r, g, b, a], np.float32)

    def solid(n: int, hex_color: str) -> np.ndarray:
        return np.tile(hex_to_rgba(hex_color), (n, 1))

    COL_RIBBON, COL_PSD = "#CB2027", "#059748"
    COL_PILLAR, COL_MODIOLAR, COL_GRAY, COL_DEFAULT = "#9D722A", "#7B3A96", "#8C8C8C", "#FFFFFF"
    COL_WHITE, COL_YELLOW = "#FFFFFF", "#FFFF00"

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
        box_edges, shape_type="path", edge_width=0.1, opacity=0.15, edge_color="white", name="Bounding box"
    )

    ab = df[df["object"].isin(["apical", "basal"])][["ihc_label", "object"]]
    matched = set(ab.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index)

    label_info = {}
    for lab in matched:
        ap = df[(df["ihc_label"] == lab) & (df["object"] == "apical")].iloc[0]
        ba = df[(df["ihc_label"] == lab) & (df["object"] == "basal")].iloc[0]
        k = str(lab).strip()
        label_info[k] = {
            "ap": np.array([float(ap["pos_z"]), float(ap["pos_y"]), float(ap["pos_x"])]),
            "ba": np.array([float(ba["pos_z"]), float(ba["pos_y"]), float(ba["pos_x"])]),
        }

    tol = 1e-6
    label_to_plane_idx, used = {}, set()
    for k, rec in label_info.items():
        apz, baz = rec["ap"][0], rec["ba"][0]
        best_i, best_d = None, np.inf
        for i, poly in enumerate(planes):
            if i in used:
                continue
            zuniq = np.unique(poly[:, 0])
            if len(zuniq) != 2:
                continue
            ok = (abs(zuniq[0] - apz) < tol and abs(zuniq[1] - baz) < tol) or (
                abs(zuniq[1] - apz) < tol and abs(zuniq[0] - baz) < tol
            )
            if not ok:
                continue
            d = np.linalg.norm(poly[np.isclose(poly[:, 0], apz, atol=tol)] - rec["ap"], axis=1).min()
            if d < best_d:
                best_d, best_i = d, i
        label_to_plane_idx[k] = best_i
        if best_i is not None:
            used.add(best_i)

    def plane_normal(poly: np.ndarray):
        a0 = poly[0]
        U = poly[2] - poly[0]
        V = poly[1] - poly[0]
        n = np.cross(U, V)
        return n / np.linalg.norm(n), a0

    def plane_normal_for_label(lab_str: str):
        i = label_to_plane_idx.get(lab_str)
        if i is None:
            return None
        return plane_normal(planes[i])

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

    def colors_for_rows(rows: pd.DataFrame, obj: str) -> np.ndarray:
        n = len(rows)
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
        if obj == "apical":
            return 3
        if obj in (cfg.pillar_obj, cfg.modiolar_obj, "basal"):
            return 2
        return 1

    def update_points(layer, pts, sizes, colors):
        with layer.events.blocker_all():
            layer.data = pts
            layer.size = sizes
            layer.face_color = colors
            layer.border_color = colors
        layer.refresh()

    def update_shapes(layer, data_list):
        with layer.events.blocker_all():
            layer.data = data_list
        layer.refresh()

    def update_vectors(layer, data):
        if layer is None:
            return
        with layer.events.blocker_all():
            layer.data = data
        layer.refresh()

    def update_label_points(layer, pos3, labels):
        if layer is None:
            return
        pos3 = np.asarray(pos3, float)
        if pos3.ndim == 1:
            pos3 = pos3.reshape((0, 3)) if pos3.size == 0 else pos3.reshape((1, 3))
        elif pos3.ndim != 2 or (pos3.size > 0 and pos3.shape[1] != 3):
            pos3 = np.zeros((0, 3), float)
        n = int(pos3.shape[0])
        if labels is None:
            lab = [""] * n
        else:
            lab = [
                "" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else str(x)
                for x in labels
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
            if n == 0:
                layer.features = pd.DataFrame({"label": []})
                layer.text.string = ""
            else:
                layer.features = pd.DataFrame({"label": lab})
                layer.text.string = "{label}"
            try:
                layer.text.visible = prev_vis
            except Exception:
                pass
        layer.refresh()

    point_layers, user_vis, vis_prog = {}, {}, False

    def bind_user_vis(layer, key: str):
        if layer is None:
            return
        def _on_visible_change(event=None, _k=key, _layer=layer):
            nonlocal vis_prog
            if vis_prog:
                return
            user_vis[_k] = bool(_layer.visible)
        layer.events.visible.connect(_on_visible_change)

    plane_layer = viewer.add_shapes(
        planes,
        shape_type="rectangle",
        edge_width=1,
        edge_color="transparent",
        face_color="white",
        blending="translucent_no_depth",
        opacity=0.25,
        name="Pillar-modiolar planes",
        visible=True,
    )
    user_vis["__planes__"] = bool(plane_layer.visible)
    bind_user_vis(plane_layer, "__planes__")

    distance_paths_ribbons = None
    distance_labels_ribbons = None
    distance_paths_psds = None
    distance_labels_psds = None

    if show_rib:
        distance_paths_ribbons = viewer.add_vectors(
            np.zeros((0, 2, 3), float),
            name="Distance paths ribbons",
            edge_width=0.2,
            edge_color="white",
            opacity=0.4,
            vector_style="triangle",
            blending="translucent_no_depth",
            visible=False,
        )
        dz_text = -0.015 * max(1e-9, (zmax - zmin))
        label_text_cfg_r = {
            "string": "{label}",
            "size": 0.3,
            "color": "white",
            "anchor": "center",
            "translation": np.array([0.0, dz_text, 0.0], float),
            "scaling": True,
        }
        distance_labels_ribbons = viewer.add_points(
            np.zeros((0, 3), float),
            name="Distance labels ribbons",
            size=0.001,
            face_color="transparent",
            border_color="transparent",
            opacity=1.0,
            visible=False,
            features=pd.DataFrame({"label": []}),
            text=label_text_cfg_r,
        )

    if show_psd:
        distance_paths_psds = viewer.add_vectors(
            np.zeros((0, 2, 3), float),
            name="Distance paths psds",
            edge_width=0.2,
            edge_color="white",
            opacity=0.4,
            vector_style="triangle",
            blending="translucent_no_depth",
            visible=False,
        )
        dz_text = -0.015 * max(1e-9, (zmax - zmin))
        label_text_cfg_p = {
            "string": "{label}",
            "size": 0.3,
            "color": "white",
            "anchor": "center",
            "translation": np.array([0.0, dz_text, 0.0], float),
            "scaling": True,
        }
        distance_labels_psds = viewer.add_points(
            np.zeros((0, 3), float),
            name="Distance labels psds",
            size=0.001,
            face_color="transparent",
            border_color="transparent",
            opacity=1.0,
            visible=False,
            features=pd.DataFrame({"label": []}),
            text=label_text_cfg_p,
        )

    if distance_paths_ribbons is not None:
        user_vis["__vec_ribbons__"] = False
    if distance_paths_psds is not None:
        user_vis["__vec_psds__"] = False
    if distance_labels_ribbons is not None:
        user_vis["__lbl_ribbons__"] = False
    if distance_labels_psds is not None:
        user_vis["__lbl_psds__"] = False

    bind_user_vis(distance_paths_ribbons, "__vec_ribbons__")
    bind_user_vis(distance_paths_psds, "__vec_psds__")
    bind_user_vis(distance_labels_ribbons, "__lbl_ribbons__")
    bind_user_vis(distance_labels_psds, "__lbl_psds__")

    for obj in [o for o in df["object"].unique() if o != cfg.ihc_obj]:
        rows = df[df["object"] == obj]
        pts = rows[["pos_z", "pos_y", "pos_x"]].to_numpy(float)
        base = base_size_for(obj)
        if obj == ribbons:
            sizes = base * (rows["volume"].astype(float).to_numpy() / max_vol_ribbon)
        elif obj == psds:
            sizes = base * (rows["volume"].astype(float).to_numpy() / max_vol_psd)
        else:
            sizes = base
        colors = colors_for_rows(rows, obj)
        layer = viewer.add_points(
            pts,
            size=sizes,
            border_color=colors,
            face_color=colors,
            name=str(obj),
            blending="translucent",
            visible=True,
        )
        layer.metadata["object"] = obj
        point_layers[obj] = layer
        user_vis[obj] = True
        bind_user_vis(layer, obj)

    labels = sorted(
        [
            s
            for s in df["ihc_label"].dropna().astype(str).str.strip().unique()
            if s.lower() not in {"nan", "none", ""}
        ]
    )
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

    def build_distance_for(obj_name: str, sel_labels):
        mask = (df["object"] == obj_name) & (df["ihc_label"].astype(str).isin(sel_labels))
        rows = df.loc[mask]
        if rows.empty:
            return np.zeros((0, 2, 3), float), np.zeros((0, 3), float), []
        normals = {}
        for lab in rows["ihc_label"].astype(str).unique():
            pn = plane_normal_for_label(lab)
            if pn is not None:
                normals[lab] = pn
        vec_list, mids, txt = [], [], []
        for _, r in rows.iterrows():
            key = str(r["ihc_label"]).strip()
            if key not in normals:
                continue
            n_hat, a0 = normals[key]
            if not np.all(np.isfinite(n_hat)):
                continue
            P = np.array([float(r["pos_z"]), float(r["pos_y"]), float(r["pos_x"])], float)
            d = float(r["dist_to_plane"])
            if not np.isfinite(d) or abs(d) < d_eps:
                continue
            sgn = np.sign(np.dot(P - a0, n_hat)) or 1.0
            V = -sgn * d * n_hat
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
        return [lab for lab, cb in checks.items() if cb.value]

    plane_last_nonempty_data = list(planes)

    def update_view(selected):
        nonlocal vis_prog, plane_last_nonempty_data
        EMPTY_VEC = np.zeros((0, 2, 3), float)
        EMPTY_PTS = np.zeros((0, 3), float)
        EMPTY_TXT = []
        if not selected:
            df_points = df[unlabeled_mask]
            idxs = []
            vec_r, mid_r, txt_r = EMPTY_VEC, EMPTY_PTS, EMPTY_TXT
            vec_p, mid_p, txt_p = EMPTY_VEC, EMPTY_PTS, EMPTY_TXT
        else:
            labeled = df[
                ~unlabeled_mask & df["ihc_label"].astype(str).str.strip().isin(selected)
            ]
            df_points = pd.concat([labeled, df[unlabeled_mask]], ignore_index=True)
            idxs = [
                label_to_plane_idx[l]
                for l in selected
                if (l in label_to_plane_idx and label_to_plane_idx[l] is not None)
            ]
            if idxs:
                new_data = [planes[i] for i in idxs]
                update_shapes(plane_layer, new_data)
                plane_last_nonempty_data = new_data
            if show_rib:
                vec_r, mid_r, txt_r = build_distance_for(ribbons, selected)
            else:
                vec_r, mid_r, txt_r = EMPTY_VEC, EMPTY_PTS, EMPTY_TXT
            if show_psd:
                vec_p, mid_p, txt_p = build_distance_for(psds, selected)
            else:
                vec_p, mid_p, txt_p = EMPTY_VEC, EMPTY_PTS, EMPTY_TXT
        vis_prog = True
        desired_plane_vis = bool(user_vis.get("__planes__", True)) and bool(selected) and bool(idxs)
        plane_layer.visible = desired_plane_vis
        vis_prog = False
        for obj, layer in point_layers_order:
            sel = df_points[df_points["object"] == obj]
            n = len(sel)
            if n > 0:
                pts = sel[["pos_z", "pos_y", "pos_x"]].to_numpy(float)
                base = base_size_for(obj)
                if obj == ribbons:
                    sizes = base * (sel["volume"].astype(float).to_numpy() / max_vol_ribbon)
                    colors = colors_for_rows(sel, obj)
                elif obj == psds:
                    sizes = base * (sel["volume"].astype(float).to_numpy() / max_vol_psd)
                    colors = colors_for_rows(sel, obj)
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
            layer.visible = bool(user_vis.get(obj, True) and (n > 0))
            vis_prog = False
        update_vectors(distance_paths_ribbons, vec_r)
        update_vectors(distance_paths_psds, vec_p)
        update_label_points(distance_labels_ribbons, mid_r, txt_r)
        update_label_points(distance_labels_psds, mid_p, txt_p)
        has_r = len(vec_r) > 0
        has_p = len(vec_p) > 0
        vis_prog = True
        if distance_paths_ribbons is not None:
            distance_paths_ribbons.visible = bool(user_vis.get("__vec_ribbons__", False) and has_r)
        if distance_labels_ribbons is not None:
            distance_labels_ribbons.visible = bool(user_vis.get("__lbl_ribbons__", False) and has_r)
        if distance_paths_psds is not None:
            distance_paths_psds.visible = bool(user_vis.get("__vec_psds__", False) and has_p)
        if distance_labels_psds is not None:
            distance_labels_psds.visible = bool(user_vis.get("__lbl_psds__", False) and has_p)
        vis_prog = False

    def apply():
        if not batching:
            update_view(selected_labels_now())

    for cb in checks.values():
        cb.changed.connect(apply)
    if show_rib:
        cb_pil_rib.changed.connect(apply)
        cb_mod_rib.changed.connect(apply)
    if show_psd:
        cb_pil_psd.changed.connect(apply)
        cb_mod_psd.changed.connect(apply)

    def set_all(val: bool):
        nonlocal batching
        batching = True
        for cb in checks.values():
            cb.value = val
        batching = False
        update_view(selected_labels_now())

    btn_all_on.clicked.connect(lambda: set_all(True))
    btn_all_off.clicked.connect(lambda: set_all(False))

    update_view(labels)
    viewer.fit_to_view()
    return viewer
