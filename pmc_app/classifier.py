from __future__ import annotations

import numpy as np
import pandas as pd


def identify_poles(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Determine which of the two anchors ('apical'/'basal') per IHC label is closer
    to most synapses (ribbons+PSDs), and relabel accordingly.
    """
    out = df.copy()
    ribbons = cfg.ribbons_obj
    psds = cfg.psds_obj

    anchors = out[out["object"].isin(["apical", "basal"])]
    have_both = (
        anchors.groupby("ihc_label")["object"]
        .nunique()
        .loc[lambda s: s >= 2]
        .index
    )

    for label in have_both:
        ap_mask = (out["ihc_label"] == label) & (out["object"] == "apical")
        ba_mask = (out["ihc_label"] == label) & (out["object"] == "basal")
        ap = out[ap_mask].iloc[0]
        ba = out[ba_mask].iloc[0]

        ap_xyz = np.array([float(ap["pos_x"]), float(ap["pos_y"]), float(ap["pos_z"])], float)
        ba_xyz = np.array([float(ba["pos_x"]), float(ba["pos_y"]), float(ba["pos_z"])], float)

        syn_mask = (out["ihc_label"] == label) & (out["object"].isin([ribbons, psds]))
        gxyz = out.loc[syn_mask, ["pos_x", "pos_y", "pos_z"]].to_numpy(float)

        d_ap2 = ((gxyz - ap_xyz) ** 2).sum(axis=1)
        d_ba2 = ((gxyz - ba_xyz) ** 2).sum(axis=1)

        closer_ap = (d_ap2 < d_ba2).sum()
        closer_ba = (d_ba2 < d_ap2).sum()

        if closer_ap > closer_ba:
            out.loc[ap_mask, "object"] = "basal"
            out.loc[ba_mask, "object"] = "apical"
        elif closer_ba > closer_ap:
            out.loc[ap_mask, "object"] = "apical"
            out.loc[ba_mask, "object"] = "basal"
    return out


# -----------------------  PLANE BUILDERS  ----------------------------------- #

def build_pm_planes(df: pd.DataFrame, cfg) -> tuple[list[np.ndarray], list[str]]:
    """
    PM rectangles (one per IHC) in ZYX.  Returns (polys, labels).
    Each label is copied from the *basal* row used to build the polygon.
    """
    ribbons, psds = cfg.ribbons_obj, cfg.psds_obj
    eps = 1e-9

    ab = df[df["object"].isin(["apical", "basal"])][["ihc_label", "object"]]
    matched = set(ab.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index)

    polys: list[np.ndarray] = []
    labels: list[str] = []

    for label in matched:
        ap = df[(df["ihc_label"] == label) & (df["object"] == "apical")].iloc[0]
        ba = df[(df["ihc_label"] == label) & (df["object"] == "basal")].iloc[0]
        x1, y1, z1 = float(ap["pos_x"]), float(ap["pos_y"]), float(ap["pos_z"])
        x2, y2, z2 = float(ba["pos_x"]), float(ba["pos_y"]), float(ba["pos_z"])

        # U along apical→basal (ZYX)
        U = np.array([z2 - z1, y2 - y1, x2 - x1], float)

        # Lateral axis v_hat in XY
        vx, vy = (x2 - x1), (y2 - y1)
        norm_xy = (vx * vx + vy * vy) ** 0.5
        px, py = (-vy / max(norm_xy, eps), vx / max(norm_xy, eps))
        v_hat = np.array([0.0, py, px], float)

        # Size across V from synapse spread
        gmask = (df["ihc_label"] == label) & (df["object"].isin([ribbons, psds]))
        gxy   = df.loc[gmask, ["pos_x", "pos_y"]].to_numpy(float)
        apxy  = np.array([x1, y1], float)
        dots  = (gxy - apxy) @ np.array([px, py], float) if len(gxy) else np.array([0.0])
        vmin  = float(dots.min()) if dots.size else 0.0
        vmax  = float(dots.max()) if dots.size else 0.0

        a0 = np.array([z1, y1, x1], float)   # apical base
        a  = a0 + v_hat * vmin               # apical, V=min
        V  = v_hat * (vmax - vmin)

        # c1..c4 = apical max -> apical min -> basal min -> basal max
        c1 = a + V
        c2 = a
        c3 = a + U
        c4 = a + U + V

        polys.append(np.vstack([c1, c2, c3, c4]).astype(float))
        labels.append(str(ba["ihc_label"]))   # *** label from BASAL ***
    return polys, labels


def build_hc_planes(df: pd.DataFrame, cfg) -> tuple[list[np.ndarray], list[str]]:
    """
    HC rectangles (one per IHC) in ZYX. Returns (polys, labels).
    Each label is copied from the *basal* row used to build the polygon.
    """
    ribbons, psds = cfg.ribbons_obj, cfg.psds_obj
    eps = 1e-9

    ab = df[df["object"].isin(["apical", "basal"])][["ihc_label", "object"]]
    matched = set(ab.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index)

    polys: list[np.ndarray] = []
    labels: list[str] = []

    for label in matched:
        ap = df[(df["ihc_label"] == label) & (df["object"] == "apical")].iloc[0]
        ba = df[(df["ihc_label"] == label) & (df["object"] == "basal")].iloc[0]
        x1, y1, z1 = float(ap["pos_x"]), float(ap["pos_y"]), float(ap["pos_z"])
        x2, y2, z2 = float(ba["pos_x"]), float(ba["pos_y"]), float(ba["pos_z"])

        # U, v_hat
        U = np.array([z2 - z1, y2 - y1, x2 - x1], float)
        vx, vy = (x2 - x1), (y2 - y1)
        norm_xy = (vx * vx + vy * vy) ** 0.5
        px, py = (-vy / max(norm_xy, eps), vx / max(norm_xy, eps))
        v_hat = np.array([0.0, py, px], float)

        # PM normal -> HC thickness direction
        n_hat = np.cross(U, v_hat)
        n_hat /= (np.linalg.norm(n_hat) + eps)

        # Lateral spread V
        gmask = (df["ihc_label"] == label) & (df["object"].isin([ribbons, psds]))
        gpos  = df.loc[gmask, ["pos_z", "pos_y", "pos_x"]].to_numpy(float)
        gxy   = df.loc[gmask, ["pos_x", "pos_y"]].to_numpy(float)

        apxy  = np.array([x1, y1], float)
        dotsV = (gxy - apxy) @ np.array([px, py], float) if len(gxy) else np.array([0.0])
        vmin  = float(dotsV.min()) if dotsV.size else 0.0
        vmax  = float(dotsV.max()) if dotsV.size else 0.0
        V     = v_hat * (vmax - vmin)

        # Thickness W around basal center
        v_mid = 0.5 * (vmin + vmax)
        a0    = np.array([z1, y1, x1], float)
        center = a0 + U + v_hat * v_mid
        dotsW  = (gpos - center) @ n_hat if len(gpos) else np.array([0.0])
        wmin   = float(dotsW.min()) if dotsW.size else 0.0
        wmax   = float(dotsW.max()) if dotsW.size else 0.0
        if (wmax - wmin) < 1e-6 * (np.linalg.norm(V) + 1.0):
            half = 0.25 * max(np.linalg.norm(V), 1.0)
            wmin, wmax = -half, +half
        W = n_hat * (wmax - wmin)

        anchor = a0 + U + v_hat * vmin + n_hat * wmin
        c1 = anchor + V + W
        c2 = anchor + W
        c3 = anchor
        c4 = anchor + V

        polys.append(np.vstack([c1, c2, c3, c4]).astype(float))
        labels.append(str(ba["ihc_label"]))   # *** label from BASAL ***
    return polys, labels



# -----------------------  CLASSIFICATION / DISTANCES  ----------------------- #

def classify_synapses(df: pd.DataFrame, cfg, planes=None, hc_planes=None) -> pd.DataFrame:
    """
    Compute:
      - 'localization' (pillar/modiolar) via the PM plane,
      - 'pillar_modiolar_axis' (distance to PM rectangle, perpendicular to PM),
      - 'habenular_cuticular_axis' (distance to HC rectangle, perpendicular to
      HC).

    Notes
    -----
    • Geometry is derived per IHC from apical/basal anchors and the local synapse spread,
      so it matches the polygons rendered by the PM/HC builders.
    • Distances are perpendicular to their respective planes; for finite rectangles we
      clamp to edges/corners when the perpendicular lands outside the rectangle.
    • Only synapses (cfg.ribbons_obj, cfg.psds_obj) receive distances/localization.
    """
    out = df.copy()

    # Initialize outputs
    out["localization"] = pd.Series(pd.NA, dtype="string")

    eps = 1e-9
    ribbons = cfg.ribbons_obj
    psds = cfg.psds_obj

    # Required anchor points for global side inference
    pil_row = out[out["object"] == cfg.pillar_obj].iloc[0]
    mod_row = out[out["object"] == cfg.modiolar_obj].iloc[0]
    P_pil = np.array([[float(pil_row["pos_z"]), float(pil_row["pos_y"]), float(pil_row["pos_x"])]], dtype=float)
    P_mod = np.array([[float(mod_row["pos_z"]), float(mod_row["pos_y"]), float(mod_row["pos_x"])]], dtype=float)

    # IHCs having both anchors
    ab = out[out["object"].isin(["apical", "basal"])][["ihc_label", "object"]]
    matched = set(ab.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index)

    per_label: dict[str, dict] = {}
    pillar_side_counts = {0: 0, 1: 0}

    # --- helpers -------------------------------------------------------------

    def geometry_from_label(ap_row, ba_row, group_xy):
        """
        For a given IHC:
          U     : apical→basal vector (ZYX)
          v_hat : lateral unit vector in XY (perp to ap→ba in XY), lifted to ZYX
          n_hat : PM plane normal (unit), i.e., U × v_hat (not normalized yet)
          a0    : apical base (ZYX)
          vmin/vmax: extents of synapses along v_hat (scalar)
        """
        x1, y1, z1 = float(ap_row["pos_x"]), float(ap_row["pos_y"]), float(ap_row["pos_z"])
        x2, y2, z2 = float(ba_row["pos_x"]), float(ba_row["pos_y"]), float(ba_row["pos_z"])

        # U along apical→basal (ZYX ordering)
        U = np.array([z2 - z1, y2 - y1, x2 - x1], dtype=float)

        # Lateral axis v_hat in XY (perpendicular to ap→ba projection)
        vx, vy = (x2 - x1), (y2 - y1)
        norm_xy = (vx * vx + vy * vy) ** 0.5
        px, py = (-vy / max(norm_xy, eps), vx / max(norm_xy, eps))  # unit in XY
        v_hat = np.array([0.0, py, px], dtype=float)                # lift to ZYX

        # PM normal (unit)
        n_hat = np.cross(U, v_hat)
        n_hat = n_hat / (np.linalg.norm(n_hat) + eps)

        # Extents along v_hat from synapse spread
        ap_xy = np.array([x1, y1], dtype=float)
        dots = (group_xy - ap_xy) @ np.array([px, py], dtype=float) if len(group_xy) else np.array([0.0])
        vmin = float(dots.min()) if dots.size else 0.0
        vmax = float(dots.max()) if dots.size else 0.0

        a0 = np.array([z1, y1, x1], dtype=float)  # apical base in ZYX
        return a0, U, v_hat, n_hat, vmin, vmax

    def side_sign(P_zyx, a0, n_hat):
        """Which side of the PM plane a point lies on: 0 for -, 1 for +."""
        s = (P_zyx - a0) @ n_hat
        return (s >= 0.0).astype(int)

    def point_to_rect_distance(P, A, U, V):
        """
        Minimum distance from point P to rectangle {A + sU + tV | s,t∈[0,1]}.
        Distance is perpendicular to the plane when the perpendicular foot
        falls inside; otherwise it's distance to the nearest edge/corner.
        """
        q = P - A
        u2 = float(U @ U)
        v2 = float(V @ V)
        uv = float(U @ V)
        rhs = np.array([q @ U, q @ V], dtype=float)
        G = np.array([[u2, uv], [uv, v2]], dtype=float)
        try:
            s_star, t_star = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            s_star, t_star = -1.0, -1.0

        if 0.0 <= s_star <= 1.0 and 0.0 <= t_star <= 1.0:
            cp = A + s_star * U + t_star * V
            return float(np.linalg.norm(P - cp))

        # clamp to edges
        s0 = min(1.0, max(0.0, float((q @ U) / (u2 + eps))))
        d0 = np.linalg.norm(P - (A + s0 * U))
        q1 = q - V
        s1 = min(1.0, max(0.0, float((q1 @ U) / (u2 + eps))))
        d1 = np.linalg.norm(P - (A + s1 * U + V))
        t0 = min(1.0, max(0.0, float((q @ V) / (v2 + eps))))
        d2 = np.linalg.norm(P - (A + t0 * V))
        q2 = q - U
        t1 = min(1.0, max(0.0, float((q2 @ V) / (v2 + eps))))
        d3 = np.linalg.norm(P - (A + U + t1 * V))
        return float(min(d0, d1, d2, d3))

    # --- Pass 1: per-label PM geometry and side inference --------------------

    for label in matched:
        ap = out[(out["ihc_label"] == label) & (out["object"] == "apical")].iloc[0]
        ba = out[(out["ihc_label"] == label) & (out["object"] == "basal")].iloc[0]

        # Synapses belonging to this label
        m_syn = (out["ihc_label"] == label) & (out["object"].isin([ribbons, psds]))
        group_xy = out.loc[m_syn, ["pos_x", "pos_y"]].to_numpy(float)

        a0, U, v_hat, n_hat, vmin, vmax = geometry_from_label(ap, ba, group_xy)

        # PM rectangle axes:
        #   anchor at apical, V=vmin
        a_pm = a0 + v_hat * vmin
        V_pm = v_hat * (vmax - vmin)

        # For side inference (pillar/modiolar) against PM plane
        P_syn = out.loc[m_syn, ["pos_z", "pos_y", "pos_x"]].to_numpy(float)
        syn_sides = side_sign(P_syn, a0, n_hat)

        pillar_side = side_sign(P_pil, a0, n_hat)[0]
        modiolar_side = side_sign(P_mod, a0, n_hat)[0]
        if pillar_side != modiolar_side:
            pillar_side_counts[pillar_side] += 1

        per_label[str(label)] = {
            "mask_syn": m_syn,
            "syn_sides": syn_sides,
            "pillar_side": int(pillar_side),
            "modiolar_side": int(modiolar_side),
            "a0": a0,            # apical base (ZYX)
            "U": U,              # apical→basal (ZYX)
            "v_hat": v_hat,      # lateral unit (ZYX)
            "n_hat": n_hat,      # PM normal (unit)
            "vmin": vmin,
            "vmax": vmax,
            "a_pm": a_pm,        # PM rect anchor (apical, V=vmin)
            "V_pm": V_pm,        # PM rect lateral span
        }

    # Global fallback for pillar/modiolar side if a label is ambiguous
    global_pillar_side = 0 if pillar_side_counts[0] >= pillar_side_counts[1] else 1
    global_modiolar_side = 1 - global_pillar_side

    # --- Pass 2: assign localization and PM distances ------------------------

    for _, pack in per_label.items():
        pil_side = pack["pillar_side"]
        mod_side = pack["modiolar_side"]
        syn_sides = pack["syn_sides"]

        if pil_side != mod_side:
            pillar_side_id, modiolar_side_id = pil_side, mod_side
        else:
            pillar_side_id, modiolar_side_id = global_pillar_side, global_modiolar_side

        loc_str = np.where(
            syn_sides == modiolar_side_id,
            "modiolar",
            np.where(syn_sides == pillar_side_id, "pillar", pd.NA),
        )

        idx = out.index[pack["mask_syn"]]
        out.loc[idx, "localization"] = pd.Series(loc_str, index=idx)

        # PM distances (finite rectangle spanned by U and V_pm, anchored at a_pm)
        P = out.loc[idx, ["pos_z", "pos_y", "pos_x"]].to_numpy(float)
        if len(P):
            a_pm = pack["a_pm"]
            U = pack["U"]
            V_pm = pack["V_pm"]
            d = np.empty(len(idx), dtype=float)
            for i in range(len(idx)):
                d[i] = point_to_rect_distance(P[i], a_pm, U, V_pm)

            # make pillar-side distances negative
            loc_vals = out.loc[idx, "localization"].astype("string")
            is_pillar = loc_vals.fillna("").str.lower().eq("pillar").to_numpy()
            sign = np.where(is_pillar, -1.0, 1.0)
            d_signed = d * sign

            out.loc[idx, "pillar_modiolar_axis"] = d_signed

    # --- Pass 3: HC distances (plane at basal; axes = V (lateral),
    # W (PM normal)) --

    for _, pack in per_label.items():
        idx = out.index[pack["mask_syn"]]
        if not len(idx):
            continue

        a0   = pack["a0"]
        U    = pack["U"]
        vmin = pack["vmin"]
        vmax = pack["vmax"]
        v_hat = pack["v_hat"]
        n_hat = pack["n_hat"]

        # Lateral span along V identical to PM
        V = v_hat * (vmax - vmin)

        # Center line at basal, mid-V, to estimate thickness along W (n_hat)
        v_mid = 0.5 * (vmin + vmax)
        base_center = a0 + U + v_hat * v_mid

        gpos = out.loc[idx, ["pos_z", "pos_y", "pos_x"]].to_numpy(float)
        dots_w = (gpos - base_center) @ n_hat if len(gpos) else np.array([0.0])
        if dots_w.size:
            wmin, wmax = float(dots_w.min()), float(dots_w.max())
        else:
            wmin, wmax = -0.5, 0.5  # tiny slice fallback

        # If thickness is degenerate, make a small slab relative to |V|
        if (wmax - wmin) < 1e-6 * (np.linalg.norm(V) + 1.0):
            half = 0.25 * max(np.linalg.norm(V), 1.0)
            wmin, wmax = -half, +half

        W = n_hat * (wmax - wmin)

        # HC rectangle anchor: basal (a0+U), V=vmin (relative to apical), W=wmin
        a_hc = a0 + U + v_hat * vmin + n_hat * wmin

        # Distances to HC rectangle (perpendicular to HC plane)
        d_hc = np.empty(len(idx), dtype=float)
        for i in range(len(idx)):
            d_hc[i] = point_to_rect_distance(gpos[i], a_hc, V, W)
        out.loc[idx, "habenular_cuticular_axis"] = d_hc

    return out

