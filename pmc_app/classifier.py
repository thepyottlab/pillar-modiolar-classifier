from __future__ import annotations

import numpy as np
import pandas as pd


def build_planes(df: pd.DataFrame, cfg) -> list[np.ndarray]:
    """Build finite rectangles (planes) per IHC label using apical/basal anchors."""
    ab = df[df["object"].isin(["apical", "basal"])][["ihc_label", "object"]]
    matched = set(ab.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index)
    polys: list[np.ndarray] = []
    ribbons = cfg.ribbons_obj
    psds = cfg.psds_obj

    for label in matched:
        ap = df[(df["ihc_label"] == label) & (df["object"] == "apical")].iloc[0]
        ba = df[(df["ihc_label"] == label) & (df["object"] == "basal")].iloc[0]
        x1, y1, z1 = float(ap["pos_x"]), float(ap["pos_y"]), float(ap["pos_z"])
        x2, y2, z2 = float(ba["pos_x"]), float(ba["pos_y"]), float(ba["pos_z"])
        vx, vy = x2 - x1, y2 - y1
        norm = (vx * vx + vy * vy) ** 0.5
        px, py = (-vy / norm, vx / norm) if norm > 1e-9 else (0.0, 1.0)

        group = df[(df["ihc_label"] == label) & (df["object"].isin([ribbons, psds]))]
        gxy = group[["pos_x", "pos_y"]].to_numpy(dtype=float)
        apxy = np.array([x1, y1], float)

        if norm <= 1e-9 and len(gxy):
            dx_loc = gxy[:, 0].ptp()
            dy_loc = gxy[:, 1].ptp()
            px, py = ((0.0, 1.0) if dx_loc >= dy_loc and dx_loc > 1e-9 else ((1.0, 0.0) if dy_loc > 1e-9 else (0.0, 1.0)))

        dots = (gxy - apxy).dot(np.array([px, py], float)) if len(gxy) else np.array([0.0, 0.0])
        min_dot, max_dot = ((float(dots.min()), float(dots.max())) if len(gxy) else (0.0, 0.0))
        c1x, c1y = x1 + px * max_dot, y1 + py * max_dot
        c2x, c2y = x1 + px * min_dot, y1 + py * min_dot
        c3x, c3y = x2 + px * min_dot, y2 + py * min_dot
        c4x, c4y = x2 + px * max_dot, y2 + py * max_dot
        polys.append(np.array([[z1, c1y, c1x], [z1, c2y, c2x], [z2, c3y, c3x], [z2, c4y, c4x]], float))
    return polys


def classify_synapses(df: pd.DataFrame, cfg, planes=None) -> pd.DataFrame:
    """Add 'localization' (pillar/modiolar) and 'dist_to_plane' (float) per synapse.

    Assumptions:
    - apical != basal (non-zero U).
    - synapse spread defines lateral span (non-zero V), per label.
    """
    out = df.copy()
    out["localization"] = pd.Series(pd.NA, dtype="string")
    out["dist_to_plane"] = np.nan

    eps = 1e-9
    ribbons = cfg.ribbons_obj
    psds = cfg.psds_obj

    pil_row = out[out["object"] == cfg.pillar_obj].iloc[0]
    mod_row = out[out["object"] == cfg.modiolar_obj].iloc[0]
    P_pil = np.array([[float(pil_row["pos_z"]), float(pil_row["pos_y"]), float(pil_row["pos_x"])]], dtype=float)
    P_mod = np.array([[float(mod_row["pos_z"]), float(mod_row["pos_y"]), float(mod_row["pos_x"])]], dtype=float)

    ab = out[out["object"].isin(["apical", "basal"])][["ihc_label", "object"]]
    matched = set(ab.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index)

    per_label: dict[str, dict] = {}
    pillar_side_counts = {0: 0, 1: 0}

    def geometry_from_label(ap_row, ba_row, group_xy):
        x1, y1, z1 = float(ap_row["pos_x"]), float(ap_row["pos_y"]), float(ap_row["pos_z"])
        x2, y2, z2 = float(ba_row["pos_x"]), float(ba_row["pos_y"]), float(ba_row["pos_z"])
        U = np.array([z2 - z1, y2 - y1, x2 - x1], dtype=float)
        vx, vy = (x2 - x1), (y2 - y1)
        norm_xy = (vx * vx + vy * vy) ** 0.5
        px, py = (-vy / max(norm_xy, eps), vx / max(norm_xy, eps))
        v_hat = np.array([0.0, py, px], dtype=float)
        n_hat = np.cross(U, v_hat)
        n_hat = n_hat / (np.linalg.norm(n_hat) + eps)
        a0 = np.array([z1, y1, x1], dtype=float)
        ap_xy = np.array([x1, y1], dtype=float)

        dots = (group_xy - ap_xy) @ np.array([px, py], dtype=float) if len(group_xy) else np.array([0.0])
        vmin = float(dots.min()) if dots.size else 0.0
        vmax = float(dots.max()) if dots.size else 0.0
        return a0, U, v_hat, n_hat, float(px), float(py), ap_xy, vmin, vmax

    def side_sign(P_zyx, a0, n_hat):
        s = (P_zyx - a0) @ n_hat
        return (s >= 0.0).astype(int)

    def point_to_rect_distance(P, a0, U, V, a):
        q = P - a
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
            cp = a + s_star * U + t_star * V
            return float(np.linalg.norm(P - cp))

        s0 = min(1.0, max(0.0, float((q @ U) / (u2 + eps))))
        d0 = np.linalg.norm(P - (a + s0 * U))
        q1 = q - V
        s1 = min(1.0, max(0.0, float((q1 @ U) / (u2 + eps))))
        d1 = np.linalg.norm(P - (a + s1 * U + V))
        t0 = min(1.0, max(0.0, float((q @ V) / (v2 + eps))))
        d2 = np.linalg.norm(P - (a + t0 * V))
        q2 = q - U
        t1 = min(1.0, max(0.0, float((q2 @ V) / (v2 + eps))))
        d3 = np.linalg.norm(P - (a + U + t1 * V))
        return float(min(d0, d1, d2, d3))

    for label in matched:
        ap = out[(out["ihc_label"] == label) & (out["object"] == "apical")].iloc[0]
        ba = out[(out["ihc_label"] == label) & (out["object"] == "basal")].iloc[0]
        m_syn = (out["ihc_label"] == label) & (out["object"].isin([ribbons, psds]))
        group_xy = out[m_syn][["pos_x", "pos_y"]].to_numpy(float)

        a0, U, v_hat, n_hat, px, py, ap_xy, vmin, vmax = geometry_from_label(ap, ba, group_xy)

        P_syn = out.loc[m_syn, ["pos_z", "pos_y", "pos_x"]].to_numpy(dtype=float)
        syn_sides = side_sign(P_syn, a0, n_hat)

        pillar_side = side_sign(P_pil, a0, n_hat)[0]
        modiolar_side = side_sign(P_mod, a0, n_hat)[0]
        if pillar_side != modiolar_side:
            pillar_side_counts[pillar_side] += 1

        V = v_hat * (vmax - vmin)
        a = a0 + v_hat * vmin

        per_label[str(label)] = {
            "mask_syn": m_syn,
            "syn_sides": syn_sides,
            "pillar_side": int(pillar_side),
            "modiolar_side": int(modiolar_side),
            "a0": a0,
            "U": U,
            "V": V,
            "a": a,
        }

    global_pillar_side = 0 if pillar_side_counts[0] >= pillar_side_counts[1] else 1
    global_modiolar_side = 1 - global_pillar_side

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
        # align by index to be safe
        out.loc[idx, "localization"] = pd.Series(loc_str, index=idx)

        # distances
        P = out.loc[idx, ["pos_z", "pos_y", "pos_x"]].to_numpy(dtype=float)
        if len(P) == 0:
            continue

        d = np.empty(len(idx), dtype=float)
        for i in range(len(idx)):
            d[i] = point_to_rect_distance(P[i], pack["a0"], pack["U"], pack["V"], pack["a"])
        out.loc[idx, "dist_to_plane"] = d

    # final, pandas nullable string dtype
    out["localization"] = out["localization"].astype("string")
    return out
