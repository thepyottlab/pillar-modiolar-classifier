"""Plane building and per-synapse pillar-modiolar classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmc_app.models import FinderConfig

F = np.float64
EPS: float = float(np.finfo(np.float64).eps)


def identify_poles(df: pd.DataFrame, cfg: FinderConfig) -> pd.DataFrame | None:
    """Relabel apical/basal anchors per IHC using proximity to synapses.

    For each IHC label with both anchors present, the anchor that is closer to
    the majority of synapses (ribbons and PSDs) is relabeled as basal; the other
    becomes apical.

    Args:
        df: Unified table with objects and positions.
        cfg: Configuration (used for object names).

    Returns:
        pd.DataFrame: Copy with 'apical'/'basal' possibly swapped per IHC.
    """
    out = df.copy()
    ribbons = cfg.ribbons_obj
    psds = cfg.psds_obj

    anchors = out[out["object"].isin(["apical", "basal"])]
    have_both = anchors.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index

    for label in have_both:
        ap_mask = (out["ihc_label"] == label) & (out["object"] == "apical")
        ba_mask = (out["ihc_label"] == label) & (out["object"] == "basal")
        ap = out[ap_mask].iloc[0]
        ba = out[ba_mask].iloc[0]

        ap_xyz = np.array([float(ap["pos_x"]), float(ap["pos_y"]), float(ap["pos_z"])], dtype=F)
        ba_xyz = np.array([float(ba["pos_x"]), float(ba["pos_y"]), float(ba["pos_z"])], dtype=F)

        syn_mask = (out["ihc_label"] == label) & (out["object"].isin([ribbons, psds]))
        gxyz = out.loc[syn_mask, ["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=F)

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


def build_pm_planes(df: pd.DataFrame, cfg: FinderConfig) -> tuple[list[np.ndarray], list[str]]:
    """Build pillar–modiolar (PM) rectangles per IHC in ZYX order.

    The lateral extent is derived from synapse spread perpendicular to the
    apical-basal/cuticular-habenular direction. Each label is copied from the
    basal row used.

    Args:
        df: Unified table.
        cfg: Configuration (for object names).

    Returns:
        tuple[list[np.ndarray], list[str]]: Rectangles and their IHC labels.
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

        # U: apical-basal direction (in ZYX space)
        U_vec = np.array([z2 - z1, y2 - y1, x2 - x1], float)
        U_len = float(np.linalg.norm(U_vec))
        U_hat = U_vec / (U_len if U_len > eps else 1.0)

        # v_hat: lateral (perp. in XY), used to span rectangle width
        vx, vy = (x2 - x1), (y2 - y1)
        norm_xy = (vx * vx + vy * vy) ** 0.5
        px, py = (-vy / max(norm_xy, eps), vx / max(norm_xy, eps))
        v_hat = np.array([0.0, py, px], float)

        gmask = (df["ihc_label"] == label) & (df["object"].isin([ribbons, psds]))
        gxy = df.loc[gmask, ["pos_x", "pos_y"]].to_numpy(float)
        gpos = df.loc[gmask, ["pos_z", "pos_y", "pos_x"]].to_numpy(float)
        apxy = np.array([x1, y1], float)

        if len(gxy):
            dots = (gxy - apxy) @ np.array([px, py], float)
            vmin = float(dots.min())
            vmax = float(dots.max())
        else:
            vmin = vmax = 0.0

        a0 = np.array([z1, y1, x1], float)

        if len(gpos):
            t = (gpos - a0) @ U_hat
            u_extent = max(U_len, float(np.max(t)))
        else:
            u_extent = U_len

        U = U_hat * u_extent
        a = a0 + v_hat * vmin
        V = v_hat * (vmax - vmin)

        c1 = a + V
        c2 = a
        c3 = a + U
        c4 = a + U + V

        polys.append(np.vstack([c1, c2, c3, c4]).astype(float))
        labels.append(str(ba["ihc_label"]))

    return polys, labels


def build_hc_planes(df: pd.DataFrame, cfg: FinderConfig) -> tuple[list[np.ndarray], list[str]]:
    """Build habenular–cuticular (HC) rectangles per IHC in ZYX order.

    For each IHC, derive the lateral axis as in the PM plane and the thickness
    axis as the PM plane normal. Rectangles are anchored at the basal side.

    Args:
        df: Unified table.
        cfg: Configuration.

    Returns:
        tuple[list[np.ndarray], list[str]]: Rectangles and their IHC labels.
    """
    ribbons, psds = cfg.ribbons_obj, cfg.psds_obj

    ab = df[df["object"].isin(["apical", "basal"])][["ihc_label", "object"]]
    matched = set(ab.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index)

    polys: list[np.ndarray] = []
    labels: list[str] = []

    for label in matched:
        ap = df[(df["ihc_label"] == label) & (df["object"] == "apical")].iloc[0]
        ba = df[(df["ihc_label"] == label) & (df["object"] == "basal")].iloc[0]
        x1, y1, z1 = float(ap["pos_x"]), float(ap["pos_y"]), float(ap["pos_z"])
        x2, y2, z2 = float(ba["pos_x"]), float(ba["pos_y"]), float(ba["pos_z"])

        U = np.array([z2 - z1, y2 - y1, x2 - x1], dtype=F)
        vx, vy = (x2 - x1), (y2 - y1)
        norm_xy = (vx * vx + vy * vy) ** 0.5
        px, py = (-vy / max(norm_xy, EPS), vx / max(norm_xy, EPS))
        v_hat = np.array([0.0, py, px], dtype=F)

        # N (HC normal) = U × v_hat — points "through" the HC thickness
        n_hat = np.cross(U, v_hat).astype(F, copy=False)
        n_hat = n_hat / (float(np.linalg.norm(n_hat)) + EPS)

        gmask = (df["ihc_label"] == label) & (df["object"].isin([ribbons, psds]))
        gpos = df.loc[gmask, ["pos_z", "pos_y", "pos_x"]].to_numpy(dtype=F)
        gxy = df.loc[gmask, ["pos_x", "pos_y"]].to_numpy(dtype=F)

        apxy = np.array([x1, y1], dtype=F)
        if len(gxy):
            dotsV = (gxy - apxy) @ np.array([px, py], dtype=F)
            vmin = dotsV.min().item()
            vmax = dotsV.max().item()
        else:
            vmin = vmax = 0.0
        V = v_hat * (vmax - vmin)

        v_mid = 0.5 * (vmin + vmax)
        a0 = np.array([z1, y1, x1], dtype=F)
        center = a0 + U + v_hat * v_mid  # noqa: F841

        if len(gpos):
            dotsW = (gpos - center) @ n_hat
            wmin = dotsW.min().item()
            wmax = dotsW.max().item()
        else:
            wmin, wmax = -0.5, 0.5

        norm_v = np.linalg.norm(V).item()
        if (wmax - wmin) < 1e-6 * (norm_v + 1.0):
            half = 0.25 * max(norm_v, 1.0)
            wmin, wmax = -half, +half

        W = n_hat * (wmax - wmin)

        # anchor at basal edge, low side of W
        anchor = a0 + U + v_hat * vmin + n_hat * wmin
        c1 = anchor + V + W
        c2 = anchor + W
        c3 = anchor
        c4 = anchor + V

        polys.append(np.vstack([c1, c2, c3, c4]))
        labels.append(str(ba["ihc_label"]))

    return polys, labels


def classify_synapses(
    df: pd.DataFrame,
    cfg: FinderConfig,
    planes: tuple[list[np.ndarray], list[str]] | None = None,
    hc_planes: tuple[list[np.ndarray], list[str]] | None = None,
) -> pd.DataFrame | None:
    """Classify synapses and compute distances along PM and HC axes.

    Adds:
        * ``localization``: pillar / modiolar side inferred per IHC using the PM plane.
        * ``pillar_modiolar_axis``: signed distance to the PM rectangle (perpendicular).
        * ``habenular_cuticular_axis``: signed distance to the HC rectangle (perpendicular).

    The PM/HC geometries are derived per IHC from apical/basal anchors and the
    local synapse spread. For finite rectangles, distances clamp to edges/corners.

    Args:
        df: Unified table.
        cfg: Configuration with object names.
        planes: Optional precomputed PM rectangles and labels.
        hc_planes: Optional precomputed HC rectangles and labels.

    Returns:
        pd.DataFrame: Copy with classification columns added.
    """
    out = df.copy()
    out["localization"] = pd.Series(pd.NA, dtype="string")

    ribbons = cfg.ribbons_obj
    psds = cfg.psds_obj

    pil_row = out[out["object"] == cfg.pillar_obj].iloc[0]
    mod_row = out[out["object"] == cfg.modiolar_obj].iloc[0]
    P_pil = np.array(
        [[float(pil_row["pos_z"]), float(pil_row["pos_y"]), float(pil_row["pos_x"])]], dtype=F
    )
    P_mod = np.array(
        [[float(mod_row["pos_z"]), float(mod_row["pos_y"]), float(mod_row["pos_x"])]], dtype=F
    )

    pm_map: dict[str, np.ndarray] = {}
    if planes is not None:
        pm_polys, pm_labels = planes
        for i, lab in enumerate(pm_labels):
            pm_map[str(lab)] = pm_polys[i]

    hc_map: dict[str, np.ndarray] = {}
    if hc_planes is not None:
        hc_polys, hc_labels = hc_planes
        for i, lab in enumerate(hc_labels):
            hc_map[str(lab)] = hc_polys[i]

    ab = out[out["object"].isin(["apical", "basal"])][["ihc_label", "object"]]
    matched = set(ab.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index)

    per_label: dict[str, dict] = {}
    pillar_side_counts = {0: 0, 1: 0}

    def geometry_from_label(
        ap_row: pd.Series,
        ba_row: pd.Series,
        group_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Return geometry primitives for a single IHC."""
        x1, y1, z1 = float(ap_row["pos_x"]), float(ap_row["pos_y"]), float(ap_row["pos_z"])
        x2, y2, z2 = float(ba_row["pos_x"]), float(ba_row["pos_y"]), float(ba_row["pos_z"])

        U = np.array([z2 - z1, y2 - y1, x2 - x1], dtype=F)

        # v_hat: lateral along XY, orthogonal to apical→basal in XY
        vx, vy = (x2 - x1), (y2 - y1)
        norm_xy = (vx * vx + vy * vy) ** 0.5
        px, py = (-vy / max(norm_xy, EPS), vx / max(norm_xy, EPS))
        v_hat = np.array([0.0, py, px], dtype=F)

        # PM plane normal = U × v_hat
        n_hat = np.cross(U, v_hat)
        n_hat = n_hat / (float(np.linalg.norm(n_hat)) + EPS)

        ap_xy = np.array([x1, y1], dtype=F)
        if len(group_xy):
            dots = (group_xy - ap_xy) @ np.array([px, py], dtype=F)
            vmin = dots.min().item()
            vmax = dots.max().item()
        else:
            vmin = vmax = 0.0

        a0 = np.array([z1, y1, x1], dtype=F)
        return a0, U, v_hat, n_hat, vmin, vmax

    def side_sign(
        P_zyx: np.ndarray,
        A: np.ndarray,
        n_hat: np.ndarray,
    ) -> np.ndarray:
        """Return pillar/modiolar side as 0/1 relative to a PM plane."""
        s = (P_zyx - A) @ n_hat
        return (s >= 0.0).astype(int)

    def point_to_rect_distance(
        P: np.ndarray,
        A: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
    ) -> float:
        """Return minimum distance from point to rectangle spanned by U, V."""
        q = P - A
        u2 = float(U @ U)
        v2 = float(V @ V)
        uv = float(U @ V)
        rhs = np.array([q @ U, q @ V], dtype=F)
        G = np.array([[u2, uv], [uv, v2]], dtype=F)
        try:
            s_star, t_star = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            s_star, t_star = -1.0, -1.0

        # inside rectangle?
        if 0.0 <= s_star <= 1.0 and 0.0 <= t_star <= 1.0:
            cp = A + s_star * U + t_star * V
            return float(np.linalg.norm(P - cp))

        # else clamp to edges
        s0 = min(1.0, max(0.0, float((q @ U) / (u2 + EPS))))
        d0 = np.linalg.norm(P - (A + s0 * U))
        q1 = q - V
        s1 = min(1.0, max(0.0, float((q1 @ U) / (u2 + EPS))))
        d1 = np.linalg.norm(P - (A + s1 * U + V))
        t0 = min(1.0, max(0.0, float((q @ V) / (v2 + EPS))))
        d2 = np.linalg.norm(P - (A + t0 * V))
        q2 = q - U
        t1 = min(1.0, max(0.0, float((q2 @ V) / (v2 + EPS))))
        d3 = np.linalg.norm(P - (A + U + t1 * V))
        dists = np.array([d0, d1, d2, d3], dtype=F)
        return float(dists.min())

    for label in matched:
        ap = out[(out["ihc_label"] == label) & (out["object"] == "apical")].iloc[0]
        ba = out[(out["ihc_label"] == label) & (out["object"] == "basal")].iloc[0]

        m_syn = (out["ihc_label"] == label) & (out["object"].isin([ribbons, psds]))
        group_xy = out.loc[m_syn, ["pos_x", "pos_y"]].to_numpy(dtype=F)

        lab_key = str(label)

        if lab_key in pm_map:
            poly_pm = pm_map[lab_key]
            A_pm = poly_pm[1].astype(F)
            U_pm = (poly_pm[2] - poly_pm[1]).astype(F)
            V_pm = (poly_pm[0] - poly_pm[1]).astype(F)
            n_hat = np.cross(U_pm, V_pm).astype(F, copy=False)
            n_hat = n_hat / (float(np.linalg.norm(n_hat)) + EPS)
            v_hat = V_pm / (float(np.linalg.norm(V_pm)) + EPS)
            a0 = A_pm
            U = U_pm
            vmin = 0.0
            vmax = float(np.linalg.norm(V_pm))
        else:
            a0, U, v_hat, n_hat, vmin, vmax = geometry_from_label(ap, ba, group_xy)
            A_pm = a0 + v_hat * vmin
            V_pm = v_hat * (vmax - vmin)
            U_pm = U

        P_syn = out.loc[m_syn, ["pos_z", "pos_y", "pos_x"]].to_numpy(dtype=F)
        syn_sides = side_sign(P_syn, A_pm, n_hat)

        pillar_side = side_sign(P_pil, A_pm, n_hat)[0]
        modiolar_side = side_sign(P_mod, A_pm, n_hat)[0]
        if pillar_side != modiolar_side:
            pillar_side_counts[pillar_side] += 1

        per_label[lab_key] = {
            "mask_syn": m_syn,
            "syn_sides": syn_sides,
            "pillar_side": int(pillar_side),
            "modiolar_side": int(modiolar_side),
            "a_pm": A_pm,
            "U_pm": U_pm,
            "V_pm": V_pm,
            "a0": a0,
            "U": U,
            "v_hat": v_hat,
            "n_hat": n_hat,
            "vmin": vmin,
            "vmax": vmax,
        }

        if lab_key in hc_map:
            poly_hc = hc_map[lab_key]
            A_hc = poly_hc[2].astype(F)
            V_hc = (poly_hc[3] - poly_hc[2]).astype(F)
            W_hc = (poly_hc[1] - poly_hc[2]).astype(F)
            N_hc = np.cross(V_hc, W_hc).astype(F, copy=False)
            N_hc = N_hc / (float(np.linalg.norm(N_hc)) + EPS)
            per_label[lab_key].update({"a_hc": A_hc, "V_hc": V_hc, "W_hc": W_hc, "N_hc": N_hc})

    # choose a global side mapping if anchors happen to be colinear
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
        out.loc[idx, "localization"] = pd.Series(loc_str, index=idx)

        # PM signed distance: negative on pillar side, positive on modiolar side
        P = out.loc[idx, ["pos_z", "pos_y", "pos_x"]].to_numpy(dtype=F)
        if len(P):
            A_pm = pack["a_pm"]
            U_pm = pack["U_pm"]
            V_pm = pack["V_pm"]
            d = np.empty(len(idx), dtype=F)
            for i in range(len(idx)):
                d[i] = point_to_rect_distance(P[i], A_pm, U_pm, V_pm)

            loc_vals = out.loc[idx, "localization"].astype("string")
            is_pillar = loc_vals.fillna("").str.lower().eq("pillar").to_numpy()
            sign_pm = np.where(is_pillar, -1.0, 1.0)
            out.loc[idx, "pillar_modiolar_axis"] = d * sign_pm

    # HC signed distance: negative below the HC plane (toward -N), positive above
    for _lab_key, pack in per_label.items():
        idx = out.index[pack["mask_syn"]]
        if not len(idx):
            continue

        if ("a_hc" in pack) and ("V_hc" in pack) and ("W_hc" in pack) and ("N_hc" in pack):
            A_hc = pack["a_hc"]
            V_hc = pack["V_hc"]
            W_hc = pack["W_hc"]
            N_hc = pack["N_hc"]
        else:
            a0 = pack["a0"]
            U = pack["U"]
            vmin = pack["vmin"]
            vmax = pack["vmax"]
            v_hat = pack["v_hat"]
            n_hat = pack["n_hat"]
            V_hc = v_hat * (vmax - vmin)
            v_mid = 0.5 * (vmin + vmax)
            base_center = a0 + U + v_hat * v_mid  # noqa: F841
            A_hc = a0 + U + v_hat * vmin + n_hat * 0.0
            W_hc = n_hat * 1.0
            N_hc = np.cross(V_hc, W_hc).astype(F, copy=False)
            N_hc = N_hc / (float(np.linalg.norm(N_hc)) + EPS)

        gpos = out.loc[idx, ["pos_z", "pos_y", "pos_x"]].to_numpy(dtype=F)
        d_hc = np.empty(len(idx), dtype=F)
        if len(gpos):
            for i in range(len(idx)):
                d_abs = point_to_rect_distance(gpos[i], A_hc, V_hc, W_hc)
                s = float((gpos[i] - A_hc) @ N_hc)
                d_hc[i] = d_abs if s < 0.0 else -d_abs
            out.loc[idx, "habenular_cuticular_axis"] = d_hc

    return out
