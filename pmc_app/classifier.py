"""Plane building and per-synapse pillar-modiolar classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmc_app.models import FinderConfig

F = np.float64
EPS: float = float(np.finfo(np.float64).eps)


def identify_poles(df: pd.DataFrame, cfg: FinderConfig) -> pd.DataFrame:
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
    have_both = (
        anchors.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index
    )

    for label in have_both:
        ap_mask = (out["ihc_label"] == label) & (out["object"] == "apical")
        ba_mask = (out["ihc_label"] == label) & (out["object"] == "basal")
        ap = out[ap_mask].iloc[0]
        ba = out[ba_mask].iloc[0]

        ap_xyz = np.array(
            [float(ap["pos_x"]), float(ap["pos_y"]), float(ap["pos_z"])], dtype=F
        )
        ba_xyz = np.array(
            [float(ba["pos_x"]), float(ba["pos_y"]), float(ba["pos_z"])], dtype=F
        )

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


def build_pm_planes(
    df: pd.DataFrame, cfg: FinderConfig
) -> tuple[list[np.ndarray], list[str]]:
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
    matched = set(
        ab.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index
    )

    polys: list[np.ndarray] = []
    labels: list[str] = []

    for label in matched:
        ap = df[(df["ihc_label"] == label) & (df["object"] == "apical")].iloc[0]
        ba = df[(df["ihc_label"] == label) & (df["object"] == "basal")].iloc[0]
        x1, y1, z1 = float(ap["pos_x"]), float(ap["pos_y"]), float(ap["pos_z"])
        x2, y2, z2 = float(ba["pos_x"]), float(ba["pos_y"]), float(ba["pos_z"])

        # u: apical-basal direction (in ZYX space)
        u_vec = np.array([z2 - z1, y2 - y1, x2 - x1], float)
        u_len = float(np.linalg.norm(u_vec))
        u_hat = u_vec / (u_len if u_len > eps else 1.0)

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
            t = (gpos - a0) @ u_hat
            u_extent = max(u_len, float(np.max(t)))
        else:
            u_extent = u_len

        u = u_hat * u_extent
        a = a0 + v_hat * vmin
        v = v_hat * (vmax - vmin)

        c1 = a + v
        c2 = a
        c3 = a + u
        c4 = a + u + v

        polys.append(np.vstack([c1, c2, c3, c4]).astype(float))
        labels.append(str(ba["ihc_label"]))

    return polys, labels


def build_hc_planes(
    df: pd.DataFrame, cfg: FinderConfig
) -> tuple[list[np.ndarray], list[str]]:
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
    matched = set(
        ab.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index
    )

    polys: list[np.ndarray] = []
    labels: list[str] = []

    for label in matched:
        ap = df[(df["ihc_label"] == label) & (df["object"] == "apical")].iloc[0]
        ba = df[(df["ihc_label"] == label) & (df["object"] == "basal")].iloc[0]
        x1, y1, z1 = float(ap["pos_x"]), float(ap["pos_y"]), float(ap["pos_z"])
        x2, y2, z2 = float(ba["pos_x"]), float(ba["pos_y"]), float(ba["pos_z"])

        u = np.array([z2 - z1, y2 - y1, x2 - x1], dtype=F)
        vx, vy = (x2 - x1), (y2 - y1)
        norm_xy = (vx * vx + vy * vy) ** 0.5
        px, py = (-vy / max(norm_xy, EPS), vx / max(norm_xy, EPS))
        v_hat = np.array([0.0, py, px], dtype=F)

        # N (HC normal) = u × v_hat — points "through" the HC thickness
        n_hat = np.cross(u, v_hat).astype(F, copy=False)
        n_hat = n_hat / (float(np.linalg.norm(n_hat)) + EPS)

        gmask = (df["ihc_label"] == label) & (df["object"].isin([ribbons, psds]))
        gpos = df.loc[gmask, ["pos_z", "pos_y", "pos_x"]].to_numpy(dtype=F)
        gxy = df.loc[gmask, ["pos_x", "pos_y"]].to_numpy(dtype=F)

        apxy = np.array([x1, y1], dtype=F)
        if len(gxy):
            dots_v = (gxy - apxy) @ np.array([px, py], dtype=F)
            vmin = dots_v.min().item()
            vmax = dots_v.max().item()
        else:
            vmin = vmax = 0.0
        v = v_hat * (vmax - vmin)

        v_mid = 0.5 * (vmin + vmax)
        a0 = np.array([z1, y1, x1], dtype=F)
        center = a0 + u + v_hat * v_mid

        if len(gpos):
            dots_w = (gpos - center) @ n_hat
            wmin = dots_w.min().item()
            wmax = dots_w.max().item()
        else:
            wmin, wmax = -0.5, 0.5

        norm_v = np.linalg.norm(v).item()
        if (wmax - wmin) < 1e-6 * (norm_v + 1.0):
            half = 0.25 * max(norm_v, 1.0)
            wmin, wmax = -half, +half

        w = n_hat * (wmax - wmin)

        # anchor at basal edge, low side of w
        anchor = a0 + u + v_hat * vmin + n_hat * wmin
        c1 = anchor + v + w
        c2 = anchor + w
        c3 = anchor
        c4 = anchor + v

        polys.append(np.vstack([c1, c2, c3, c4]))
        labels.append(str(ba["ihc_label"]))

    return polys, labels


def classify_synapses(
    df: pd.DataFrame,
    cfg: FinderConfig,
    planes: tuple[list[np.ndarray], list[str]] | None = None,
    hc_planes: tuple[list[np.ndarray], list[str]] | None = None,
) -> pd.DataFrame:
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
    p_pil = np.array(
        [[float(pil_row["pos_z"]), float(pil_row["pos_y"]), float(pil_row["pos_x"])]],
        dtype=F,
    )
    p_mod = np.array(
        [[float(mod_row["pos_z"]), float(mod_row["pos_y"]), float(mod_row["pos_x"])]],
        dtype=F,
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
    matched = set(
        ab.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index
    )

    per_label: dict[str, dict] = {}
    pillar_side_counts = {0: 0, 1: 0}

    def geometry_from_label(
        ap_row: pd.Series,
        ba_row: pd.Series,
        group_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Return geometry primitives for a single IHC."""
        x1, y1, z1 = (
            float(ap_row["pos_x"]),
            float(ap_row["pos_y"]),
            float(ap_row["pos_z"]),
        )
        x2, y2, z2 = (
            float(ba_row["pos_x"]),
            float(ba_row["pos_y"]),
            float(ba_row["pos_z"]),
        )

        u = np.array([z2 - z1, y2 - y1, x2 - x1], dtype=F)

        # v_hat: lateral along XY, orthogonal to apical→basal in XY
        vx, vy = (x2 - x1), (y2 - y1)
        norm_xy = (vx * vx + vy * vy) ** 0.5
        px, py = (-vy / max(norm_xy, EPS), vx / max(norm_xy, EPS))
        v_hat = np.array([0.0, py, px], dtype=F)

        # PM plane normal = u × v_hat
        n_hat = np.cross(u, v_hat)
        n_hat = n_hat / (float(np.linalg.norm(n_hat)) + EPS)

        ap_xy = np.array([x1, y1], dtype=F)
        if len(group_xy):
            dots = (group_xy - ap_xy) @ np.array([px, py], dtype=F)
            vmin = dots.min().item()
            vmax = dots.max().item()
        else:
            vmin = vmax = 0.0

        a0 = np.array([z1, y1, x1], dtype=F)
        return a0, u, v_hat, n_hat, vmin, vmax

    def side_sign(
        p_zyx: np.ndarray,
        a: np.ndarray,
        n_hat: np.ndarray,
    ) -> np.ndarray:
        """Return pillar/modiolar side as 0/1 relative to a PM plane."""
        s = (p_zyx - a) @ n_hat
        return (s >= 0.0).astype(int)

    def point_to_rect_distance(
        p: np.ndarray,
        a: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
    ) -> float:
        """Return minimum distance from point to rectangle spanned by u, V."""
        q = p - a
        u2 = float(u @ u)
        v2 = float(v @ v)
        uv = float(u @ v)
        rhs = np.array([q @ u, q @ v], dtype=F)
        g = np.array([[u2, uv], [uv, v2]], dtype=F)
        try:
            s_star, t_star = np.linalg.solve(g, rhs)
        except np.linalg.LinAlgError:
            s_star, t_star = -1.0, -1.0

        # inside rectangle?
        if 0.0 <= s_star <= 1.0 and 0.0 <= t_star <= 1.0:
            cp = a + s_star * u + t_star * v
            return float(np.linalg.norm(p - cp))

        # else clamp to edges
        s0 = min(1.0, max(0.0, float((q @ u) / (u2 + EPS))))
        d0 = np.linalg.norm(p - (a + s0 * u))
        q1 = q - v
        s1 = min(1.0, max(0.0, float((q1 @ u) / (u2 + EPS))))
        d1 = np.linalg.norm(p - (a + s1 * u + v))
        t0 = min(1.0, max(0.0, float((q @ v) / (v2 + EPS))))
        d2 = np.linalg.norm(p - (a + t0 * v))
        q2 = q - u
        t1 = min(1.0, max(0.0, float((q2 @ v) / (v2 + EPS))))
        d3 = np.linalg.norm(p - (a + u + t1 * v))
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
            a_pm = poly_pm[1].astype(F)
            u_pm = (poly_pm[2] - poly_pm[1]).astype(F)
            v_pm = (poly_pm[0] - poly_pm[1]).astype(F)
            n_hat = np.cross(u_pm, v_pm).astype(F, copy=False)
            n_hat = n_hat / (float(np.linalg.norm(n_hat)) + EPS)
            v_hat = v_pm / (float(np.linalg.norm(v_pm)) + EPS)
            a0 = a_pm
            u = u_pm
            vmin = 0.0
            vmax = float(np.linalg.norm(v_pm))
        else:
            a0, u, v_hat, n_hat, vmin, vmax = geometry_from_label(ap, ba, group_xy)
            a_pm = a0 + v_hat * vmin
            v_pm = v_hat * (vmax - vmin)
            u_pm = u

        p_syn = out.loc[m_syn, ["pos_z", "pos_y", "pos_x"]].to_numpy(dtype=F)
        syn_sides = side_sign(p_syn, a_pm, n_hat)

        pillar_side = side_sign(p_pil, a_pm, n_hat)[0]
        modiolar_side = side_sign(p_mod, a_pm, n_hat)[0]
        if pillar_side != modiolar_side:
            pillar_side_counts[pillar_side] += 1

        per_label[lab_key] = {
            "mask_syn": m_syn,
            "syn_sides": syn_sides,
            "pillar_side": int(pillar_side),
            "modiolar_side": int(modiolar_side),
            "a_pm": a_pm,
            "u_pm": u_pm,
            "v_pm": v_pm,
            "a0": a0,
            "u": u,
            "v_hat": v_hat,
            "n_hat": n_hat,
            "vmin": vmin,
            "vmax": vmax,
        }

        if lab_key in hc_map:
            poly_hc = hc_map[lab_key]
            a_hc = poly_hc[2].astype(F)
            v_hc = (poly_hc[3] - poly_hc[2]).astype(F)
            w_hc = (poly_hc[1] - poly_hc[2]).astype(F)
            n_hc = np.cross(v_hc, w_hc).astype(F, copy=False)
            n_hc = n_hc / (float(np.linalg.norm(n_hc)) + EPS)
            per_label[lab_key].update(
                {"a_hc": a_hc, "v_hc": v_hc, "w_hc": w_hc, "n_hc": n_hc}
            )

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
            np.where(
                syn_sides == pillar_side_id, "pillar", np.array(pd.NA, dtype=object)
            ),
        ).astype(object)

        idx = out.index[pack["mask_syn"]]
        out.loc[idx, "localization"] = pd.Series(loc_str, index=idx)

        # PM signed distance: negative on pillar side, positive on modiolar side
        p = out.loc[idx, ["pos_z", "pos_y", "pos_x"]].to_numpy(dtype=F)
        if len(p):
            a_pm = pack["a_pm"]
            u_pm = pack["u_pm"]
            v_pm = pack["v_pm"]
            d = np.empty(len(idx), dtype=F)
            for i in range(len(idx)):
                d[i] = point_to_rect_distance(p[i], a_pm, u_pm, v_pm)

            loc_vals = out.loc[idx, "localization"].astype("string")
            is_pillar = loc_vals.fillna("").str.lower().eq("pillar").to_numpy()
            sign_pm = np.where(is_pillar, -1.0, 1.0)
            out.loc[idx, "pillar_modiolar_axis"] = d * sign_pm

    # HC signed distance: negative below the HC plane (toward -N), positive above
    for _lab_key, pack in per_label.items():
        idx = out.index[pack["mask_syn"]]
        if not len(idx):
            continue

        if (
            ("a_hc" in pack)
            and ("v_hc" in pack)
            and ("w_hc" in pack)
            and ("n_hc" in pack)
        ):
            a_hc = pack["a_hc"]
            v_hc = pack["v_hc"]
            w_hc = pack["w_hc"]
            n_hc = pack["n_hc"]
        else:
            a0 = pack["a0"]
            u = pack["u"]
            vmin = pack["vmin"]
            vmax = pack["vmax"]
            v_hat = pack["v_hat"]
            n_hat = pack["n_hat"]
            v_hc = v_hat * (vmax - vmin)
            v_mid = 0.5 * (vmin + vmax)
            base_center = a0 + u + v_hat * v_mid  # noqa: F841
            a_hc = a0 + u + v_hat * vmin + n_hat * 0.0
            w_hc = n_hat * 1.0
            n_hc = np.cross(v_hc, w_hc).astype(F, copy=False)
            n_hc = n_hc / (float(np.linalg.norm(n_hc)) + EPS)

        gpos = out.loc[idx, ["pos_z", "pos_y", "pos_x"]].to_numpy(dtype=F)
        d_hc = np.empty(len(idx), dtype=F)
        if len(gpos):
            for i in range(len(idx)):
                d_abs = point_to_rect_distance(gpos[i], a_hc, v_hc, w_hc)
                s = float((gpos[i] - a_hc) @ n_hc)
                d_hc[i] = d_abs if s < 0.0 else -d_abs
            out.loc[idx, "habenular_cuticular_axis"] = d_hc

    return out
