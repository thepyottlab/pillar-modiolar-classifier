import napari
import numpy as np


def draw_objects(df, cfg):
    viewer = napari.Viewer(ndisplay=3)

    ab = df[df["object"].isin(["apical", "basal"])][["ihc_label", "object"]]
    matched = set(ab.groupby("ihc_label")["object"].nunique().loc[lambda s: s >= 2].index)

    polys = []
    ribbons = cfg.ribbons_obj
    psds = cfg.psds_obj

    for label in matched:
        ap = df[(df["ihc_label"] == label) & (df["object"] == "apical")].iloc[0]
        ba = df[(df["ihc_label"] == label) & (df["object"] == "basal")].iloc[0]
        x1, y1, z1 = float(ap["pos_x"]), float(ap["pos_y"]), float(ap["pos_z"])
        x2, y2, z2 = float(ba["pos_x"]), float(ba["pos_y"]), float(ba["pos_z"])

        vx, vy = x2 - x1, y2 - y1
        norm = (vx * vx + vy * vy) ** 0.5
        px, py = -vy / norm, vx / norm


        group = df[
            (df["ihc_label"] == label) & (df["object"].isin([ribbons, psds]))]
        gxy = group[["pos_x", "pos_y"]].to_numpy(dtype=float)
        apxy = np.array([x1, y1], float)
        if norm <= 1e-9:
            dx_loc = gxy[:, 0].max() - gxy[:, 0].min()
            dy_loc = gxy[:, 1].max() - gxy[:, 1].min()
            if dx_loc >= dy_loc and dx_loc > 1e-9:
                px, py = 0.0, 1.0
            elif dy_loc > 1e-9:
                px, py = 1.0, 0.0
            else:
                px, py = 0.0, 1.0
        dots = (gxy - apxy).dot(np.array([px, py], float))
        min_dot, max_dot = float(dots.min()), float(dots.max())

        c1x, c1y = x1 + px * max_dot, y1 + py * max_dot
        c2x, c2y = x1 + px * min_dot, y1 + py * min_dot
        c3x, c3y = x2 + px * min_dot, y2 + py * min_dot
        c4x, c4y = x2 + px * max_dot, y2 + py * max_dot

        polys.append(np.array(
            [[z1, c1y, c1x], [z1, c2y, c2x], [z2, c3y, c3x], [z2, c4y, c4x]],
            float))

    viewer.add_shapes(
        polys,
        shape_type="rectangle",
        edge_width=0.1,
        edge_color='white',
        face_color='white',
        opacity=0.3,
        name="Pillar-modiolar plane"
    )

    PALETTE = {
        cfg.ribbons_obj: "#CB2027",
        cfg.psds_obj: "#059748",
        cfg.ihc_obj: "#8C8C8C",
        cfg.pillar_obj: "#9D722A",
        cfg.modiolar_obj: "#7B3A96",
        "apical": "#256DAB",
        "basal": "#265DAB",
    }

    objs = df["object"].unique()

    for obj in objs:
        g = df[df["object"] == obj]
        coords = g[["pos_z", "pos_y", "pos_x"]].to_numpy(dtype=float)

        if obj == cfg.ihc_obj:
            size = 3
            show = False
        elif obj in (cfg.pillar_obj, cfg.modiolar_obj, "apical", "basal"):
            size = 2
            show = True
        else:
            size = 1
            show = True

        color = PALETTE.get(obj, "white")

        viewer.add_points(
            coords,
            size=size,
            face_color=color,
            name=str(obj),
            blending="additive",
            visible = show
        )

    return viewer