from pathlib import Path
from models import FinderConfig
from finder import find_groups
from parser import parse_group
from processing import process_volume_df, process_position_df, merge_dfs
from visualizer import draw_objects

import napari


def demo():
    cfg = FinderConfig(folder=Path("examples"))
    groups = find_groups(cfg)
    print("found:", list(groups.keys()))
    for gid, g in groups.items():
        ribbons_df, psds_df, positions_df = parse_group(g, cfg)

        ribbons_df = process_volume_df(ribbons_df)
        psds_df = process_volume_df(psds_df)
        positions_df, cfg = process_position_df(positions_df, cfg)

        df = merge_dfs(ribbons_df, psds_df, positions_df)

        viewer = draw_objects(df, cfg)
        napari.run()

        print(df)



if __name__ == "__main__":
    demo()


