"""Entry point for the pillar-modiolar classifier demo application."""

from pathlib import Path

import napari

from finder import find_groups
from models import FinderConfig
from parser import parse_group
from processing import merge_dfs, process_position_df, process_volume_df
from visualizer import draw_objects


def demo() -> None:
    """Run the Napari demo using the files contained in the examples folder."""

    cfg = FinderConfig(folder=Path("examples"))
    groups = find_groups(cfg)
    print("found:", list(groups.keys()))
    for group_id, group in groups.items():
        ribbons_df, psds_df, positions_df = parse_group(group, cfg)

        ribbons_df = process_volume_df(ribbons_df)
        psds_df = process_volume_df(psds_df)
        positions_df, cfg = process_position_df(positions_df, cfg)

        df = merge_dfs(ribbons_df, psds_df, positions_df)

        viewer = draw_objects(df, cfg)
        napari.run()

        print(group_id)
        print(df)


if __name__ == "__main__":
    demo()


