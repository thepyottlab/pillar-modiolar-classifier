from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .exporter import export_df_csv
from .finder import find_groups
from .gui import launch_gui
from .models import FinderConfig
from .parser import parse_group
from .processing import merge_dfs, process_position_df, process_volume_df

app = typer.Typer(help="Pillar–Modiolar Classifier CLI")


@app.command()
def gui() -> None:
    """Launch the GUI (same as `python -m pmc_app`)."""
    launch_gui()


@app.command()
def export_all(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    extension: str = typer.Option(".xls", help="File extension to scan"),
    ribbons: str = typer.Option("ribbon"),
    psds: str = typer.Option("psd"),
    positions: str = typer.Option("sum"),
    ribbons_obj: str = typer.Option("ribbons"),
    psds_obj: str = typer.Option("PSDs"),
    ihc_obj: str = typer.Option("IHC"),
    pillar_obj: str = typer.Option("pillar"),
    modiolar_obj: str = typer.Option("modiolar"),
    case_insensitive: bool = typer.Option(True),
    ribbons_only: bool = typer.Option(False),
    psds_only: bool = typer.Option(False),
    out_dir: Path = typer.Option(Path("exports"), exists=False, file_okay=False, dir_okay=True, writable=True),
):
    """Headless classification and CSV export for all detected groups."""
    cfg = FinderConfig(
        folder=folder,
        ribbons=ribbons,
        psds=psds,
        positions=positions,
        extensions=extension,
        ribbons_obj=ribbons_obj,
        psds_obj=psds_obj,
        ihc_obj=ihc_obj,
        pillar_obj=pillar_obj,
        modiolar_obj=modiolar_obj,
        case_insensitive=case_insensitive,
        ribbons_only=ribbons_only,
        psds_only=psds_only,
    )
    groups = find_groups(cfg)
    if not groups:
        typer.echo("No groups found.")
        raise typer.Exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    for gid, group in groups.items():
        rib, psd, pos = parse_group(group, cfg)
        rib = process_volume_df(rib)
        psd = process_volume_df(psd)
        pos, _ = process_position_df(pos, cfg)
        df = merge_dfs(rib, psd, pos)

        from .classifier import build_planes, classify_synapses

        planes = build_planes(df, cfg)
        df = classify_synapses(df, cfg, planes)
        export_df_csv(df, gid, out_dir)

    typer.echo(f"Exported {len(groups)} group(s) to {out_dir}")


if __name__ == "__main__":
    app()
