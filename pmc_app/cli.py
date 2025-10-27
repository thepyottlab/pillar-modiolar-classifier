"""Command-line interface for the Pillar–Modiolar Classifier."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import typer

from .classifier import build_hc_planes, build_pm_planes, classify_synapses
from .exporter import export_df_csv
from .finder import find_groups
from .gui import launch_gui
from .logging_config import configure_logging
from .models import FinderConfig
from .parser import parse_group
from .processing import merge_dfs, process_position_df, process_volume_df

configure_logging()
logger = logging.getLogger(__name__)
app = typer.Typer(help="Pillar–Modiolar Classifier CLI")


@app.command()
def gui() -> None:
    """Launch the GUI (same as `python -m pmc_app`)."""
    launch_gui()


@app.command()
def export_all(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    extension: str = typer.Option(".xls", help="File extension to scan"),
    ribbons: str = typer.Option("rib"),
    psds: str = typer.Option("psd"),
    positions: str = typer.Option("pos"),
    ribbons_obj: str = typer.Option("ribbons"),
    psds_obj: str = typer.Option("PSDs"),
    pillar_obj: str = typer.Option("pillar"),
    modiolar_obj: str = typer.Option("modiolar"),
    case_insensitive: bool = typer.Option(True),
    ribbons_only: bool = typer.Option(False),
    psds_only: bool = typer.Option(False),
    identify_poles: bool = typer.Option(True),
    out_dir: Path = typer.Option(
        Path(os.environ.get("LOCALAPPDATA")) / "Pillar-Modiolar Classifier",
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
    ),
) -> None:
    """Headless classification and CSV export for all detected groups.

    Raises:
        typer.Exit: If no groups are found.
    """
    cfg = FinderConfig(
        folder=folder,
        ribbons=ribbons,
        psds=psds,
        positions=positions,
        extensions=extension,
        ribbons_obj=ribbons_obj,
        psds_obj=psds_obj,
        pillar_obj=pillar_obj,
        modiolar_obj=modiolar_obj,
        case_insensitive=case_insensitive,
        ribbons_only=ribbons_only,
        psds_only=psds_only,
        identify_poles=identify_poles,
    )

    groups = find_groups(cfg)
    if not groups:
        logger.info("No groups found.")
        raise typer.Exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    for gid, group in groups.items():
        rib, psd, pos = parse_group(group, cfg)
        rib = process_volume_df(rib)
        psd = process_volume_df(psd)
        pos = process_position_df(pos, cfg)
        df = merge_dfs(rib, psd, pos, cfg)

        pm_bundle = build_pm_planes(df, cfg)
        hc_bundle = build_hc_planes(df, cfg)

        df = classify_synapses(df, cfg, pm_bundle, hc_bundle)
        export_df_csv(df, gid, out_dir)
        logger.info("Exported %s to %s", gid, out_dir)

    logger.info("Exported %d group(s) to %s", len(groups), out_dir)


if __name__ == "__main__":
    app()
