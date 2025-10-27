"""Command-line interface for the Pillar–Modiolar Classifier."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

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


def _default_export_dir() -> Path:
    """Return a sensible per-OS default export directory as a Path."""
    env = os.environ.get("LOCALAPPDATA")
    base = Path(env) if env else (Path.home() / "AppData" / "Local")
    return base / "Pillar-Modiolar Classifier"


@app.command()
def gui() -> None:
    """Launch the GUI (same as `python -m pmc_app`)."""
    launch_gui()


@app.command()
def export_all(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Root folder containing your group files.",
        ),
    ],
    extension: Annotated[
        str, typer.Option(help="File extension to scan (e.g., .xls, .xlsx)")
    ] = ".xls",
    ribbons: Annotated[str, typer.Option(help="Substring token for ribbon files")] = "rib",
    psds: Annotated[str, typer.Option(help="Substring token for PSD files")] = "psd",
    positions: Annotated[str, typer.Option(help="Substring token for position files")] = "pos",
    ribbons_obj: Annotated[
        str, typer.Option(help="Object name for ribbons in the sheets")
    ] = "ribbons",
    psds_obj: Annotated[str, typer.Option(help="Object name for PSDs in the sheets")] = "PSDs",
    pillar_obj: Annotated[str, typer.Option(help="Object name for pillar anchors")] = "pillar",
    modiolar_obj: Annotated[
        str, typer.Option(help="Object name for modiolar anchors")
    ] = "modiolar",
    case_insensitive: Annotated[bool, typer.Option(help="Case-insensitive token matching")] = True,
    ribbons_only: Annotated[bool, typer.Option(help="Process only ribbons")] = False,
    psds_only: Annotated[bool, typer.Option(help="Process only PSDs")] = False,
    identify_poles: Annotated[
        bool, typer.Option(help="Relabel apical/basal anchors per IHC")
    ] = True,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            exists=False,
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="Output directory for CSV exports (created if needed).",
        ),
    ] = None,
) -> None:
    """Headless classification and CSV export for all detected groups.

    Raises:
        typer.Exit: If no groups are found.
    """
    if out_dir is None:
        out_dir = _default_export_dir()

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
        pos = process_position_df(pos)
        df = merge_dfs(rib, psd, pos, cfg)

        pm_bundle = build_pm_planes(df, cfg)
        hc_bundle = build_hc_planes(df, cfg)

        df = classify_synapses(df, cfg, pm_bundle, hc_bundle)
        export_df_csv(df, gid, out_dir)
        logger.info("Exported %s to %s", gid, out_dir)

    logger.info("Exported %d group(s) to %s", len(groups), out_dir)


if __name__ == "__main__":
    app()
