"""Helpers to resolve packaged resource file paths (PyInstaller-safe)."""

from __future__ import annotations

import sys
from importlib.resources import files
from pathlib import Path
from typing import Optional


def resource_path(name: str, *, package: str = "pmc_app.resources") -> str:
    """Return a filesystem path to a packaged resource in a PyInstaller-safe way.

    Resolves a resource whether the app runs from source or from a PyInstaller bundle.

    Args:
        name: Resource filename (e.g., "icon.ico").
        package: Dotted package path that contains the resources.

    Returns:
        str: A concrete path on disk to the resource.
    """
    meipass: Optional[str] = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass, package.replace(".", "/"), name)
        if candidate.exists():
            return str(candidate)

    resource = files(package) / name
    try:
        from importlib.resources import as_file
        with as_file(resource) as tmp:
            return str(tmp)
    except Exception:
        return str(resource)
