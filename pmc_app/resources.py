from __future__ import annotations

import sys
from importlib.resources import files
from pathlib import Path
from typing import Optional


def resource_path(name: str, *, package: str = "pmc_app.resources") -> str:
    """Return a filesystem path to a packaged resource, PyInstaller-safe.

    - Dev: resolves via importlib.resources.
    - PyInstaller: respects sys._MEIPASS if present.

    Args:
        name: Resource filename, e.g., "icon.ico".
        package: Package containing resources.

    Returns:
        String path to the resource on disk (always exists if bundled).
    """
    # PyInstaller onefile/onedir
    meipass: Optional[str] = getattr(sys, "_MEIPASS", None)
    if meipass:
        p = Path(meipass, package.replace(".", "/"), name)
        if p.exists():
            return str(p)

    # importlib.resources fallback
    resource = files(package) / name
    # In some cases resources may not be materialized on disk; convert safely
    try:
        from importlib.resources import as_file  # Py3.9+

        with as_file(resource) as tmp:
            return str(tmp)
    except Exception:
        return str(resource)
