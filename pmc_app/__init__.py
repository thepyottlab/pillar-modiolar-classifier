"""Public package entry points and metadata for the Pillar–Modiolar Classifier."""

from __future__ import annotations

from importlib import import_module

__all__ = ["launch_gui", "__version__"]
__version__ = "1.4.0"


def launch_gui() -> None:
    """Launch the GUI application.

    Imported lazily to avoid importing Qt/Napari at package import time.
    """
    return import_module(".gui", __package__).launch_gui()
