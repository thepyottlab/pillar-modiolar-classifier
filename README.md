\# Pillar–Modiolar Classifier (Standalone GUI)



A small Napari-based app that:

\- Loads groups of `.xls` files with suffix tokens (e.g., `"A ribbon.xls"`, `"A psd.xls"`, `"A sum.xls"`),

\- Builds per-IHC pillar–modiolar planes,

\- Classifies synapses (pillar vs. modiolar),

\- Computes distances to the finite plane,

\- Visualizes in 3D and exports CSVs.



> Designed for Windows; compatible with PyInstaller / Inno Setup.



---



\## Quickstart (dev)



```bash

\# 1) Create venv and install

python -m venv .venv

\# Windows:

. .venv/Scripts/activate

\# macOS/Linux:

\# . .venv/bin/activate



pip install -U pip

pip install -e ".\[dev]"   # installs runtime + dev tools



\# 2) Run tests / style

pytest -q

ruff check .

black --check .

mypy



\# 3) Launch the GUI

python -m pmc\_app

\# or: pmc gui



