<img width="1824" height="1143" alt="image" src="https://github.com/user-attachments/assets/1cb0b600-c8b3-41d1-82ca-275e69b6eb8a" />


# Pillar–Modiolar Classifier

A small **Napari-based** desktop app to help analyze inner hair cell synapses. It:

* Loads groups of `.xls` files named with suffix tokens (e.g., `ID A ribbon.xls`, `ID A psd.xls`, `ID A sum.xls`)
* Builds per-IHC pillar–modiolar planes
* Classifies synapses (pillar vs. modiolar)
* Computes signed distances to the finite plane
* Visualizes results in 3D and exports CSVs

> **Target platform:** Windows (10/11).
> Works only with data exported from Imaris following the provided protocol.

---

## Contents

* [Install](#install)
* [Quick start](#quick-start)
* [File naming & expected inputs](#file-naming--expected-inputs)
* [Using the viewer](#using-the-viewer)
* [Exported CSVs](#exported-csvs)
* [Tips & troubleshooting](#tips--troubleshooting)
* [Run from source](#run-from-source)

---

## Install

1. **Read the workflow**
   See `Protocol Pillar Modiolar Imaris Classification.docx` for the steps to export the correct tables from Imaris.

2. **Use the Windows installer**
   Download the installer from the repository’s **Releases** page and run it.
   (The installer bundles Python and dependencies; no separate setup required.)

> For source installs, see [Run from source](#run-from-source).

---

## Quick start

1. Launch the app.
2. Choose the folder containing your exported `.xls` files.
3. Set the **file extension** and the **sheet identifiers** used in your filenames to distinguish:

   * Ribbon volumes
   * PSD volumes
   * Position table
4. Use **Ribbons only** or **PSDs only** if you want to restrict processing.
5. Click **Load files** — detected IDs appear under **Loaded IDs**.
6. Set the **Object names** to match your Imaris labels (IHC, pillar, modiolar, ribbons, PSDs).
7. Click **Assess selected (open in viewer)** to visualize, or use the **Export** section to write classified CSVs with plane distances.

The log shows progress and a compact progress bar while a dataset is opening or re-rendering.

---

## File naming & expected inputs

The app discovers groups by filename suffix tokens. For example, for ID `A`:

```
ID A ribbon.xls   # ribbon volumes
ID A psd.xls      # PSD volumes
ID A sum.xls      # positions (apical/basal/IHC etc.)
```

* The **suffix tokens** (e.g., `ribbon`, `psd`, `sum`) are configurable in the UI.
* Filenames must end with `"<space><token><extension>"`.
  Example: `Some Prefix ID A ribbon.xls`
* The app can run **Ribbons only** or **PSDs only**, but still needs the **positions** table.

---

## Using the viewer

* The viewer shows:

  * Points for ribbons, PSDs, and IHC/reference objects
  * Per-IHC finite planes (pillar–modiolar)
  * Optional distance vectors/labels
* Right panel:

  * Toggle per-IHC labels to filter the scene
  * “Show all / Hide all” for quick selection changes
  * Highlight options for pillar/modiolar on ribbons/PSDs
* Navigation:

  * Use **Prev/Next** in the main window to switch loaded groups
  * The viewer stays open; layers and controls are rebuilt for the new group

---

## Exported CSVs

* **Per-group export** or **Export all**
* Each CSV contains the unified table with classifications and **distance to the finite plane** for each synapse
* Choose an output folder

---

## Tips & troubleshooting

* **Object names must match Imaris**: Ensure IHC, pillar, and modiolar labels in the app match the object names used in your exported tables.
* **Suffix tokens**: If “Load files” finds no groups, double-check your suffix tokens and file extension in the **Import** section.
* **Viewer closed manually**: Just click **Assess selected** again, the app will reopen the viewer.
* **Ribbons-only / PSDs-only**: These modes hide the other modality’s UI toggles and processing where appropriate.
* **Plane definition**: The pillar–modiolar plane is derived from the apical and basal spots on each inner hair cell and bounded by the outermost ribbon and PSD coordinates in the XY plane. The Z-position of the plane is held constant, meaning its accuracy depends on the cochlea being mounted flat. If the Z-stack is uneven (e.g., the inner hair cells are on a fold) or the inner hair cells are rotated or tilted, the plane may not accurately separate the pillar and modiolar sides.

---

## Run from source

Requirements:

* Windows 10/11
* Python 3.12

```bash
# clone your fork or this repo
git clone https://github.com/thepyottlab/pillar-modiolar-classifier
cd pillar-modiolar-classifier

# create and activate a venv (PowerShell example)
python -m venv .venv
.venv\Scripts\activate

# install in editable mode with deps
python.exe -m pip install -U pip
pip install -e .[all]   # or simply: pip install -e .

# launch the GUI
python -m pmc_app

# Using the command line interface
 python -m pmc_app.cli gui  # Launch the GUI
 pmc_app.cli export_all "C:\folder_path" # Classify and export all files to an automatically created exports folder
```

---


If you run into issues, please open an issue with:

* A short description, steps to reproduce
* A redacted example of your filenames (and tokens you used)
* The log output shown by the app
