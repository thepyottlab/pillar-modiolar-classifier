# Pillar–Modiolar Classifier

A small **Napari-based** desktop app to help analyze IHC synapses. It:

* Loads groups of `.xls` files named with suffix tokens (e.g., `ID A ribbon.xls`, `ID A psd.xls`, `ID A sum.xls`)
* Builds per-IHC pillar–modiolar planes
* Classifies synapses (pillar vs. modiolar)
* Computes signed distances to the finite plane
* Visualizes results in 3D and exports CSVs

> **Target platform:** Windows (10/11).
> Works best with data exported from Imaris following the provided protocol.

---

## Contents

* [Install](#install)
* [Quick start](#quick-start)
* [File naming & expected inputs](#file-naming--expected-inputs)
* [Using the viewer](#using-the-viewer)
* [Exported CSVs](#exported-csvs)
* [Tips & troubleshooting](#tips--troubleshooting)
* [Run from source](#run-from-source)
* [Build a standalone EXE (optional)](#build-a-standalone-exe-optional)
* [Tests](#tests)
* [Acknowledgements](#acknowledgements)

---

## Install

1. **Read the workflow**
   See `Protocol Pillar Modiolar Imaris Classification.docx` for the steps to export the correct tables from Imaris.

2. **Use the Windows installer**
   Download the installer from the repository’s **Releases** page and run it.
   (The installer bundles Python and dependencies; no separate setup required.)

> Prefer source installs? See [Run from source](#run-from-source).

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
  * A bounding box to keep orientation stable
* Right panel:

  * Toggle per-IHC labels to filter the scene
  * “Show all / Hide all” for quick selection changes
  * Highlight options for pillar/modiolar on ribbons/PSDs
* Navigation:

  * Use **Prev/Next** in the main window to switch loaded groups
  * The viewer stays open; layers and controls are rebuilt for the new group

> The progress bar in the **Log** panel indicates parsing, processing, classification, and rendering steps.

---

## Exported CSVs

* **Per-group export** or **Export all**
* Each CSV contains the unified table with classifications and **distance to the finite plane** for each synapse
* Choose an output folder, or set the optional **Export folder** field

---

## Tips & troubleshooting

* **Object names must match Imaris**: Ensure IHC, pillar, and modiolar labels in the app match the object names used in your exported tables.
* **Suffix tokens**: If “Load files” finds no groups, double-check your suffix tokens and file extension in the **Import** section.
* **Viewer closed manually?** Just click **Assess selected** again — the app will reopen/reuse the viewer safely.
* **Ribbons-only / PSDs-only**: These modes hide the other modality’s UI toggles and processing where appropriate.

---

## Run from source

Requirements (typical):

* Windows 10/11
* Python 3.9+ (3.10/3.11 also fine)
* A virtual environment is recommended

```bash
# clone your fork or this repo
git clone https://github.com/TomNaber/pillar-modiolar-classifier
cd pillar-modiolar-classifier

# create and activate a venv (PowerShell example)
python -m venv .venv
.venv\Scripts\activate

# install in editable mode with deps
pip install -U pip
pip install -e .[all]   # or simply: pip install -e .

# launch the GUI
python -m pmc_app
```

There’s also a small CLI (Typer):

```bash
# show CLI help
python -m pmc_app.cli --help

# open GUI via CLI entry
python -m pmc_app.cli gui

# (example) export help
python -m pmc_app.cli export --help
```

---

## Build a standalone EXE (optional)

If you need to build your own Windows executable with PyInstaller:

```bash
pyinstaller --noconfirm --onedir --windowed ^
  --name "PillarModiolarClassifier" ^
  --icon pmc_app/resources/icon.ico ^
  --collect-all napari --collect-all vispy --collect-all magicgui --collect-all qtpy ^
  --add-data "pmc_app/resources;pmc_app/resources" ^
  pmc_app\__main__.py
```

* Run from the project root in a virtual environment where the app is installed.
* The build artifacts go to `dist/` (folder-based app).

---

## Tests

```bash
# from the project root
pytest -q
```

Test modules cover group detection, normalization/merge, and plane/localization edge cases.

---

## Acknowledgements

Built on the shoulders of:

* [napari](https://napari.org/)
* [magicgui](https://pyapp-kit.github.io/magicgui/)
* [QtPy / Qt](https://github.com/spyder-ide/qtpy)
* [VisPy](https://vispy.org/)
* [NumPy](https://numpy.org/) & [pandas](https://pandas.pydata.org/)

---

If you run into issues, please open an issue with:

* A short description, steps to reproduce
* A redacted example of your filenames (and tokens you used)
* The log output shown by the app
