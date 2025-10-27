"""GUI for the Pillar–Modiolar Classifier built with magicgui/Qt and napari."""

from __future__ import annotations

import configparser
import logging
import os
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import napari
import pandas as pd
from magicgui.types import FileDialogMode
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    Label,
    LineEdit,
    PushButton,
    TextEdit,
)
from qtpy.QtCore import QEvent, QObject, QTimer
from qtpy.QtGui import QColor, QFontMetrics, QIcon, QPaintEvent, QPainter
from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QProgressBar,
    QSizePolicy,
    QStyle,
    QStyleOptionProgressBar,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .assets import resource_path
from .classifier import build_hc_planes, build_pm_planes, classify_synapses, identify_poles
from .exceptions import GroupValidationError
from .exporter import export_df_csv, prompt_export_dir
from .finder import find_groups
from .logging_config import configure_logging
from .models import FinderConfig, Group
from .parser import parse_group
from .processing import merge_dfs, process_position_df, process_volume_df
from .visualizer import draw_objects


@dataclass(frozen=True)
class TooltipText:
    """Static tooltip copy for the GUI."""
    case_insensitive: str = "Ignore letter case when matching file names, extensions, and identifiers."
    file_extension: str = "File extension of the Imaris exports."
    ribbons_sheet_id: str = "Suffix/identifier used for presynaptic ribbon volume sheets."
    psds_sheet_id: str = "Suffix/identifier used for glutamate-receptor patch (PSD) volume sheets."
    positions_sheet_id: str = "Identifier for the group-level export sheet that contains object positions."
    identify_poles: str = (
        "On: mark the spot closest to most synapses as basal. "
        "Off: mark the spot that was created first as apical (lower ID)."
    )
    ribbons_only: str = "Select when no glutamate-receptor patch volumes were analyzed."
    psds_only: str = "Select when no presynaptic ribbon volumes were analyzed."
    update_all_fields: str = "Store all values in ‘Import’ and ‘Object names’."
    ribbons_obj_id: str = "Object name for presynaptic ribbons in Imaris."
    psds_obj_id: str = "Object name for glutamate-receptor patches in Imaris."
    pillar_obj_id: str = "Object name of the spot on the pillar side of the inner hair cell in Imaris."
    modiolar_obj_id: str = "Object name of the spot on the modiolar side of the inner hair cell in Imaris."


TOOLTIP_WRAP_CH = 72
TOOLTIP_DELAY_MS = 600
TT = TooltipText()

APP_WIDTH = 667
APP_HEIGHT = 1080
TOP_PANEL_PADDING = 0
BOTTOM_PANEL_PADDING = 0

GROUP_BORDER_THICKNESS = 1
GROUP_BORDER_RADIUS = 12
GROUP_BORDER_COLOR = "#C8CDD3"
BETWEEN_CARD_PADDING = 10
MARGINS_APP_AREA = (10, 20, 10, 20)

SPC_IMPORT = 0
SPC_OBJECTS = 5
SPC_NAV = 9
SPC_EXPORT = 9
SPC_LOG = 0

ROW_IMPORT_SPACING = 8
ROW_OBJECTS_SPACING = 8
ROW_NAV_SPACING = 8
ROW_EXPORT_SPACING = 8

M_IMPORT_FOLDER_ROW = (7, 0, 16, 0)
M_IMPORT_FIELDS_ROW = (16, 2, 16, 0)
M_IMPORT_FILTERS_ROW = (16, 2, 16, 0)
M_OBJECTS_ROW = (16, 0, 16, 0)
M_LOADED_ROW = (16, 0, 16, 0)
M_NAV_BUTTONS_ROW = (16, 0, 16, 0)
M_EXPORT_PATH_ROW = (16, 0, 16, 0)
M_EXPORT_BUTTONS_ROW = (16, 0, 18, 0)
M_LOG_TEXT_ROW = (7, 0, 7, 0)

S_IMPORT_FOLDER_OBJECTS = 4
S_CASE_IDENTIFY = 12
S_IMPORT_FIELDS_OBJECTS = 8
S_IMPORT_FILTERS_OBJECTS = 8
S_OBJECTS_OBJECTS = 8
S_NAV_BUTTONS_OBJECTS = 8
S_EXPORT_BUTTONS_OBJECTS = 10

ACTION_BUTTONS_GAP = 8
GROUP_MARGINS = (20, 27, 20, 38)
LOG_GROUP_MARGINS = (20, 27, 20, 20)

IMPORT_FIELD_MIN_W = 120
OBJECT_FIELD_MIN_W = 80
FOLDER_MIN_W = 320
EXPORT_PATH_MIN_W = 420

NAV_BTN_W = 162
ASSESS_BTN_W = 220
EXPORT_BTN_W = 274

FIELD_LABEL_SP = 0

PROG_MARGIN_TOP = 0
PROG_MARGIN_BOTTOM = 0
PROG_MARGIN_LEFT = 16
PROG_MARGIN_RIGHT = 16
PROG_CORNER_RADIUS = 9
PROG_BORDER_COLOR = "#C8CDD3"
PROG_BG_COLOR = "#FFFFFF"
PROG_CHUNK_COLOR = "#228B22"

CONFIG_VENDOR = "ThePyottLab"
CONFIG_APP = "PillarModiolarClassifier"
CONFIG_FILENAME = "config.ini"


def _qss_light(dropdown_png_path: str, border_color: str, border_radius: int, border_thickness: int) -> str:
    """Return a light QSS theme string parameterized by shared values."""
    return f"""
QMainWindow {{ background: #F6F7F9; }}
QWidget     {{ font-size: 10pt; }}
QGroupBox, QLabel {{ color: #222; }}
QLineEdit, QComboBox, QTextEdit, QPlainTextEdit {{
    background: #FFFFFF;
    border: 1px solid #CCD3DB;
    border-radius: 8px;
    padding: 5px 7px;
}}
QWidget[group="true"] {{
    background: #FFFFFF;
    border: {border_thickness}px solid {border_color};
    border-radius: {border_radius}px;
}}
QWidget[group="true"] > * {{
    margin: 0px;
}}
QComboBox {{
    padding-right: 28px;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border-left: 1px solid #CCD3DB;
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #f6f8fb, stop:1 #e9eef5);
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}}
QComboBox::down-arrow {{
    image: {dropdown_png_path};
    width: 10px; height: 10px;
}}
QPushButton {{
    background: #FFFFFF;
    border: 1px solid #C8CDD3;
    border-radius: 9px;
    padding: 5px 10px;
}}
QPushButton:hover  {{ background: #f0f3f7; }}
QPushButton:pressed{{ background: #e7ebf1; }}
QPushButton#import {{
    background: #1E66FF;
    border: 1px solid #1A56E0;
    color: #FFFFFF;
    font-weight: 600;
    border-radius: 9px;
    padding: 5px 10px;
}}
QPushButton#import:hover {{
    background: #1753E6;
    border-color: #1347C7;
}}
QPushButton#import:pressed {{
    background: #0F43C9;
    border-color: #0C36A7;
}}
QCheckBox {{ spacing: 6px; margin-left: 0px; }}
""".strip()


def _wrap_tt(text: str, width: int = TOOLTIP_WRAP_CH) -> str:
    """Soft-wrap tooltip text at roughly `width` characters."""
    return textwrap.fill(text, width=width)


class ToolTipDelayFilter(QObject):
    """Event filter that implements a consistent tooltip delay."""

    def __init__(self, delay_ms: int = TOOLTIP_DELAY_MS, parent: QObject | None = None):
        """Initialize the filter."""
        super().__init__(parent)
        self._delay: int = delay_ms
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._target: QWidget | None = None
        self._timer.timeout.connect(self._show)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Filter tooltip-related events for the installed widgets."""
        t = event.type()

        if t in (QEvent.Enter, QEvent.HoverEnter):
            text = getattr(obj, "toolTip", lambda: "")()
            if text:
                self._target = obj
                self._timer.start(self._delay)
            return False

        if t in (QEvent.Leave, QEvent.HoverLeave, QEvent.FocusOut, QEvent.MouseButtonPress):
            self._timer.stop()
            self._target = None
            QToolTip.hideText()
            return False

        if t == QEvent.ToolTip:
            # suppress default immediate tooltip
            return True

        return False

    def _show(self) -> None:
        """Show the delayed tooltip for the current target, if any."""
        tgt = self._target
        if tgt is not None:
            pos = tgt.mapToGlobal(tgt.rect().center())
            QToolTip.showText(pos, tgt.toolTip(), tgt)


def _user_config_dir() -> Path:
    """Return and create (if missing) the user config directory."""
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = Path.home() / "AppData" / "Local"
    else:
        base = Path(base)
    cfg_dir = base / "Pillar-Modiolar Classifier"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir


def _user_config_path() -> Path:
    """Return the user config file path."""
    return _user_config_dir() / CONFIG_FILENAME


def _read_config_from(path: Path) -> configparser.ConfigParser | None:
    """Read a config file if it exists; return None on error or absence."""
    try:
        if not path.exists():
            return None
        cp = configparser.ConfigParser()
        cp.read(path)
        return cp
    except Exception:
        return None


def _ensure_parent_dir(p: Path) -> None:
    """Create the parent directory for a path if needed."""
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def header(text: str, gap_px: int) -> Label:
    """Return a styled header label widget."""
    lab = Label(value=text)
    lab.native.setStyleSheet(f"font-weight: 600; font-size: 11pt; margin: 0 0 {gap_px}px 11px;")
    return lab


def vfield(caption: str, widget) -> Container:
    """Return a vertical label+field container with consistent margins."""
    cap = Label(value=caption)
    cap.native.setStyleSheet(f"margin-bottom: {FIELD_LABEL_SP}px;")
    c = Container(widgets=[cap, widget], layout="vertical", labels=False)
    lay = c.native.layout()
    lay.setContentsMargins(0, 0, 0, 0)
    c.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return c


def set_layout(container: Container, margins=(0, 0, 0, 0), spacing=0) -> None:
    """Apply margins/spacing to a magicgui container."""
    lay = container.native.layout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    l, t, r, b = margins
    lay.setContentsMargins(l, t, r, b)
    lay.setSpacing(spacing)


def equal_stretch(row_container: Container) -> None:
    """Ensure equal stretch factors for a row of child widgets."""
    lay = row_container.native.layout()
    for i in range(lay.count()):
        lay.setStretch(i, 1)


def mark_group(container: Container) -> None:
    """Mark a container as a 'group' for QSS styling."""
    container.native.setProperty("group", True)
    container.native.style().unpolish(container.native)
    container.native.style().polish(container.native)


class ContrastProgressBar(QProgressBar):
    """Progress bar that preserves text contrast over the filled chunk."""

    def paintEvent(self, event: QPaintEvent) -> None:
        """Custom paint that flips text color over the filled region."""
        opt = QStyleOptionProgressBar()
        self.initStyleOption(opt)
        original_text = opt.text
        opt.text = ""
        painter = QPainter(self)
        self.style().drawControl(QStyle.CE_ProgressBar, opt, painter, self)
        text = original_text if self.isTextVisible() else ""
        if text:
            fm = QFontMetrics(self.font())
            r = self.rect()
            total_w = fm.horizontalAdvance(text)
            x = r.x() + max(0, (r.width() - total_w) // 2)
            y = r.y() + (r.height() + fm.ascent() - fm.descent()) // 2
            rng = max(1, self.maximum() - self.minimum())
            frac = float(self.value() - self.minimum()) / float(rng)
            chunk_x = r.x() + int(r.width() * frac)
            run_x = x
            for ch in text:
                cw = fm.horizontalAdvance(ch)
                cx = run_x + cw // 2
                painter.setPen(QColor(255, 255, 255) if cx <= chunk_x else QColor(0, 0, 0))
                painter.drawText(run_x, y, ch)
                run_x += cw
        painter.end()


class App:
    """Top-level GUI application controller."""

    def __init__(self) -> None:
        """Create the UI, wire events, and load persisted inputs."""
        self._pylogger = logging.getLogger("pmc.gui")

        self.w_folder = FileEdit(label="Folder", mode=FileDialogMode.EXISTING_DIRECTORY, value=None)
        self.w_case_insensitive = CheckBox(label="Case insensitive", value=True)

        self.w_extensions = LineEdit(label="", value=".xls")
        self.w_ribbons = LineEdit(label="", value="rib")
        self.w_psds = LineEdit(label="", value="psd")
        self.w_positions = LineEdit(label="", value="pos")

        self.w_identify_poles = CheckBox(label="Identify poles", value=True)

        self.w_ribbons_only = CheckBox(label="Ribbons only", value=False)
        self.w_psds_only = CheckBox(label="PSDs only", value=False)
        self.btn_update_all = PushButton(text="Update all fields")
        self.btn_load = PushButton(text="Load files")
        self.btn_load.native.setObjectName("import")

        self.w_ribbons_obj = LineEdit(label="", value="ribbons")
        self.w_psds_obj = LineEdit(label="", value="PSDs")
        self.w_pillar_obj = LineEdit(label="", value="pillar")
        self.w_modiolar_obj = LineEdit(label="", value="modiolar")

        self.cbo_group = ComboBox(label="Loaded IDs", choices=[])
        self.btn_prev = PushButton(text="◀ Prev")
        self.btn_next = PushButton(text="Next ▶")
        self.btn_assess = PushButton(text="Assess selected (open in viewer)")

        self.w_export_dir = FileEdit(
            label="Export folder (optional)",
            mode=FileDialogMode.EXISTING_DIRECTORY,
            value=None,
        )
        self.btn_export_selected = PushButton(text="Classify and export selected to csv…")
        self.btn_export_all = PushButton(text="Classify and export all to csv…")

        self.txt_log = TextEdit(value="", tooltip="Logs and progress")
        self.cbo_group.native.setEditable(False)

        self.w_case_insensitive.native.setToolTip(_wrap_tt(TT.case_insensitive))
        self.w_extensions.native.setToolTip(_wrap_tt(TT.file_extension))
        self.w_ribbons.native.setToolTip(_wrap_tt(TT.ribbons_sheet_id))
        self.w_psds.native.setToolTip(_wrap_tt(TT.psds_sheet_id))
        self.w_positions.native.setToolTip(_wrap_tt(TT.positions_sheet_id))
        self.w_identify_poles.native.setToolTip(_wrap_tt(TT.identify_poles))
        self.w_ribbons_only.native.setToolTip(_wrap_tt(TT.ribbons_only))
        self.w_psds_only.native.setToolTip(_wrap_tt(TT.psds_only))
        self.btn_update_all.native.setToolTip(_wrap_tt(TT.update_all_fields))
        self.w_ribbons_obj.native.setToolTip(_wrap_tt(TT.ribbons_obj_id))
        self.w_psds_obj.native.setToolTip(_wrap_tt(TT.psds_obj_id))
        self.w_pillar_obj.native.setToolTip(_wrap_tt(TT.pillar_obj_id))
        self.w_modiolar_obj.native.setToolTip(_wrap_tt(TT.modiolar_obj_id))

        # Sizing
        def _min_w(widget, w: int, expanding: bool = False) -> None:
            widget.native.setMinimumWidth(w)
            if expanding:
                widget.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            else:
                widget.native.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        _min_w(self.w_folder, FOLDER_MIN_W, expanding=True)
        _min_w(self.w_export_dir, EXPORT_PATH_MIN_W, expanding=True)
        for lew in (self.w_extensions, self.w_ribbons, self.w_psds, self.w_positions):
            _min_w(lew, IMPORT_FIELD_MIN_W, expanding=True)
        for lew in (self.w_ribbons_obj, self.w_psds_obj, self.w_pillar_obj, self.w_modiolar_obj):
            _min_w(lew, OBJECT_FIELD_MIN_W, expanding=True)

        for b, w in ((self.btn_prev, NAV_BTN_W), (self.btn_next, NAV_BTN_W), (self.btn_assess, ASSESS_BTN_W)):
            b.native.setMinimumWidth(w)
            b.native.setMaximumWidth(w)
            b.native.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        for b in (self.btn_export_selected, self.btn_export_all):
            b.native.setMinimumWidth(EXPORT_BTN_W)
            b.native.setMaximumWidth(EXPORT_BTN_W)
            b.native.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # --- Layout -------------------------------------------------------------------
        source_controls = Container(widgets=[self.w_case_insensitive], layout="horizontal", labels=False)
        set_layout(source_controls, margins=(0, 0, 0, 0), spacing=S_CASE_IDENTIFY)

        folder_row = Container(
            widgets=[Container(widgets=[self.w_folder], layout="horizontal", labels=False), source_controls],
            layout="horizontal",
            labels=False,
        )
        set_layout(folder_row, margins=M_IMPORT_FOLDER_ROW, spacing=S_IMPORT_FOLDER_OBJECTS)
        lay = folder_row.native.layout()
        lay.setStretch(0, 1)
        lay.setStretch(1, 0)

        import_fields_row = Container(
            widgets=[
                vfield("File extension", self.w_extensions),
                vfield("Ribbon volume sheet ID", self.w_ribbons),
                vfield("PSD volume sheet ID", self.w_psds),
                vfield("Position sheet ID", self.w_positions),
            ],
            layout="horizontal",
            labels=False,
        )
        set_layout(import_fields_row, margins=M_IMPORT_FIELDS_ROW, spacing=S_IMPORT_FIELDS_OBJECTS)
        equal_stretch(import_fields_row)

        right_buttons = Container(widgets=[self.btn_update_all, self.btn_load], layout="horizontal", labels=False)
        set_layout(right_buttons, margins=(0, 0, 0, 0), spacing=ACTION_BUTTONS_GAP)

        import_filters_row = Container(
            widgets=[self.w_identify_poles, self.w_ribbons_only, self.w_psds_only, right_buttons],
            layout="horizontal",
            labels=False,
        )
        set_layout(import_filters_row, margins=M_IMPORT_FILTERS_ROW, spacing=S_IMPORT_FILTERS_OBJECTS)
        lay = import_filters_row.native.layout()
        lay.insertStretch(3, 1)

        grp_import = Container(
            widgets=[header("Import", SPC_IMPORT), folder_row, import_fields_row, import_filters_row],
            layout="vertical",
            labels=False,
        )
        set_layout(grp_import, margins=GROUP_MARGINS, spacing=ROW_IMPORT_SPACING)
        mark_group(grp_import)

        object_row = Container(
            widgets=[
                vfield("Ribbon object ID", self.w_ribbons_obj),
                vfield("PSD object ID", self.w_psds_obj),
                vfield("Pillar object ID", self.w_pillar_obj),
                vfield("Modiolar object ID", self.w_modiolar_obj),
            ],
            layout="horizontal",
            labels=False,
        )
        set_layout(object_row, margins=M_OBJECTS_ROW, spacing=S_OBJECTS_OBJECTS)
        equal_stretch(object_row)

        grp_objects = Container(widgets=[header("Object names", SPC_OBJECTS), object_row], layout="vertical", labels=False)
        set_layout(grp_objects, margins=GROUP_MARGINS, spacing=ROW_OBJECTS_SPACING)
        mark_group(grp_objects)

        loaded_row = Container(widgets=[self.cbo_group], layout="horizontal", labels=False)
        set_layout(loaded_row, margins=M_LOADED_ROW, spacing=0)
        loaded_row.native.layout().setStretch(0, 1)

        nav_buttons = Container(widgets=[self.btn_prev, self.btn_assess, self.btn_next], layout="horizontal", labels=False)
        set_layout(nav_buttons, margins=M_NAV_BUTTONS_ROW, spacing=S_NAV_BUTTONS_OBJECTS)
        lay = nav_buttons.native.layout()
        lay.insertStretch(0, 1)
        lay.addStretch(1)

        grp_nav = Container(widgets=[header("Loaded IDs", SPC_NAV), loaded_row, nav_buttons], layout="vertical", labels=False)
        set_layout(grp_nav, margins=GROUP_MARGINS, spacing=ROW_NAV_SPACING)
        mark_group(grp_nav)

        export_path_row = Container(widgets=[self.w_export_dir], layout="horizontal", labels=False)
        set_layout(export_path_row, margins=M_EXPORT_PATH_ROW, spacing=0)
        export_path_row.native.layout().setStretch(0, 1)

        export_buttons = Container(widgets=[self.btn_export_selected, self.btn_export_all], layout="horizontal", labels=False)
        set_layout(export_buttons, margins=M_EXPORT_BUTTONS_ROW, spacing=S_EXPORT_BUTTONS_OBJECTS)

        grp_export = Container(
            widgets=[header("Export", SPC_EXPORT), export_path_row, export_buttons],
            layout="vertical",
            labels=False,
        )
        set_layout(grp_export, margins=GROUP_MARGINS, spacing=ROW_EXPORT_SPACING)
        mark_group(grp_export)

        log_text_row = Container(widgets=[self.txt_log], layout="horizontal", labels=False)
        log_text_row.native.setContentsMargins(*M_LOG_TEXT_ROW)

        grp_log = Container(widgets=[header("Log", SPC_LOG), log_text_row], layout="vertical", labels=False)
        set_layout(grp_log, margins=LOG_GROUP_MARGINS, spacing=ROW_EXPORT_SPACING)
        mark_group(grp_log)

        self._progress_wrap = QWidget(grp_log.native)
        self._progress_wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        wrap_layout = QVBoxLayout(self._progress_wrap)
        wrap_layout.setContentsMargins(PROG_MARGIN_LEFT, PROG_MARGIN_TOP, PROG_MARGIN_RIGHT, PROG_MARGIN_BOTTOM)
        wrap_layout.setSpacing(0)

        self._progress = ContrastProgressBar(grp_log.native)
        self._progress.setTextVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setStyleSheet(
            f"""
            QProgressBar {{
                border: 1px solid {PROG_BORDER_COLOR};
                border-radius: {PROG_CORNER_RADIUS}px;
                background: {PROG_BG_COLOR};
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {PROG_CHUNK_COLOR};
                border-radius: {PROG_CORNER_RADIUS}px;
            }}
            """
        )
        self._progress.hide()
        wrap_layout.addWidget(self._progress)
        grp_log.native.layout().insertWidget(2, self._progress_wrap)

        self.panel = Container(widgets=[grp_import, grp_objects, grp_nav, grp_export, grp_log], layout="vertical", labels=False)
        set_layout(self.panel, margins=MARGINS_APP_AREA, spacing=BETWEEN_CARD_PADDING)

        # tooltip delay filter
        self._tt_filter = ToolTipDelayFilter(parent=self.panel.native)
        for w in [
            self.w_case_insensitive.native,
            self.w_extensions.native,
            self.w_ribbons.native,
            self.w_psds.native,
            self.w_positions.native,
            self.w_identify_poles.native,
            self.w_ribbons_only.native,
            self.w_psds_only.native,
            self.btn_update_all.native,
            self.w_ribbons_obj.native,
            self.w_psds_obj.native,
            self.w_pillar_obj.native,
            self.w_modiolar_obj.native,
        ]:
            w.installEventFilter(self._tt_filter)

        self.cbo_group.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.txt_log.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.cfg: FinderConfig | None = None
        self.groups: dict[str, Group] = {}
        self.df_cache: dict[str, pd.DataFrame] = {}
        self.current_viewer: napari.Viewer | None = None
        self._window: QMainWindow | None = None
        self._progress_total: int | None = None
        self._progress_value: int = 0

        self._load_persisted_inputs()
        self.cfg = self._snap_cfg()

        self._wire_return_updates()
        self.w_ribbons_only.changed.connect(lambda v: self._checkbox_update("ribbons_only", "Ribbons only", bool(v), invalidate=True))
        self.w_psds_only.changed.connect(lambda v: self._checkbox_update("psds_only", "PSDs only", bool(v), invalidate=True))
        self.w_identify_poles.changed.connect(lambda v: self._checkbox_update("identify_poles", "Identify poles", bool(v)))
        self.w_case_insensitive.changed.connect(lambda v: self._checkbox_update("case_insensitive", "Case insensitive", bool(v)))

        self._update_mode_ui()

        self.btn_update_all.changed.connect(lambda *_: self.on_update_all_fields())
        self.btn_load.changed.connect(lambda *_: self.on_load_groups())
        self.btn_assess.changed.connect(lambda *_: self.on_assess_selected())
        self.btn_export_selected.changed.connect(lambda *_: self.on_export_selected())
        self.btn_export_all.changed.connect(lambda *_: self.on_export_all())
        self.btn_prev.changed.connect(lambda *_: self.on_prev())
        self.btn_next.changed.connect(lambda *_: self.on_next())


    def log(self, *args, level: int | None = None) -> None:
        """Append a log line to the GUI and emit via the Python logger.

        - If message starts with ``[error]``, ``[warn]``, or ``[debug]`` (case-insensitive),
          the corresponding logging level is used and the prefix is stripped.
        - Otherwise INFO level is used (or an explicit ``level`` if provided).
        """
        raw = " ".join(str(a) for a in args)
        msg = raw
        lvl = level or logging.INFO

        prefix_map = {
            "[error]": logging.ERROR,
            "[warn]": logging.WARNING,
            "[warning]": logging.WARNING,
            "[debug]": logging.DEBUG,
        }
        low = raw.strip().lower()
        for prefix, plvl in prefix_map.items():
            if low.startswith(prefix):
                lvl = plvl
                msg = raw[len(prefix):].lstrip()
                break

        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts} | {msg}"
        prev = self.txt_log.value or ""
        self.txt_log.value = line + "\n" + prev
        self.txt_log.native.verticalScrollBar().setValue(0)

        self._pylogger.log(lvl, msg)


    def _process_events(self) -> None:
        """Safely flush the Qt event loop (instance method form)."""
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _progress_start(self, title: str, total: int | None = None) -> None:
        """Start a progress section with an optional determinate range."""
        self._progress_total = total
        self._progress_value = 0
        self._progress.show()
        if total is None:
            self._progress.setRange(0, 0)
        else:
            self._progress.setRange(0, total)
            self._progress.setValue(0)
        self._progress.setFormat(f"{title} %p%")
        self._process_events()

    def _progress_tick(self, message: str | None = None) -> None:
        """Advance progress by one step and optionally log a message."""
        if self._progress_total is None:
            return
        self._progress_value += 1
        self._progress.setValue(self._progress_value)
        if message:
            self._progress.setFormat(f"{message} %p%")
            self.log(message)
        self._process_events()

    def _progress_finish(self, message: str | None = None) -> None:
        """Finish progress display and reset internal counters."""
        if message:
            self._progress.setFormat(message)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._process_events()
        self._progress.hide()
        self._progress_total = None
        self._progress_value = 0


    def _remember_enabled(self) -> bool:
        """Return whether input persistence is enabled (default True)."""
        cp = _read_config_from(_user_config_path())
        try:
            if cp is None:
                return True
            return cp.has_section("inputs") and cp.getboolean("inputs", "remember_input_fields", fallback=True)
        except Exception:
            return True

    def _persist_inputs_if_enabled(self) -> None:
        """Persist current input fields to the user config when allowed."""
        if not (self.cfg and self.cfg.remember_input_fields):
            return
        cfg_path = _user_config_path()
        _ensure_parent_dir(cfg_path)
        cp = configparser.ConfigParser()
        if cfg_path.exists():
            try:
                cp.read(cfg_path)
            except Exception:
                cp = configparser.ConfigParser()
        if "inputs" not in cp:
            cp["inputs"] = {}
        snap = asdict(self._snap_cfg())
        snap["folder"] = str(snap["folder"])
        for k, v in snap.items():
            cp["inputs"][k] = str(v)
        with cfg_path.open("w", encoding="utf-8") as f:
            cp.write(f)

    def _load_persisted_inputs(self) -> None:
        """Load input values from the user config, migrating legacy files if needed."""
        user_cfg = _read_config_from(_user_config_path())
        if user_cfg is None:
            cp = configparser.ConfigParser()
            cp["inputs"] = {}
            snap = asdict(FinderConfig(folder=Path(".")))
            snap["folder"] = str(Path("."))
            for k, v in snap.items():
                cp["inputs"][k] = str(v)
            cfg_path = _user_config_path()
            _ensure_parent_dir(cfg_path)
            try:
                with cfg_path.open("w", encoding="utf-8") as f:
                    cp.write(f)
            except Exception:
                pass
            user_cfg = cp

        if "inputs" not in user_cfg:
            return
        sec = user_cfg["inputs"]

        def _set(le: LineEdit, key: str, default: str) -> None:
            le.native.setText(sec.get(key, fallback=default))

        folder_txt = sec.get("folder", fallback="")
        if folder_txt:
            self.w_folder.value = folder_txt

        _set(self.w_extensions, "extensions", ".xls")
        _set(self.w_ribbons, "ribbons", "ribbon")
        _set(self.w_psds, "psds", "psd")
        _set(self.w_positions, "positions", "sum")

        _set(self.w_ribbons_obj, "ribbons_obj", "ribbons")
        _set(self.w_psds_obj, "psds_obj", "PSDs")
        _set(self.w_pillar_obj, "pillar_obj", "pillar")
        _set(self.w_modiolar_obj, "modiolar_obj", "modiolar")

        self.w_case_insensitive.value = sec.getboolean("case_insensitive", fallback=True)
        self.w_ribbons_only.value = sec.getboolean("ribbons_only", fallback=False)
        self.w_psds_only.value = sec.getboolean("psds_only", fallback=False)
        self.w_identify_poles.value = sec.getboolean("identify_poles", fallback=True)
        self._update_mode_ui()

    def _update_mode_ui(self) -> None:
        """Enable/disable inputs based on ribbons-only or PSDs-only toggles."""
        rib_only = bool(self.w_ribbons_only.value)
        psd_only = bool(self.w_psds_only.value)
        self.w_psds.enabled = not rib_only
        self.w_psds_obj.enabled = not rib_only
        self.w_ribbons.enabled = not psd_only
        self.w_ribbons_obj.enabled = not psd_only

    def _get_text(self, widget: LineEdit) -> str:
        """Return trimmed text from a ``LineEdit`` widget."""
        return widget.native.text().strip()

    def _wire_return_updates(self) -> None:
        """Apply edits on Enter for relevant fields, with cache invalidation rules."""
        mapping: dict[LineEdit, tuple[str, str, bool]] = {
            self.w_extensions: ("extensions", "File extension", True),
            self.w_ribbons: ("ribbons", "Ribbon volume sheet ID", True),
            self.w_psds: ("psds", "PSD volume sheet ID", True),
            self.w_positions: ("positions", "Position sheet ID", True),
            self.w_ribbons_obj: ("ribbons_obj", "Ribbon object ID", False),
            self.w_psds_obj: ("psds_obj", "PSDs object ID", False),
            self.w_pillar_obj: ("pillar_obj", "Pillar object ID", False),
            self.w_modiolar_obj: ("modiolar_obj", "Modiolar object ID", False),
        }
        for widget, (attr, label, invalidate) in mapping.items():
            widget.native.returnPressed.connect(
                lambda w=widget, a=attr, lbl=label, inv=invalidate: self._apply_edit_update(w, a, lbl, invalidate=inv)
            )

    def _apply_edit_update(self, widget: LineEdit, cfg_attr: str, label: str, *, invalidate: bool) -> None:
        """Persist a text edit, optionally invalidating caches."""
        value = self._get_text(widget)
        if self.cfg is None:
            self.cfg = self._snap_cfg()
        setattr(self.cfg, cfg_attr, value)
        if invalidate:
            self._invalidate_cache()
        self._persist_inputs_if_enabled()
        self.log(f"{label} changed to '{value}'")

    def _checkbox_update(self, cfg_attr: str, label: str, state: bool, *, invalidate: bool = False) -> None:
        """Persist a checkbox change, optionally invalidating caches."""
        if self.cfg is None:
            self.cfg = self._snap_cfg()
        if cfg_attr == "ribbons_only" and state and self.w_psds_only.value:
            self.w_psds_only.value = False
        if cfg_attr == "psds_only" and state and self.w_ribbons_only.value:
            self.w_ribbons_only.value = False
        setattr(self.cfg, cfg_attr, state)
        if invalidate:
            self._invalidate_cache()
        self._persist_inputs_if_enabled()
        self._update_mode_ui()
        self.log(f"{label} {'enabled' if state else 'disabled'}")

    def on_update_all_fields(self) -> None:
        """Apply all current UI fields to the live configuration."""
        self.cfg = self._snap_cfg()
        self._persist_inputs_if_enabled()
        self.log("All fields updated")

    def _invalidate_cache(self) -> None:
        """Clear the per-group dataframe cache."""
        self.df_cache.clear()

    def _snap_cfg(self) -> FinderConfig:
        """Snapshot current UI values into a ``FinderConfig`` instance."""
        folder_path = Path(self.w_folder.value) if self.w_folder.value else Path(".")
        return FinderConfig(
            folder=folder_path,
            ribbons=self._get_text(self.w_ribbons),
            psds=self._get_text(self.w_psds),
            positions=self._get_text(self.w_positions),
            extensions=self._get_text(self.w_extensions),
            ribbons_obj=self._get_text(self.w_ribbons_obj),
            psds_obj=self._get_text(self.w_psds_obj),
            pillar_obj=self._get_text(self.w_pillar_obj),
            modiolar_obj=self._get_text(self.w_modiolar_obj),
            case_insensitive=bool(self.w_case_insensitive.value),
            ribbons_only=bool(self.w_ribbons_only.value),
            psds_only=bool(self.w_psds_only.value),
            identify_poles=bool(self.w_identify_poles.value),
            remember_input_fields=self._remember_enabled(),
        )

    def _make_cfg(self) -> FinderConfig:
        """Construct a ``FinderConfig`` from UI values with basic validation."""
        folder_v = self.w_folder.value
        if folder_v is None:
            raise ValueError("Please select a folder.")
        folder_path = Path(folder_v)
        return FinderConfig(
            folder=folder_path,
            ribbons=self._get_text(self.w_ribbons),
            psds=self._get_text(self.w_psds),
            positions=self._get_text(self.w_positions),
            extensions=self._get_text(self.w_extensions),
            ribbons_obj=self._get_text(self.w_ribbons_obj),
            psds_obj=self._get_text(self.w_psds_obj),
            pillar_obj=self._get_text(self.w_pillar_obj),
            modiolar_obj=self._get_text(self.w_modiolar_obj),
            case_insensitive=bool(self.w_case_insensitive.value),
            ribbons_only=bool(self.w_ribbons_only.value),
            psds_only=bool(self.w_psds_only.value),
            identify_poles=bool(self.w_identify_poles.value),
            remember_input_fields=self._remember_enabled(),
        )


    def _build_df_for_group(self, gid: str) -> tuple[pd.DataFrame, object]:
        """Build the processed dataframe and plane bundles for a group id."""
        if gid not in self.groups:
            raise KeyError(f"Unknown group id: {gid}")
        g = self.groups[gid]
        ribbons_df, psds_df, positions_df = parse_group(g, self.cfg)
        ribbons_df = process_volume_df(ribbons_df)
        psds_df = process_volume_df(psds_df)
        positions_df = process_position_df(positions_df, self.cfg)
        df = merge_dfs(ribbons_df, psds_df, positions_df, self.cfg)
        msg = self._ensure_required_objects_in_df(df, self.cfg, gid)

        if self.cfg.identify_poles:
            df = identify_poles(df, self.cfg)

        pm_bundle = build_pm_planes(df, self.cfg)
        hc_bundle = build_hc_planes(df, self.cfg)

        df = classify_synapses(df, self.cfg, pm_bundle, hc_bundle)
        return df, (pm_bundle, hc_bundle)

    def _ensure_required_objects_in_df(self, df: pd.DataFrame, cfg: FinderConfig, gid: str) -> str:
        """Ensure that required object categories exist; return a summary message."""
        if "object" not in df.columns:
            raise RuntimeError("Expected column 'object' not found in data.")

        def _norm(s: str) -> str:
            s = str(s).strip()
            return s.lower() if cfg.case_insensitive else s

        present = {_norm(x) for x in df["object"].astype(str).unique()}
        required_map: dict[str, str] = {
            "Pillar object ID": cfg.pillar_obj,
            "Modiolar object ID": cfg.modiolar_obj,
        }
        if not cfg.psds_only:
            required_map["Ribbon object ID"] = cfg.ribbons_obj
        if not cfg.ribbons_only:
            required_map["PSDs object ID"] = cfg.psds_obj
        missing_msgs: list[str] = []
        group_files: list[str] = []
        if gid in self.groups:
            group_files = [Path(p).name for p in self.groups[gid].file_paths.values()]
        for label, val in required_map.items():
            if _norm(val) not in present:
                if group_files:
                    missing_msgs.append(f"{label} '{val}' not found in any loaded file(s): " + ", ".join(group_files))
                else:
                    missing_msgs.append(f"{label} '{val}' not found in dataset.")
        if missing_msgs:
            raise RuntimeError("\n".join(missing_msgs))

        n_rib = len(df.loc[df["object"] == "Unclassified " + cfg.ribbons_obj])
        n_psd = len(df.loc[df["object"] == "Unclassified " + cfg.psds_obj])

        msg = f"{n_rib} ribbon(s) and {n_psd} PSD(s) were not allocated to any IHCs and are excluded from classification."
        return msg


    def _confirm_export_dir(self) -> Path | None:
        """Return the selected export directory, prompting if empty."""
        val = self.w_export_dir.value
        if val and str(val) != ".":
            return Path(val)
        return prompt_export_dir()

    def _safe_close_viewer(self) -> None:
        """Close the existing viewer if present, ignoring errors."""
        v = self.current_viewer
        self.current_viewer = None
        if not v:
            return
        try:
            v.close()
        except Exception:
            pass

    def on_load_groups(self) -> None:
        """Load and index groups from the selected folder."""
        self.cfg = self._make_cfg()
        self._persist_inputs_if_enabled()
        try:
            self.groups = find_groups(self.cfg) or {}
            gids = sorted(self.groups.keys())
            self.cbo_group.choices = gids
            if gids:
                self.cbo_group.value = gids[0]
                self.log(f"Loaded {len(gids)} ID(s).")
            else:
                self.log(
                    "No groups found. Check your suffixes/extensions and folder.")
        except GroupValidationError as e:
            self.groups = e.groups or {}
            gids = sorted(self.groups.keys())
            self.cbo_group.choices = gids
            if gids:
                self.cbo_group.value = gids[0]
                self.log(f"Loaded {len(gids)} ID(s).")
            self.log(f"[warn] {e}")
        except Exception as e:
            self.log(f"[error] Identify files: {e}")

    def on_assess_selected(self) -> None:
        """Open the currently selected group in the napari viewer."""
        self.cfg = self._make_cfg()
        self._persist_inputs_if_enabled()
        gid = self.cbo_group.value
        if not gid:
            self.log("Select a group first.")
            return
        try:
            if gid in self.df_cache:
                total_steps = 3
                self._progress_start(f"Opening {gid}…", total_steps)
                df = self.df_cache[gid]
                if self.cfg.identify_poles:
                    df = identify_poles(df, self.cfg)
                pm_bundle = build_pm_planes(df, self.cfg)
                hc_bundle = build_hc_planes(df, self.cfg)
                self._progress_tick()
                msg = self._ensure_required_objects_in_df(df, self.cfg, gid)
                self._progress_tick()
            else:
                total_steps = 8
                self._progress_start(f"Opening {gid}…", total_steps)
                g = self.groups[gid]
                ribbons_df, psds_df, positions_df = parse_group(g, self.cfg)
                self._progress_tick()
                ribbons_df = process_volume_df(ribbons_df)
                self._progress_tick()
                psds_df = process_volume_df(psds_df)
                self._progress_tick()
                positions_df = process_position_df(positions_df, self.cfg)
                self._progress_tick()
                df = merge_dfs(ribbons_df, psds_df, positions_df, self.cfg)
                self._progress_tick()
                msg = self._ensure_required_objects_in_df(df, self.cfg, gid)
                self._progress_tick(msg)
                if self.cfg.identify_poles:
                    df = identify_poles(df, self.cfg)
                pm_bundle = build_pm_planes(df, self.cfg)
                hc_bundle = build_hc_planes(df, self.cfg)
                df = classify_synapses(df, self.cfg, pm_bundle, hc_bundle)
                self._progress_tick("Classified synapses.")
                self.df_cache[gid] = df
        except Exception as e:
            self._progress_finish("Failed")
            self.log(f"[error] Could not build/verify dataset for '{gid}': {e}")
            return
        try:
            self._progress_tick("Rendering viewer...")
            self.current_viewer = self._open_viewer(df, (pm_bundle, hc_bundle))
            self._progress_finish("Done")
        except Exception as e:
            msg = str(e)
            if ("QtViewer has been deleted" in msg) or ("wrapped C/C++ object" in msg and "QtViewer" in msg):
                try:
                    self._safe_close_viewer()
                    self.current_viewer = self._open_viewer(df, (pm_bundle, hc_bundle))
                    self._progress_finish("Done")
                except Exception as e2:
                    self._progress_finish("Failed")
                    self.log(f"[error] Opening viewer failed: {e2}")
                    self.current_viewer = None
                    return
            else:
                self._progress_finish("Failed")
                self.log(f"[error] Opening viewer failed: {e}")
                self.current_viewer = None
                return
        self.log(f"Opened group '{gid}' in napari.")

    def _open_viewer(self, df: pd.DataFrame, plane_bundles) -> napari.Viewer:
        """Create or reuse a viewer and draw the current dataset."""
        reuse = self.current_viewer if (self.current_viewer is not None and getattr(self.current_viewer, "window", None) is not None) else None
        pm_bundle, hc_bundle = plane_bundles
        viewer = draw_objects(df, self.cfg, pm_bundle, hc_bundle, viewer=reuse)
        return viewer

    def on_export_selected(self) -> None:
        """Classify and export the currently selected group to CSV."""
        self.cfg = self._make_cfg()
        self._persist_inputs_if_enabled()
        gid = self.cbo_group.value
        if not gid:
            self.log("Select a group first.")
            return
        out_dir = self._confirm_export_dir()
        self.w_export_dir.value = out_dir
        if not out_dir:
            self.log("Export cancelled (no folder selected).")
            return
        try:
            if gid in self.df_cache:
                df = self.df_cache[gid]
            else:
                df, _ = self._build_df_for_group(gid)
                self.df_cache[gid] = df
            _ = self._ensure_required_objects_in_df(df, self.cfg, gid)
        except Exception as e:
            self.log(f"[error] Cannot export '{gid}': {e}")
            return
        try:
            out_path = export_df_csv(df, gid, out_dir)
            self.log(f"Exported '{gid}' to {out_path}")
        except Exception as e:
            self.log(f"[error] Export failed for '{gid}': {e}")

    def on_export_all(self) -> None:
        """Classify and export all loaded groups to CSV."""
        self.cfg = self._make_cfg()
        self._persist_inputs_if_enabled()
        if not self.groups:
            self.log("No groups loaded.")
            return
        out_dir = self._confirm_export_dir()
        self.w_export_dir.value = out_dir
        if not out_dir:
            self.log("Export cancelled (no folder selected).")
            return
        total = len(self.groups)
        ok = 0
        for i, gid in enumerate(sorted(self.groups.keys()), start=1):
            try:
                if gid in self.df_cache:
                    df = self.df_cache[gid]
                else:
                    df, _ = self._build_df_for_group(gid)
                    self.df_cache[gid] = df
                _ = self._ensure_required_objects_in_df(df, self.cfg, gid)
                out_path = export_df_csv(df, gid, out_dir)
                ok += 1
                self.log(f"[{i}/{total}] Exported '{gid}' to {out_path}")
            except Exception as e:
                self.log(f"[{i}/{total}] [error] Export failed for '{gid}': {e}")
        self.log(f"Finished export: {ok}/{total} succeeded.")

    def on_prev(self) -> None:
        """Navigate to the previous loaded group and open it."""
        choices = list(self.cbo_group.choices or [])
        if not choices:
            return
        idx = choices.index(self.cbo_group.value)
        if idx > 0:
            self.cbo_group.value = choices[idx - 1]
            self.on_assess_selected()

    def on_next(self) -> None:
        """Navigate to the next loaded group and open it."""
        choices = list(self.cbo_group.choices or [])
        if not choices:
            return
        idx = choices.index(self.cbo_group.value)
        if idx < len(choices) - 1:
            self.cbo_group.value = choices[idx + 1]
            self.on_assess_selected()


    def show(self) -> None:
        """Create and show the main window if needed."""
        if getattr(self, "_window", None) is None:
            self._window = QMainWindow()
            self._window.setWindowTitle("Pillar–Modiolar Classifier")
            icon_path = resource_path("icon.ico")
            self._window.setWindowIcon(QIcon(icon_path))
            central = QWidget(self._window)
            lay = QVBoxLayout(central)
            lay.setContentsMargins(8, TOP_PANEL_PADDING, 8, BOTTOM_PANEL_PADDING)
            lay.setSpacing(0)
            lay.addWidget(self.panel.native)
            self._window.setCentralWidget(central)
            self._window.resize(APP_WIDTH, APP_HEIGHT)
            self._window.setMinimumSize(APP_WIDTH, APP_HEIGHT)
            chev = resource_path("chevron_down.png").replace("\\", "/")
            arrow_path = f'url("{chev}")'
            self._window.setStyleSheet(
                _qss_light(
                    arrow_path,
                    border_color=GROUP_BORDER_COLOR,
                    border_radius=GROUP_BORDER_RADIUS,
                    border_thickness=GROUP_BORDER_THICKNESS,
                )
            )
        self._window.show()


def launch_gui() -> None:
    """Launch the GUI and start the napari event loop.

    Ensures logging is configured once (idempotent) so that all self.log(...)
    messages also appear in the terminal.
    """
    configure_logging()  # <-- sets root StreamHandler -> terminal
    logging.getLogger("pmc.gui").info("Launching Pillar–Modiolar Classifier GUI")
    app = App()
    app.show()
    napari.run()
