'''
UI main
'''
# Standard import
import sys
import os
import json
import copy
import logging
import math
import time
from pathlib import Path

# PySide 6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QCheckBox, QFileDialog, QHBoxLayout, QLineEdit,
    QPlainTextEdit, QGroupBox, QFormLayout, QGridLayout, QFrame,
    QSizePolicy, QComboBox, QScrollArea, QStackedWidget, QDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)
from PySide6.QtGui import (QTextCharFormat, QColor, QTextCursor, QPixmap,
                          QImage, QIcon, QIntValidator)
from PySide6.QtCore import Qt, Signal

# Local import
from src.utils.logger import logger
from src.utils.ui import (
    validate_numerical_input, clear_debug_canvas,
    create_error_label, SingleKeyEdit, QtLogHandler, create_advance_setting_gbox,
    ResponsiveImageLabel,
)
from src.utils.common import (
    load_yaml, override_cfg, is_mac, save_yaml, get_cfg_diff,
    load_yaml_with_comments, retain_explicit_config_values
)
from src.engine.OpenVinoMonsterDetector import MODEL_HEIGHT, MODEL_WIDTH
from src.engine.NameTagProfileRepository import NameTagProfileRepository
from src.app_paths import writable_path
from src.config_compat import (
    ConfigCompatibilityError,
    migrate_health_monitor_config,
    validate_custom_config,
)
from src.runtime_policy import RUNTIME_POLICY
from src.ui.announcement import ANNOUNCEMENT
from src.engine.ReadinessDiagnostics import CHECKS, initial_rows

ADV_SETTINGS_HIDE = ['key', 'bot'] # cfg tile here will not shown in advanced settings tabs

APP_QSS = """
QMainWindow, QWidget#AppRoot { background: #0b1220; color: #dbeafe; }
QWidget { color: #dbeafe; }
QLabel, QCheckBox, QRadioButton { color: #dbeafe; background: transparent; }
QFrame#Sidebar { background: #0f172a; border-right: 1px solid #1e293b; }
QLabel#Brand { color: #f8fafc; font-size: 19px; font-weight: 700; }
QLabel#PageTitle { color: #f8fafc; font-size: 22px; font-weight: 700; }
QLabel#Subtle { color: #94a3b8; }
QLabel#StatusPill { border-radius: 10px; padding: 4px 10px; font-weight: 700; }
QFrame#AnnouncementHero { background: #10243a; border: 1px solid #0e7490;
  border-radius: 10px; }
QLabel#AnnouncementStatus { color: #67e8f9; font-size: 26px; font-weight: 700; }
QLabel#AnnouncementGroup { color: #f8fafc; font-size: 24px; font-weight: 700; }
QPushButton#NavButton { border: 0; border-radius: 6px; padding: 11px 14px;
  text-align: left; color: #94a3b8; background: transparent; font-size: 14px; }
QPushButton#NavButton:hover { background: #17243a; color: #e2e8f0; }
QPushButton#NavButton:checked { background: #153047; color: #22d3ee;
  border-left: 3px solid #22d3ee; }
QGroupBox { background: #111c2e; border: 1px solid #23324a; border-radius: 8px;
  margin-top: 13px; padding: 14px 10px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px;
  color: #67e8f9; }
QLineEdit, QComboBox, QKeySequenceEdit, QPlainTextEdit { background: #0b1424;
  color: #e2e8f0; border: 1px solid #334155; border-radius: 5px; padding: 5px; }
QComboBox QAbstractItemView { background: #111c2e; color: #e2e8f0;
  selection-background-color: #164e63; selection-color: #ecfeff; }
QTableWidget { background: #0b1424; color: #e2e8f0; gridline-color: #23324a;
  border: 1px solid #334155; selection-background-color: #164e63; }
QHeaderView::section { background: #17243a; color: #dbeafe;
  border: 0; border-right: 1px solid #23324a; padding: 5px; }
QTableCornerButton::section { background: #17243a; border: 0; }
QPushButton { background: #164e63; color: #ecfeff; border: 1px solid #0e7490;
  border-radius: 6px; padding: 7px 12px; }
QPushButton:hover { background: #155e75; }
QPushButton:disabled, QLineEdit:disabled, QComboBox:disabled,
QCheckBox:disabled, QLabel:disabled { color: #94a3b8;
  background: #111827; border-color: #1f2937; }
QScrollArea { border: 0; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QToolTip { color: #e2e8f0; background: #111827; border: 1px solid #334155; }
"""

class MainWindow(QMainWindow):
    '''
    MainWindow
    '''
    request_close = Signal()

    def __init__(self, controller=None):
        super().__init__()

        # UI window Icon
        self.setWindowIcon(QIcon("media/icon.png"))

        self.controller = controller # autoBotController
        self.selected_map = ""
        self.route_studio_window = None
        self.nametag_profile_repository = NameTagProfileRepository()
        self.nametag_calibration_active = False
        self._readiness_busy = False
        self._readiness_report = None
        self._readiness_live = False

        # Load default yaml and platform yaml as base config
        _, self.comments, self.comments_section = load_yaml_with_comments("config/config_default.yaml")
        self.cfg_base = load_yaml("config/config_default.yaml")
        if is_mac():
            self.cfg_base = override_cfg(self.cfg_base,
                                         load_yaml("config/config_macOS.yaml"))
        self.cfg = copy.deepcopy(self.cfg_base)
        # Immutable per-session reset target. Loading a profile replaces this
        # snapshot so "restore defaults" returns to that profile's values,
        # rather than generic settings for a different client.
        self.cfg_loaded_defaults = copy.deepcopy(self.cfg)
        self.path_cfg_custom = str(writable_path("config", "config_custom.yaml"))
        # Runtime snapshots are process-local.  Two accidentally opened main
        # windows must never race on one shared .config_tmp.yaml.
        self.runtime_cfg_path = str(
            writable_path("config", f".config_tmp_{os.getpid()}.yaml")
        )
        os.makedirs(os.path.dirname(self.path_cfg_custom) or ".", exist_ok=True)

        # Load database
        self.data = load_yaml("config/config_data.yaml")

        # Window Settings
        self.setWindowTitle("冒险岛自动练级助手")
        self.setMinimumSize(1080, 700)
        self.resize(1280, 820)
        self.setStyleSheet(APP_QSS)

        # Setup fixed pages inside one professional console shell.
        self.tab_main = self.setup_main_tab()
        self.tab_advance_setting = self.setup_advance_setting_tab()
        self.tab_game_window_viz = self.setup_game_window_viz_tab()
        self.tab_route_map_viz = self.setup_route_map_viz_tab()
        self.tab_announcement = self.setup_announcement_tab()

        self._build_application_shell()

        # Populate every editor from the default profile before an optional
        # remembered custom profile is loaded.  A first-run installation has
        # no ui_state.json and must still start with complete, valid values.
        self.apply_config_to_ui()

        # Load previous stored UI state
        self.load_ui_state()

        # Any edit invalidates a paused check, even if it has not yet emitted
        # the advanced-settings conversion signal.
        for editor in self.findChildren(QLineEdit):
            editor.textChanged.connect(self.invalidate_readiness)
        for editor in self.findChildren(QComboBox):
            editor.currentTextChanged.connect(self.invalidate_readiness)
        for editor in self.findChildren(QCheckBox):
            editor.toggled.connect(self.invalidate_readiness)

        # Signal
        self.request_close.connect(self.close)

    def _build_application_shell(self):
        root = QWidget()
        root.setObjectName("AppRoot")
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(205)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(16, 22, 16, 20)
        side.setSpacing(7)
        brand = QLabel("冒险岛助手")
        brand.setObjectName("Brand")
        side.addWidget(brand)
        subtitle = QLabel("YOLO 自动控制台")
        subtitle.setObjectName("Subtle")
        side.addWidget(subtitle)
        side.addSpacing(20)

        self.page_stack = QStackedWidget()
        self.page_titles = (
            "运行中心", "高级设置", "窗口监控", "路线监控", "公告与交流"
        )
        pages = (
            self.tab_main, self.tab_advance_setting,
            self.tab_game_window_viz, self.tab_route_map_viz,
            self.tab_announcement,
        )
        self.nav_buttons = []
        for index, (title, page) in enumerate(zip(self.page_titles, pages)):
            button = QPushButton(title)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, i=index: self.page_stack.setCurrentIndex(i))
            side.addWidget(button)
            self.nav_buttons.append(button)
            self.page_stack.addWidget(page)
        side.addStretch()
        fixed_policy = QLabel("内部策略\n主循环 10 FPS\n捕获 15 FPS\nSendInput")
        fixed_policy.setObjectName("Subtle")
        side.addWidget(fixed_policy)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(22, 16, 22, 20)
        header = QHBoxLayout()
        self.header_page_title = QLabel(self.page_titles[0])
        self.header_page_title.setObjectName("PageTitle")
        self.header_config = QLabel("配置：默认")
        self.header_config.setObjectName("Subtle")
        self.header_status = QLabel("未运行")
        self.header_status.setObjectName("StatusPill")
        header.addWidget(self.header_page_title)
        header.addSpacing(12)
        header.addWidget(self.header_config)
        header.addStretch()
        header.addWidget(self.header_status)
        content_layout.addLayout(header)
        content_layout.addWidget(self.page_stack, 1)

        shell.addWidget(sidebar)
        shell.addWidget(content, 1)
        self.setCentralWidget(root)
        self.page_stack.currentChanged.connect(self.on_tab_changed)
        self.page_stack.setCurrentIndex(0)
        self.nav_buttons[0].setChecked(True)
        self._set_run_status("未运行", "#94a3b8", "#1e293b")

    def _set_run_status(self, text, foreground, background):
        self.header_status.setText(text)
        self.header_status.setStyleSheet(
            f"color: {foreground}; background: {background};"
        )

    def setup_main_tab(self):
        '''
        Init Main Tab with scrollable area
        '''
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        scroll_layout.setSpacing(12)

        # Control group box
        self.control_gbox = self.create_control_gbox()
        scroll_layout.addWidget(self.control_gbox, 0, 0)

        # Attack setting group box
        self.attack_gbox = self.create_attack_gbox()
        scroll_layout.addWidget(self.attack_gbox, 0, 1)

        # Key bindings group box
        self.key_binding_gbox = self.create_key_binding_gbox()
        scroll_layout.addWidget(self.key_binding_gbox, 2, 0)

        # Pet function group box
        self.pet_skill_gbox = self.create_pet_skill_gbox()
        scroll_layout.addWidget(self.pet_skill_gbox, 2, 1)

        # Map selection group box
        self.map_selection_gbox = self.create_map_selection_gbox()
        scroll_layout.addWidget(self.map_selection_gbox, 3, 0)

        # Logger output window
        self.log_gbox = self.create_log_gbox()
        scroll_layout.addWidget(self.log_gbox, 3, 1)
        self.readiness_gbox = self.create_readiness_gbox()
        scroll_layout.addWidget(self.readiness_gbox, 1, 0, 1, 2)
        scroll_layout.setRowStretch(4, 1)

        scroll_area.setWidget(scroll_widget)

        tab_main = QWidget()
        layout = QVBoxLayout(tab_main)
        layout.addWidget(scroll_area)

        return tab_main

    def create_readiness_gbox(self):
        box = QGroupBox("运行准备与实时识别")
        layout = QVBoxLayout(box)
        buttons = QGridLayout()
        self.button_check_readiness = QPushButton("检查当前画面 / 重新检查")
        self.button_check_readiness.clicked.connect(self.check_current_frame)
        self.button_cancel_readiness = QPushButton("取消检查 / 启动")
        self.button_cancel_readiness.setEnabled(False)
        self.button_cancel_readiness.clicked.connect(self.cancel_readiness)
        self.button_readiness_name = QPushButton("标定人物名字")
        self.button_readiness_name.clicked.connect(self.open_nametag_calibration)
        self.button_readiness_map = QPushButton("地图 / ROI / 路线编辑")
        self.button_readiness_map.clicked.connect(self.open_route_studio)
        self.button_readiness_settings = QPushButton("识别设置")
        self.button_readiness_settings.clicked.connect(lambda: self.page_stack.setCurrentIndex(1))
        for index, button in enumerate((self.button_check_readiness, self.button_cancel_readiness,
                                        self.button_readiness_name, self.button_readiness_map,
                                        self.button_readiness_settings)):
            buttons.addWidget(button, index // 3, index % 3)
        layout.addLayout(buttons)
        self.readiness_summary = QLabel("尚未检查。先检查当前画面，再按 F1；检查不会移动、攻击或喝药。")
        self.readiness_summary.setWordWrap(True)
        self.readiness_summary.setTextFormat(Qt.PlainText)
        layout.addWidget(self.readiness_summary)
        body = QHBoxLayout()
        self.readiness_table = QTableWidget(len(CHECKS), 3)
        self.readiness_table.setHorizontalHeaderLabels(["检查项", "状态", "结果与失败原因（选中查看恢复方法）"])
        self.readiness_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.readiness_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.readiness_table.verticalHeader().setVisible(False)
        self.readiness_table.verticalHeader().setDefaultSectionSize(28)
        self.readiness_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.readiness_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.readiness_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.readiness_table.setMinimumHeight(320)
        self.readiness_table.setMaximumHeight(340)
        for row, (_key, title, hint) in enumerate(CHECKS):
            for col, text in enumerate((title, "未检查", "尚未检查")):
                self.readiness_table.setItem(row, col, QTableWidgetItem(text))
        self.readiness_table.currentCellChanged.connect(self._readiness_row_selected)
        body.addWidget(self.readiness_table, 3)
        self.readiness_name_preview = QLabel("名字框 / 脚点预览\n尚未取得有效识别")
        self.readiness_name_preview.setAlignment(Qt.AlignCenter)
        self.readiness_name_preview.setMinimumWidth(200)
        self.readiness_name_preview.setMaximumWidth(320)
        self.readiness_name_preview.setWordWrap(True)
        body.addWidget(self.readiness_name_preview, 1)
        layout.addLayout(body)
        self.readiness_recovery = QLabel("选中检查项，查看对应恢复方法。名字匹配成功后仍需人工确认紫色脚点位置。")
        self.readiness_recovery.setWordWrap(True)
        self.readiness_recovery.setTextFormat(Qt.PlainText)
        layout.addWidget(self.readiness_recovery)
        self.readiness_last_issue = QLabel("")
        self.readiness_last_issue.setTextFormat(Qt.PlainText)
        self.readiness_last_issue.setWordWrap(True)
        layout.addWidget(self.readiness_last_issue)
        return box

    def _readiness_row_selected(self, row, *_args):
        if 0 <= row < len(CHECKS):
            self.readiness_recovery.setText(
                f"{CHECKS[row][1]}：{self.readiness_table.item(row, 2).text()}\n恢复方法：{CHECKS[row][2]}。"
                "运行中请先暂停，再进行标定或编辑。")

    def invalidate_readiness(self, *_args):
        if self._readiness_report is None or self._readiness_busy or self.button_start_pause.isChecked():
            return
        self._readiness_report = None
        self.readiness_summary.setText("设置已改变，之前的检查结果已失效；请重新检查当前画面。")
        self.readiness_name_preview.clear()
        self.readiness_name_preview.setText("设置已改变，等待重新识别")
        for row in range(len(CHECKS)):
            self.readiness_table.item(row, 1).setText("待重检")
            self.readiness_table.item(row, 1).setForeground(QColor("#94a3b8"))

    def update_readiness_report(self, report, live=False):
        self._readiness_report = report
        self._readiness_live = live
        states = {"pending": ("等待", "#fbbf24"), "pass": ("通过", "#4ade80"),
                  "fail": ("失败", "#f87171"), "warn": ("注意", "#fbbf24"),
                  "unused": ("不适用", "#94a3b8")}
        indexed = {item.key: item for item in report.rows}
        failures = []
        for row, (key, title, _hint) in enumerate(CHECKS):
            item = indexed.get(key)
            if item is None:
                continue
            text, color = states[item.status]
            self.readiness_table.item(row, 1).setText(text)
            self.readiness_table.item(row, 1).setForeground(QColor(color))
            self.readiness_table.item(row, 2).setText(item.detail)
            self.readiness_table.item(row, 2).setToolTip(item.detail)
            if item.status == "fail":
                failures.append(f"{title}：{item.detail}")
        stamp = time.strftime("%H:%M:%S")
        self.readiness_summary.setText(f"{'实时' if live else '只读采样'} {stamp}｜{report.summary}")
        if failures:
            self.readiness_last_issue.setText("最近失败（保留供排查）：" + "；".join(failures))
        name = indexed.get("name")
        if report.name_preview:
            pixmap = QPixmap()
            if pixmap.loadFromData(report.name_preview, "PNG"):
                self.readiness_name_preview.setPixmap(pixmap.scaled(300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif name is None or name.status != "pass":
            self.readiness_name_preview.clear()
            self.readiness_name_preview.setText("当前无有效名字预览\n不使用旧人物位置")
        self._readiness_row_selected(self.readiness_table.currentRow())
        if live:
            self._mark_monitor_history("识别异常：监控可能保留上一有效帧，请查看运行中心的失败原因" if failures else
                                       "实时识别中；画面与识别状态请结合更新时间查看")
            attention = any(item.status in {"pending", "warn", "fail"} for item in report.rows)
            title = "运行中·识别异常" if failures else "运行中·待确认" if attention else "运行中·识别监测"
            self._set_run_status(title, "#fbbf24" if attention else "#4ade80", "#1e293b")

    def _mark_monitor_history(self, text):
        self.game_viz_state.setText(text)
        self.route_viz_state.setText(text)

    def mark_readiness_stale(self):
        self.readiness_summary.setText("识别状态超过 1 秒未更新；下表与预览仅供排查，不能视为当前识别成功。")
        self._mark_monitor_history("诊断未更新：保留历史画面，非当前有效识别")
        self._set_run_status("等待识别更新", "#fbbf24", "#422006")
        for row in range(len(CHECKS)):
            if self.readiness_table.item(row, 1).text() == "通过":
                self.readiness_table.item(row, 1).setText("已过期")
                self.readiness_table.item(row, 1).setForeground(QColor("#94a3b8"))

    def _set_readiness_busy(self, busy):
        self._readiness_busy = busy
        self.button_start_pause.setEnabled(not busy)
        self.button_cancel_readiness.setEnabled(busy)
        self.button_check_readiness.setEnabled(not busy and not self.button_start_pause.isChecked())
        self.button_route_studio.setEnabled(not busy and not self.button_start_pause.isChecked())
        self.set_gbox_enabled(not busy and not self.button_start_pause.isChecked())

    def check_current_frame(self):
        if self._readiness_busy or self.button_start_pause.isChecked():
            return
        if self.route_studio_window is not None or self.nametag_calibration_active:
            self.readiness_summary.setText("请先关闭路线编辑器或人物标定窗口，再检查画面。")
            return
        candidate, error = self.build_candidate_config_from_ui()
        if candidate is None:
            self.readiness_summary.setText(error)
            return
        try:
            validate_custom_config(candidate)
        except (ValueError, TypeError) as exc:
            self.readiness_summary.setText(f"配置无效：{exc}")
            return
        if not self.validate_attack_detection_ranges(candidate):
            return
        self._reset_readiness(candidate)
        self._set_readiness_busy(True)
        if not self.controller.check_readiness(candidate):
            self._set_readiness_busy(False)
            self.readiness_summary.setText(getattr(self.controller, "last_start_error", "无法开始检查，请稍后重试。"))

    def _reset_readiness(self, cfg):
        from src.engine.ReadinessDiagnostics import ReadinessReport
        self.readiness_last_issue.clear()
        self.update_readiness_report(ReadinessReport(tuple(initial_rows(cfg).values()), time.monotonic(), "准备中，请稍候…"))

    def cancel_readiness(self):
        self.controller.cancel_preparation()
        self.button_cancel_readiness.setEnabled(False)
        self.readiness_summary.setText("正在取消，等待资源安全释放；不会开始自动运行。")

    def finish_readiness_operation(self, kind, result, cancelled=False):
        if kind == "start":
            self._finish_start_ui(result)
        self._set_readiness_busy(False)
        if kind == "check" and (result != 0 or cancelled):
            self.readiness_summary.setText("检查已取消；未开始自动运行。" if cancelled else
                                          getattr(self.controller, "last_start_error", "检查失败"))

    def setup_advance_setting_tab(self):
        tab_advance_setting = QWidget()
        categories = (
            ("战斗", ("directional_attack", "aoe_skill", "health_monitor", "combat_tracking")),
            ("移动与路线", (
                "teleport", "edge_teleport", "route", "watchdog", "patrol",
                "fixed_platform",
            )),
            ("识别", ("nametag", "monster_detect", "minimap", "game_window")),
            ("运营", ("channel_change", "scheduled_channel_switching", "ui_coords", "profiler")),
        )
        grid = QGridLayout()
        grid.setSpacing(14)
        self.advance_settings_gboxes = {}
        for category_index, (category_name, sections) in enumerate(categories):
            column_widget = QWidget()
            column_layout = QVBoxLayout(column_widget)
            column_layout.setContentsMargins(0, 0, 0, 0)
            category_label = QLabel(category_name)
            category_label.setStyleSheet("color:#22d3ee; font-size:16px; font-weight:700;")
            column_layout.addWidget(category_label)
            for title in sections:
                hidden_keys = ()
                if title == "health_monitor":
                    hidden_keys = (
                        "auto_hp_enabled", "auto_mp_enabled",
                        "add_hp_percent", "add_mp_percent", "fps_limit",
                    )
                elif title == "nametag":
                    hidden_keys = ("name",)
                gbox = create_advance_setting_gbox(
                    title, self.cfg, self.comments, self.comments_section,
                    hidden_keys=hidden_keys,
                )
                if title in {"nametag", "monster_detect"}:
                    if title == "nametag":
                        profile_combo = QComboBox()
                        profile_combo.setEditable(True)
                        profile_combo.setToolTip(
                            "选择人物名字定位配置；支持在主程序内标定新角色。"
                        )
                        gbox.layout().insertRow(0, "人物配置", profile_combo)
                        gbox._field_refs["name"] = profile_combo
                        self.nametag_profile_combo = profile_combo
                        self.refresh_nametag_profiles(
                            preferred=self.cfg["nametag"].get("name", "")
                        )
                        calibration_button = QPushButton("标定与管理人物名字")
                        calibration_button.clicked.connect(
                            self.open_nametag_calibration
                        )
                        gbox.layout().addRow(calibration_button)
                        gbox._calibration_button = calibration_button
                        self.nametag_calibration_button = calibration_button
                    button_text = (
                        "恢复名字检测默认值" if title == "nametag"
                        else "恢复怪物检测默认值"
                    )
                    reset_button = QPushButton(button_text)
                    reset_button.clicked.connect(
                        lambda _checked=False, section=title:
                        self.restore_detection_defaults(section)
                    )
                    gbox.layout().addRow(reset_button)
                    gbox._reset_button = reset_button
                if title == "monster_detect":
                    min_box_side = gbox._field_refs["yolo_min_monster_box_side"]
                    min_box_side.setValidator(
                        QIntValidator(0, min(MODEL_WIDTH, MODEL_HEIGHT), min_box_side)
                    )
                    min_box_side.setToolTip(
                        "过滤最短边小于此值的怪物框；0 表示关闭过滤。"
                    )
                    exclusion_width = gbox._field_refs[
                        "yolo_player_exclusion_width"
                    ]
                    exclusion_width.setValidator(
                        QIntValidator(0, MODEL_WIDTH, exclusion_width)
                    )
                    exclusion_width.setToolTip(
                        "以名称定位点为底部中心的排除区宽度；"
                        "宽度或高度为 0 时关闭过滤。"
                    )
                    exclusion_height = gbox._field_refs[
                        "yolo_player_exclusion_height"
                    ]
                    exclusion_height.setValidator(
                        QIntValidator(0, MODEL_HEIGHT, exclusion_height)
                    )
                    exclusion_height.setToolTip(
                        "从名称定位点向上延伸的排除区高度；"
                        "宽度或高度为 0 时关闭过滤。"
                    )
                    max_det = gbox._field_refs["yolo_max_det"]
                    max_det.setReadOnly(True)
                    max_det.setToolTip("内部固定为 50。")
                    model_info = QLabel(
                        f"输入尺寸：{MODEL_WIDTH}×{MODEL_HEIGHT}\n类别：monster / player"
                    )
                    model_info.setObjectName("Subtle")
                    gbox.layout().addRow("模型规格", model_info)
                if title == "fixed_platform":
                    width_px = gbox._field_refs["width_px"]
                    width_px.setValidator(QIntValidator(10, 1000, width_px))
                    width_px.setToolTip(
                        "以启动时人物的小地图位置为中心，设置左右巡逻的总宽度；"
                        "允许 10–1000 px。"
                    )
                self.advance_settings_gboxes[title] = gbox
                column_layout.addWidget(gbox)
            column_layout.addStretch()
            row, column = divmod(category_index, 2)
            grid.addWidget(column_widget, row, column)

        scroll_area = QScrollArea()
        container = QWidget()
        container.setLayout(grid)
        scroll_area.setWidget(container)
        scroll_area.setWidgetResizable(True)

        # Final layout for the tab
        final_layout = QVBoxLayout()
        final_layout.addWidget(scroll_area)
        tab_advance_setting.setLayout(final_layout)

        return tab_advance_setting

    def setup_game_window_viz_tab(self):
        tab_game_window_viz = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        tab_game_window_viz.setLayout(layout)

        legend_bar = QWidget()
        legend_bar.setStyleSheet(
            "background-color: #111c2e; border: 1px solid #23324a; "
            "border-radius: 4px;"
        )
        legend = QHBoxLayout(legend_bar)
        legend.setContentsMargins(10, 5, 10, 5)
        legend.setSpacing(7)
        for color, text in (
            ("#006cff", "蓝框：怪物检测范围"),
            ("#ff3030", "红框：人物攻击范围"),
            ("#ff3030", "红色中线：左右方向分界"),
        ):
            swatch = QLabel("  ")
            swatch.setFixedSize(18, 12)
            swatch.setStyleSheet(
                f"background-color: {color}; border: none; border-radius: 2px;"
            )
            legend.addWidget(swatch)
            text_label = QLabel(text)
            text_label.setStyleSheet("border: none; color: #dbeafe;")
            legend.addWidget(text_label)
            legend.addSpacing(12)
        legend.addStretch()
        layout.addWidget(legend_bar, 0)

        self.game_viz_state = QLabel("尚未运行：等待实时画面")
        layout.addWidget(self.game_viz_state)
        self.game_viz_metrics = QLabel("性能指标：等待监控帧")
        self.game_viz_metrics.setStyleSheet("color: #9fb3c8; padding: 2px 4px;")
        layout.addWidget(self.game_viz_metrics, 0)

        # Create a large QLabel as a canvas
        self.debug_canvas = ResponsiveImageLabel()
        clear_debug_canvas(self.debug_canvas)
        layout.addWidget(self.debug_canvas, 1)

        return tab_game_window_viz

    def setup_route_map_viz_tab(self):
        tab_route_map_viz_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        tab_route_map_viz_tab.setLayout(layout)

        self.route_viz_state = QLabel("尚未运行：等待实时画面")
        layout.addWidget(self.route_viz_state)
        self.route_viz_metrics = QLabel("性能指标：等待监控帧")
        self.route_viz_metrics.setStyleSheet("color: #9fb3c8; padding: 2px 4px;")
        layout.addWidget(self.route_viz_metrics, 0)

        # Create a large QLabel as a canvas
        self.route_map_canvas = ResponsiveImageLabel()
        clear_debug_canvas(self.route_map_canvas)
        layout.addWidget(self.route_map_canvas, 1)

        return tab_route_map_viz_tab

    def setup_announcement_tab(self):
        """Build a read-only page whose content is compiled into the app."""
        page = QWidget()
        page.setObjectName("AnnouncementPage")
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        hero = QFrame()
        hero.setObjectName("AnnouncementHero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        status = QLabel(ANNOUNCEMENT.status)
        status.setObjectName("AnnouncementStatus")
        status.setAlignment(Qt.AlignCenter)
        free_notice = QLabel(ANNOUNCEMENT.free_notice)
        free_notice.setWordWrap(True)
        free_notice.setAlignment(Qt.AlignCenter)
        free_notice.setTextInteractionFlags(Qt.TextSelectableByMouse)
        hero_layout.addWidget(status)
        hero_layout.addWidget(free_notice)
        layout.addWidget(hero)

        group_box = QGroupBox("官方交流QQ群")
        group_layout = QVBoxLayout(group_box)
        self.announcement_qq_label = QLabel(ANNOUNCEMENT.qq_group)
        self.announcement_qq_label.setObjectName("AnnouncementGroup")
        self.announcement_qq_label.setAlignment(Qt.AlignCenter)
        self.announcement_qq_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        group_purpose = QLabel(ANNOUNCEMENT.group_purpose)
        group_purpose.setAlignment(Qt.AlignCenter)
        group_purpose.setWordWrap(True)
        self.button_copy_qq_group = QPushButton("复制群号")
        self.button_copy_qq_group.clicked.connect(self.copy_announcement_qq_group)
        self.announcement_copy_status = QLabel("")
        self.announcement_copy_status.setObjectName("Subtle")
        self.announcement_copy_status.setAlignment(Qt.AlignCenter)
        group_layout.addWidget(self.announcement_qq_label)
        group_layout.addWidget(group_purpose)
        group_layout.addWidget(
            self.button_copy_qq_group, alignment=Qt.AlignHCenter
        )
        group_layout.addWidget(self.announcement_copy_status)
        layout.addWidget(group_box)

        risk_box = QGroupBox("使用风险提示")
        risk_layout = QVBoxLayout(risk_box)
        risk_notice = QLabel(ANNOUNCEMENT.risk_notice)
        risk_notice.setWordWrap(True)
        risk_notice.setTextInteractionFlags(Qt.TextSelectableByMouse)
        risk_layout.addWidget(risk_notice)
        layout.addWidget(risk_box)
        layout.addStretch()

        scroll.setWidget(content)
        page_layout.addWidget(scroll)
        return page

    def copy_announcement_qq_group(self):
        QApplication.clipboard().setText(ANNOUNCEMENT.qq_group)
        self.announcement_copy_status.setText(
            f"已复制QQ群号：{ANNOUNCEMENT.qq_group}"
        )

    def save_ui_state(self):
        path = str(writable_path("ui_state.json"))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "last_config_path": self.path_cfg_custom
        }
        with open(path, 'w') as f:
            json.dump(state, f)
        logger.info(f"[UI] Save UI state to {path}")

    def load_ui_state(self):
        path = str(writable_path("ui_state.json"))
        if os.path.exists(path):
            with open(path, 'r') as f:
                state = json.load(f)
                # Load config
                self.load_config(create_error_label(),
                                 state.get("last_config_path"))
        logger.info(f"[UI] Load UI state from {path}")


    def create_attack_gbox(self):
        '''
        Create attack group box using two-column QFormLayouts
        '''
        gbox = QGroupBox("⚔️ 攻击与检测范围")

        # Left column
        form_left = QFormLayout()
        self.attack_mode = QComboBox()
        self.attack_mode.addItem("方向攻击", "directional")
        self.attack_mode.addItem("范围技能", "aoe_skill")
        self.attack_mode.setFixedWidth(120)

        # Connect dropdown change to handler
        self.attack_mode.currentIndexChanged.connect(
            self.update_atk_config_trigger_by_drop_list
        )

        # Set default value
        form_left.addRow("攻击模式：", self.attack_mode)

        self.attack_range_x = QLineEdit()
        self.attack_range_x.setFixedWidth(60)
        self.attack_range_x.setValidator(QIntValidator(1, 9999, self))
        form_left.addRow("攻击横向单侧距离 (px)：", self.attack_range_x)

        self.detect_range_x = QLineEdit()
        self.detect_range_x.setFixedWidth(60)
        self.detect_range_x.setValidator(QIntValidator(1, 9999, self))
        form_left.addRow("检测横向半径 (px)：", self.detect_range_x)

        # Right column
        form_right = QFormLayout()
        self.attack_cooldown = QLineEdit()
        self.attack_cooldown.setFixedWidth(60)
        form_right.addRow("攻击冷却 (秒)：", self.attack_cooldown)

        self.attack_range_y = QLineEdit()
        self.attack_range_y.setFixedWidth(60)
        self.attack_range_y.setValidator(QIntValidator(1, 9999, self))
        form_right.addRow("攻击纵向总高度 (px)：", self.attack_range_y)

        self.detect_range_y = QLineEdit()
        self.detect_range_y.setFixedWidth(60)
        self.detect_range_y.setValidator(QIntValidator(1, 9999, self))
        form_right.addRow("检测纵向半径 (px)：", self.detect_range_y)

        # Combine left and right forms
        columns = QHBoxLayout()
        columns.addLayout(form_left)
        columns.addSpacing(20)
        columns.addLayout(form_right)

        # Field validation
        error_label = create_error_label()
        self.attack_range_error_label = error_label
        self.attack_range_x.editingFinished.connect(
            lambda: validate_numerical_input(self.attack_range_x.text(), error_label, 1, 9999))
        self.attack_range_y.editingFinished.connect(
            lambda: validate_numerical_input(self.attack_range_y.text(), error_label, 1, 9999))
        self.detect_range_x.editingFinished.connect(
            lambda: validate_numerical_input(self.detect_range_x.text(), error_label, 1, 9999))
        self.detect_range_y.editingFinished.connect(
            lambda: validate_numerical_input(self.detect_range_y.text(), error_label, 1, 9999))
        self.attack_cooldown.editingFinished.connect(
            lambda: validate_numerical_input(self.attack_cooldown.text(), error_label, 0, 9999))

        # Final layout
        layout = QVBoxLayout()
        layout.addWidget(error_label)
        layout.addLayout(columns)
        gbox.setLayout(layout)
        return gbox

    def create_key_binding_gbox(self):
        gbox = QGroupBox("🎮 按键绑定")
        hbox = QHBoxLayout()

        # Left Column
        form_left = QFormLayout()
        self.basic_attack_key = SingleKeyEdit()
        self.basic_attack_key.setFixedWidth(100)
        form_left.addRow("普通攻击：", self.basic_attack_key)

        self.teleport_key = SingleKeyEdit()
        self.teleport_key.setFixedWidth(100)
        form_left.addRow("瞬移：", self.teleport_key)

        # Right Column
        form_right = QFormLayout()
        self.aoe_skill_key = SingleKeyEdit()
        self.aoe_skill_key.setFixedWidth(100)
        form_right.addRow("范围技能：", self.aoe_skill_key)

        self.jump_key = SingleKeyEdit()
        self.jump_key.setFixedWidth(100)
        form_right.addRow("跳跃：", self.jump_key)

        self.return_home_key = SingleKeyEdit()
        self.return_home_key.setFixedWidth(100)
        form_right.addRow("回城：", self.return_home_key)

        # Combine left and right column form
        hbox.addLayout(form_left)
        hbox.addSpacing(20)  # space between columns
        hbox.addLayout(form_right)
        gbox.setLayout(hbox)
        return gbox

    def create_pet_skill_gbox(self):
        gbox = QGroupBox("补给")
        layout_form = QFormLayout()

        # Auto Add HP checkbox
        hp_row = QHBoxLayout()
        self.checkbox_auto_add_hp = QCheckBox("自动补充生命值")
        self.checkbox_auto_add_hp.stateChanged.connect(self.toggle_auto_add_hp)
        hp_row.addWidget(self.checkbox_auto_add_hp)
        # Auto Add HP settings
        self.hp_input_widget, self.add_hp_percent, self.add_hp_key = self.create_hp_mp_widget("HP")
        self.hp_input_widget.setVisible(False) # Hide setting on default
        hp_row.addWidget(self.hp_input_widget)
        # Validation HP percent
        error_label_hp = create_error_label()
        self.add_hp_percent.editingFinished.connect(
            lambda: validate_numerical_input(self.add_hp_percent.text(), error_label_hp, 0, 100))
        # Add to layout
        layout_form.addWidget(error_label_hp)
        layout_form.addRow(hp_row)

        # Auto add MP checkbox
        mp_row = QHBoxLayout()
        self.checkbox_auto_add_mp = QCheckBox("自动补充魔法值")
        self.checkbox_auto_add_mp.stateChanged.connect(self.toggle_auto_add_mp)
        mp_row.addWidget(self.checkbox_auto_add_mp)
        # Auto Add MPHP settings
        self.mp_input_widget, self.add_mp_percent, self.add_mp_key = self.create_hp_mp_widget("MP")
        self.mp_input_widget.setVisible(False)
        mp_row.addWidget(self.mp_input_widget)
        # Validation MP percent
        error_label_mp = create_error_label()
        self.add_mp_percent.editingFinished.connect(
            lambda: validate_numerical_input(self.add_mp_percent.text(), error_label_mp, 0, 100))
        # Add to layout
        layout_form.addWidget(error_label_mp)
        layout_form.addRow(mp_row)

        gbox.setLayout(layout_form)
        return gbox

    def create_map_selection_gbox(self):
        '''
        Create a stable, non-editable map selector.
        '''
        gbox = QGroupBox("🗺️ 地图")

        self.map_combo = QComboBox()
        self.map_combo.setEditable(False)
        self.label_map_info = QLabel("请选择地图：")
        self.map_combo.currentIndexChanged.connect(self.on_map_selected)
        self.refresh_map_selector()

        layout = QVBoxLayout()
        layout.addWidget(self.label_map_info)
        layout.addWidget(self.map_combo)
        gbox.setLayout(layout)
        return gbox

    def refresh_map_selector(self, preferred_map=None, minimap_dir="minimaps"):
        """Rescan runnable map projects without recreating the main window."""
        current = str(
            preferred_map
            or getattr(self, "selected_map", "")
            or self.map_combo.currentData()
            or ""
        )
        root = Path(minimap_dir)
        projects = []
        if root.is_dir():
            for directory in root.iterdir():
                if not directory.is_dir() or directory.name.startswith("."):
                    continue
                if not (directory / "map.png").is_file():
                    continue
                if not any(directory.glob("route*.png")):
                    continue
                projects.append(directory.name)

        previous_block = self.map_combo.blockSignals(True)
        try:
            self.map_combo.clear()
            translations = self.data.get("eng_to_cn", {})
            for name in sorted(projects, key=str.casefold):
                visible = str(translations.get(name, name))
                display_text = name if visible == name else f"{name} ({visible})"
                self.map_combo.addItem(display_text, name)
            selected_index = self.map_combo.findData(current)
            if selected_index < 0 and self.map_combo.count():
                selected_index = 0
            if selected_index >= 0:
                self.map_combo.setCurrentIndex(selected_index)
        finally:
            self.map_combo.blockSignals(previous_block)

        if self.map_combo.count():
            self.on_map_selected(self.map_combo.currentIndex())
        else:
            self.selected_map = ""
            self.label_map_info.setText("没有可运行地图：请先保存底图和路线。")
        return self.map_combo.count()

    def create_log_gbox(self):
        gbox = QGroupBox("📜 技术日志")
        layout = QVBoxLayout()

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)

        layout.addWidget(self.log_output)
        gbox.setLayout(layout)

        # Create Qt logger handler
        self.qt_log_handler = QtLogHandler()
        self.qt_log_handler.log_signal.connect(self.append_log)
        self.qt_log_handler.setFormatter(
            logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', '%H:%M:%S'))

        # Add it to your logger
        logger.addHandler(self.qt_log_handler)

        return gbox

    def create_control_gbox(self):
        '''
        Creates a group box with hotkey instructions and a dropdown to select mode
        '''
        gbox = QGroupBox("🕹️ 自动助手控制")
        layout = QVBoxLayout()
        layout.setSpacing(6)

        # Load Config Section
        self.load_config_error_label = create_error_label()
        layout.addWidget(self.load_config_error_label)  # Add above the button

        load_config_layout = QHBoxLayout()
        load_config_layout.setSpacing(8)
        load_config_layout.setAlignment(Qt.AlignLeft)

        self.button_load_config = QPushButton("📂 加载配置")
        self.button_load_config.clicked.connect(
            lambda: self.load_config(self.load_config_error_label)
        )
        self.label_config_path = QLabel("（尚未加载配置）")

        load_config_layout.addWidget(self.button_load_config)
        load_config_layout.addWidget(self.label_config_path)

        # --- Control Buttons ---
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        button_layout.setAlignment(Qt.AlignLeft)

        # Start / Pause Button
        self.button_start_pause = QPushButton("▶ 开始 (F1)")
        self.button_start_pause.setCheckable(True)
        self.button_start_pause.clicked.connect(self.toggle_start_ui)
        self.button_route_studio = QPushButton("路线录制与编辑")
        self.button_route_studio.clicked.connect(self.open_route_studio)

        # Bot Mode Dropdown
        layout_bot_mode = QHBoxLayout()
        layout_bot_mode.setSpacing(8)
        self.bot_mode = QComboBox()
        self.bot_mode.addItem("普通", "normal")
        self.bot_mode.addItem("辅助", "aux")
        self.bot_mode.addItem("巡逻", "patrol")
        self.bot_mode.addItem("固定平台区域攻击", "fixed_platform")
        self.bot_mode.addItem("持续定向攻击", "continuous_attack")

        layout_bot_mode.addWidget(QLabel("运行模式："))
        layout_bot_mode.addWidget(self.bot_mode)
        layout_bot_mode.setAlignment(Qt.AlignLeft)

        button_layout.addWidget(self.button_start_pause)
        button_layout.addWidget(self.button_route_studio)
        button_layout.addLayout(layout_bot_mode)

        layout.addLayout(button_layout)
        layout.addLayout(load_config_layout)

        gbox.setLayout(layout)
        return gbox
        logger.info(f"[UI] Map selected: {map_name}")

    def create_attack_widget(self):
        '''
        Create a wedge widget: "Press key ➜ [KEY] for Mode A/B"
        '''
        layout_main = QHBoxLayout()

        layout_main.addWidget(QLabel("攻击按键："))
        key_input = SingleKeyEdit()
        key_input.setFixedWidth(100)
        layout_main.addWidget(key_input)

        # Horizontal Range
        layout_main.addWidget(QLabel("横向范围："))
        range_x = QLineEdit()
        range_x.setPlaceholderText("50")
        range_x.setFixedWidth(60)
        layout_main.addWidget(range_x)

        # Vertical Range
        layout_main.addWidget(QLabel("纵向范围："))
        range_y = QLineEdit()
        range_y.setPlaceholderText("50")
        range_y.setFixedWidth(60)
        layout_main.addWidget(range_y)

        # CoolDown
        layout_main.addWidget(QLabel("冷却时间（秒）："))
        cooldown = QLineEdit()
        cooldown.setPlaceholderText("0.1")
        cooldown.setFixedWidth(60)
        layout_main.addWidget(cooldown)

        container = QWidget()
        container.setLayout(layout_main)

        return container, key_input, range_x, range_y, cooldown

    def create_hp_mp_widget(self, title):
        '''
        Creates a widget for Auto HP/MP usage with tight label-field alignment
        '''
        container = QWidget()
        layout_main = QVBoxLayout()
        layout_main.setContentsMargins(0, 0, 0, 0)
        layout_main.setSpacing(2)

        # Input line
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(2)
        input_layout.setAlignment(Qt.AlignLeft)

        label_1 = QLabel(f"当 {title} 低于：")
        label_1.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

        percent = QLineEdit()
        percent.setPlaceholderText("50")
        percent.setFixedWidth(60)

        label_2 = QLabel("% 时，按下")
        label_2.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

        input_key = SingleKeyEdit()
        input_key.setFixedWidth(100)

        label_3 = QLabel("键。")
        label_3.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

        # Add to layout
        input_layout.addWidget(label_1)
        input_layout.addWidget(percent)
        input_layout.addWidget(label_2)
        input_layout.addWidget(input_key)
        input_layout.addWidget(label_3)

        layout_main.addLayout(input_layout)
        container.setLayout(layout_main)

        return container, percent, input_key

    def load_config(self, error_label, path=None):
        '''
        Load config_custom.yaml from file and apply settings to UI
        '''
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择配置文件", "", "YAML 文件 (*.yaml);;所有文件 (*)"
            )
            if not path:
                return  # User canceled

        # Validation
        if not path.endswith(".yaml"):
            error_label.setText("仅支持 .yaml 配置文件。")
            error_label.setVisible(True)
            return

        # Validate the file name
        if "config_default.yaml" in path or "config_data.yaml" in path or \
           "config_macOS.yaml" in path:
            error_label.setText(f"不能将 {path} 作为自定义配置加载。")
            error_label.setVisible(True)
            return

        # Load customized yaml
        try:
            custom_cfg = migrate_health_monitor_config(
                validate_custom_config(load_yaml(path))
            )
            self.cfg = override_cfg(copy.deepcopy(self.cfg_base), custom_cfg)
        except FileNotFoundError:
            logger.warning(f"[UI] Unable to find config file: {path}")
            return
        except ConfigCompatibilityError as exc:
            error_label.setText(str(exc))
            error_label.setVisible(True)
            logger.error(str(exc))
            return
        except (TypeError, ValueError) as exc:
            error_label.setText(f"健康监控配置无效：{exc}")
            error_label.setVisible(True)
            logger.error(f"[UI] Health config invalid: {exc}")
            return
        else:
            error_label.setVisible(False)
            self.label_config_path.setText(path)
            self.label_config_path.setStyleSheet("color: #4ade80")
            self.header_config.setText(f"配置：{os.path.basename(path)}")
            self.path_cfg_custom = path
            self.cfg_loaded_defaults = copy.deepcopy(self.cfg)

        # Re-apply config to UI
        self.apply_config_to_ui()

    def restore_detection_defaults(self, section):
        """Restore one detection section to the currently loaded profile."""
        if section not in {"nametag", "monster_detect"}:
            raise ValueError(f"Unsupported reset section: {section}")
        defaults = self.cfg_loaded_defaults.get(section)
        if defaults is None:
            defaults = self.cfg_base[section]
        self.cfg[section] = copy.deepcopy(defaults)
        self.update_advance_setting_ui_from_cfg()
        self.apply_config_to_ui()
        logger.info(f"[UI] Restore loaded defaults: {section}")

    def update_atk_config_trigger_by_drop_list(self):
        '''
        Update attack config fields based on selected attack mode from drop-down.
        '''
        mode = self.attack_mode.currentData()
        if mode == "directional":
            atk_cfg = self.cfg["directional_attack"]
        elif mode == "aoe_skill":
            atk_cfg = self.cfg["aoe_skill"]
        else:
            atk_cfg = None

        if atk_cfg:
            self.attack_range_x.setText(str(atk_cfg["range_x"]))
            self.attack_cooldown.setText(str(atk_cfg["cooldown"]))
            self.attack_range_y.setText(str(atk_cfg["range_y"]))

    def update_detection_range_mode(self, *_args):
        """Display the immutable YOLO input half extents."""
        self.detect_range_x.setText(str(MODEL_WIDTH // 2))
        self.detect_range_y.setText(str(MODEL_HEIGHT // 2))
        self.detect_range_x.setEnabled(False)
        self.detect_range_y.setEnabled(False)
        tooltip = "YOLO 模型输入固定为 1280×224，以人物脚部为中心。"
        self.detect_range_x.setToolTip(tooltip)
        self.detect_range_y.setToolTip(tooltip)

    def apply_config_to_ui(self):
        # === Attack Section ===
        atk_cfg = None
        if self.cfg["bot"]["attack"] == "directional":
            self.attack_mode.setCurrentIndex(0)
            atk_cfg = self.cfg["directional_attack"]
        elif self.cfg["bot"]["attack"] == "aoe_skill":
            self.attack_mode.setCurrentIndex(1)
            atk_cfg = self.cfg["aoe_skill"]
        self.attack_range_x.setText(str(atk_cfg["range_x"]))
        self.attack_cooldown.setText(str(atk_cfg["cooldown"]))
        self.attack_range_y.setText(str(atk_cfg["range_y"]))

        self.update_detection_range_mode()

        # === Key Bindings ===
        key_cfg = self.cfg["key"]
        self.basic_attack_key.set_key(key_cfg["directional_attack"])
        self.teleport_key.set_key(key_cfg["teleport"])
        self.aoe_skill_key.set_key(key_cfg["aoe_skill"])
        self.jump_key.set_key(key_cfg["jump"])
        self.return_home_key.set_key(key_cfg["return_home"])

        # === HP/MP Section ===
        hm_cfg = self.cfg["health_monitor"]
        self.checkbox_auto_add_hp.setChecked(hm_cfg["auto_hp_enabled"])
        self.add_hp_percent.setText(str(hm_cfg["add_hp_percent"]))
        self.add_hp_key.set_key(self.cfg["key"]["add_hp"])
        self.checkbox_auto_add_mp.setChecked(hm_cfg["auto_mp_enabled"])
        self.add_mp_percent.setText(str(hm_cfg["add_mp_percent"]))
        self.add_mp_key.set_key(self.cfg["key"]["add_mp"])

        # === Bot Mode ===
        index = self.bot_mode.findData(self.cfg["bot"]["mode"])
        self.bot_mode.setCurrentIndex(max(0, index))

        # Map Selection
        map_index = self.map_combo.findData(self.cfg["bot"]["map"])
        if map_index >= 0:
            self.map_combo.setCurrentIndex(map_index)
            self.on_map_selected(map_index)

        # Advance settings
        self.update_advance_setting_ui_from_cfg()

    def set_gbox_enabled(self, enabled: bool):
        self.bot_mode.setEnabled(enabled)
        self.button_load_config.setEnabled(enabled)
        self.button_readiness_name.setEnabled(enabled)
        self.button_readiness_map.setEnabled(enabled)
        gray_style = "color: lightgray;" if not enabled else ""
        gboxs = [
            self.attack_gbox,
            self.key_binding_gbox,
            self.pet_skill_gbox,
            self.map_selection_gbox,
        ]
        gboxs += list(self.advance_settings_gboxes.values())
        # Apply disable + style
        for gbox in gboxs:
            gbox.setEnabled(enabled)
            if hasattr(gbox, "setStyleSheet"):
                gbox.setStyleSheet(gray_style)

    def on_tab_changed(self, index):
        tab = self.page_stack.widget(index)
        self.header_page_title.setText(self.page_titles[index])
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        settings_collected = self.commit_current_ui_settings()
        if tab is self.tab_game_window_viz:
            self.controller.enable_bot_viz("game")

        elif tab is self.tab_route_map_viz:
            self.controller.enable_bot_viz("route")

        elif tab is self.tab_main:
            self.controller.disable_bot_viz()
            if settings_collected:
                self.apply_config_to_ui()

        elif tab is self.tab_advance_setting:
            self.controller.disable_bot_viz()
            if settings_collected:
                self.apply_config_to_ui()

        elif tab is self.tab_announcement:
            self.controller.disable_bot_viz()

        else:
            logger.error(f"[UI] Unexpected tab index: {index}")
            self.controller.disable_bot_viz()

        logger.info(f"[UI] user change page to {self.page_titles[index]}")

    def on_map_selected(self, index):
        map_name = self.map_combo.itemData(index)
        if not map_name:
            return
        self.selected_map = map_name

        map_path = os.path.join("minimaps", map_name)
        self.label_map_info.setText(f"已选择地图：{map_path}")

    def on_mode_checkbox_toggle(self, toggled_checkbox):
        '''
        Mutural exclusive checkbox
        '''
        if not toggled_checkbox.isChecked():
            # Prevent unchecking — always keep one mode selected
            toggled_checkbox.setChecked(True)
            return

        if toggled_checkbox == self.checkbox_attack:
            self.checkbox_aoe_skill.blockSignals(True)
            self.checkbox_aoe_skill.setChecked(False)
            self.checkbox_aoe_skill.blockSignals(False)

        elif toggled_checkbox == self.checkbox_aoe_skill:
            self.checkbox_attack.blockSignals(True)
            self.checkbox_attack.setChecked(False)
            self.checkbox_attack.blockSignals(False)

    def toggle_auto_add_hp(self, state):
        '''
        Callback function for auto add hp checkbox
        '''
        enabled = Qt.CheckState(state) == Qt.Checked
        self.hp_input_widget.setVisible(enabled)
        logger.debug("[toggle_auto_add_hp] Checkbox toggled: "
                    f"{'Enabled' if enabled else 'Disabled'}")

    def toggle_auto_add_mp(self, state):
        '''
        Callback function for auto add mp checkbox
        '''
        enabled = Qt.CheckState(state) == Qt.Checked
        self.mp_input_widget.setVisible(enabled)
        logger.debug("[toggle_auto_add_mp] Checkbox toggled: "
                    f"{'Enabled' if enabled else 'Disabled'}")

    def toggle_start_ui(self):
        if self._readiness_busy:
            return
        if self.button_start_pause.isChecked(): # When start autobot
            if self.route_studio_window is not None:
                self.button_start_pause.setChecked(False)
                self.load_config_error_label.setText(
                    "请先关闭“路线录制与编辑”窗口。"
                )
                self.load_config_error_label.setVisible(True)
                return
            candidate_cfg, error = self.build_candidate_config_from_ui()
            if candidate_cfg is None:
                self.attack_range_error_label.setText(error)
                self.attack_range_error_label.setVisible(True)
                self.button_start_pause.setChecked(False)
                return
            if not self.validate_attack_detection_ranges(candidate_cfg):
                self.button_start_pause.setChecked(False)
                return
            self.cfg = candidate_cfg

            # Save UI config to tmp file
            cfg_path = self.runtime_cfg_path
            try:
                save_yaml(self.cfg, cfg_path)
            except Exception as exc:
                self._finish_start_ui(-1, f"无法保存启动配置：{exc}")
                return

            # Start AutoBot
            self._reset_readiness(candidate_cfg)
            if hasattr(self.controller, "start_bot_async"):
                self._set_readiness_busy(True)
                self._set_run_status("加载与检查中", "#fbbf24", "#422006")
                if not self.controller.start_bot_async(cfg_path):
                    self.finish_readiness_operation("start", -1)
            else:
                self._finish_start_ui(self.controller.start_bot(cfg_path))

        else: # When pause autobot
            self.button_start_pause.setText("▶ 开始 (F1)")
            self.button_route_studio.setEnabled(True)
            self.controller.pause_bot()
            self._set_run_status("未运行", "#94a3b8", "#1e293b")
            self.set_gbox_enabled(True)
            self.button_check_readiness.setEnabled(True)
            self.readiness_summary.setText("已暂停；下表与监控保留最后结果，非实时。可重新检查、标定或编辑地图。")
            self._mark_monitor_history("已暂停：保留最后画面，非实时")

    def _finish_start_ui(self, ret, message=None):
        if ret == 0:
            self._mark_monitor_history("等待本次运行的新画面；此前图像仅为历史参考")
            self.button_start_pause.setText("⏸ 暂停 (F1)")
            self.button_route_studio.setEnabled(False)
            self.button_check_readiness.setEnabled(False)
            self._set_run_status("已启动·等待识别", "#fbbf24", "#422006")
            self.readiness_summary.setText("资源已加载，等待当前画面识别；启动成功不代表所有识别均通过。")
            self.load_config_error_label.setVisible(False)
            self.set_gbox_enabled(False)
        else:
            self.button_start_pause.setChecked(False)
            self.button_start_pause.setText("▶ 开始 (F1)")
            self._set_run_status("启动失败", "#f87171", "#451a1a")
            message = message or getattr(self.controller, "last_start_error", "") or "启动失败；请检查当前画面以查看逐项问题。"
            self.load_config_error_label.setText(message)
            self.load_config_error_label.setVisible(True)
            self.readiness_summary.setText(message)
            self.readiness_last_issue.setText("启动失败（保留）：" + message)
            self.set_gbox_enabled(True)
            self.button_check_readiness.setEnabled(True)

    def handle_runtime_stopped(self, reason):
        """Restore editable UI state after a worker-initiated safety stop."""
        signals_were_blocked = self.button_start_pause.blockSignals(True)
        try:
            self.button_start_pause.setChecked(False)
        finally:
            self.button_start_pause.blockSignals(signals_were_blocked)
        self.button_start_pause.setText("▶ 开始 (F1)")
        self.button_route_studio.setEnabled(True)
        self.set_gbox_enabled(True)
        self.button_check_readiness.setEnabled(True)

        if reason == "return_home":
            self._set_run_status("已回城停止", "#fbbf24", "#422006")
            message = "血量持续过低，已执行回城并停止自动助手。"
        elif reason in {"input_error", "engine_error"}:
            self._set_run_status("错误", "#f87171", "#451a1a")
            message = "自动助手因运行异常已安全停止，请查看日志。"
        elif reason == "route_ladder_failed":
            self._set_run_status("挂梯失败", "#f87171", "#451a1a")
            message = "挂梯连续失败，已安全停止；请检查起跳点、梯子中心和挂梯方向。"
        else:
            self._set_run_status("已停止", "#94a3b8", "#1e293b")
            message = "自动助手已停止。"
        self.load_config_error_label.setText(message)
        self.load_config_error_label.setVisible(True)
        self.readiness_summary.setText(message + " 下表与监控保留最后结果，非实时。请先排查再重新启动。")
        self.readiness_last_issue.setText("最近停止原因（保留）：" + message)
        self._mark_monitor_history("已安全停止：保留最后画面，非实时。" + message)

    def refresh_nametag_profiles(self, preferred=None):
        """Refresh the editable profile selector without losing typed text."""
        combo = getattr(self, "nametag_profile_combo", None)
        if combo is None:
            return
        selected = combo.currentText().strip() if preferred is None else preferred
        names = self.nametag_profile_repository.list_profiles()
        combo.blockSignals(True)
        try:
            combo.clear()
            for name in names:
                combo.addItem(name, name)
            index = combo.findData(selected)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                combo.setEditText(str(selected or ""))
        finally:
            combo.blockSignals(False)

    def _persist_current_custom_config(self):
        """Persist the current diff to the selected writable YAML atomically."""
        if "config_default.yaml" in self.path_cfg_custom:
            self.path_cfg_custom = str(
                writable_path("config", "config_custom.yaml")
            )
        cfg_diff = get_cfg_diff(self.cfg_base, self.cfg)
        try:
            explicit_profile = load_yaml(self.path_cfg_custom)
        except (FileNotFoundError, TypeError, ValueError):
            explicit_profile = {}
        cfg_diff = retain_explicit_config_values(
            cfg_diff, self.cfg, explicit_profile
        )
        save_yaml(cfg_diff, self.path_cfg_custom)

    def open_nametag_calibration(self):
        """Open the in-app calibrator only while all input control is idle."""
        if self._readiness_busy:
            return
        self.invalidate_readiness()
        if self.button_start_pause.isChecked():
            self.load_config_error_label.setText(
                "请先暂停自动助手，再标定人物名字。"
            )
            self.load_config_error_label.setVisible(True)
            return
        if self.route_studio_window is not None:
            self.load_config_error_label.setText(
                "请先关闭“路线录制与编辑”窗口，再标定人物名字。"
            )
            self.load_config_error_label.setVisible(True)
            return
        if self.nametag_calibration_active:
            return
        if not self.commit_current_ui_settings():
            return

        from src.ui.nametag_calibration import NameTagCalibrationDialog

        self.nametag_calibration_active = True
        self.button_start_pause.setEnabled(False)
        self.button_route_studio.setEnabled(False)
        dialog = None
        try:
            dialog = NameTagCalibrationDialog(
                cfg=self.cfg,
                current_profile=self.cfg["nametag"].get("name", ""),
                repository=self.nametag_profile_repository,
                parent=self,
            )
            result = dialog.exec()
        except RuntimeError as exc:
            self.load_config_error_label.setText(str(exc))
            self.load_config_error_label.setVisible(True)
            return
        finally:
            self.nametag_calibration_active = False
            if not self.button_start_pause.isChecked():
                self.button_start_pause.setEnabled(True)
                self.button_route_studio.setEnabled(True)

        if result != QDialog.Accepted or dialog is None or not dialog.applied_name:
            return
        applied_name = dialog.applied_name
        self.cfg["nametag"]["name"] = applied_name
        self.refresh_nametag_profiles(preferred=applied_name)
        try:
            self._persist_current_custom_config()
        except (OSError, TypeError, ValueError) as exc:
            self.load_config_error_label.setText(
                "人物名字资料已保存，但当前配置未能持久化；"
                f"本次界面仍已选中该配置。错误：{exc}"
            )
            self.load_config_error_label.setVisible(True)
            logger.error(f"[人物名字标定] 配置保存失败：{exc}")
            return
        self.cfg_loaded_defaults["nametag"]["name"] = applied_name
        self.load_config_error_label.setText(
            f"人物名字配置“{applied_name}”已保存并应用。"
        )
        self.load_config_error_label.setVisible(True)

    def open_route_studio(self):
        if self._readiness_busy:
            return
        self.invalidate_readiness()
        """Open the shared editor while enforcing exclusive control ownership."""
        if self.button_start_pause.isChecked():
            self.load_config_error_label.setText(
                "请先暂停自动助手，再打开路线录制器。"
            )
            self.load_config_error_label.setVisible(True)
            return
        if self.route_studio_window is not None:
            self.route_studio_window.show()
            self.route_studio_window.raise_()
            self.route_studio_window.activateWindow()
            return
        from src.ui.route_studio import RouteStudioWindow

        if not self.commit_current_ui_settings():
            return
        try:
            self.route_studio_window = RouteStudioWindow(
                cfg=self.cfg,
                initial_map=self.selected_map,
                parent=self,
            )
        except RuntimeError as exc:
            self.route_studio_window = None
            self.load_config_error_label.setText(str(exc))
            self.load_config_error_label.setVisible(True)
            return
        self.route_studio_window.setAttribute(Qt.WA_DeleteOnClose, True)
        self.route_studio_window.studio_closed.connect(self._route_studio_closed)
        self.route_studio_window.project_saved.connect(
            self._route_project_saved
        )
        self.button_start_pause.setEnabled(False)
        self.route_studio_window.show()

    def _route_studio_closed(self):
        self.route_studio_window = None
        self.refresh_map_selector(preferred_map=self.selected_map)
        if not self.button_start_pause.isChecked():
            self.button_start_pause.setEnabled(True)

    def _route_project_saved(self, map_id):
        """Make a newly saved runnable map selectable immediately."""
        self.refresh_map_selector(preferred_map=str(map_id))

    def update_cfg_from_main_ui(self, target_cfg=None):
        '''
        Collect setting from UI framework
        '''
        cfg = self.cfg if target_cfg is None else target_cfg
        # Bot control gbox
        cfg["bot"]["mode"] = self.bot_mode.currentData()
        # Attack setting gbox
        attack_mode = self.attack_mode.currentData()
        if attack_mode == "directional":
            cfg["bot"]["attack"] = "directional"
            cfg["key"]["directional_attack"] = self.basic_attack_key.get_key()
            atk_cfg = cfg["directional_attack"]
        elif attack_mode == "aoe_skill":
            cfg["bot"]["attack"] = "aoe_skill"
            cfg["key"]["aoe_skill"] = self.basic_attack_key.get_key()
            atk_cfg = cfg["aoe_skill"]
        else:
            logger.error(
                "[update_cfg_from_main_ui] Unsupported attack mode: "
                f"{cfg['bot']['attack']}"
            )
            atk_cfg = None
        if atk_cfg is not None:
            try:
                atk_cfg["range_x"] = int(self.attack_range_x.text())
                atk_cfg["range_y"] = int(self.attack_range_y.text())
                atk_cfg["cooldown"] = float(self.attack_cooldown.text())
            except ValueError:
                # Tab switching may occur while a field is being edited. Start
                # performs strict validation and will not launch in this state.
                pass
        # Key binding gbox
        cfg["key"]["teleport"] = self.teleport_key.get_key()
        cfg["key"]["aoe_skill"] = self.aoe_skill_key.get_key()
        cfg["key"]["jump"] = self.jump_key.get_key()
        cfg["key"]["return_home"] = self.return_home_key.get_key()
        # Auto Add HP
        cfg["health_monitor"]["auto_hp_enabled"] = \
            self.checkbox_auto_add_hp.isChecked()
        cfg["health_monitor"]["add_hp_percent"] = int(self.add_hp_percent.text())
        cfg["key"]["add_hp"] = self.add_hp_key.get_key()
        # Auto Add MP
        cfg["health_monitor"]["auto_mp_enabled"] = \
            self.checkbox_auto_add_mp.isChecked()
        cfg["health_monitor"]["add_mp_percent"] = int(self.add_mp_percent.text())
        cfg["key"]["add_mp"] = self.add_mp_key.get_key()
        # Map selection
        cfg["bot"]["map"] = self.selected_map

    @staticmethod
    def _parse_advanced_value(text, template):
        stripped = text.strip()
        if isinstance(template, bool):
            raise TypeError("Boolean values must use a checkbox")
        if isinstance(template, int):
            if stripped == "":
                raise ValueError("empty integer")
            return int(stripped)
        if isinstance(template, float):
            if stripped == "":
                raise ValueError("empty float")
            value = float(stripped)
            if not math.isfinite(value):
                raise ValueError("non-finite float")
            return value
        if isinstance(template, str):
            return text.strip()
        raise TypeError(f"Unsupported advanced setting type: {type(template).__name__}")

    def collect_advance_settings_from_ui(self, candidate_cfg):
        """Parse every advanced widget into *candidate_cfg* atomically."""
        for section, gbox in self.advance_settings_gboxes.items():
            refs = getattr(gbox, "_field_refs", {})
            for key, widget in refs.items():
                template = candidate_cfg[section][key]
                try:
                    if isinstance(widget, QCheckBox):
                        value = widget.isChecked()
                    elif isinstance(widget, QComboBox):
                        value = (
                            widget.currentText().strip()
                            if widget.isEditable() else widget.currentData()
                        )
                        if value is None:
                            raise ValueError("no selected option")
                    elif isinstance(widget, list):
                        if len(widget) != len(template):
                            raise ValueError("list length mismatch")
                        parsed = [
                            self._parse_advanced_value(line.text(), item)
                            for line, item in zip(widget, template)
                        ]
                        value = tuple(parsed) if isinstance(template, tuple) else parsed
                    elif isinstance(widget, QLineEdit):
                        value = self._parse_advanced_value(widget.text(), template)
                    else:
                        raise TypeError(type(widget).__name__)
                except (TypeError, ValueError) as exc:
                    logger.debug(
                        f"[UI] Invalid advanced setting {section}.{key}: {exc}"
                    )
                    return False, (
                        f"高级设置 {section}.{key} 的值无效，已阻止启动。"
                    )
                candidate_cfg[section][key] = value
        return True, ""

    def build_candidate_config_from_ui(self):
        """Return a fully parsed UI snapshot without partially replacing cfg."""
        candidate_cfg = copy.deepcopy(self.cfg)
        try:
            self.update_cfg_from_main_ui(candidate_cfg)
        except (TypeError, ValueError) as exc:
            logger.debug(f"[UI] Invalid main setting: {exc}")
            return None, "主界面存在无效数值，已阻止启动。"
        valid, error = self.collect_advance_settings_from_ui(candidate_cfg)
        return (candidate_cfg, "") if valid else (None, error)

    def commit_current_ui_settings(self):
        candidate_cfg, error = self.build_candidate_config_from_ui()
        if candidate_cfg is None:
            self.attack_range_error_label.setText(error)
            self.attack_range_error_label.setVisible(True)
            return False
        self.cfg = candidate_cfg
        return True

    def validate_attack_detection_ranges(self, candidate_cfg=None):
        """Validate the four pixel range fields before starting the bot."""
        cfg = self.cfg if candidate_cfg is None else candidate_cfg
        if cfg["bot"].get("mode") == "continuous_attack":
            if cfg["bot"].get("attack") != "directional":
                self.attack_range_error_label.setText(
                    "持续定向攻击模式仅支持方向攻击，已阻止启动。"
                )
                self.attack_range_error_label.setVisible(True)
                return False
            if not str(cfg["key"].get("directional_attack", "")).strip():
                self.attack_range_error_label.setText(
                    "持续定向攻击模式的攻击键不能为空，已阻止启动。"
                )
                self.attack_range_error_label.setVisible(True)
                return False
            if not str(cfg["key"].get("jump", "")).strip():
                self.attack_range_error_label.setText(
                    "持续定向攻击模式的跳跃键不能为空，已阻止启动。"
                )
                self.attack_range_error_label.setVisible(True)
                return False
        labels = (
            (self.attack_range_x, "攻击横向单侧距离"),
            (self.attack_range_y, "攻击纵向总高度"),
            (self.detect_range_x, "检测横向半径"),
            (self.detect_range_y, "检测纵向半径"),
        )
        for widget, label in labels:
            try:
                value = int(widget.text().strip())
            except ValueError:
                value = 0
            if not 1 <= value <= 9999:
                self.attack_range_error_label.setText(
                    f"{label}必须是 1–9999 px 之间的整数，已阻止启动。"
                )
                self.attack_range_error_label.setVisible(True)
                return False
        health_cfg = cfg["health_monitor"]
        for kind, label in (("hp", "生命值"), ("mp", "魔法值")):
            enabled = bool(health_cfg[f"auto_{kind}_enabled"])
            threshold = int(health_cfg[f"add_{kind}_percent"])
            if not 0 <= threshold <= 100 or (enabled and threshold == 0):
                allowed = "1–100" if enabled else "0–100"
                self.attack_range_error_label.setText(
                    f"{label}补给阈值必须是 {allowed} 之间的整数，已阻止启动。"
                )
                self.attack_range_error_label.setVisible(True)
                return False
            if enabled and not str(cfg["key"].get(f"add_{kind}", "")).strip():
                self.attack_range_error_label.setText(
                    f"已启用自动补充{label}，补给按键不能为空。"
                )
                self.attack_range_error_label.setVisible(True)
                return False
            cooldown = float(health_cfg[f"add_{kind}_cooldown"])
            if not 0.1 <= cooldown <= 60.0:
                self.attack_range_error_label.setText(
                    f"{label}补给冷却必须在 0.1–60 秒之间，已阻止启动。"
                )
                self.attack_range_error_label.setVisible(True)
                return False
        if (health_cfg.get("force_heal") or
                health_cfg.get("return_home_if_no_potion")) and not \
                health_cfg["auto_hp_enabled"]:
            self.attack_range_error_label.setText(
                "强制补给或无药回城要求先启用自动补充生命值。"
            )
            self.attack_range_error_label.setVisible(True)
            return False
        if (health_cfg.get("return_home_if_no_potion") and
                not str(cfg["key"].get("return_home", "")).strip()):
            self.attack_range_error_label.setText(
                "已启用无药回城，回城按键不能为空。"
            )
            self.attack_range_error_label.setVisible(True)
            return False
        watchdog_timeout = float(health_cfg["return_home_watch_dog_timeout"])
        if not 0.1 <= watchdog_timeout <= 3600.0:
            self.attack_range_error_label.setText(
                "无药回城等待时间必须在 0.1–3600 秒之间，已阻止启动。"
            )
            self.attack_range_error_label.setVisible(True)
            return False
        refs = self.advance_settings_gboxes["monster_detect"]._field_refs
        confidence_widget = refs.get("yolo_confidence")
        try:
            confidence = float(confidence_widget.text().strip())
        except (AttributeError, ValueError):
            confidence = -1.0
        if not 0.05 <= confidence <= 0.95:
            self.attack_range_error_label.setText(
                "YOLO 置信度必须在 0.05–0.95 之间，已阻止启动。"
            )
            self.attack_range_error_label.setVisible(True)
            return False
        min_box_side_widget = refs.get("yolo_min_monster_box_side")
        try:
            min_box_side = int(min_box_side_widget.text().strip())
        except (AttributeError, ValueError):
            min_box_side = -1
        max_min_box_side = min(MODEL_WIDTH, MODEL_HEIGHT)
        if not 0 <= min_box_side <= max_min_box_side:
            self.attack_range_error_label.setText(
                f"YOLO 怪物框最小边长必须是 0–{max_min_box_side} px "
                "之间的整数，已阻止启动。"
            )
            self.attack_range_error_label.setVisible(True)
            return False
        exclusion_fields = (
            ("yolo_player_exclusion_width", MODEL_WIDTH, "宽度"),
            ("yolo_player_exclusion_height", MODEL_HEIGHT, "高度"),
        )
        for field, maximum, label in exclusion_fields:
            widget = refs.get(field)
            try:
                value = int(widget.text().strip())
            except (AttributeError, ValueError):
                value = -1
            if not 0 <= value <= maximum:
                self.attack_range_error_label.setText(
                    f"YOLO 人物排除区{label}必须是 0–{maximum} px "
                    "之间的整数，已阻止启动。"
                )
                self.attack_range_error_label.setVisible(True)
                return False
        fixed_platform_refs = self.advance_settings_gboxes["fixed_platform"]._field_refs
        width_px_widget = fixed_platform_refs.get("width_px")
        try:
            width_px = int(width_px_widget.text().strip())
        except (AttributeError, ValueError):
            width_px = 0
        if not 10 <= width_px <= 1000:
            self.attack_range_error_label.setText(
                "固定平台巡逻总宽度必须是 10–1000 px 之间的整数，已阻止启动。"
            )
            self.attack_range_error_label.setVisible(True)
            return False
        if int(cfg["monster_detect"].get("yolo_max_det", 0)) != RUNTIME_POLICY.yolo_max_det:
            self.attack_range_error_label.setText("YOLO 最大检测数必须保持为 50。")
            self.attack_range_error_label.setVisible(True)
            return False
        self.attack_range_error_label.setVisible(False)
        return True

    def update_debug_canvas(self, img):
        if img is None:
            return

        height, width, _ = img.shape
        qimg = QImage(
            img.data, width, height, int(img.strides[0]), QImage.Format_BGR888
        ).copy()
        pixmap = QPixmap.fromImage(qimg)

        self.debug_canvas.set_source_pixmap(pixmap)

    def update_route_map_canvas(self, img):
        if img is None:
            return

        height, width, _ = img.shape
        qimg = QImage(
            img.data, width, height, int(img.strides[0]), QImage.Format_BGR888
        ).copy()
        pixmap = QPixmap.fromImage(qimg)

        self.route_map_canvas.set_source_pixmap(pixmap)

    def update_visualization_metrics(self, mode, metrics):
        """Show worker and UI timing without coupling it to the image payload."""
        text = (
            "控制 P95 {control_p95_ms:.1f} ms | "
            "YOLO 推理 P95 {infer_p95_ms:.1f} ms | "
            "预览 P95 {viz_p95_ms:.1f} ms | "
            "UI P95 {ui_p95_ms:.1f} ms | 显示 {display_fps:.1f} FPS | "
            "帧龄 P95 {preview_age_p95_ms:.0f} ms | 覆盖 {dropped:.0f}"
        ).format(
            control_p95_ms=float(metrics.get("control_p95_ms", 0.0)),
            infer_p95_ms=float(metrics.get("infer_p95_ms", 0.0)),
            viz_p95_ms=float(metrics.get("viz_p95_ms", 0.0)),
            ui_p95_ms=float(metrics.get("ui_p95_ms", metrics.get("ui_ms", 0.0))),
            display_fps=float(metrics.get("display_fps", 0.0)),
            preview_age_p95_ms=float(
                metrics.get("preview_age_p95_ms", metrics.get("preview_age_ms", 0.0))
            ),
            dropped=float(metrics.get("dropped", 0.0)),
        )
        label = self.game_viz_metrics if mode == "game" else self.route_viz_metrics
        label.setText(text)

    def update_advance_setting_ui_from_cfg(self):
        '''
        Updates UI fields in an existing gbox to reflect the latest cfg[title].
        '''
        for title in self.cfg:
            if title in ADV_SETTINGS_HIDE: # skip hide settings
                continue
            refs = getattr(self.advance_settings_gboxes[title], "_field_refs", {})
            for key, value in self.cfg[title].items():
                widget = refs.get(key)
                if widget is None:
                    continue  # unknown field, skip

                # Checkbox
                if isinstance(value, bool) and isinstance(widget, QCheckBox):
                    widget.setChecked(value)

                # List of QLineEdits
                elif isinstance(value, (list, tuple)) and isinstance(widget, list):
                    for line, v in zip(widget, value):
                        line.setText(str(v))

                # Single numeric value
                elif isinstance(value, (int, float)) and isinstance(widget, QLineEdit):
                    widget.setText(str(value))

                # Droplist
                elif isinstance(value, str) and isinstance(widget, QComboBox):
                    index = widget.findData(value)
                    if index != -1:
                        widget.setCurrentIndex(index)
                    elif widget.isEditable():
                        widget.setEditText(value)

                # String
                elif isinstance(value, str) and isinstance(widget, QLineEdit):
                    widget.setText(value)

    def append_log(self, message: str, level: int):
        color = QColor("white")
        if level >= logging.ERROR:
            color = QColor("red")
        elif level >= logging.WARNING:
            color = QColor("orange")

        fmt = QTextCharFormat()
        fmt.setForeground(color)

        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(message + "\n", fmt)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def closeEvent(self, event):
        '''
        Call when user close the UI window
        '''
        if self._readiness_busy:
            self.cancel_readiness()
            self.readiness_summary.setText("正在释放检查资源，请完成后再次关闭窗口。")
            event.ignore()
            return
        if self.route_studio_window is not None:
            self.route_studio_window.close()
            if self.route_studio_window is not None:
                event.ignore()
                return

        # Collect every current field before persisting.  Invalid drafts do
        # not replace the last valid configuration during shutdown.
        self.commit_current_ui_settings()
        # Save current UI config to config_XXXX.yaml
        if "config_default.yaml" not in self.path_cfg_custom:
            self._persist_current_custom_config()

        # Save your UI state (e.g., last loaded config path)
        self.save_ui_state()

        # Terminate all bot threads
        self.controller.terminate_bot()

        try:
            Path(self.runtime_cfg_path).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(f"[UI] 无法清理运行临时配置：{exc}")

        event.accept()  # Continue with the close

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
