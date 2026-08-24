'''
UI main
'''
# Standard import
import sys
import os
import json
import copy
import logging

# PySide 6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QCheckBox, QFileDialog, QHBoxLayout, QLineEdit,
    QPlainTextEdit, QGroupBox, QFormLayout, QGridLayout, QFrame,
    QSizePolicy, QComboBox, QScrollArea, QStackedWidget
)
from PySide6.QtGui import (QTextCharFormat, QColor, QTextCursor, QPixmap,
                          QImage, QIntValidator)
from PySide6.QtCore import Qt, Signal

# Local import
from src.utils.logger import logger
from src.utils.ui import (
    validate_numerical_input, clear_debug_canvas,
    create_error_label, SingleKeyEdit, QtLogHandler, create_advance_setting_gbox,
    ResponsiveImageLabel,
)
from src.utils.common import (
    load_yaml, override_cfg, is_mac, save_yaml, get_cfg_diff, load_yaml_with_comments
)
from src.engine.OpenVinoMonsterDetector import MODEL_HEIGHT, MODEL_WIDTH
from src.app_paths import writable_path
from src.config_compat import ConfigCompatibilityError, validate_custom_config
from src.runtime_policy import RUNTIME_POLICY

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

        self.controller = controller # autoBotController
        self.selected_map = ""

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

        self._build_application_shell()

        # Load previous stored UI state
        self.load_ui_state()

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
        self.page_titles = ("运行中心", "高级设置", "窗口监控", "路线监控")
        pages = (
            self.tab_main, self.tab_advance_setting,
            self.tab_game_window_viz, self.tab_route_map_viz,
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
        scroll_layout.addWidget(self.key_binding_gbox, 1, 0)

        # Pet function group box
        self.pet_skill_gbox = self.create_pet_skill_gbox()
        scroll_layout.addWidget(self.pet_skill_gbox, 1, 1)

        # Map selection group box
        self.map_selection_gbox = self.create_map_selection_gbox()
        scroll_layout.addWidget(self.map_selection_gbox, 2, 0)

        # Logger output window
        self.log_gbox = self.create_log_gbox()
        scroll_layout.addWidget(self.log_gbox, 2, 1)
        scroll_layout.setRowStretch(3, 1)

        scroll_area.setWidget(scroll_widget)

        tab_main = QWidget()
        layout = QVBoxLayout(tab_main)
        layout.addWidget(scroll_area)

        return tab_main

    def setup_advance_setting_tab(self):
        tab_advance_setting = QWidget()
        categories = (
            ("战斗", ("directional_attack", "aoe_skill", "health_monitor", "combat_tracking")),
            ("移动与路线", ("teleport", "edge_teleport", "route", "watchdog", "patrol")),
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
                gbox = create_advance_setting_gbox(
                    title, self.cfg, self.comments, self.comments_section
                )
                if title in {"nametag", "monster_detect"}:
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
                    max_det = gbox._field_refs["yolo_max_det"]
                    max_det.setReadOnly(True)
                    max_det.setToolTip("内部固定为 50。")
                    model_info = QLabel(
                        f"输入尺寸：{MODEL_WIDTH}×{MODEL_HEIGHT}\n类别：monster / player"
                    )
                    model_info.setObjectName("Subtle")
                    gbox.layout().addRow("模型规格", model_info)
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

        # Create a large QLabel as a canvas
        self.debug_canvas = ResponsiveImageLabel()
        clear_debug_canvas(self.debug_canvas)
        layout.addWidget(self.debug_canvas, 1)

        return tab_game_window_viz

    def setup_route_map_viz_tab(self):
        tab_route_map_viz_tab = QWidget()
        layout = QVBoxLayout()
        tab_route_map_viz_tab.setLayout(layout)

        # Create a large QLabel as a canvas
        self.route_map_canvas = ResponsiveImageLabel()
        clear_debug_canvas(self.route_map_canvas)
        layout.addWidget(self.route_map_canvas, 1)

        return tab_route_map_viz_tab

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

        # Load map list from directory
        self.map_combo = QComboBox()
        self.map_combo.setEditable(False)
        self.label_map_info = QLabel("请选择地图：")
        minimap_dir = "minimaps"
        if os.path.isdir(minimap_dir):
            map_names = os.listdir(minimap_dir)
        else:
            # The source distribution intentionally ships without screenshots.
            # Keep the UI usable and show the known map IDs as setup targets.
            map_names = sorted(self.data.get("map_mobs_mapping", {}))
            self.label_map_info.setText("未检测到本地地图资源，请先录制地图：")
        for name in map_names:
            if name.startswith("."):
                continue  # Skip hidden/system files

            if name in self.data["eng_to_cn"]:
                name_cn = self.data["eng_to_cn"][name]
                full_path = os.path.join(minimap_dir, name)
                if not os.path.isdir(minimap_dir) or os.path.isdir(full_path):
                    # self.list_widget_maps.addItem(f"{name} ({name_cn})")
                    display_text = f"{name} ({name_cn})"
                    self.map_combo.addItem(display_text, name)
        self.map_combo.currentIndexChanged.connect(self.on_map_selected)
        if self.map_combo.count():
            self.on_map_selected(self.map_combo.currentIndex())

        layout = QVBoxLayout()
        layout.addWidget(self.label_map_info)
        layout.addWidget(self.map_combo)
        gbox.setLayout(layout)
        return gbox

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

        # Bot Mode Dropdown
        layout_bot_mode = QHBoxLayout()
        layout_bot_mode.setSpacing(8)
        self.bot_mode = QComboBox()
        self.bot_mode.addItem("普通", "normal")
        self.bot_mode.addItem("辅助", "aux")
        self.bot_mode.addItem("巡逻", "patrol")

        layout_bot_mode.addWidget(QLabel("运行模式："))
        layout_bot_mode.addWidget(self.bot_mode)
        layout_bot_mode.setAlignment(Qt.AlignLeft)

        button_layout.addWidget(self.button_start_pause)
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
            custom_cfg = validate_custom_config(load_yaml(path))
            self.cfg = override_cfg(copy.deepcopy(self.cfg_base), custom_cfg)
        except FileNotFoundError:
            logger.warning(f"[UI] Unable to find config file: {path}")
            return
        except ConfigCompatibilityError as exc:
            error_label.setText(str(exc))
            error_label.setVisible(True)
            logger.error(str(exc))
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
        self.checkbox_auto_add_hp.setChecked(hm_cfg["add_hp_percent"] > 0)
        self.add_hp_percent.setText(str(hm_cfg["add_hp_percent"]))
        self.add_hp_key.set_key(self.cfg["key"]["add_hp"])
        self.checkbox_auto_add_mp.setChecked(hm_cfg["add_mp_percent"] > 0)
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
        if tab is self.tab_game_window_viz:
            self.controller.enable_bot_viz("game")

        elif tab is self.tab_route_map_viz:
            self.controller.enable_bot_viz("route")

        elif tab is self.tab_main:
            self.controller.disable_bot_viz()
            self.apply_config_to_ui()

        elif tab is self.tab_advance_setting:
            self.controller.disable_bot_viz()
            self.update_cfg_from_main_ui()
            self.apply_config_to_ui()

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
        if self.button_start_pause.isChecked(): # When start autobot
            if not self.validate_attack_detection_ranges():
                self.button_start_pause.setChecked(False)
                return
            self.update_cfg_from_main_ui()

            # Save UI config to tmp file
            cfg_path = "config/.config_tmp.yaml"
            save_yaml(self.cfg, cfg_path)

            # Start AutoBot
            ret = self.controller.start_bot(cfg_path)

            if ret == 0: # Start success
                self.button_start_pause.setText("⏸ 暂停 (F1)")
                self._set_run_status("运行中", "#4ade80", "#143525")
                self.set_gbox_enabled(False)
            else:
                # Start failed
                self.button_start_pause.setChecked(False)
                self._set_run_status("错误", "#f87171", "#451a1a")

        else: # When pause autobot
            self.button_start_pause.setText("▶ 开始 (F1)")
            self._set_run_status("未运行", "#94a3b8", "#1e293b")
            self.controller.pause_bot()
            self.set_gbox_enabled(True)
            clear_debug_canvas(self.debug_canvas) # Set debug viz to null
            clear_debug_canvas(self.route_map_canvas) # Set debug viz to null

    def update_cfg_from_main_ui(self):
        '''
        Collect setting from UI framework
        '''
        # Bot control gbox
        self.cfg["bot"]["mode"] = self.bot_mode.currentData()
        # Attack setting gbox
        attack_mode = self.attack_mode.currentData()
        if attack_mode == "directional":
            self.cfg["bot"]["attack"] = "directional"
            self.cfg["key"]["directional_attack"] = self.basic_attack_key.get_key()
            atk_cfg = self.cfg["directional_attack"]
        elif attack_mode == "aoe_skill":
            self.cfg["bot"]["attack"] = "aoe_skill"
            self.cfg["key"]["aoe_skill"] = self.basic_attack_key.get_key()
            atk_cfg = self.cfg["aoe_skill"]
        else:
            logger.error(f"[update_cfg_from_main_ui] Unsupported attack mode: {self.cfg['bot']['attack']}")
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
        self.cfg["key"]["teleport"] = self.teleport_key.get_key()
        self.cfg["key"]["aoe_skill"] = self.aoe_skill_key.get_key()
        self.cfg["key"]["jump"] = self.jump_key.get_key()
        self.cfg["key"]["return_home"] = self.return_home_key.get_key()
        # Auto Add HP
        if self.checkbox_auto_add_hp.isChecked():
            self.cfg["health_monitor"]["add_hp_percent"] = int(self.add_hp_percent.text())
            self.cfg["key"]["add_hp"] = self.add_hp_key.get_key()
        else:
            self.cfg["health_monitor"]["add_hp_percent"] = 0
        # Auto Add MP
        if self.checkbox_auto_add_mp.isChecked():
            self.cfg["health_monitor"]["add_mp_percent"] = int(self.add_mp_percent.text())
            self.cfg["key"]["add_mp"] = self.add_mp_key.get_key()
        else:
            self.cfg["health_monitor"]["add_mp_percent"] = 0
        # Map selection
        self.cfg["bot"]["map"] = self.selected_map

    def validate_attack_detection_ranges(self):
        """Validate the four pixel range fields before starting the bot."""
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
        if int(self.cfg["monster_detect"].get("yolo_max_det", 0)) != RUNTIME_POLICY.yolo_max_det:
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
        qimg = QImage(img.data, width, height, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)

        self.route_map_canvas.set_source_pixmap(pixmap)

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
        # Collect current UI setting and update to self.cfg
        self.update_cfg_from_main_ui()
        # Save current UI config to config_XXXX.yaml
        if "config_default.yaml" not in self.path_cfg_custom:
            cfg_diff = get_cfg_diff(self.cfg_base, self.cfg)
            save_yaml(cfg_diff, self.path_cfg_custom)

        # Save your UI state (e.g., last loaded config path)
        self.save_ui_state()

        # Terminate all bot threads
        self.controller.terminate_bot()

        event.accept()  # Continue with the close

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
