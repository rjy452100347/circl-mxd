import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit

from src.ui.ui import APP_QSS, MainWindow
from src.utils.logger import logger
from src.utils.ui import ResponsiveImageLabel


class _Controller:
    def __init__(self):
        self.viz_events = []

    def enable_bot_viz(self, mode="game"):
        self.viz_events.append(("enable", mode))

    def disable_bot_viz(self):
        self.viz_events.append(("disable", None))

    def terminate_bot(self):
        self.viz_events.append(("terminate", None))


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt_app, monkeypatch):
    monkeypatch.setattr(MainWindow, "load_ui_state", lambda self: None)
    controller = _Controller()
    widget = MainWindow(controller)
    widget.apply_config_to_ui()
    yield widget
    logger._logger.removeHandler(widget.qt_log_handler)
    widget.deleteLater()


def test_professional_shell_has_four_chinese_sidebar_pages(window):
    assert window.windowTitle() == "冒险岛自动练级助手"
    assert window.page_titles == ("运行中心", "高级设置", "窗口监控", "路线监控")
    assert [button.text() for button in window.nav_buttons] == list(window.page_titles)
    assert window.page_stack.count() == 4
    assert window.minimumWidth() == 1080
    assert window.minimumHeight() == 700
    assert window.width() == 1280
    assert window.height() == 820
    assert "#0b1220" in APP_QSS
    assert "#0f172a" in APP_QSS
    assert "#111c2e" in APP_QSS
    assert "#22d3ee" in APP_QSS
    assert "QLabel, QCheckBox, QRadioButton { color: #dbeafe" in APP_QSS
    assert "QComboBox QAbstractItemView" in APP_QSS


def test_stable_enums_and_map_values_do_not_depend_on_visible_chinese(window):
    assert window.attack_mode.itemText(0) == "方向攻击"
    assert window.attack_mode.itemData(0) == "directional"
    assert window.bot_mode.itemText(0) == "普通"
    assert window.bot_mode.itemData(0) == "normal"
    assert isinstance(window.map_combo, QComboBox)
    assert not window.map_combo.isEditable()
    assert window.map_combo.count() > 0
    for index in range(window.map_combo.count()):
        value = window.map_combo.itemData(index)
        assert isinstance(value, str) and value
        assert value.isascii()


def test_removed_features_have_no_main_ui_entry(window):
    for attribute in (
        "button_screenshot", "button_record", "party_key",
        "buff_skill_gbox", "input_method",
    ):
        assert not hasattr(window, attribute)

    all_text = " ".join(
        widget.text()
        for widget in window.findChildren(type(window.header_status))
        if hasattr(widget, "text")
    )
    for removed in ("保存截图", "开始录制", "组队血条", "增益技能", "邮件通知"):
        assert removed not in all_text


def test_advanced_sections_match_the_reduced_product_surface(window):
    expected = {
        "directional_attack", "aoe_skill", "health_monitor", "combat_tracking",
        "teleport", "edge_teleport", "route", "watchdog", "patrol",
        "nametag", "monster_detect", "minimap", "game_window",
        "channel_change", "scheduled_channel_switching", "ui_coords", "profiler",
    }
    assert set(window.advance_settings_gboxes) == expected
    assert "system" not in window.advance_settings_gboxes
    assert "route_recoder" not in window.advance_settings_gboxes


def test_yolo_card_has_only_current_deployment_controls(window):
    refs = window.advance_settings_gboxes["monster_detect"]._field_refs
    assert set(refs) == {
        "openvino_deployment_root", "yolo_variant",
        "yolo_confidence", "yolo_max_det",
    }
    variant = refs["yolo_variant"]
    assert isinstance(variant, QComboBox)
    assert variant.currentData() == "int8_v2"
    assert variant.itemText(variant.findData("int8_v2")) == "INT8 v2（默认）"
    max_det = refs["yolo_max_det"]
    assert isinstance(max_det, QLineEdit)
    assert max_det.text() == "50"
    assert max_det.isReadOnly()
    assert window.detect_range_x.text() == "640"
    assert window.detect_range_y.text() == "112"
    assert not window.detect_range_x.isEnabled()
    assert not window.detect_range_y.isEnabled()


def test_attack_fields_and_combat_tracking_round_trip(window):
    window.attack_mode.setCurrentIndex(window.attack_mode.findData("directional"))
    window.attack_range_x.setText("525")
    window.attack_range_y.setText("105")
    window.attack_cooldown.setText("0.9")
    window.update_cfg_from_main_ui()

    assert window.cfg["directional_attack"]["range_x"] == 525
    assert window.cfg["directional_attack"]["range_y"] == 105
    tracking = window.advance_settings_gboxes["combat_tracking"]._field_refs
    tracking["pursuit_vertical_tolerance"].setText("72")
    tracking["target_lost_grace_seconds"].setText("0.35")
    assert window.cfg["combat_tracking"] == {
        "pursuit_vertical_tolerance": 72,
        "target_lost_grace_seconds": 0.35,
    }


def test_yolo_confidence_validation_and_fixed_detection_range(window):
    confidence = window.advance_settings_gboxes["monster_detect"]._field_refs[
        "yolo_confidence"
    ]
    confidence.setText("0.01")
    assert not window.validate_attack_detection_ranges()
    assert "0.05–0.95" in window.attack_range_error_label.text()

    confidence.setText("0.60")
    assert window.validate_attack_detection_ranges()


def test_detection_reset_buttons_restore_loaded_profile_only(window):
    original_name = window.cfg_loaded_defaults["nametag"]["name"]
    original_confidence = window.cfg_loaded_defaults["monster_detect"][
        "yolo_confidence"
    ]
    window.cfg["nametag"]["name"] = "changed"
    window.cfg["monster_detect"]["yolo_confidence"] = 0.25
    window.cfg["directional_attack"]["range_x"] = 777

    name_group = window.advance_settings_gboxes["nametag"]
    mob_group = window.advance_settings_gboxes["monster_detect"]
    assert name_group._reset_button.text() == "恢复名字检测默认值"
    assert mob_group._reset_button.text() == "恢复怪物检测默认值"

    window.restore_detection_defaults("nametag")
    window.restore_detection_defaults("monster_detect")

    assert window.cfg["nametag"]["name"] == original_name
    assert window.cfg["monster_detect"]["yolo_confidence"] == original_confidence
    assert window.cfg["directional_attack"]["range_x"] == 777


def test_running_lock_does_not_disable_navigation_or_monitor_pages(window):
    window.set_gbox_enabled(False)
    assert not window.attack_gbox.isEnabled()
    assert all(button.isEnabled() for button in window.nav_buttons)
    assert window.page_stack.isEnabled()
    window.set_gbox_enabled(True)


def test_sidebar_switches_existing_viz_channel_without_new_render_loop(window):
    controller = window.controller
    controller.viz_events.clear()

    window.page_stack.setCurrentIndex(2)
    window.page_stack.setCurrentIndex(3)
    window.page_stack.setCurrentIndex(0)

    assert controller.viz_events == [
        ("enable", "game"),
        ("enable", "route"),
        ("disable", None),
    ]


def test_viz_canvas_rescales_source_when_window_size_changes(qt_app):
    canvas = ResponsiveImageLabel()
    source = QPixmap(800, 400)
    source.fill()
    canvas.resize(400, 300)
    canvas.show()
    canvas.set_source_pixmap(source)
    qt_app.processEvents()

    first = canvas.pixmap().size()
    canvas.resize(800, 600)
    qt_app.processEvents()
    second = canvas.pixmap().size()

    assert (first.width(), first.height()) == (400, 200)
    assert (second.width(), second.height()) == (800, 400)
    canvas.close()


def test_window_viz_preserves_bgr_colour_in_qt_pixmap(window):
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0] = (17, 83, 211)

    window.update_debug_canvas(frame)

    colour = window.debug_canvas._source_pixmap.toImage().pixelColor(0, 0)
    assert (colour.red(), colour.green(), colour.blue()) == (211, 83, 17)
