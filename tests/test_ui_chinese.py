import os
import copy
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QLineEdit, QPlainTextEdit, QLabel,
)

from src.ui.ui import APP_QSS, MainWindow
from src.utils.logger import logger
from src.utils.common import load_yaml
from src.utils.ui import ResponsiveImageLabel


class _Controller:
    def __init__(self):
        self.viz_events = []
        self.start_calls = []
        self.start_result = 0
        self.pause_calls = 0

    def enable_bot_viz(self, mode="game"):
        self.viz_events.append(("enable", mode))

    def disable_bot_viz(self):
        self.viz_events.append(("disable", None))

    def terminate_bot(self):
        self.viz_events.append(("terminate", None))

    def start_bot(self, path):
        self.start_calls.append((path, copy.deepcopy(load_yaml(path))))
        return self.start_result

    def pause_bot(self):
        self.pause_calls += 1


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
    for path, _cfg in controller.start_calls:
        Path(path).unlink(missing_ok=True)


def test_professional_shell_has_five_chinese_sidebar_pages(window):
    assert window.windowTitle() == "冒险岛自动练级助手"
    assert window.page_titles == (
        "运行中心", "高级设置", "窗口监控", "路线监控", "公告与交流"
    )
    assert [button.text() for button in window.nav_buttons] == list(window.page_titles)
    assert window.page_stack.count() == 5
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


def test_main_window_exposes_editable_nametag_profile_manager(window):
    refs = window.advance_settings_gboxes["nametag"]._field_refs
    assert isinstance(refs["name"], QComboBox)
    assert refs["name"].isEditable()
    assert window.nametag_calibration_button.text() == "标定与管理人物名字"

    refs["name"].setEditText("新中文角色")
    candidate = copy.deepcopy(window.cfg)
    valid, error = window.collect_advance_settings_from_ui(candidate)
    assert valid, error
    assert candidate["nametag"]["name"] == "新中文角色"

    window.set_gbox_enabled(False)
    assert not window.nametag_calibration_button.isEnabled()
    window.set_gbox_enabled(True)


def test_nametag_profile_value_survives_f1_start(window):
    combo = window.advance_settings_gboxes["nametag"]._field_refs["name"]
    combo.setEditText("classic_cn_player")

    window.button_start_pause.click()

    assert window.controller.start_calls[-1][1]["nametag"]["name"] == "classic_cn_player"
    assert combo.currentText() == "classic_cn_player"
    assert not window.advance_settings_gboxes["nametag"].isEnabled()
    window.button_start_pause.click()
    assert combo.currentText() == "classic_cn_player"


def test_calibration_apply_selects_and_persists_profile(
    window, monkeypatch, tmp_path
):
    import src.ui.nametag_calibration as calibration_module

    class Dialog:
        def __init__(self, **_kwargs):
            self.applied_name = "测试人物甲"

        def exec(self):
            return QDialog.Accepted

    monkeypatch.setattr(calibration_module, "NameTagCalibrationDialog", Dialog)
    window.path_cfg_custom = str(tmp_path / "character.yaml")

    window.open_nametag_calibration()

    assert window.cfg["nametag"]["name"] == "测试人物甲"
    assert window.nametag_profile_combo.currentText() == "测试人物甲"
    assert load_yaml(window.path_cfg_custom)["nametag"]["name"] == "测试人物甲"
    assert window.button_start_pause.isEnabled()
    assert window.button_route_studio.isEnabled()


def test_announcement_page_is_read_only_and_uses_compiled_content(window):
    page = window.tab_announcement
    text = " ".join(label.text() for label in page.findChildren(QLabel))

    assert "内测期间免费" in text
    assert "860498805" in text
    assert "问题反馈、地图路线交流和内测通知" in text
    assert "不是游戏官方产品" in text
    assert "不提供反作弊绕过能力" in text
    assert page.findChildren(QLineEdit) == []
    assert page.findChildren(QPlainTextEdit) == []
    assert "announcement" not in window.cfg_base


def test_announcement_group_copy_writes_only_the_group_number(window):
    QApplication.clipboard().clear()

    window.button_copy_qq_group.click()

    assert QApplication.clipboard().text() == "860498805"
    assert window.announcement_copy_status.text() == "已复制QQ群号：860498805"


def test_applying_or_locking_configuration_cannot_change_announcement(window):
    before = window.announcement_qq_label.text()
    window.cfg["bot"]["mode"] = "patrol"
    window.apply_config_to_ui()
    window.set_gbox_enabled(False)

    assert before == window.announcement_qq_label.text() == "860498805"
    assert window.tab_announcement.isEnabled()
    assert window.button_copy_qq_group.isEnabled()


def test_runtime_config_snapshot_is_scoped_to_current_process(window):
    assert Path(window.runtime_cfg_path).name == f".config_tmp_{os.getpid()}.yaml"


def test_stable_enums_and_unicode_map_values_do_not_depend_on_visible_labels(window):
    assert window.attack_mode.itemText(0) == "方向攻击"
    assert window.attack_mode.itemData(0) == "directional"
    assert window.bot_mode.itemText(0) == "普通"
    assert window.bot_mode.itemData(0) == "normal"
    fixed_platform_index = window.bot_mode.findData("fixed_platform")
    assert fixed_platform_index >= 0
    assert window.bot_mode.itemText(fixed_platform_index) == "固定平台区域攻击"
    continuous_index = window.bot_mode.findData("continuous_attack")
    assert continuous_index >= 0
    assert window.bot_mode.itemText(continuous_index) == "持续定向攻击"
    assert isinstance(window.map_combo, QComboBox)
    assert not window.map_combo.isEditable()
    # The public source snapshot intentionally ships without map assets.
    assert window.map_combo.count() == 0
    window.map_combo.addItem("示例地图", "sample_map")
    assert window.map_combo.count() > 0
    for index in range(window.map_combo.count()):
        value = window.map_combo.itemData(index)
        assert isinstance(value, str) and value


def test_map_selector_refresh_adds_saved_unicode_project_without_restart(
    window, tmp_path
):
    complete = tmp_path / "测试新地图"
    complete.mkdir()
    (complete / "map.png").write_bytes(b"map")
    (complete / "route1.png").write_bytes(b"route")
    incomplete = tmp_path / "只有底图"
    incomplete.mkdir()
    (incomplete / "map.png").write_bytes(b"map")

    count = window.refresh_map_selector(
        preferred_map="测试新地图", minimap_dir=tmp_path
    )

    assert count == 1
    assert window.map_combo.currentData() == "测试新地图"
    assert window.selected_map == "测试新地图"
    assert window.map_combo.findData("只有底图") == -1


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
        "fixed_platform",
        "nametag", "monster_detect", "minimap", "game_window",
        "channel_change", "scheduled_channel_switching", "ui_coords", "profiler",
    }
    assert set(window.advance_settings_gboxes) == expected
    assert "system" not in window.advance_settings_gboxes
    assert "route_recoder" not in window.advance_settings_gboxes


def test_fixed_platform_card_has_default_and_integer_range(window):
    group = window.advance_settings_gboxes["fixed_platform"]
    refs = group._field_refs

    assert group.title() == "固定平台区域攻击"
    assert set(refs) == {"width_px"}
    width_px = refs["width_px"]
    assert isinstance(width_px, QLineEdit)
    assert width_px.text() == "80"
    assert width_px.validator().bottom() == 10
    assert width_px.validator().top() == 1000
    assert "10–1000" in width_px.toolTip()


def test_fixed_platform_mode_and_width_are_saved_on_f1_start(window):
    mode_index = window.bot_mode.findData("fixed_platform")
    width_px = window.advance_settings_gboxes["fixed_platform"]._field_refs[
        "width_px"
    ]
    window.bot_mode.setCurrentIndex(mode_index)
    width_px.setText("160")

    window.button_start_pause.click()

    assert window.button_start_pause.isChecked()
    assert not window.advance_settings_gboxes["fixed_platform"].isEnabled()
    assert len(window.controller.start_calls) == 1
    _, started_cfg = window.controller.start_calls[0]
    assert started_cfg["bot"]["mode"] == "fixed_platform"
    assert started_cfg["fixed_platform"]["width_px"] == 160
    assert window.bot_mode.currentData() == "fixed_platform"
    assert width_px.text() == "160"

    window.button_start_pause.click()

    assert not window.button_start_pause.isChecked()
    assert window.advance_settings_gboxes["fixed_platform"].isEnabled()
    assert window.bot_mode.currentData() == "fixed_platform"
    assert width_px.text() == "160"


def test_continuous_attack_mode_is_saved_and_locked_on_f1_start(window):
    mode_index = window.bot_mode.findData("continuous_attack")
    window.bot_mode.setCurrentIndex(mode_index)
    window.attack_mode.setCurrentIndex(
        window.attack_mode.findData("directional")
    )

    window.button_start_pause.click()

    assert window.button_start_pause.isChecked()
    assert len(window.controller.start_calls) == 1
    _, started_cfg = window.controller.start_calls[0]
    assert started_cfg["bot"]["mode"] == "continuous_attack"
    assert started_cfg["bot"]["attack"] == "directional"
    assert not window.attack_gbox.isEnabled()
    assert window.bot_mode.currentData() == "continuous_attack"

    window.button_start_pause.click()

    assert not window.button_start_pause.isChecked()
    assert window.attack_gbox.isEnabled()
    assert window.bot_mode.currentData() == "continuous_attack"


def test_continuous_attack_rejects_aoe_without_locking_settings(window):
    window.bot_mode.setCurrentIndex(
        window.bot_mode.findData("continuous_attack")
    )
    window.attack_mode.setCurrentIndex(window.attack_mode.findData("aoe_skill"))

    window.button_start_pause.click()

    assert not window.button_start_pause.isChecked()
    assert window.controller.start_calls == []
    assert window.attack_gbox.isEnabled()
    assert "仅支持方向攻击" in window.attack_range_error_label.text()


@pytest.mark.parametrize(
    ("field", "message"),
    [("attack", "攻击键不能为空"), ("jump", "跳跃键不能为空")],
)
def test_continuous_attack_rejects_empty_required_key(window, field, message):
    window.bot_mode.setCurrentIndex(
        window.bot_mode.findData("continuous_attack")
    )
    window.attack_mode.setCurrentIndex(
        window.attack_mode.findData("directional")
    )
    key_widget = window.basic_attack_key if field == "attack" else window.jump_key
    key_widget.set_key("")

    window.button_start_pause.click()

    assert not window.button_start_pause.isChecked()
    assert window.controller.start_calls == []
    assert window.attack_gbox.isEnabled()
    assert message in window.attack_range_error_label.text()


@pytest.mark.parametrize("invalid", ("", "9", "1001", "80.5", "-1"))
def test_invalid_fixed_platform_width_blocks_start_atomically(window, invalid):
    width_px = window.advance_settings_gboxes["fixed_platform"]._field_refs[
        "width_px"
    ]
    original_cfg = copy.deepcopy(window.cfg)
    width_px.blockSignals(True)
    width_px.setText(invalid)
    width_px.blockSignals(False)

    window.button_start_pause.click()

    assert not window.button_start_pause.isChecked()
    assert window.advance_settings_gboxes["fixed_platform"].isEnabled()
    assert window.controller.start_calls == []
    assert window.cfg == original_cfg
    assert width_px.text() == invalid
    assert not window.attack_range_error_label.isHidden()
    if invalid in {"", "80.5"}:
        assert "fixed_platform.width_px" in window.attack_range_error_label.text()
    else:
        assert "10–1000" in window.attack_range_error_label.text()


def test_yolo_card_has_only_current_deployment_controls(window):
    refs = window.advance_settings_gboxes["monster_detect"]._field_refs
    assert set(refs) == {
        "openvino_deployment_root", "yolo_variant",
        "yolo_confidence", "yolo_min_monster_box_side",
        "yolo_player_exclusion_width", "yolo_player_exclusion_height",
        "yolo_max_det",
    }
    variant = refs["yolo_variant"]
    assert isinstance(variant, QComboBox)
    assert variant.currentData() == "int8_v2"
    assert variant.itemText(variant.findData("int8_v2")) == \
        "INT8 Mixed + 检测头 FP（默认）"
    max_det = refs["yolo_max_det"]
    assert isinstance(max_det, QLineEdit)
    assert max_det.text() == "50"
    assert max_det.isReadOnly()
    min_box_side = refs["yolo_min_monster_box_side"]
    assert isinstance(min_box_side, QLineEdit)
    assert min_box_side.text() == "10"
    assert min_box_side.validator().bottom() == 0
    assert min_box_side.validator().top() == 224
    assert "0 表示关闭" in min_box_side.toolTip()
    exclusion_width = refs["yolo_player_exclusion_width"]
    exclusion_height = refs["yolo_player_exclusion_height"]
    assert exclusion_width.text() == "80"
    assert exclusion_width.validator().bottom() == 0
    assert exclusion_width.validator().top() == 1280
    assert exclusion_height.text() == "100"
    assert exclusion_height.validator().bottom() == 0
    assert exclusion_height.validator().top() == 224
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


def test_yolo_min_monster_box_side_round_trip_and_validation(window):
    min_box_side = window.advance_settings_gboxes["monster_detect"]._field_refs[
        "yolo_min_monster_box_side"
    ]
    min_box_side.setText("24")
    assert window.cfg["monster_detect"]["yolo_min_monster_box_side"] == 24
    assert window.validate_attack_detection_ranges()

    for invalid in ("-1", "225", "10.5"):
        min_box_side.blockSignals(True)
        min_box_side.setText(invalid)
        min_box_side.blockSignals(False)
        assert not window.validate_attack_detection_ranges()
        assert "0–224" in window.attack_range_error_label.text()

    min_box_side.setText("10")
    assert window.validate_attack_detection_ranges()


@pytest.mark.parametrize(
    ("field", "invalid", "expected"),
    [
        ("yolo_player_exclusion_width", "", "yolo_player_exclusion_width"),
        ("yolo_player_exclusion_width", "80.5", "yolo_player_exclusion_width"),
        ("yolo_player_exclusion_width", "1281", "0–1280"),
        ("yolo_player_exclusion_height", "-1", "0–224"),
        ("yolo_player_exclusion_height", "225", "0–224"),
    ],
)
def test_yolo_player_exclusion_validation_blocks_start_atomically(
    window, field, invalid, expected
):
    refs = window.advance_settings_gboxes["monster_detect"]._field_refs
    original_cfg = copy.deepcopy(window.cfg)
    refs[field].blockSignals(True)
    refs[field].setText(invalid)
    refs[field].blockSignals(False)

    window.button_start_pause.click()

    assert not window.button_start_pause.isChecked()
    assert window.controller.start_calls == []
    assert window.cfg == original_cfg
    assert refs[field].text() == invalid
    assert expected in window.attack_range_error_label.text()

def test_start_collects_current_yolo_values_and_pause_preserves_them(window):
    refs = window.advance_settings_gboxes["monster_detect"]._field_refs
    deployment_root = "C:/test-assets/deployment"
    refs["openvino_deployment_root"].setText(deployment_root)
    refs["yolo_confidence"].setText("0.55")
    refs["yolo_min_monster_box_side"].setText("17")
    refs["yolo_player_exclusion_width"].setText("96")
    refs["yolo_player_exclusion_height"].setText("112")

    window.button_start_pause.click()

    assert window.button_start_pause.isChecked()
    assert not window.advance_settings_gboxes["monster_detect"].isEnabled()
    assert len(window.controller.start_calls) == 1
    _, started_cfg = window.controller.start_calls[0]
    assert started_cfg["monster_detect"]["openvino_deployment_root"] == deployment_root
    assert started_cfg["monster_detect"]["yolo_variant"] == "int8_v2"
    assert started_cfg["monster_detect"]["yolo_confidence"] == 0.55
    assert started_cfg["monster_detect"]["yolo_min_monster_box_side"] == 17
    assert started_cfg["monster_detect"]["yolo_player_exclusion_width"] == 96
    assert started_cfg["monster_detect"]["yolo_player_exclusion_height"] == 112
    assert refs["yolo_confidence"].text() == "0.55"
    assert refs["yolo_min_monster_box_side"].text() == "17"
    assert refs["yolo_player_exclusion_width"].text() == "96"
    assert refs["yolo_player_exclusion_height"].text() == "112"

    window.button_start_pause.click()

    assert not window.button_start_pause.isChecked()
    assert window.advance_settings_gboxes["monster_detect"].isEnabled()
    assert window.controller.pause_calls == 1
    assert refs["openvino_deployment_root"].text() == deployment_root
    assert refs["yolo_confidence"].text() == "0.55"
    assert refs["yolo_min_monster_box_side"].text() == "17"
    assert refs["yolo_player_exclusion_width"].text() == "96"
    assert refs["yolo_player_exclusion_height"].text() == "112"


def test_collects_every_advanced_widget_type_from_current_ui(window):
    health = window.advance_settings_gboxes["health_monitor"]._field_refs
    monster = window.advance_settings_gboxes["monster_detect"]._field_refs
    minimap = window.advance_settings_gboxes["minimap"]._field_refs

    checkbox_key = next(
        key for key, widget in health.items() if isinstance(widget, QCheckBox)
    )
    health[checkbox_key].setChecked(not health[checkbox_key].isChecked())
    monster["yolo_variant"].setCurrentIndex(
        monster["yolo_variant"].findData("int8_v2")
    )
    monster["yolo_confidence"].setText("0.61")
    monster["yolo_min_monster_box_side"].setText("23")
    monster["openvino_deployment_root"].setText("F:/models/current")
    roi_key = next(key for key, widget in minimap.items() if isinstance(widget, list))
    roi_widgets = minimap[roi_key]
    for index, line in enumerate(roi_widgets):
        line.setText(str(10 + index))

    candidate = copy.deepcopy(window.cfg)
    valid, error = window.collect_advance_settings_from_ui(candidate)

    assert valid, error
    assert isinstance(candidate["health_monitor"][checkbox_key], bool)
    assert candidate["health_monitor"][checkbox_key] == health[checkbox_key].isChecked()
    assert candidate["monster_detect"]["yolo_variant"] == "int8_v2"
    assert candidate["monster_detect"]["yolo_confidence"] == 0.61
    assert candidate["monster_detect"]["yolo_min_monster_box_side"] == 23
    assert candidate["monster_detect"]["openvino_deployment_root"] == "F:/models/current"
    assert candidate["minimap"][roi_key] == list(range(10, 10 + len(roi_widgets)))


def test_health_main_controls_are_the_only_source_for_switches_and_thresholds(window):
    health_refs = window.advance_settings_gboxes["health_monitor"]._field_refs

    assert "auto_hp_enabled" not in health_refs
    assert "auto_mp_enabled" not in health_refs
    assert "add_hp_percent" not in health_refs
    assert "add_mp_percent" not in health_refs
    assert "fps_limit" not in health_refs
    assert "暂停攻击" in health_refs["force_heal"].toolTip()

    window.checkbox_auto_add_hp.setChecked(True)
    window.add_hp_percent.setText("37")
    window.add_hp_key.set_key("f")
    window.checkbox_auto_add_mp.setChecked(False)
    window.add_mp_percent.setText("28")
    window.add_mp_key.set_key("g")

    candidate, error = window.build_candidate_config_from_ui()

    assert not error
    assert candidate["health_monitor"]["auto_hp_enabled"] is True
    assert candidate["health_monitor"]["add_hp_percent"] == 37
    assert candidate["key"]["add_hp"] == "f"
    assert candidate["health_monitor"]["auto_mp_enabled"] is False
    assert candidate["health_monitor"]["add_mp_percent"] == 28
    assert candidate["key"]["add_mp"] == "g"


def test_f1_start_preserves_health_values_and_pause_keeps_them_editable(window):
    window.checkbox_auto_add_hp.setChecked(True)
    window.add_hp_percent.setText("42")
    window.add_hp_key.set_key("f")
    window.checkbox_auto_add_mp.setChecked(False)
    window.add_mp_percent.setText("31")
    window.add_mp_key.set_key("g")

    window.button_start_pause.click()

    assert window.button_start_pause.isChecked()
    _, started_cfg = window.controller.start_calls[-1]
    assert started_cfg["health_monitor"]["auto_hp_enabled"] is True
    assert started_cfg["health_monitor"]["add_hp_percent"] == 42
    assert started_cfg["key"]["add_hp"] == "f"
    assert started_cfg["health_monitor"]["auto_mp_enabled"] is False
    assert started_cfg["health_monitor"]["add_mp_percent"] == 31
    assert started_cfg["key"]["add_mp"] == "g"
    assert "enable" not in started_cfg["health_monitor"]
    assert not window.attack_gbox.isEnabled()
    assert window.add_hp_percent.text() == "42"

    window.button_start_pause.click()

    assert not window.button_start_pause.isChecked()
    assert window.attack_gbox.isEnabled()
    assert window.add_hp_percent.text() == "42"
    assert window.add_hp_key.get_key() == "f"
    assert window.add_mp_percent.text() == "31"
    assert window.add_mp_key.get_key() == "g"


def test_return_home_runtime_stop_unlocks_main_window_without_pause_callback(window):
    window.button_start_pause.setChecked(True)
    window.button_start_pause.setText("⏸ 暂停 (F1)")
    window.button_route_studio.setEnabled(False)
    window.set_gbox_enabled(False)

    window.handle_runtime_stopped("return_home")

    assert not window.button_start_pause.isChecked()
    assert window.button_start_pause.text() == "▶ 开始 (F1)"
    assert window.button_route_studio.isEnabled()
    assert window.attack_gbox.isEnabled()
    assert window.controller.pause_calls == 0
    assert "已执行回城" in window.load_config_error_label.text()
    assert not window.load_config_error_label.isHidden()


def test_ladder_failure_runtime_stop_unlocks_main_window_with_chinese_error(window):
    window.button_start_pause.setChecked(True)
    window.button_start_pause.setText("⏸ 暂停 (F1)")
    window.button_route_studio.setEnabled(False)
    window.set_gbox_enabled(False)

    window.handle_runtime_stopped("route_ladder_failed")

    assert not window.button_start_pause.isChecked()
    assert window.button_route_studio.isEnabled()
    assert window.attack_gbox.isEnabled()
    assert "挂梯连续失败" in window.load_config_error_label.text()


@pytest.mark.parametrize(
    ("enabled_attr", "threshold_attr", "key_attr", "threshold", "key", "message"),
    (
        ("checkbox_auto_add_hp", "add_hp_percent", "add_hp_key", "0", "f", "生命值补给阈值"),
        ("checkbox_auto_add_mp", "add_mp_percent", "add_mp_key", "50", "", "补给按键不能为空"),
    ),
)
def test_invalid_enabled_health_setting_blocks_start_atomically(
    window, enabled_attr, threshold_attr, key_attr, threshold, key, message
):
    original_cfg = copy.deepcopy(window.cfg)
    getattr(window, enabled_attr).setChecked(True)
    getattr(window, threshold_attr).setText(threshold)
    getattr(window, key_attr).set_key(key)

    window.button_start_pause.click()

    assert not window.button_start_pause.isChecked()
    assert window.controller.start_calls == []
    assert window.cfg == original_cfg
    assert getattr(window, threshold_attr).text() == threshold
    assert getattr(window, key_attr).get_key() == key
    assert message in window.attack_range_error_label.text()


def test_negative_return_home_watchdog_blocks_start(window):
    health_refs = window.advance_settings_gboxes["health_monitor"]._field_refs
    window.checkbox_auto_add_hp.setChecked(True)
    health_refs["return_home_if_no_potion"].setChecked(True)
    health_refs["return_home_watch_dog_timeout"].setText("-1")

    window.button_start_pause.click()

    assert not window.button_start_pause.isChecked()
    assert window.controller.start_calls == []
    assert "无药回城等待时间" in window.attack_range_error_label.text()


def test_invalid_advanced_number_blocks_start_without_partial_candidate(window):
    refs = window.advance_settings_gboxes["monster_detect"]._field_refs
    original_cfg = copy.deepcopy(window.cfg)
    confidence = refs["yolo_confidence"]
    min_side = refs["yolo_min_monster_box_side"]
    confidence.blockSignals(True)
    min_side.blockSignals(True)
    confidence.setText("0.73")
    min_side.setText("")
    confidence.blockSignals(False)
    min_side.blockSignals(False)

    window.button_start_pause.click()

    assert not window.button_start_pause.isChecked()
    assert window.advance_settings_gboxes["monster_detect"].isEnabled()
    assert window.controller.start_calls == []
    assert window.cfg == original_cfg
    assert confidence.text() == "0.73"
    assert min_side.text() == ""
    assert not window.attack_range_error_label.isHidden()
    assert "monster_detect.yolo_min_monster_box_side" in \
        window.attack_range_error_label.text()


def test_failed_controller_start_keeps_settings_editable_and_unchanged(window):
    refs = window.advance_settings_gboxes["monster_detect"]._field_refs
    refs["yolo_confidence"].setText("0.66")
    window.controller.start_result = -1

    window.button_start_pause.click()

    assert not window.button_start_pause.isChecked()
    assert window.advance_settings_gboxes["monster_detect"].isEnabled()
    assert refs["yolo_confidence"].text() == "0.66"
    assert window.cfg["monster_detect"]["yolo_confidence"] == 0.66


def test_page_switch_explicitly_collects_advanced_value_without_signal(window):
    window.page_stack.setCurrentIndex(1)
    confidence = window.advance_settings_gboxes["monster_detect"]._field_refs[
        "yolo_confidence"
    ]
    confidence.blockSignals(True)
    confidence.setText("0.57")
    confidence.blockSignals(False)

    window.page_stack.setCurrentIndex(0)

    assert window.cfg["monster_detect"]["yolo_confidence"] == 0.57
    assert confidence.text() == "0.57"


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


def test_route_viz_owns_qimage_after_numpy_source_is_released(window):
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0] = (17, 83, 211)

    window.update_route_map_canvas(frame)
    del frame

    colour = window.route_map_canvas._source_pixmap.toImage().pixelColor(0, 0)
    assert (colour.red(), colour.green(), colour.blue()) == (211, 83, 17)


def test_monitor_pages_show_control_infer_viz_display_and_drop_metrics(window):
    metrics = {
        "control_p95_ms": 72.5,
        "infer_p95_ms": 18.4,
        "viz_p95_ms": 6.2,
        "ui_ms": 3.1,
        "display_fps": 5.0,
        "preview_age_ms": 103.0,
        "dropped": 4,
    }

    window.update_visualization_metrics("game", metrics)

    text = window.game_viz_metrics.text()
    assert "控制 P95 72.5 ms" in text
    assert "YOLO 推理 P95 18.4 ms" in text
    assert "预览 P95 6.2 ms" in text
    assert "显示 5.0 FPS" in text
    assert "覆盖 4" in text
