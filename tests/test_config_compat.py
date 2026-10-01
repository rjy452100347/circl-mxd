import copy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.config_compat import (
    ConfigCompatibilityError,
    REMOVED_SECTIONS,
    find_removed_config_paths,
    migrate_health_monitor_config,
    validate_custom_config,
)
from src.utils.common import retain_explicit_config_values


def test_profile_save_retains_explicit_value_equal_to_default():
    diff = {"key": {"teleport": ""}}
    current = {"key": {"jump": "space", "teleport": ""}}
    explicit = {"key": {"jump": "space", "teleport": ""}}

    saved = retain_explicit_config_values(diff, current, explicit)

    assert saved["key"] == {"jump": "space", "teleport": ""}


def test_profile_save_does_not_restore_removed_legacy_field():
    current = {"health_monitor": {"auto_hp_enabled": True}}
    explicit = {"health_monitor": {"enable": False}}

    saved = retain_explicit_config_values({}, current, explicit)

    assert saved == {}
from src.runtime_policy import RUNTIME_POLICY
from src.utils.common import load_yaml, override_cfg


@pytest.mark.parametrize("section", sorted(REMOVED_SECTIONS))
def test_each_removed_top_level_section_is_rejected(section):
    with pytest.raises(ConfigCompatibilityError) as error:
        validate_custom_config({section: {}})
    assert section in error.value.paths
    assert "配置包含已删除的项目" in str(error.value)


def test_all_nested_legacy_conflicts_are_reported_together():
    config = {
        "nametag": {"enable": True},
        "monster_detect": {
            "mode": "contour_only",
            "search_range_x": 600,
            "with_enemy_hp_bar": True,
        },
        "key": {"party": "p"},
    }

    paths = find_removed_config_paths(config)

    assert paths == [
        "key.party",
        "monster_detect.mode",
        "monster_detect.search_range_x",
        "monster_detect.with_enemy_hp_bar",
        "nametag.enable",
    ]


@pytest.mark.parametrize(
    "name",
    (
        "config_default.yaml",
        "config_classic_cn.yaml",
        "config_cleric.yaml",
        "config_macOS.yaml",
    ),
)
def test_shipped_profiles_contain_no_removed_settings(name):
    config = load_yaml(str(Path("config") / name))
    assert validate_custom_config(config) is config


def test_legacy_profile_inherits_fixed_platform_defaults():
    base = load_yaml(str(Path("config") / "config_default.yaml"))
    legacy_profile = {"bot": {"mode": "normal"}}

    merged = override_cfg(copy.deepcopy(base), legacy_profile)

    assert merged["fixed_platform"] == {"width_px": 80}


def test_legacy_health_global_switch_migrates_without_remaining_in_saved_config():
    legacy = {
        "health_monitor": {
            "enable": False,
            "add_hp_percent": 60,
            "add_mp_percent": 40,
        }
    }

    migrated = migrate_health_monitor_config(legacy)

    assert migrated["health_monitor"]["auto_hp_enabled"] is False
    assert migrated["health_monitor"]["auto_mp_enabled"] is False
    assert "enable" not in migrated["health_monitor"]


def test_partial_legacy_health_profile_only_migrates_fields_it_owns():
    base = load_yaml(str(Path("config") / "config_default.yaml"))
    partial = {"health_monitor": {"add_hp_percent": 70, "force_heal": True}}

    migrated = migrate_health_monitor_config(partial)
    merged = override_cfg(copy.deepcopy(base), migrated)

    assert migrated["health_monitor"]["auto_hp_enabled"] is True
    assert "auto_mp_enabled" not in migrated["health_monitor"]
    assert merged["health_monitor"]["auto_mp_enabled"] is True


def test_new_health_switches_take_precedence_over_legacy_switch():
    config = {
        "health_monitor": {
            "enable": False,
            "auto_hp_enabled": True,
            "add_hp_percent": 55,
            "add_mp_percent": 0,
        }
    }

    migrate_health_monitor_config(config)

    assert config["health_monitor"]["auto_hp_enabled"] is True
    assert config["health_monitor"]["auto_mp_enabled"] is False
    assert "enable" not in config["health_monitor"]


def test_legacy_disabled_override_stays_disabled_when_merged_for_cli():
    base = load_yaml(str(Path("config") / "config_default.yaml"))
    custom = {"health_monitor": {"enable": False}}

    merged = override_cfg(
        copy.deepcopy(base), migrate_health_monitor_config(custom)
    )

    assert merged["health_monitor"]["auto_hp_enabled"] is False
    assert merged["health_monitor"]["auto_mp_enabled"] is False


def test_malformed_string_legacy_switch_fails_closed():
    config = {"health_monitor": {"enable": "false", "add_hp_percent": 50}}

    migrate_health_monitor_config(config)

    assert config["health_monitor"]["auto_hp_enabled"] is False
    assert config["health_monitor"]["auto_mp_enabled"] is False


def test_malformed_new_health_switch_is_rejected():
    config = {"health_monitor": {"auto_hp_enabled": "false"}}

    with pytest.raises(ValueError, match="必须是布尔值"):
        migrate_health_monitor_config(config)


def test_runtime_policy_is_fixed_and_matches_product_contract():
    assert (
        RUNTIME_POLICY.main_fps,
        RUNTIME_POLICY.capture_fps,
        RUNTIME_POLICY.keyboard_fps,
    ) == (10, 15, 30)
    assert RUNTIME_POLICY.command_timeout_seconds == 0.5
    assert RUNTIME_POLICY.frame_stale_timeout_seconds == 0.75
    assert RUNTIME_POLICY.hotkey_debounce_seconds == 1.0
    assert RUNTIME_POLICY.route_recorder_fps == 10
    assert RUNTIME_POLICY.preview_fps == 5
    assert RUNTIME_POLICY.preview_max_width == 1280
    assert RUNTIME_POLICY.auto_dice_fps == 1
    assert RUNTIME_POLICY.client_language == "cn"
    assert (RUNTIME_POLICY.yolo_input_width, RUNTIME_POLICY.yolo_input_height) == (
        1280,
        224,
    )
    assert RUNTIME_POLICY.yolo_max_det == 50

    with pytest.raises(FrozenInstanceError):
        RUNTIME_POLICY.main_fps = 11

