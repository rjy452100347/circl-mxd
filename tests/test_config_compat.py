from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.config_compat import (
    ConfigCompatibilityError,
    REMOVED_SECTIONS,
    find_removed_config_paths,
    validate_custom_config,
)
from src.runtime_policy import RUNTIME_POLICY
from src.utils.common import load_yaml


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
    assert RUNTIME_POLICY.auto_dice_fps == 1
    assert RUNTIME_POLICY.client_language == "cn"
    assert (RUNTIME_POLICY.yolo_input_width, RUNTIME_POLICY.yolo_input_height) == (
        1280,
        224,
    )
    assert RUNTIME_POLICY.yolo_max_det == 50

    with pytest.raises(FrozenInstanceError):
        RUNTIME_POLICY.main_fps = 11

