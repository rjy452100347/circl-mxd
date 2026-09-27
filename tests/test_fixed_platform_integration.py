import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot


def _config(width=80):
    return {
        "bot": {
            "mode": "fixed_platform",
            "route_only": True,
            "route_attack": True,
        },
        "fixed_platform": {"width_px": width},
    }


def test_fixed_platform_runtime_ignores_route_flags_without_mutating_profile():
    profile = _config()
    original = copy.deepcopy(profile)

    runtime = MapleStoryAutoBot.prepare_runtime_config(profile)

    assert profile == original
    assert runtime["bot"]["mode"] == "fixed_platform"
    assert runtime["bot"]["route_only"] is False
    assert runtime["bot"]["route_attack"] is False
    assert runtime["fixed_platform"]["width_px"] == 80


@pytest.mark.parametrize("width", [None, True, 9, 1001, 80.5, "80"])
def test_engine_rejects_invalid_fixed_platform_width(width):
    with pytest.raises(ValueError, match="10–1000"):
        MapleStoryAutoBot.prepare_runtime_config(_config(width))


def test_engine_rejects_unknown_main_program_mode():
    config = _config()
    config["bot"]["mode"] = "unknown"

    with pytest.raises(ValueError, match="不支持的运行模式"):
        MapleStoryAutoBot.prepare_runtime_config(config)


@pytest.mark.parametrize(
    ("mode", "state"),
    [
        ("normal", "hunting"),
        ("aux", "aux"),
        ("patrol", "patrol"),
        ("fixed_platform", "fixed_platform"),
        ("continuous_attack", "continuous_attack"),
    ],
)
def test_start_and_channel_change_share_mode_to_state_mapping(mode, state):
    class FSM:
        selected = None

        def set_init_state(self, value):
            self.selected = value

    bot = object.__new__(MapleStoryAutoBot)
    bot.cfg = {"bot": {"mode": mode}}
    bot.fsm = FSM()

    bot.set_state_for_configured_mode()

    assert bot.fsm.selected == state


def test_existing_window_preview_draws_minimap_anchor_and_platform_bounds():
    bot = object.__new__(MapleStoryAutoBot)
    bot.cfg = {"bot": {"mode": "fixed_platform"}}
    bot.img_frame_debug = np.zeros((120, 240, 3), dtype=np.uint8)
    bot.img_minimap = np.zeros((30, 100, 3), dtype=np.uint8)
    bot.loc_minimap = (20, 10)
    bot.fixed_platform_state = SimpleNamespace(
        get_diagnostics=lambda: {
            "anchor_x": 50,
            "left_x": 10,
            "right_x": 90,
            "direction": "left",
            "localized": True,
            "requested_width": 80,
            "actual_width": 80,
        }
    )

    bot.draw_fixed_platform_debug()

    # BGR: orange bounds and cyan anchor are drawn in the existing game canvas.
    assert tuple(bot.img_frame_debug[10, 30]) == (0, 165, 255)
    assert tuple(bot.img_frame_debug[10, 70]) == (255, 255, 0)
    assert tuple(bot.img_frame_debug[10, 110]) == (0, 165, 255)


def test_fixed_platform_window_info_does_not_require_route_image():
    bot = object.__new__(MapleStoryAutoBot)
    bot.cfg = {"bot": {"mode": "fixed_platform"}}
    bot.img_frame_debug = np.zeros((120, 240, 3), dtype=np.uint8)
    bot.frame = np.zeros((120, 240, 3), dtype=np.uint8)
    bot.img_minimap = np.zeros((30, 100, 3), dtype=np.uint8)
    bot.loc_minimap = (20, 10)
    bot.t_last_frame = time.time() - 0.1
    bot.fsm = SimpleNamespace(state=SimpleNamespace(name="fixed_platform"))
    bot.kb = SimpleNamespace(is_enable=True)
    bot.img_route = None

    bot.update_info_on_img_frame_debug()
