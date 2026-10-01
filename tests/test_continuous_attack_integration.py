import copy
import importlib
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot

engine_module = importlib.import_module("src.engine.MapleStoryAutoLevelUp")


def config(attack="directional", attack_key="w", jump_key="space"):
    return {
        "bot": {
            "mode": "continuous_attack",
            "attack": attack,
            "route_only": True,
            "route_attack": True,
        },
        "key": {
            "directional_attack": attack_key,
            "jump": jump_key,
        },
    }


def test_runtime_disables_routes_without_mutating_saved_profile():
    profile = config()
    original = copy.deepcopy(profile)

    runtime = MapleStoryAutoBot.prepare_runtime_config(profile)

    assert profile == original
    assert runtime["bot"]["mode"] == "continuous_attack"
    assert runtime["bot"]["route_only"] is False
    assert runtime["bot"]["route_attack"] is False


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (config(attack="aoe_skill"), "仅支持方向攻击"),
        (config(attack_key=""), "攻击键不能为空"),
        (config(jump_key=""), "跳跃键不能为空"),
    ],
)
def test_runtime_rejects_invalid_continuous_attack_input(profile, message):
    with pytest.raises(ValueError, match=message):
        MapleStoryAutoBot.prepare_runtime_config(profile)


def test_mode_maps_to_dedicated_fsm_state():
    bot = object.__new__(MapleStoryAutoBot)
    bot.cfg = {"bot": {"mode": "continuous_attack"}}
    bot.fsm = SimpleNamespace(selected=None)
    bot.fsm.set_init_state = lambda value: setattr(bot.fsm, "selected", value)

    bot.set_state_for_configured_mode()

    assert bot.fsm.selected == "continuous_attack"


def test_locked_direction_frame_skips_name_detection_and_clears_old_boxes():
    bot = object.__new__(MapleStoryAutoBot)
    bot.continuous_attack_state = SimpleNamespace(direction_locked=True)
    bot.monsters = [{"stale": True}]
    bot.current_nametag_valid = True
    bot.last_visual_pause_reason = "old"
    bot.is_first_frame = True
    bot.profiler = SimpleNamespace(marks=[])
    bot.profiler.mark = bot.profiler.marks.append
    bot.fsm = SimpleNamespace(calls=0)
    bot.fsm.do_state_stuff = lambda: setattr(bot.fsm, "calls", bot.fsm.calls + 1)
    bot.get_player_location_by_nametag = lambda: pytest.fail(
        "locked mode must not run name localization"
    )

    result = bot._run_continuous_attack_frame()

    assert result == 0
    assert bot.monsters == []
    assert bot.current_nametag_valid is False
    assert bot.fsm.calls == 1
    assert bot.last_visual_pause_reason == ""
    assert bot.is_first_frame is False


def test_run_once_enters_lightweight_branch_before_minimap(monkeypatch):
    bot = object.__new__(MapleStoryAutoBot)
    bot.cfg = {"bot": {"mode": "continuous_attack", "route_only": False}}
    bot.profiler = SimpleNamespace(start=lambda: None, mark=lambda _name: None)
    bot.current_minimap_roi_valid = True
    bot.current_minimap_player_valid = True
    bot.current_nametag_valid = True
    bot.is_need_show_debug_window = False
    bot._claim_visualization_mode = lambda: None
    bot.control_frame_sequence = 0
    bot.get_img_frame = lambda: np.zeros((700, 1280, 3), dtype=np.uint8)
    bot.health_monitor = None
    bot._run_continuous_attack_frame = lambda: 17
    monkeypatch.setattr(
        engine_module,
        "get_minimap_loc_size",
        lambda *_args, **_kwargs: pytest.fail("minimap must not be accessed"),
    )

    assert bot.run_once() == 17
    assert bot.control_frame_sequence == 1


def test_continuous_window_info_does_not_touch_minimap_or_route():
    bot = object.__new__(MapleStoryAutoBot)
    bot.cfg = {"bot": {"mode": "continuous_attack"}}
    bot.img_frame_debug = np.zeros((700, 1280, 3), dtype=np.uint8)
    bot.frame = np.zeros((700, 1280, 3), dtype=np.uint8)
    bot.t_last_frame = time.time() - 0.1
    bot.fsm = SimpleNamespace(state=SimpleNamespace(name="continuous_attack"))
    bot.kb = SimpleNamespace(is_enable=True)
    bot.health_monitor = None
    bot.img_minimap = None
    bot.img_route = None

    bot.update_info_on_img_frame_debug()

    assert bot.img_frame_debug.shape == (700, 1280, 3)


def test_lock_frame_can_draw_detection_but_later_frames_cannot():
    bot = object.__new__(MapleStoryAutoBot)
    bot.cfg = {"bot": {"mode": "continuous_attack"}}
    bot.img_frame_debug = np.zeros((120, 240, 3), dtype=np.uint8)
    bot.img_frame = bot.img_frame_debug
    bot.is_ui = True
    bot.loc_player = (120, 60)
    bot.current_nametag_valid = True
    bot.continuous_attack_state = SimpleNamespace(direction_locked=True)
    bot.monsters = [{"position": (80, 40), "size": (10, 10)}]
    bot.get_monster_search_range = lambda: (0, 0, 240, 120)
    calls = []
    bot.draw_monster_detection_debug = lambda *_args, **_kwargs: calls.append(
        "detection"
    )
    bot.draw_combat_ranges_debug = lambda *_args, **_kwargs: calls.append(
        "ranges"
    )
    bot.draw_player_exclusion_debug = lambda *_args, **_kwargs: None
    bot.draw_fixed_platform_debug = lambda *_args, **_kwargs: None
    bot.draw_continuous_attack_debug = lambda *_args, **_kwargs: None

    assert bot.get_frame_debug_for_viz() is not None
    assert calls == ["detection", "ranges"]

    calls.clear()
    bot.img_frame_debug = np.zeros((120, 240, 3), dtype=np.uint8)
    bot.current_nametag_valid = False
    bot.monsters = []
    assert bot.get_frame_debug_for_viz() is not None
    assert calls == []
