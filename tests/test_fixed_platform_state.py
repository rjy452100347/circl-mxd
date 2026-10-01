import numpy as np
import pytest

import src.states.fixed_platform as fixed_platform_module
from src.states.fixed_platform import FixedPlatformState


class _Keyboard:
    def __init__(self):
        self.commands = []
        self.release_calls = 0

    def set_command(self, command):
        self.commands.append(command)

    def release_all_key(self):
        self.release_calls += 1


class _Bot:
    def __init__(self):
        self.cfg = {
            "bot": {"mode": "fixed_platform", "attack": "directional"},
            "fixed_platform": {"width_px": 80},
            "directional_attack": {
                "range_x": 100,
                "range_y": 60,
                "cooldown": 0.0,
            },
            "aoe_skill": {"range_x": 160, "range_y": 80, "cooldown": 0.0},
        }
        self.current_nametag_valid = True
        self.current_minimap_player_valid = True
        self.loc_player_minimap = (100, 20)
        self.loc_player = (200, 100)
        self.img_minimap = np.zeros((50, 240, 3), dtype=np.uint8)
        self.monsters = []
        self.is_disable_control = False
        self.t_last_attack = -100.0
        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        self.debug_attack_direction = "none"
        self.last_visual_pause_reason = ""
        self.kb = _Keyboard()
        self.observation_calls = 0
        self.stuck = False
        self.stuck_calls = 0

    def pause_for_visual(self, reason):
        self.last_visual_pause_reason = reason
        self.cmd_move_x, self.cmd_move_y, self.cmd_action = "stop", "stop", "none"
        self.kb.set_command("stop stop none")
        self.kb.release_all_key()

    def is_player_stuck(self):
        self.stuck_calls += 1
        return self.stuck

    def update_monster_observations(self):
        self.observation_calls += 1

    def get_attack_range(self, is_left=True):
        if self.cfg["bot"]["attack"] == "aoe_skill":
            half_x = self.cfg["aoe_skill"]["range_x"] // 2
            half_y = self.cfg["aoe_skill"]["range_y"] // 2
            return (
                self.loc_player[0] - half_x,
                self.loc_player[1] - half_y,
                self.loc_player[0] + half_x,
                self.loc_player[1] + half_y,
            )
        half_y = self.cfg["directional_attack"]["range_y"] // 2
        if is_left:
            x0, x1 = self.loc_player[0] - 100, self.loc_player[0]
        else:
            x0, x1 = self.loc_player[0], self.loc_player[0] + 100
        return x0, self.loc_player[1] - half_y, x1, self.loc_player[1] + half_y

    def get_nearest_monster(self, is_left=True):
        x0, y0, x1, y1 = self.get_attack_range(is_left=is_left)
        candidates = []
        for monster in self.monsters:
            mx, my = monster["position"]
            width, height = monster["size"]
            if min(x1, mx + width) <= max(x0, mx):
                continue
            if min(y1, my + height) <= max(y0, my):
                continue
            distance = abs(mx + width // 2 - self.loc_player[0])
            candidates.append((distance, monster))
        return min(candidates, default=(None, None))[-1]


def _state(bot=None):
    bot = bot or _Bot()
    state = FixedPlatformState("fixed_platform", bot)
    state.on_enter()
    return state, bot


def _monster(x, y=90, width=20, height=20):
    return {"position": (x, y), "size": (width, height), "score": 0.1}


def test_on_enter_resets_anchor_direction_localization_and_attack_gap():
    state, _ = _state()
    state.anchor_x = 10
    state.left_x = 1
    state.right_x = 20
    state.direction = "right"
    state.localized = True
    state._neutral_attack_required = True

    state.on_enter()

    assert state.get_diagnostics() == {
        "anchor_x": None,
        "left_x": None,
        "right_x": None,
        "direction": "left",
        "localized": False,
        "requested_width": 80,
        "actual_width": None,
    }
    assert not state._neutral_attack_required


@pytest.mark.parametrize(
    ("nametag_valid", "minimap_valid"),
    [(False, True), (True, False), (False, False)],
)
def test_invalid_localization_stops_and_skips_detection(
    nametag_valid, minimap_valid
):
    state, bot = _state()
    bot.current_nametag_valid = nametag_valid
    bot.current_minimap_player_valid = minimap_valid

    state.on_frame()

    assert bot.last_visual_pause_reason == "fixed_platform_localization_invalid"
    assert (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action) == (
        "stop", "stop", "none"
    )
    assert bot.kb.commands == ["stop stop none"]
    assert bot.kb.release_calls == 1
    assert bot.observation_calls == 0
    assert bot.monsters == []
    assert bot.debug_attack_direction == "none"
    assert not state.get_diagnostics()["localized"]


def test_first_valid_frame_captures_centered_bounds_and_detects_once():
    state, bot = _state()

    state.on_frame()

    assert state.get_diagnostics() == {
        "anchor_x": 100,
        "left_x": 60,
        "right_x": 140,
        "direction": "left",
        "localized": True,
        "requested_width": 80,
        "actual_width": 80,
    }
    assert bot.observation_calls == 1
    assert bot.kb.commands == ["left none none"]


@pytest.mark.parametrize(
    ("anchor", "expected_left", "expected_right"),
    [(5, 0, 45), (235, 195, 239)],
)
def test_bounds_are_clipped_without_shifting_the_anchor_interval(
    anchor, expected_left, expected_right
):
    state, bot = _state()
    bot.loc_player_minimap = (anchor, 20)

    state.on_frame()

    assert state.anchor_x == anchor
    assert state.left_x == expected_left
    assert state.right_x == expected_right


def test_turns_at_margin_and_when_outside_bounds():
    state, bot = _state()
    state.on_frame()

    for player_x, expected in [(62, "right"), (150, "left"), (50, "right"), (138, "left")]:
        bot.loc_player_minimap = (player_x, 20)
        state.on_frame()
        assert state.direction == expected
        assert bot.cmd_move_x == expected

    assert bot.observation_calls == 5


def test_stuck_only_reverses_and_boundaries_have_priority():
    state, bot = _state()
    state.on_frame()
    bot.stuck = True

    bot.loc_player_minimap = (100, 20)
    state.on_frame()
    assert state.direction == "right"
    assert (bot.cmd_move_y, bot.cmd_action) == ("none", "none")

    state.direction = "left"
    bot.loc_player_minimap = (61, 20)
    state.on_frame()
    assert state.direction == "right"

    state.direction = "right"
    bot.loc_player_minimap = (139, 20)
    state.on_frame()
    assert state.direction == "left"


def test_no_monster_never_periodically_or_blindly_attacks(monkeypatch):
    state, bot = _state()
    monkeypatch.setattr(fixed_platform_module.time, "time", lambda: 1000.0)
    bot.t_last_attack = -1000.0

    state.on_frame()
    state.on_frame()

    assert bot.observation_calls == 2
    assert bot.kb.commands == ["left none none", "left none none"]
    assert bot.t_last_attack == -1000.0


def test_directional_attack_is_single_frame_with_neutral_gap(monkeypatch):
    state, bot = _state()
    now = [10.0]
    monkeypatch.setattr(fixed_platform_module.time, "time", lambda: now[0])
    bot.monsters = [_monster(120)]

    state.on_frame()
    assert (bot.cmd_move_x, bot.cmd_action) == ("left", "attack_left")
    assert bot.t_last_attack == 10.0

    now[0] = 11.0
    state.on_frame()
    assert (bot.cmd_move_x, bot.cmd_action) == ("left", "none")

    now[0] = 12.0
    state.on_frame()
    assert (bot.cmd_move_x, bot.cmd_action) == ("left", "attack_left")
    assert bot.observation_calls == 3


def test_attack_toward_monster_does_not_take_patrol_movement(monkeypatch):
    state, bot = _state()
    monkeypatch.setattr(fixed_platform_module.time, "time", lambda: 10.0)
    bot.monsters = [_monster(260)]

    state.on_frame()

    assert state.direction == "left"
    assert (bot.cmd_move_x, bot.cmd_action) == ("left", "attack_right")
    assert bot.kb.commands[-1] == "left none attack_right"


def test_directional_monster_outside_attack_boxes_does_not_trigger(monkeypatch):
    state, bot = _state()
    monkeypatch.setattr(fixed_platform_module.time, "time", lambda: 10.0)
    bot.monsters = [_monster(400)]

    state.on_frame()

    assert bot.cmd_action == "none"
    assert bot.t_last_attack == -100.0


def test_directional_box_overlap_does_not_attack_toward_wrong_center_side(
    monkeypatch,
):
    state, bot = _state()
    monkeypatch.setattr(fixed_platform_module.time, "time", lambda: 10.0)
    # This box overlaps the left attack half, but its center is to the right.
    bot.monsters = [_monster(195, width=20)]

    state.on_frame()

    assert bot.cmd_action == "attack_right"


@pytest.mark.parametrize(
    ("monster", "expected_action"),
    [(_monster(150), "attack"), (_monster(400), "none")],
)
def test_aoe_requires_positive_intersection_with_existing_box(
    monkeypatch, monster, expected_action
):
    state, bot = _state()
    monkeypatch.setattr(fixed_platform_module.time, "time", lambda: 10.0)
    bot.cfg["bot"]["attack"] = "aoe_skill"
    bot.monsters = [monster]

    state.on_frame()

    assert bot.cmd_action == expected_action


def test_cooldown_prevents_attack_even_with_visible_target(monkeypatch):
    state, bot = _state()
    monkeypatch.setattr(fixed_platform_module.time, "time", lambda: 10.0)
    bot.cfg["directional_attack"]["cooldown"] = 5.0
    bot.t_last_attack = 8.0
    bot.monsters = [_monster(120)]

    state.on_frame()

    assert bot.cmd_action == "none"
    assert bot.t_last_attack == 8.0


def test_disable_control_still_detects_but_never_attacks(monkeypatch):
    state, bot = _state()
    monkeypatch.setattr(fixed_platform_module.time, "time", lambda: 10.0)
    bot.is_disable_control = True
    bot.monsters = [_monster(120)]

    state.on_frame()

    assert bot.observation_calls == 1
    assert bot.cmd_action == "none"
    assert bot.t_last_attack == -100.0
    assert bot.kb.commands == ["left none none"]


def test_localization_recovery_keeps_original_bounds():
    state, bot = _state()
    state.on_frame()
    original_bounds = (state.anchor_x, state.left_x, state.right_x)

    bot.current_nametag_valid = False
    bot.loc_player_minimap = (180, 20)
    state.on_frame()
    bot.current_nametag_valid = True
    state.on_frame()

    assert (state.anchor_x, state.left_x, state.right_x) == original_bounds
    assert state.localized
    assert bot.last_visual_pause_reason == ""
