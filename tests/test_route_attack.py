import time
from collections import deque
from types import SimpleNamespace

import numpy as np

from src.engine.MapleStoryAutoLevelUp import (
    MapleStoryAutoBot,
)
from src.engine.RouteCombatArbiter import RouteIntent


def _route_attack_bot(command="right none none"):
    bot = object.__new__(MapleStoryAutoBot)
    bot.cfg = {
        "bot": {
            "mode": "normal",
            "attack": "directional",
            "route_only": True,
            "route_attack": True,
        },
        "directional_attack": {
            "range_x": 120,
            "range_y": 60,
            "cooldown": 0.0,
        },
        "aoe_skill": {"range_x": 120, "range_y": 60, "cooldown": 0.0},
        "key": {"directional_attack": "q"},
        "route": {"traversal_commit_seconds": 0.45},
        "monster_detect": {
            "yolo_confidence": 0.60,
            "yolo_max_det": 50,
        },
        "combat_tracking": {
            "pursuit_vertical_tolerance": 70,
            "target_lost_grace_seconds": 0.3,
        },
    }
    bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action = command.split()
    bot.loc_player = (150, 100)
    bot.img_frame = np.zeros((220, 320, 3), dtype=np.uint8)
    bot.img_frame_debug = None
    bot.is_disable_control = False
    bot.t_last_attack = time.time() - 10
    bot.monsters_info = {
        "test": [(np.zeros((10, 20, 3), dtype=np.uint8), None)]
    }
    bot.monsters = []
    bot.monster_detection_times = deque(maxlen=300)
    bot.monster_frame_pipeline_times = deque(maxlen=300)
    bot._test_detections = []
    bot._detection_calls = 0

    class Detector:
        last_timing = SimpleNamespace(
            crop_ms=0.1, infer_ms=5.0, convert_ms=0.1,
            total_ms=5.2, monsters=0, players=0,
        )
        last_players = []
        last_player_foot = None

        def detect(self, _frame, _player):
            bot._detection_calls += 1
            result = list(bot._test_detections)
            self.last_timing.monsters = len(result)
            return result

    bot.monster_detector = Detector()
    bot.last_route_intent = RouteIntent.from_command(command)
    bot.is_on_ladder = False
    bot.route_navigator = None
    return bot


def _seed_target(bot, target, last_seen=0.0, last_state="visible_pursuit"):
    bot.combat_target = target
    bot.combat_target_last_seen_at = last_seen
    bot.combat_target_last_direction = bot._monster_direction(target)
    bot.combat_target_last_state = last_state


def test_first_visible_target_behind_immediately_stops_and_attacks_left():
    bot = _route_attack_bot("right none none")
    detected = {
        "name": "test",
        "position": (100, 90),
        "size": (20, 20),
        "score": 0.1,
    }
    bot._test_detections = [detected]

    bot.update_cmd_by_mob_detection(bot.last_route_intent)

    assert bot.cmd_move_x == "stop"
    assert bot.cmd_move_y == "none"
    assert bot.cmd_action == "attack_left"


def test_route_attack_prefers_route_direction_when_both_sides_are_close():
    bot = _route_attack_bot("right none none")
    bot.monsters = [
        {"name": "test", "position": (90, 90), "size": (20, 20), "score": 0.1},
        {"name": "test", "position": (190, 90), "size": (20, 20), "score": 0.1},
    ]

    left = bot.get_nearest_monster(is_left=True)
    right = bot.get_nearest_monster(is_left=False)

    assert bot.get_attack_direction(left, right) == "right"


def test_nearest_same_level_target_is_reselected_every_frame():
    bot = _route_attack_bot("right none none")
    left = {
        "name": "monster", "position": (0, 90), "size": (20, 20),
        "confidence": 0.8, "score": 0.2,
    }
    right = {
        "name": "monster", "position": (280, 90), "size": (20, 20),
        "confidence": 0.9, "score": 0.1,
    }
    bot.monsters = [left, right]

    first = bot.build_route_combat_intent(bot.last_route_intent, now=0.0)
    assert first.target is right
    assert first.state == "visible_pursuit"
    assert first.direction == "right"

    right["position"] = (400, 90)
    second = bot.build_route_combat_intent(bot.last_route_intent, now=0.1)
    assert second.target is left
    assert second.state == "visible_pursuit"
    assert second.direction == "left"


def test_attack_range_uses_any_positive_box_intersection():
    bot = _route_attack_bot()
    edge_touch_only = {
        "name": "monster", "position": (10, 90), "size": (20, 20),
        "score": 0.1,
    }
    one_pixel_overlap = {
        "name": "monster", "position": (11, 90), "size": (20, 20),
        "score": 0.1,
    }

    assert not bot.is_monster_in_attack_range(edge_touch_only)
    assert bot.is_monster_in_attack_range(one_pixel_overlap)


def test_yolo_observation_uses_fixed_strip_backend_without_legacy_fusion():
    bot = _route_attack_bot()
    detected = {
        "name": "monster",
        "position": (220, 80),
        "size": (30, 25),
        "confidence": 0.9,
        "score": 0.1,
        "class_id": 0,
        "detector": "openvino_yolo",
    }

    class Detector:
        last_timing = SimpleNamespace(
            crop_ms=0.1, infer_ms=5.0, convert_ms=0.1,
            total_ms=5.2, monsters=1, players=0,
        )

        def detect(self, frame, player):
            assert frame is bot.img_frame
            assert player == bot.loc_player
            return [detected]

    bot.monster_detector = Detector()
    bot.monster_detection_times = deque(maxlen=300)
    assert bot.update_monster_observations()
    assert bot.monsters == [detected]
    assert list(bot.monster_detection_times) == [5.2]


def test_yolo_search_range_matches_visible_model_strip():
    bot = _route_attack_bot()
    bot.img_frame = np.zeros((500, 1600, 3), dtype=np.uint8)
    bot.loc_player = (800, 250)

    assert bot.get_monster_search_range() == (160, 138, 1440, 362)


def test_viz_rendering_follows_every_control_loop_frame():
    bot = _route_attack_bot()
    bot.is_need_show_debug_window = True
    bot.t_last_viz_frame = 1.0

    assert bot.is_viz_frame_due(now=1.01)
    bot.is_need_show_debug_window = False
    assert not bot.is_viz_frame_due(now=2.0)


def test_viz_mode_switch_forces_one_fresh_frame():
    bot = _route_attack_bot()
    bot.is_need_show_debug_window = False
    bot.is_show_debug_window = False
    bot.t_last_viz_frame = 99.0

    bot.enable_viz("route")

    assert bot.viz_mode == "route"
    assert bot.t_last_viz_frame == 0.0
    assert bot.is_need_show_debug_window is True


def test_route_attack_detects_but_does_not_interrupt_mount_or_vertical_action():
    bot = _route_attack_bot("none up mount")
    bot._test_detections = []

    bot.update_cmd_by_mob_detection()

    assert bot._detection_calls == 1
    assert (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action) == (
        "none", "up", "mount"
    )


def test_disable_control_still_detects_on_vertical_route_without_attacking():
    bot = _route_attack_bot("none up mount")
    bot.is_disable_control = True
    detected = {
        "name": "test",
        "position": (160, 90),
        "size": (20, 20),
        "score": 0.1,
    }
    bot._test_detections = [detected]

    bot.update_cmd_by_mob_detection()

    assert bot._detection_calls == 1
    assert bot.monsters == [detected]
    assert bot.debug_attack_direction == "right"
    assert (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action) == (
        "none", "up", "mount"
    )


def test_disable_control_runs_yolo_each_frame_without_changing_route_command():
    bot = _route_attack_bot("right none none")
    bot.is_disable_control = True
    cached = {
        "name": "test",
        "position": (160, 90),
        "size": (20, 20),
        "score": 0.1,
    }
    bot._test_detections = [cached]

    bot.update_cmd_by_mob_detection()
    bot.update_cmd_by_mob_detection()

    assert bot._detection_calls == 2
    assert bot.monsters == [cached]
    assert (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action) == (
        "right", "none", "none"
    )


def test_first_same_level_target_outside_attack_range_owns_movement():
    bot = _route_attack_bot("right none none")
    target = {
        "name": "test",
        "position": (0, 90),
        "size": (20, 20),
        "score": 0.1,
    }
    bot._test_detections = [target]

    intent = bot.update_cmd_by_mob_detection(bot.last_route_intent)

    assert intent.state == "visible_pursuit"
    assert (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action) == (
        "left", "none", "none"
    )


def test_critical_route_keeps_control_while_nearest_target_is_observed():
    bot = _route_attack_bot("none up mount")
    target = {
        "name": "test",
        "position": (0, 90),
        "size": (20, 20),
        "score": 0.1,
    }
    bot._test_detections = [target]
    critical = RouteIntent("none", "up", "mount", "mount_pending")

    intent = bot.update_cmd_by_mob_detection(critical)

    assert intent.state == "traversal_wait"
    assert bot.combat_target is target
    assert (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action) == (
        "none", "up", "mount"
    )

    cruise = RouteIntent("right", "none", "none")
    intent = bot.update_cmd_by_mob_detection(cruise)

    assert intent.state == "visible_pursuit"
    assert (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action) == (
        "left", "none", "none"
    )


def test_different_platform_target_never_takes_horizontal_control():
    bot = _route_attack_bot("right none none")
    target = {
        "name": "test",
        "position": (285, 190),
        "size": (20, 20),
        "score": 0.1,
    }
    bot._test_detections = [target]

    bot.update_cmd_by_mob_detection(bot.last_route_intent)
    intent = bot.update_cmd_by_mob_detection(bot.last_route_intent)

    assert intent.state == "observe"
    assert (bot.cmd_move_x, bot.cmd_action) == ("right", "none")


def test_short_pursuit_dropout_moves_without_blind_attack_then_expires():
    bot = _route_attack_bot("right none none")
    target = {
        "name": "test",
        "position": (0, 90),
        "size": (20, 20),
        "score": 0.1,
    }
    bot.monsters = [target]
    bot.apply_route_combat_intent(bot.last_route_intent, now=0.1)
    assert bot.cmd_move_x == "left"

    bot.monsters = []
    bot.t_last_attack = -10.0
    grace = bot.apply_route_combat_intent(bot.last_route_intent, now=0.3)

    assert grace.state == "lost_grace_pursuit"
    assert (bot.cmd_move_x, bot.cmd_action) == ("left", "none")

    expired = bot.apply_route_combat_intent(bot.last_route_intent, now=0.41)

    assert expired.state == "observe"
    assert bot.combat_target is None
    assert (bot.cmd_move_x, bot.cmd_action) == ("right", "none")


def test_short_attack_dropout_stays_stopped_and_respects_attack_cooldown():
    bot = _route_attack_bot("right none none")
    target = {
        "name": "test",
        "position": (100, 90),
        "size": (20, 20),
        "score": 0.1,
    }
    bot.monsters = [target]
    visible = bot.apply_route_combat_intent(bot.last_route_intent, now=0.0)
    assert visible.state == "visible_attack"

    bot.monsters = []
    bot.t_last_attack = -10.0
    grace = bot.apply_route_combat_intent(bot.last_route_intent, now=0.2)

    assert grace.state == "lost_grace_attack"
    assert (bot.cmd_move_x, bot.cmd_action) == ("stop", "attack_left")


def test_visible_target_never_times_out_when_distance_does_not_improve():
    bot = _route_attack_bot("right none none")
    target = {
        "name": "test",
        "position": (0, 90),
        "size": (20, 20),
        "score": 0.1,
    }
    bot.monsters = [target]

    first = bot.build_route_combat_intent(bot.last_route_intent, now=0.0)
    much_later = bot.build_route_combat_intent(bot.last_route_intent, now=100.0)

    assert first.state == "visible_pursuit"
    assert much_later.state == "visible_pursuit"
    assert much_later.target is target


def test_first_detection_owns_control_and_short_dropout_preserves_pursuit():
    bot = _route_attack_bot("right none none")
    target = {
        "name": "test",
        "position": (0, 90),
        "size": (20, 20),
        "score": 0.1,
    }

    bot.monsters = [target]
    first = bot.build_route_combat_intent(bot.last_route_intent, now=0.0)
    bot.monsters = []
    missed = bot.build_route_combat_intent(bot.last_route_intent, now=0.2)
    bot.monsters = [target]
    visible_again = bot.build_route_combat_intent(bot.last_route_intent, now=0.4)

    assert first.state == "visible_pursuit"
    assert missed.state == "lost_grace_pursuit"
    assert visible_again.state == "visible_pursuit"


def test_visible_combat_skips_route_decision_and_cannot_consume_jump_event():
    bot = _route_attack_bot("right none none")
    target = {
        "name": "test",
        "position": (0, 90),
        "size": (20, 20),
        "score": 0.1,
    }
    bot.update_monster_observations = lambda: setattr(bot, "monsters", [target])

    def route_must_not_run():
        raise AssertionError("RouteNavigator was sampled during combat ownership")

    bot.update_cmd_by_route = route_must_not_run

    route_intent, combat_intent = bot.update_route_only_commands(now=0.1)

    assert route_intent is None
    assert combat_intent.state == "visible_pursuit"
    assert (bot.cmd_move_x, bot.cmd_action) == ("left", "none")


def test_ladder_keeps_route_control_until_visual_state_confirms_exit():
    bot = _route_attack_bot("none up mount")
    target = {
        "name": "test",
        "position": (0, 90),
        "size": (20, 20),
        "score": 0.1,
    }
    bot.update_monster_observations = lambda: setattr(bot, "monsters", [target])
    bot.is_on_ladder = True
    bot.route_commit_until = 0.0
    calls = []

    def committed_route():
        calls.append("route")
        intent = RouteIntent("none", "up", "mount", "mount_pending")
        bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action = (
            intent.move_x, intent.move_y, intent.action
        )
        bot.last_route_intent = intent
        return intent

    bot.update_cmd_by_route = committed_route

    route_intent, combat_intent = bot.update_route_only_commands(now=5.0)

    assert calls == ["route"]
    assert route_intent.action == "mount"
    assert combat_intent.state == "traversal_wait"
    assert (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action) == (
        "none", "up", "mount"
    )

    bot.is_on_ladder = False
    route_intent, combat_intent = bot.update_route_only_commands(now=5.5)

    assert calls == ["route"]
    assert route_intent is None
    assert combat_intent.state == "visible_pursuit"
    assert (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action) == (
        "left", "none", "none"
    )


def test_fresh_visible_target_replaces_previous_target_immediately():
    bot = _route_attack_bot("right none none")
    lost_target = {
        "name": "test",
        "position": (0, 90),
        "size": (20, 20),
        "score": 0.1,
    }
    other_target = {
        "name": "other",
        "position": (260, 90),
        "size": (20, 20),
        "score": 0.1,
    }
    _seed_target(bot, lost_target)
    bot.t_last_attack = -10.0
    bot.update_monster_observations = lambda: setattr(bot, "monsters", [other_target])
    bot.update_cmd_by_route = lambda: (_ for _ in ()).throw(
        AssertionError("RouteNavigator was sampled during visible combat")
    )

    route_intent, combat_intent = bot.update_route_only_commands(now=0.5)

    assert route_intent is None
    assert combat_intent.state == "visible_attack"
    assert combat_intent.target is other_target
    assert (bot.cmd_move_x, bot.cmd_action) == ("stop", "attack_right")


def test_expired_lost_grace_resamples_current_route():
    bot = _route_attack_bot("left none none")
    target = {
        "name": "test",
        "position": (0, 90),
        "size": (20, 20),
        "score": 0.1,
    }
    _seed_target(bot, target)
    bot.update_monster_observations = lambda: setattr(bot, "monsters", [])
    calls = []

    def fresh_route():
        calls.append("route")
        intent = RouteIntent("right", "none", "none")
        bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action = (
            intent.move_x, intent.move_y, intent.action
        )
        bot.last_route_intent = intent
        return intent

    bot.update_cmd_by_route = fresh_route

    route_intent, combat_intent = bot.update_route_only_commands(now=0.31)

    assert calls == ["route"]
    assert route_intent.move_x == "right"
    assert combat_intent.state == "observe"
    assert bot.combat_target is None
    assert (bot.cmd_move_x, bot.cmd_action) == ("right", "none")


def test_fixed_yolo_strip_and_combined_directional_attack_geometry():
    bot = _route_attack_bot()
    bot.img_frame = np.zeros((800, 1600, 3), dtype=np.uint8)
    bot.loc_player = (800, 400)
    bot.cfg["directional_attack"].update({"range_x": 525, "range_y": 105})

    search = bot.get_monster_search_range()
    attack = bot.get_combined_attack_range()

    assert search == (160, 288, 1440, 512)
    assert (search[2] - search[0], search[3] - search[1]) == (1280, 224)
    assert attack == (275, 348, 1325, 453)
    assert (attack[2] - attack[0], attack[3] - attack[1]) == (1050, 105)


def test_combat_range_debug_draws_blue_red_and_direction_center_line():
    bot = _route_attack_bot()
    bot.img_frame = np.zeros((500, 900, 3), dtype=np.uint8)
    bot.img_frame_debug = np.zeros_like(bot.img_frame)
    bot.loc_player = (450, 250)
    bot.cfg["directional_attack"].update({"range_x": 300, "range_y": 100})

    bot.draw_combat_ranges_debug()

    # Blue detection border, red attack border and red center divider.
    assert tuple(bot.img_frame_debug[138, 0]) == (255, 0, 0)
    assert tuple(bot.img_frame_debug[200, 150]) == (0, 0, 255)
    assert tuple(bot.img_frame_debug[250, 450]) == (0, 0, 255)


def test_viz_composes_ranges_on_every_copy_without_mutating_source():
    bot = _route_attack_bot()
    bot.img_frame = np.zeros((500, 900, 3), dtype=np.uint8)
    bot.img_frame_debug = np.zeros_like(bot.img_frame)
    bot.loc_player = (450, 250)
    bot.cfg["ui_coords"] = {"ui_y_start": 500}
    bot.monsters = [{
        "name": "blue_snail",
        "position": (300, 200),
        "size": (20, 20),
        "score": 0.2,
    }]

    first = bot.get_frame_debug_for_viz()
    second = bot.get_frame_debug_for_viz()

    assert tuple(first[138, 0]) == (255, 0, 0)
    assert tuple(second[138, 0]) == (255, 0, 0)
    assert tuple(first[200, 300]) == (0, 255, 0)
    assert tuple(second[200, 300]) == (0, 255, 0)
    assert tuple(bot.img_frame_debug[138, 0]) == (0, 0, 0)
    assert tuple(bot.img_frame_debug[200, 300]) == (0, 0, 0)


def test_viz_preserves_original_bgr_colours_and_contiguous_layout():
    bot = _route_attack_bot()
    bot.img_frame = np.zeros((30, 40, 3), dtype=np.uint8)
    bot.img_frame_debug = np.zeros_like(bot.img_frame)
    bot.img_frame_debug[10, 10] = (17, 83, 211)
    bot.loc_player = None
    bot.cfg["ui_coords"] = {"ui_y_start": 30}

    rendered = bot.get_frame_debug_for_viz()

    assert tuple(rendered[10, 10]) == (17, 83, 211)
    assert rendered.flags.c_contiguous
