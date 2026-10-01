from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.engine.SemanticRouteNavigator import SemanticRouteNavigator


def _navigator():
    document = {
        "schema_version": 1,
        "map_id": "training",
        "route_index": 1,
        "canvas_size": [100, 60],
        "loop": True,
        "segments": [
            {"id": "walk", "type": "walk", "direction": "right",
             "points": [[10, 40], [20, 40], [30, 40]]},
            {"id": "goal", "type": "goal", "position": [35, 40], "radius": 6},
        ],
    }
    return SemanticRouteNavigator([document])


def test_json_route_is_authoritative_even_without_legacy_route_only_flag():
    bot = object.__new__(MapleStoryAutoBot)
    bot.cfg = {"bot": {"route_only": False}}
    bot.route_navigator = _navigator()
    bot.loc_player_global = (10, 40)
    bot.is_on_ladder = False
    bot.idx_routes = 0

    intent = bot.update_cmd_by_route()

    assert (intent.move_x, intent.move_y, intent.action) == (
        "right", "none", "none",
    )
    assert (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action) == (
        "right", "none", "none",
    )


def test_semantic_goal_is_not_advanced_twice_by_legacy_state_hook():
    bot = object.__new__(MapleStoryAutoBot)
    bot.route_navigator = _navigator()
    bot.cmd_action = "goal"
    bot.idx_routes = 0
    bot.img_routes = [object(), object()]

    bot.check_reach_goal()

    assert bot.idx_routes == 0


def test_ladder_failure_terminates_input_with_specific_reason():
    document = {
        "schema_version": 1,
        "map_id": "training",
        "route_index": 1,
        "canvas_size": [100, 100],
        "loop": True,
        "segments": [
            {
                "id": "ladder", "type": "ladder",
                "approach": [50, 90], "mount": [50, 90],
                "points": [[50, 90], [50, 60]], "exit": [50, 60],
                "exit_direction": "none",
            },
            {"id": "goal", "type": "goal", "position": [50, 60], "radius": 6},
        ],
    }
    navigator = SemanticRouteNavigator([document])
    navigator._traversal_state = "failed"
    navigator._ladder_attempts = 3

    class Keyboard:
        is_need_force_heal = False
        termination_reason = ""

        def __init__(self):
            self.commands = []
            self.released = False
            self.terminated = False

        def set_command(self, command):
            self.commands.append(command)

        def release_all_key(self):
            self.released = True

        def terminate(self):
            self.terminated = True

    bot = object.__new__(MapleStoryAutoBot)
    bot.route_navigator = navigator
    bot.loc_player_global = (50, 90)
    bot.is_on_ladder = False
    bot.idx_routes = 0
    bot.kb = Keyboard()

    intent = bot.update_cmd_by_route()

    assert intent.reason == "semantic_ladder_failed"
    assert bot.kb.commands[-1] == "stop stop none"
    assert bot.kb.released and bot.kb.terminated
    assert bot.kb.termination_reason == "route_ladder_failed"


def test_ladder_segment_is_route_committed_before_mount_action():
    document = {
        "schema_version": 1,
        "map_id": "training",
        "route_index": 1,
        "canvas_size": [100, 100],
        "loop": True,
        "segments": [
            {
                "id": "ladder", "type": "ladder",
                "approach": [50, 90], "mount": [50, 90],
                "points": [[50, 90], [50, 60]], "exit": [50, 60],
                "exit_direction": "none",
            },
            {"id": "goal", "type": "goal", "position": [50, 60], "radius": 6},
        ],
    }
    bot = object.__new__(MapleStoryAutoBot)
    bot.route_navigator = SemanticRouteNavigator([document])
    bot.is_on_ladder = False
    bot.route_commit_until = 0.0
    bot.route_commit_kind = None

    assert bot.is_route_traversal_committed(now=0.0)
