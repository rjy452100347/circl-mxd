from src.engine.SemanticRouteNavigator import SemanticRouteNavigator
import pytest


def _route(index=1, segments=None):
    return {
        "schema_version": 1,
        "map_id": "training",
        "route_index": index,
        "canvas_size": [200, 120],
        "loop": True,
        "segments": segments or [
            {
                "id": "walk",
                "type": "walk",
                "direction": "right",
                "points": [[10, 90], [20, 90], [30, 90], [40, 90]],
            },
            {
                "id": "goal",
                "type": "goal",
                "position": [45, 90],
                "radius": 6,
            },
        ],
    }


def test_ordered_progress_ignores_nearby_crossing_segment():
    route = _route(segments=[
        {
            "id": "bottom", "type": "walk", "direction": "right",
            "points": [[10, 80], [20, 80], [30, 80], [40, 80]],
        },
        {
            "id": "top", "type": "walk", "direction": "left",
            "points": [[40, 82], [30, 82], [20, 82], [10, 82]],
        },
        {"id": "goal", "type": "goal", "position": [5, 82], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route], reacquire_radius=18)

    decision = navigator.decide((20, 81))

    assert decision.command == "right none none"
    assert navigator.segment_position == 0


def test_walk_direction_hysteresis_filters_small_reverse_jitter():
    navigator = SemanticRouteNavigator([_route()], direction_hysteresis=3)
    first = navigator.decide((20, 90))
    second = navigator.decide((31, 90))

    assert first.command == "right none none"
    assert second.command == "right none none"


def test_global_reacquire_selects_one_unique_route_segment():
    route = _route(segments=[
        {
            "id": "first", "type": "walk", "direction": "right",
            "points": [[5, 20], [20, 20], [35, 20]],
        },
        {
            "id": "second", "type": "walk", "direction": "left",
            "points": [[150, 80], [130, 80], [110, 80]],
        },
        {"id": "goal", "type": "goal", "position": [100, 80], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route], reacquire_radius=20)

    decision = navigator.decide((132, 80))

    assert decision.command == "left none none"
    assert navigator.segment_position == 1
    assert navigator.diagnostics.reason == "semantic_walk"


def test_global_reacquire_stops_on_cross_segment_ambiguity():
    route = _route(segments=[
        {
            "id": "first", "type": "walk", "direction": "right",
            "points": [[5, 20], [20, 20], [35, 20]],
        },
        {
            "id": "second", "type": "walk", "direction": "left",
            "points": [[5, 24], [20, 24], [35, 24]],
        },
        {"id": "goal", "type": "goal", "position": [40, 24], "radius": 6},
    ])
    navigator = SemanticRouteNavigator(
        [route], reacquire_radius=18, ambiguity_margin=5
    )

    decision = navigator.decide((100, 100))
    assert decision.command == "stop stop none"
    assert decision.reason == "semantic_route_lost"

    # Move within range but equally close to both crossing segments.
    navigator.segment_position = 0
    navigator.point_position = 0
    decision = navigator.decide((20, 22))
    assert decision.command == "right none none"  # local continuity wins

    navigator.segment_position = 0
    navigator.point_position = 0
    decision = navigator.decide((80, 22))
    assert decision.command == "stop stop none"


def test_jump_is_one_event_until_visual_completion():
    route = _route(segments=[
        {
            "id": "jump", "type": "jump",
            "anchor": [20, 90], "direction": "right",
        },
        {
            "id": "walk", "type": "walk", "direction": "right",
            "points": [[22, 86], [35, 86]],
        },
        {"id": "goal", "type": "goal", "position": [40, 86], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route], event_commit_seconds=0.45)

    first = navigator.decide((20, 90), now=0.0)
    pending = navigator.decide((20, 90), now=0.2)
    completed = navigator.decide((22, 84), now=0.3)

    assert first.command == "right none jump"
    assert pending.command == "right none jump"
    assert completed.command == "right none none"
    assert navigator.segment_position == 1


def test_drop_waits_for_recorded_landing_y():
    route = _route(segments=[
        {
            "id": "drop", "type": "drop",
            "anchor": [20, 30], "landing": [20, 80],
        },
        {
            "id": "walk", "type": "walk", "direction": "right",
            "points": [[20, 80], [35, 80]],
        },
        {"id": "goal", "type": "goal", "position": [40, 80], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route])

    assert navigator.decide((20, 30), now=0).command == "none down jump"
    assert navigator.decide((20, 60), now=0.2).command == "none down jump"
    assert navigator.decide((20, 79), now=0.3).command == "right none none"


def test_ladder_align_mount_climb_exit_and_complete():
    route = _route(segments=[
        {
            "id": "ladder", "type": "ladder",
            "approach": [50, 90],
            "mount": [50, 90],
            "points": [[50, 90], [50, 60], [50, 30]],
            "exit": [56, 30],
            "exit_direction": "right",
        },
        {
            "id": "walk", "type": "walk", "direction": "right",
            "points": [[56, 30], [70, 30]],
        },
        {"id": "goal", "type": "goal", "position": [75, 30], "radius": 6},
    ])
    navigator = SemanticRouteNavigator(
        [route], ladder_retry_frames=3, ladder_success_distance=12
    )

    assert navigator.decide((42, 90)).command == "right none none"
    assert not navigator.has_pending_traversal
    assert navigator.requires_route_exclusive_control
    assert navigator.decide((50, 90), now=0.0).command == "none none none"
    assert navigator.decide((50, 90), now=0.1).command == "none up mount"
    assert navigator.has_pending_traversal
    assert navigator.decide((50, 78), is_on_ladder=True, now=0.2).command == "none up none"
    assert navigator.decide((50, 77), is_on_ladder=True, now=0.3).command == "none up none"
    assert navigator.decide((50, 76), is_on_ladder=True, now=0.4).command == "none up none"
    assert navigator.decide((50, 31), is_on_ladder=True, now=0.5).command == "right none none"
    assert navigator.decide((54, 30), is_on_ladder=False, now=0.6).command == "right none none"
    completed = navigator.decide((57, 30), is_on_ladder=False, now=0.7)

    assert completed.command == "right none none"
    assert navigator.segment_position == 1
    assert not navigator.has_pending_traversal


def test_failed_ladder_mount_stops_after_bounded_attempts():
    route = _route(segments=[
        {
            "id": "ladder", "type": "ladder",
            "approach": [50, 90],
            "mount": [50, 90],
            "points": [[50, 90], [50, 60], [50, 30]],
            "exit": [50, 30],
            "exit_direction": "none",
        },
        {"id": "goal", "type": "goal", "position": [50, 30], "radius": 6},
    ])
    navigator = SemanticRouteNavigator(
        [route], ladder_max_attempts=3, ladder_attempt_timeout=1.2
    )

    now = 0.0
    candidate_x = (50, 48, 52)
    for attempt, x in enumerate(candidate_x):
        assert navigator.decide((x, 90), now=now).command == "none none none"
        assert navigator.decide((x, 90), now=now + 0.1).command == "none up mount"
        failed = navigator.decide((x, 90), now=now + 1.5)
        if attempt < 2:
            assert failed.reason == "semantic_ladder_retry_timeout"
        now += 2.0

    assert failed.command == "stop stop none"
    assert failed.reason == "semantic_ladder_failed"
    assert navigator.segment_position == 0
    assert navigator.diagnostics.traversal_state == "failed"
    assert navigator.diagnostics.ladder_attempts == 3


def test_vertical_ladder_brakes_and_requires_two_stable_frames():
    route = _route(segments=[
        {
            "id": "ladder", "type": "ladder",
            "approach": [50, 90], "mount": [50, 90],
            "points": [[50, 90], [50, 60]], "exit": [50, 60],
            "exit_direction": "none",
        },
        {"id": "goal", "type": "goal", "position": [50, 60], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route])

    assert navigator.decide((43, 90), now=0.0).reason == "semantic_ladder_coarse_align"
    fine = navigator.decide((46, 90), now=0.1)
    brake = navigator.decide((47, 90), now=0.2)
    assert fine.reason == "semantic_ladder_fine_align"
    assert brake.reason == "semantic_ladder_fine_brake"
    assert navigator.decide((49, 90), now=0.3).command == "none none none"
    # A slide across the center resets stability instead of mounting early.
    assert navigator.decide((52, 90), now=0.4).command == "left none none"
    assert navigator.decide((50, 90), now=0.5).command == "none none none"
    assert navigator.decide((50, 90), now=0.6).command == "none up mount"


def test_forward_ladder_launch_holds_direction_then_releases_it():
    route = _route(segments=[
        {
            "id": "ladder", "type": "ladder",
            "approach": [40, 90], "mount": [50, 90],
            "mount_direction": "right",
            "points": [[50, 90], [50, 60]], "exit": [50, 60],
            "exit_direction": "none",
        },
        {"id": "goal", "type": "goal", "position": [50, 60], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route], ladder_success_distance=5)

    assert navigator.decide((35, 90), now=0.0).command == "right none none"
    assert navigator.decide((39, 90), now=0.1).command == "right up mount"
    assert navigator.decide((43, 87), now=0.2).command == "right up none"
    assert navigator.decide((49, 84), now=0.3).command == "none up none"
    assert navigator.decide((50, 83), now=0.4).command == "none up none"
    climbed = navigator.decide((50, 82), now=0.5)
    assert climbed.command == "none up none"
    assert navigator.diagnostics.traversal_state == "climbing"


def test_forward_ladder_timeout_returns_to_opposite_runup_side():
    route = _route(segments=[
        {
            "id": "ladder", "type": "ladder",
            "approach": [40, 90], "mount": [50, 90],
            "mount_direction": "right",
            "points": [[50, 90], [50, 60]], "exit": [50, 60],
            "exit_direction": "none",
        },
        {"id": "goal", "type": "goal", "position": [50, 60], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route], ladder_attempt_timeout=1.2)

    assert navigator.decide((40, 90), now=0.0).command == "right up mount"
    failed = navigator.decide((42, 90), now=1.3)
    assert failed.reason == "semantic_ladder_retry_timeout"
    reset = navigator.decide((42, 90), now=1.4)
    assert reset.command == "left none none"
    assert reset.point == (30, 90)


def test_ladder_suspension_freezes_attempt_timeout():
    route = _route(segments=[
        {
            "id": "ladder", "type": "ladder",
            "approach": [50, 90], "mount": [50, 90],
            "points": [[50, 90], [50, 60]], "exit": [50, 60],
            "exit_direction": "none",
        },
        {"id": "goal", "type": "goal", "position": [50, 60], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route], ladder_attempt_timeout=1.2)
    navigator.decide((50, 90), now=0.0)
    assert navigator.decide((50, 90), now=0.1).command == "none up mount"

    suspended = navigator.decide(
        (50, 90), now=0.8, suspend_traversal=True
    )
    assert suspended.reason == "semantic_ladder_suspended"
    resumed = navigator.decide((50, 90), now=5.0)
    assert resumed.reason == "semantic_ladder_mount_wait"
    assert navigator.diagnostics.ladder_attempts == 1


def test_vertical_mount_retries_jump_pulse_every_five_control_frames():
    route = _route(segments=[
        {
            "id": "ladder", "type": "ladder",
            "approach": [50, 90], "mount": [50, 90],
            "points": [[50, 90], [50, 60]], "exit": [50, 60],
            "exit_direction": "none",
        },
        {"id": "goal", "type": "goal", "position": [50, 60], "radius": 6},
    ])
    navigator = SemanticRouteNavigator(
        [route], ladder_retry_frames=5, ladder_attempt_timeout=2.0
    )
    navigator.decide((50, 90), now=0.0)
    assert navigator.decide((50, 90), now=0.1).command.split()[-1] == "mount"
    actions = [
        navigator.decide((50, 89), now=0.2 + index * 0.1).command.split()[-1]
        for index in range(5)
    ]
    assert actions == ["none", "none", "none", "none", "mount"]
    assert navigator.ladder_debug["pulse_count"] == 2


def test_adaptive_candidate_offset_is_cached_for_next_route_loop():
    route = _route(segments=[
        {
            "id": "ladder", "type": "ladder",
            "approach": [50, 90], "mount": [50, 90],
            "points": [[50, 90], [50, 60]], "exit": [50, 60],
            "exit_direction": "none",
        },
        {"id": "goal", "type": "goal", "position": [48, 60], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route], ladder_attempt_timeout=0.5)
    navigator.decide((50, 90), now=0.0)
    navigator.decide((50, 90), now=0.1)
    assert navigator.decide((50, 90), now=0.7).reason == "semantic_ladder_retry_timeout"
    navigator.decide((48, 90), now=0.8)
    navigator.decide((48, 90), now=0.9)
    for index, y in enumerate((78, 77, 76), start=10):
        navigator.decide((48, y), now=index / 10)
    assert navigator.ladder_debug["cached_offset"] == -2
    navigator.decide((48, 60), is_on_ladder=False, now=1.4)
    navigator.decide((48, 60), is_on_ladder=False, now=1.5)
    navigator.decide((48, 60), now=1.6)
    assert navigator.route_index == 0 and navigator.segment_position == 0
    first = navigator.decide((48, 90), now=1.7)
    assert first.reason == "semantic_ladder_settle"
    assert navigator.ladder_debug["candidate_offset"] == -2


def test_rise_then_descent_never_confirms_mount():
    route = _route(segments=[
        {
            "id": "ladder", "type": "ladder",
            "approach": [50, 90], "mount": [50, 90],
            "points": [[50, 90], [50, 60]], "exit": [50, 60],
            "exit_direction": "none",
        },
        {"id": "goal", "type": "goal", "position": [50, 60], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route])
    navigator.decide((50, 90), now=0.0)
    navigator.decide((50, 90), now=0.1)
    navigator.decide((50, 78), now=0.2)
    failed = navigator.decide((50, 79), now=0.3)
    assert failed.reason == "semantic_ladder_retry_descending"
    assert navigator.diagnostics.traversal_state == "coarse_align"


def test_short_ladder_uses_dynamic_five_pixel_success_threshold():
    route = _route(segments=[
        {
            "id": "ladder", "type": "ladder",
            "approach": [50, 90], "mount": [50, 90],
            "points": [[50, 90], [50, 85]], "exit": [50, 85],
            "exit_direction": "none",
        },
        {"id": "goal", "type": "goal", "position": [50, 85], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route], ladder_success_distance=12)
    navigator.decide((50, 90), now=0.0)
    navigator.decide((50, 90), now=0.1)
    navigator.decide((50, 85), now=0.2)
    navigator.decide((50, 84), now=0.3)
    decision = navigator.decide((50, 83), now=0.4)

    assert decision.reason == "semantic_ladder_climb"
    assert navigator.diagnostics.traversal_state == "climbing"


@pytest.mark.parametrize(
    ("direction", "approach_x", "mount_x", "start_x", "near_x", "command"),
    [
        ("right", 40, 50, 40, 51, "right up mount"),
        ("left", 60, 50, 60, 49, "left up mount"),
    ],
)
def test_forward_mount_releases_horizontal_immediately_after_crossing_center(
    direction, approach_x, mount_x, start_x, near_x, command
):
    route = _route(segments=[
        {
            "id": "ladder", "type": "ladder",
            "approach": [approach_x, 90], "mount": [mount_x, 90],
            "mount_direction": direction,
            "points": [[mount_x, 90], [mount_x, 60]],
            "exit": [mount_x, 60], "exit_direction": "none",
        },
        {"id": "goal", "type": "goal", "position": [mount_x, 60], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route])
    assert navigator.decide((start_x, 90), now=0.0).command == command
    crossed = navigator.decide((near_x, 86), now=0.1)
    assert crossed.command == "none up none"


def test_all_five_candidate_offsets_are_bounded_and_fail_closed():
    route = _route(segments=[
        {
            "id": "ladder", "type": "ladder",
            "approach": [50, 90], "mount": [50, 90],
            "points": [[50, 90], [50, 60]], "exit": [50, 60],
            "exit_direction": "none",
        },
        {"id": "goal", "type": "goal", "position": [50, 60], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route], ladder_attempt_timeout=0.5)
    offsets = []
    now = 0.0
    final = None
    for x in (50, 48, 52, 46, 54):
        offsets.append(navigator.ladder_debug["candidate_offset"])
        navigator.decide((x, 90), now=now)
        navigator.decide((x, 90), now=now + 0.1)
        final = navigator.decide((x, 90), now=now + 0.7)
        now += 1.0
    assert offsets == [0, -2, 2, -4, 4]
    assert final.reason == "semantic_ladder_failed"
    assert final.command == "stop stop none"


def test_goal_advances_routes_in_a_cycle_only_when_in_radius():
    route1 = _route(index=1)
    route2 = _route(index=2)
    navigator = SemanticRouteNavigator([route1, route2])

    navigator.segment_position = 1
    approach = navigator.decide((30, 90))
    goal = navigator.decide((45, 90))

    assert approach.command == "right none none"
    assert goal.command == "none none goal"
    assert goal.route_index == 0
    assert navigator.route_index == 1


def test_non_looping_last_route_stops_after_goal():
    route = _route()
    route["loop"] = False
    navigator = SemanticRouteNavigator([route])
    navigator.segment_position = 1

    assert navigator.decide((45, 90)).command == "none none goal"
    completed = navigator.decide((45, 90))

    assert completed.command == "stop stop none"
    assert completed.reason == "semantic_route_complete"


def test_stop_segment_emits_once_then_advances():
    route = _route(segments=[
        {"id": "stop", "type": "stop", "position": [20, 90]},
        {
            "id": "walk", "type": "walk", "direction": "right",
            "points": [[20, 90], [35, 90]],
        },
        {"id": "goal", "type": "goal", "position": [40, 90], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route])

    assert navigator.decide((20, 90)).command == "stop stop none"
    assert navigator.decide((20, 90)).command == "right none none"


def test_explicit_resume_reacquires_from_current_position_not_stale_segment():
    route = _route(segments=[
        {
            "id": "first", "type": "walk", "direction": "right",
            "points": [[5, 20], [20, 20], [35, 20]],
        },
        {
            "id": "second", "type": "walk", "direction": "left",
            "points": [[150, 80], [130, 80], [110, 80]],
        },
        {"id": "goal", "type": "goal", "position": [100, 80], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route], reacquire_radius=20)
    assert navigator.decide((20, 20)).command == "right none none"

    assert navigator.request_reacquire("combat_finished")
    decision = navigator.decide((132, 80))

    assert decision.command == "left none none"
    assert navigator.segment_position == 1


def test_non_adjacent_global_reacquire_requires_persistent_local_failure():
    route = _route(segments=[
        {"id": "first", "type": "walk", "direction": "right",
         "points": [[5, 10], [20, 10], [35, 10]]},
        {"id": "middle", "type": "walk", "direction": "right",
         "points": [[60, 45], [75, 45], [90, 45]]},
        {"id": "last", "type": "walk", "direction": "left",
         "points": [[160, 90], [145, 90], [130, 90]]},
        {"id": "goal", "type": "goal", "position": [120, 90], "radius": 6},
    ])
    navigator = SemanticRouteNavigator([route], reacquire_radius=18)
    navigator.request_reacquire("combat_finished")

    first = navigator.decide((145, 90))
    second = navigator.decide((145, 90))
    third = navigator.decide((145, 90))

    assert first.command == second.command == "stop stop none"
    assert third.command == "left none none"
    assert navigator.segment_position == 2
