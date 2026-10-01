import pytest

from src.engine.SemanticRouteRecorder import (
    RouteRecordingSample,
    SemanticRouteRecorder,
)


def _recorder(**kwargs):
    return SemanticRouteRecorder(
        map_id="training",
        route_index=1,
        canvas_size=(200, 120),
        jump_key="space",
        teleport_key="",
        **kwargs,
    )


def _sample(t, x, y, keys=(), score=0.05, valid=True):
    return RouteRecordingSample.create(
        t, (x, y) if valid else None, keys, score, valid
    )


def test_jump_uses_configured_space_and_ignores_other_character_key():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 20, 90, {"1"}))
    recorder.feed(_sample(0.1, 21, 90, set()))
    recorder.feed(_sample(0.2, 22, 90, {"space"}))
    recorder.feed(_sample(0.6, 23, 86, set()))
    document = recorder.finish_goal((30, 86))

    jumps = [segment for segment in document["segments"] if segment["type"] == "jump"]
    assert len(jumps) == 1
    assert jumps[0]["anchor"] == [22, 90]


def test_blank_teleport_binding_never_records_teleport():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 20, 90, {"e", "right"}))
    recorder.feed(_sample(0.1, 24, 90, {"e", "right"}))
    document = recorder.finish_goal((30, 90))

    assert not any(segment["type"] == "teleport" for segment in document["segments"])


def test_walk_is_ordered_one_pixel_path_and_reverse_knockback_is_ignored():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 10, 90, {"right"}))
    recorder.feed(_sample(0.1, 12, 90, {"right"}))
    recorder.feed(_sample(0.2, 11, 90, {"right"}))
    recorder.feed(_sample(0.3, 15, 90, {"right"}))
    recorder.feed(_sample(0.4, 15, 90, set()))
    document = recorder.finish_goal((20, 90))

    walk = document["segments"][0]
    assert walk["type"] == "walk"
    assert walk["direction"] == "right"
    assert walk["points"] == [[10, 90], [12, 90], [15, 90]]


def test_jump_up_within_window_and_upward_feedback_becomes_ladder():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 50, 90, {"space", "up"}))
    recorder.feed(_sample(0.1, 50, 85, {"up"}))
    recorder.feed(_sample(0.2, 50, 77, {"up"}))
    recorder.feed(_sample(0.3, 50, 60, {"up"}))
    recorder.feed(_sample(0.4, 54, 50, {"right"}))
    document = recorder.finish_goal((60, 50))

    ladder = document["segments"][0]
    assert ladder["type"] == "ladder"
    assert ladder["mount"] == [50, 90]
    assert ladder["mount_direction"] == "none"
    assert ladder["points"][0] == [50, 90]
    assert ladder["exit"] == [54, 50]
    assert ladder["exit_direction"] == "right"
    assert not any(segment["type"] == "jump" for segment in document["segments"])


def test_forward_jump_ladder_preserves_mount_direction_and_separate_mount():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 40, 90, {"space", "up", "right"}))
    recorder.feed(_sample(0.1, 44, 86, {"up", "right"}))
    recorder.feed(_sample(0.2, 49, 82, {"up", "right"}))
    # The launch direction is ignored until an up-only climbing sample arms exit.
    recorder.feed(_sample(0.3, 50, 75, {"up", "right"}))
    recorder.feed(_sample(0.4, 50, 65, {"up"}))
    recorder.feed(_sample(0.5, 54, 60, {"right"}))
    document = recorder.finish_goal((60, 60))

    ladder = document["segments"][0]
    assert ladder["type"] == "ladder"
    assert ladder["approach"] == [40, 90]
    assert ladder["mount"] == [50, 90]
    assert ladder["mount_direction"] == "right"
    assert ladder["exit_direction"] == "right"


def test_ladder_requires_two_upward_frames_not_one_vertical_jitter():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 40, 90, {"space", "up"}))
    recorder.feed(_sample(0.1, 40, 84, {"up"}))
    recorder.feed(_sample(0.2, 40, 86, {"up"}))
    recorder.feed(_sample(0.3, 40, 90, set()))
    document = recorder.finish_goal((45, 90))

    assert document["segments"][0]["type"] == "jump"


def test_jump_without_up_stays_jump_after_300ms():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 20, 90, {"space", "right"}))
    recorder.feed(_sample(0.31, 24, 86, {"right"}))
    document = recorder.finish_goal((30, 86))

    assert document["segments"][0]["type"] == "jump"
    assert document["segments"][0]["direction"] == "right"


def test_jump_plus_up_without_vertical_success_does_not_block_later_walk():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 20, 80, {"space", "up"}))
    recorder.feed(_sample(0.4, 20, 80, {"up"}))
    recorder.feed(_sample(0.5, 21, 80, {"right"}))
    recorder.feed(_sample(0.6, 30, 80, {"right"}))
    recorder.feed(_sample(0.7, 31, 80, {"right"}))
    recorder.feed(_sample(0.8, 33, 80, {"right"}))

    document = recorder.finish_goal((35, 80))

    assert [segment["type"] for segment in document["segments"]] == [
        "jump", "walk", "goal",
    ]


def test_down_jump_records_landing():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 20, 30, {"space", "down"}))
    recorder.feed(_sample(0.2, 20, 60, {"down"}))
    recorder.feed(_sample(0.4, 20, 80, set()))
    document = recorder.finish_goal((30, 80))

    drop = document["segments"][0]
    assert drop == {
        "id": drop["id"],
        "type": "drop",
        "anchor": [20, 30],
        "landing": [20, 80],
    }


def test_low_confidence_gap_breaks_continuous_walk():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 10, 90, {"right"}))
    recorder.feed(_sample(0.1, 12, 90, {"right"}))
    recorder.feed(_sample(0.2, 14, 90, {"right"}, score=0.9))
    recorder.feed(_sample(0.8, 40, 90, {"right"}))
    recorder.feed(_sample(0.9, 45, 90, {"right"}))
    recorder.feed(_sample(1.0, 45, 90, set()))
    document = recorder.finish_goal((50, 90))

    walks = [segment for segment in document["segments"] if segment["type"] == "walk"]
    assert len(walks) == 2
    assert walks[0]["points"] == [[10, 90], [12, 90]]
    assert walks[1]["points"] == [[40, 90], [45, 90]]


def test_large_position_jump_breaks_line_even_without_lost_frame():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 10, 90, {"right"}))
    recorder.feed(_sample(0.1, 12, 90, {"right"}))
    recorder.feed(_sample(0.2, 40, 90, {"right"}))
    recorder.feed(_sample(0.3, 45, 90, {"right"}))
    recorder.feed(_sample(0.4, 46, 90, {"right"}))
    recorder.feed(_sample(0.5, 46, 90, set()))
    document = recorder.finish_goal((50, 90))

    walks = [segment for segment in document["segments"] if segment["type"] == "walk"]
    assert [walk["points"] for walk in walks] == [
        [[10, 90], [12, 90]],
        [[45, 90], [46, 90]],
    ]


def test_ladder_mount_center_uses_stable_climb_samples_not_flight_arc():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 40, 90, {"space", "up", "right"}))
    recorder.feed(_sample(0.1, 44, 86, {"up", "right"}))
    recorder.feed(_sample(0.2, 48, 82, {"up", "right"}))
    recorder.feed(_sample(0.3, 51, 77, {"up"}))
    recorder.feed(_sample(0.4, 51, 72, {"up"}))
    recorder.feed(_sample(0.5, 50, 67, {"up"}))
    recorder.feed(_sample(0.6, 51, 62, {"up"}))
    recorder.feed(_sample(0.7, 54, 60, {"right"}))
    ladder = recorder.finish_goal((60, 60))["segments"][0]

    assert ladder["type"] == "ladder"
    assert ladder["approach"] == [40, 90]
    assert ladder["mount"] == [51, 90]


def test_undo_redo_and_goal_document_validation():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 10, 90, {"right"}))
    recorder.feed(_sample(0.1, 15, 90, {"right"}))
    recorder.feed(_sample(0.2, 15, 90, set()))
    assert recorder.undo()
    assert recorder.redo()
    document = recorder.finish_goal((20, 90))

    assert [segment["type"] for segment in document["segments"]] == ["walk", "goal"]
    assert document["route_index"] == 1


def test_document_without_goal_is_rejected():
    recorder = _recorder()
    recorder.feed(_sample(0.0, 10, 90, {"right"}))
    recorder.feed(_sample(0.1, 15, 90, {"right"}))
    recorder.feed(_sample(0.2, 15, 90, set()))
    with pytest.raises(Exception, match="Goal"):
        recorder.document()
