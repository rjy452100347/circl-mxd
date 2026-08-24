import numpy as np

from src.engine.RouteNavigator import RouteNavigator


PRIMARY = {
    (255, 0, 0): "left none none",
    (0, 0, 255): "right none none",
    (255, 0, 255): "none none jump",
    (255, 255, 0): "none none goal",
}
VERTICAL = {(127, 127, 127): "none up none"}


def _route(*pixels):
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    for x, y, color in pixels:
        image[y, x] = color
    return image


def test_nearest_route_pixel_drives_continuous_direction():
    navigator = RouteNavigator([_route((10, 10, (0, 0, 255)))], PRIMARY, VERTICAL, 3)

    assert navigator.decide((9, 10)).command == "right none none"


def test_jump_next_to_up_becomes_one_shot_mount_then_up_hold():
    image = _route(
        (10, 10, (255, 0, 255)),
        (10, 11, (255, 0, 255)),
        (10, 8, (127, 127, 127)),
        (10, 7, (127, 127, 127)),
    )
    navigator = RouteNavigator([image], PRIMARY, VERTICAL, 4, mount_search_range=4)

    first = navigator.decide((10, 10))
    second = navigator.decide((10, 10))

    assert first.command == "none up mount"
    assert first.event_id is not None
    assert second.command == "none up none"


def test_failed_mount_holds_up_and_retries_without_horizontal_oscillation():
    image = _route(
        (10, 10, (255, 0, 255)),
        (10, 8, (127, 127, 127)),
        (7, 10, (255, 0, 0)),
        (13, 10, (0, 0, 255)),
    )
    navigator = RouteNavigator(
        [image], PRIMARY, VERTICAL, 5,
        mount_search_range=4,
        mount_retry_frames=3,
        mount_success_distance=5,
    )

    decisions = [navigator.decide((10, 10)) for _ in range(5)]

    assert [decision.command for decision in decisions] == [
        "none up mount",
        "none up none",
        "none up none",
        "none up mount",
        "none up none",
    ]
    assert {decision.reason for decision in decisions[1:]} == {
        "mount_pending", "mount_retry"
    }


def test_goal_advances_route_once_and_cycles_back():
    route1 = _route((5, 5, (255, 255, 0)))
    route2 = _route((8, 5, (255, 255, 0)))
    navigator = RouteNavigator([route1, route2], PRIMARY, VERTICAL, 2)

    first = navigator.decide((5, 5))
    second = navigator.decide((8, 5))

    assert first.command == "none none goal"
    assert first.route_index == 0
    assert second.command == "none none goal"
    assert second.route_index == 1
    assert navigator.route_index == 0


def test_consumed_event_rearms_only_after_leaving_marker():
    image = _route(
        (10, 10, (255, 0, 255)),
        (10, 8, (127, 127, 127)),
    )
    navigator = RouteNavigator(
        [image], PRIMARY, VERTICAL, 4, event_rearm_distance=5, mount_search_range=4
    )

    assert navigator.decide((10, 10)).command == "none up mount"
    assert navigator.decide((10, 10)).command == "none up none"
    navigator.decide((17, 10))
    assert navigator.decide((10, 10)).command == "none up mount"


def test_missing_route_pixel_requests_safe_stop():
    navigator = RouteNavigator([_route()], PRIMARY, VERTICAL, 3)

    decision = navigator.decide((20, 20))

    assert decision.command == "stop stop none"
    assert decision.reason == "route_lost"


def test_goal_temporarily_suppresses_inbound_line_for_single_route_loop():
    image = _route(
        (20, 10, (255, 255, 0)),
        (20, 8, (127, 127, 127)),
        (8, 10, (255, 0, 0)),
    )
    navigator = RouteNavigator([image], PRIMARY, VERTICAL, 15, event_rearm_distance=5)

    assert navigator.decide((20, 10)).command == "none none goal"
    assert navigator.decide((20, 10)).command == "left none none"
