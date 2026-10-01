import numpy as np

from src.engine.MinimapPoseTracker import MinimapPoseTracker


class Matcher:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, _map, _minimap, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)


def _images():
    return np.zeros((20, 20, 3), dtype=np.uint8), np.zeros(
        (5, 5, 3), dtype=np.uint8
    )


def test_initial_global_then_local_tracking_uses_last_camera_and_median():
    matcher = Matcher([
        ((10, 20), 0.05, False),
        ((11, 20), 0.04, True),
        ((12, 20), 0.03, True),
    ])
    tracker = MinimapPoseTracker(matcher)
    base, minimap = _images()

    first = tracker.update(base, minimap, (2, 3))
    second = tracker.update(base, minimap, (2, 3))
    third = tracker.update(base, minimap, (2, 3))

    assert first.stable_position == (12, 23)
    assert second.stable_position == (12, 23)
    assert third.stable_position == (13, 23)
    assert matcher.calls[1]["last_result"] == (10, 20)
    assert matcher.calls[2]["last_result"] == (11, 20)


def test_global_fallback_waits_three_failures_and_confirms_far_candidate_twice():
    matcher = Matcher([
        ((0, 0), 0.01, False),
        ((50, 0), 0.01, False),
        ((50, 0), 0.01, False),
        ((50, 0), 0.01, False),
        ((51, 0), 0.01, False),
    ])
    tracker = MinimapPoseTracker(matcher)
    base, minimap = _images()
    assert tracker.update(base, minimap, (1, 1)).valid

    results = [tracker.update(base, minimap, (1, 1)) for _ in range(4)]

    assert [result.reason for result in results] == [
        "local_recovery_wait", "local_recovery_wait",
        "global_unconfirmed", "ok",
    ]
    assert results[-1].source == "global_confirmed"
    assert results[-1].stable_position == (52, 1)


def test_local_single_frame_jump_is_rejected_without_polluting_history():
    matcher = Matcher([
        ((0, 0), 0.01, False),
        ((30, 0), 0.01, True),
        ((1, 0), 0.01, True),
    ])
    tracker = MinimapPoseTracker(matcher)
    base, minimap = _images()
    assert tracker.update(base, minimap, (1, 1)).valid
    rejected = tracker.update(base, minimap, (1, 1))
    recovered = tracker.update(base, minimap, (1, 1))

    assert not rejected.valid
    assert rejected.reason == "position_jump"
    assert recovered.valid
    assert recovered.stable_position == (2, 1)


def test_reset_and_shift_origin_update_all_accepted_coordinates():
    matcher = Matcher([((5, 6), 0.01, False)])
    tracker = MinimapPoseTracker(matcher)
    base, minimap = _images()
    tracker.update(base, minimap, (2, 3))
    tracker.shift_origin(4, 7)
    assert tracker.last_camera == (9, 13)
    assert tracker.last_raw_position == (11, 16)
    tracker.reset()
    assert tracker.last_camera is None
    assert tracker.last_snapshot.reason == "reset"


def test_explicit_teleport_accepts_large_jump_and_resets_median_history():
    matcher = Matcher([
        ((0, 0), 0.01, False),
        ((40, 0), 0.01, True),
    ])
    tracker = MinimapPoseTracker(matcher)
    base, minimap = _images()
    tracker.update(base, minimap, (1, 1))
    teleported = tracker.update(
        base, minimap, (1, 1), allow_large_jump=True
    )

    assert teleported.valid
    assert teleported.raw_position == teleported.stable_position == (41, 1)
