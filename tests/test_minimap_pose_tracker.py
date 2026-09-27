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


def test_scores_between_old_thresholds_trigger_explicit_global_recovery():
    matcher = Matcher([((0, 0), .01, False)] * 2 +
                      [((0, 0), .30, True)] * 3 + [((0, 0), .01, False)])
    tracker = MinimapPoseTracker(matcher, max_score=.2)
    results = [tracker.update(*_images(), (1, 1), sequence=i, produced_at=i*.1, now=i*.1)
               for i in range(6)]
    assert [r.reason for r in results[2:5]] == ['score']*3
    assert results[-1].valid
    assert [c['search_mode'] for c in matcher.calls] == ['global', 'global', 'local', 'local', 'local', 'global']
    assert all(c['global_threshold'] == .2 for c in matcher.calls)


def test_duplicate_and_stale_frames_cannot_confirm_global_position():
    matcher = Matcher([((0, 0), .01, False, .1)] * 3)
    tracker = MinimapPoseTracker(matcher)
    update = lambda seq, at, now: tracker.update(*_images(), (1, 1), sequence=seq, produced_at=at, now=now)
    assert not update(1, 0., 0.).valid
    assert not update(1, 0., .1).valid
    assert len(matcher.calls) == 1
    assert update(1, 0., .4).reason == 'frame_stale'
    assert not update(2, .5, .5).valid
    assert update(3, .6, .6).valid


def test_repeated_platform_margin_rejects_even_perfect_score():
    tracker = MinimapPoseTracker(Matcher([((0, 0), 0., False, 0.)]*3))
    for seq in range(3):
        result = tracker.update(*_images(), (1, 1), sequence=seq)
        assert not result.valid and result.reason == 'map_ambiguous'
    assert tracker.last_camera is None


def test_motion_tolerance_scales_to_capture_interval_but_is_capped():
    tracker = MinimapPoseTracker(Matcher([((0, 0), .01, False)]*2 + [((16, 0), .01, True), ((50, 0), .01, True)]))
    for seq, at in enumerate((0., .1)):
        tracker.update(*_images(), (1, 1), sequence=seq, produced_at=at, now=at)
    assert tracker.update(*_images(), (1, 1), sequence=2, produced_at=.31, now=.31).valid
    assert tracker.update(*_images(), (1, 1), sequence=3, produced_at=1., now=1.).reason == 'position_jump'


def test_initial_global_then_local_tracking_uses_last_camera_and_median():
    matcher = Matcher([
        ((10, 20), 0.05, False),
        ((10, 20), 0.05, False),
        ((11, 20), 0.04, True),
        ((12, 20), 0.03, True),
    ])
    tracker = MinimapPoseTracker(matcher)
    base, minimap = _images()

    assert not tracker.update(base, minimap, (2, 3)).valid
    first = tracker.update(base, minimap, (2, 3))
    second = tracker.update(base, minimap, (2, 3))
    third = tracker.update(base, minimap, (2, 3))

    assert first.stable_position == (12, 23)
    assert second.stable_position == (12, 23)
    assert third.stable_position == (13, 23)
    assert matcher.calls[2]["last_result"] == (10, 20)
    assert matcher.calls[3]["last_result"] == (11, 20)


def test_global_fallback_waits_three_failures_and_confirms_far_candidate_twice():
    matcher = Matcher([
        ((0, 0), 0.01, False),
        ((0, 0), 0.01, False),
        ((50, 0), 0.01, False),
        ((50, 0), 0.01, False),
        ((50, 0), 0.01, False),
        ((51, 0), 0.01, False),
        ((51, 0), 0.01, False),
    ])
    tracker = MinimapPoseTracker(matcher)
    base, minimap = _images()
    assert not tracker.update(base, minimap, (1, 1)).valid
    assert tracker.update(base, minimap, (1, 1)).valid

    results = [tracker.update(base, minimap, (1, 1)) for _ in range(5)]

    assert [result.reason for result in results] == [
        "position_jump", "position_jump", "position_jump",
        "global_unconfirmed", "ok",
    ]
    assert results[-1].source == "global_confirmed"
    assert results[-1].stable_position == (52, 1)


def test_local_single_frame_jump_is_rejected_without_polluting_history():
    matcher = Matcher([
        ((0, 0), 0.01, False),
        ((0, 0), 0.01, False),
        ((30, 0), 0.01, True),
        ((1, 0), 0.01, True),
    ])
    tracker = MinimapPoseTracker(matcher)
    base, minimap = _images()
    assert not tracker.update(base, minimap, (1, 1)).valid
    assert tracker.update(base, minimap, (1, 1)).valid
    rejected = tracker.update(base, minimap, (1, 1))
    recovered = tracker.update(base, minimap, (1, 1))

    assert not rejected.valid
    assert rejected.reason == "position_jump"
    assert recovered.valid
    assert recovered.stable_position == (2, 1)


def test_reset_and_shift_origin_update_all_accepted_coordinates():
    matcher = Matcher([((5, 6), 0.01, False)] * 2)
    tracker = MinimapPoseTracker(matcher)
    base, minimap = _images()
    tracker.update(base, minimap, (2, 3))
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
        ((0, 0), 0.01, False),
        ((40, 0), 0.01, True),
    ])
    tracker = MinimapPoseTracker(matcher)
    base, minimap = _images()
    tracker.update(base, minimap, (1, 1))
    tracker.update(base, minimap, (1, 1))
    teleported = tracker.update(
        base, minimap, (1, 1), allow_large_jump=True
    )

    assert teleported.valid
    assert teleported.raw_position == teleported.stable_position == (41, 1)
