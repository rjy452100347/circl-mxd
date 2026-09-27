import cv2
import numpy as np
import pytest

from src.engine.PlayerVisualTracker import PlayerVisualTracker, VisualObservation


def scene(x=110, y=90, *, seed=42, shape=(320, 480), blank=False):
    """Only the textured body moves; static surrounding background is unrelated."""
    frame = np.full((*shape, 3), 24, np.uint8)
    cv2.line(frame, (0, 250), (shape[1] - 1, 250), (90, 90, 90), 2)
    if not blank:
        sprite = np.random.default_rng(seed).integers(40, 230, (72, 48), dtype=np.uint8)
        frame[y:y + 72, x:x + 48] = sprite[:, :, None]
    return frame


def initialized():
    tracker = PlayerVisualTracker()
    assert tracker.initialize(scene(), (110, 90, 48, 72), sequence=1, produced_at=1.)
    return tracker


def test_textured_body_translation_has_new_evidence_and_incremental_displacement():
    tracker = initialized()
    first = tracker.update(scene(116, 93), sequence=2, produced_at=1.1)
    assert first.valid, first.reason
    assert first.delta == pytest.approx((6., 3.), abs=.15)
    assert first.box == pytest.approx((116., 93., 48., 72.), abs=.15)
    second = tracker.update(scene(119, 95), sequence=3, produced_at=1.2)
    assert second.valid, second.reason
    assert second.delta == pytest.approx((3., 2.), abs=.15)
    assert second.points >= tracker.MIN_POINTS and second.inlier_ratio >= .65
    assert second.fb_error <= tracker.MAX_FB_ERROR


def test_empty_body_cannot_initialize_or_be_tracked_as_prediction():
    tracker = PlayerVisualTracker()
    assert not tracker.initialize(scene(blank=True), (110, 90, 48, 72), sequence=1, produced_at=1.)
    tracker = initialized()
    result = tracker.update(scene(blank=True), sequence=2, produced_at=1.1)
    assert not result.valid and not tracker.active
    assert not tracker.update(scene(), sequence=3, produced_at=1.2).valid


def test_different_character_and_major_occlusion_cannot_drift_identity():
    for image in (scene(seed=202), scene()):
        tracker = initialized()
        if np.array_equal(image, scene()):
            image[100:137, 115:155] = 255
        result = tracker.update(image, sequence=2, produced_at=1.1)
        assert not result.valid and not tracker.active


def test_only_name_and_weapon_region_changes_do_not_drive_visual_motion():
    tracker = initialized()
    frame = scene(115, 90)
    # Lower name/pet band and lateral weapon effects are outside feature core.
    frame[163:185, 95:175] = 220
    frame[100:140, 90:110] = 255
    result = tracker.update(frame, sequence=2, produced_at=1.1)
    assert result.valid, result.reason
    assert result.delta == pytest.approx((5., 0.), abs=.2)


def test_repeated_sequence_and_timestamp_never_advance_tracking():
    tracker = initialized()
    old_points = tracker._points.copy()
    assert not tracker.update(scene(116), sequence=1, produced_at=1.1).valid
    assert not tracker.update(scene(116), sequence=2, produced_at=1.).valid
    assert not tracker.update(scene(116), sequence=0, produced_at=1.1).valid
    assert tracker.active and tracker._produced_at == 1.
    np.testing.assert_array_equal(tracker._points, old_points)
    assert tracker.update(scene(116), sequence=2, produced_at=1.1).valid


def test_long_frame_gap_invalidates_until_independent_reinitialization():
    tracker = initialized()
    assert tracker.update(scene(), sequence=2, produced_at=1.3).valid
    result = tracker.update(scene(), sequence=3, produced_at=1.601)
    assert not result.valid and '300 ms' in result.reason and not tracker.active
    assert not tracker.update(scene(), sequence=4, produced_at=1.7).valid
    assert tracker.initialize(scene(), (110, 90, 48, 72), sequence=4, produced_at=1.7)


def test_coordinate_change_reset_and_invalid_inputs_fail_closed():
    tracker = initialized()
    assert not tracker.update(scene(shape=(321, 480)), sequence=2, produced_at=1.1).valid
    assert not tracker.active
    for frame, box in [(None, (1, 1, 20, 30)), (scene(), (-1, 0, 48, 72)),
                       (scene(), (450, 90, 48, 72)), (scene(), (110, 90, 8, 9)),
                       (scene(), (110, 90, float('nan'), 72))]:
        assert not tracker.initialize(frame, box, sequence=1, produced_at=1.)
    tracker = initialized()
    assert not tracker.update(scene(), sequence=2, produced_at=float('nan')).valid


def test_valid_near_boundary_and_bounded_owned_memory():
    tracker = PlayerVisualTracker()
    original = scene(0, 1, shape=(1200, 2051))
    assert tracker.initialize(original, (0, 1, 48, 72), sequence=1, produced_at=1.)
    assert tracker._gray.ndim == 2 and max(tracker._gray.shape) <= 384
    assert not np.shares_memory(tracker._gray, original)
    assert not np.shares_memory(tracker._reference, original)
    original[:] = 0  # Capturer may release/mutate its own buffer after initialization.
    result = tracker.update(scene(4, 3, shape=(1200, 2051)), sequence=2, produced_at=1.1)
    assert result.valid, result.reason
    assert result.delta == pytest.approx((4., 2.), abs=.2)
    assert sum(a.nbytes for a in vars(tracker).values() if isinstance(a, np.ndarray)) < 384 * 384 * 2 + 1024
    tracker.reset()
    assert not tracker.active and tracker._reference is None and tracker._points is None


def test_single_textured_corner_is_not_sufficient_spatial_support():
    frame = scene(blank=True)
    frame[98:108, 122:132] = np.random.default_rng(8).integers(0, 255, (10, 10, 3), dtype=np.uint8)
    tracker = PlayerVisualTracker()
    assert not tracker.initialize(frame, (110, 90, 48, 72), sequence=1, produced_at=1.)


def test_observation_is_immutable():
    observation = VisualObservation()
    with pytest.raises(AttributeError):
        observation.valid = True


def test_lk_success_flag_alone_never_proves_identity(monkeypatch):
    tracker = initialized()

    def false_success(previous, current, points, prediction, **kwargs):
        # A library returning all status=1 cannot bypass appearance comparison.
        return points.copy(), np.ones((len(points), 1), np.uint8), np.zeros((len(points), 1), np.float32)

    monkeypatch.setattr(cv2, 'calcOpticalFlowPyrLK', false_success)
    result = tracker.update(scene(seed=999), sequence=2, produced_at=1.1)
    assert not result.valid and '外观' in result.reason and not tracker.active


def test_fixed_reference_is_not_relearned_by_visual_updates():
    tracker = initialized()
    reference = tracker._reference.copy()
    # Pure flow is not allowed to continuously renew the template or its identity.
    for sequence in range(2, 5):
        result = tracker.update(scene(110 + sequence - 1, 90),
                                sequence=sequence, produced_at=1. + (sequence - 1) * .1)
        assert result.valid, result.reason
        np.testing.assert_array_equal(tracker._reference, reference)


def test_only_one_visible_character_part_is_rejected():
    tracker = initialized()
    partial = scene(blank=True)
    original = scene()
    partial[98:117, 120:147] = original[98:117, 120:147]
    result = tracker.update(partial, sequence=2, produced_at=1.1)
    assert not result.valid and not tracker.active


def test_lk_failure_clears_retained_visual_arrays(monkeypatch):
    tracker = initialized()

    def unavailable(*args, **kwargs):
        return None, None, None

    monkeypatch.setattr(cv2, 'calcOpticalFlowPyrLK', unavailable)
    assert not tracker.update(scene(), sequence=2, produced_at=1.1).valid
    assert all(not isinstance(value, np.ndarray) for value in vars(tracker).values())


@pytest.mark.parametrize('body_width', [6, 8, 10, 12, 14, 18])
@pytest.mark.parametrize('moved', [False, True], ids=['body_disappears', 'body_moves_background_stays'])
def test_textured_background_cannot_replace_missing_low_texture_body(body_width, moved):
    background = np.random.default_rng(40).integers(40, 220, (320, 480, 3), dtype=np.uint8)
    initial = background.copy()
    initial[101:138, 134 - body_width // 2:134 + body_width // 2] = 120
    current = background.copy()
    if moved:
        # Only the person moves.  Following old static background with delta=0
        # would claim the old person position remains supported.
        current[101:138, 154 - body_width // 2:154 + body_width // 2] = 120
    tracker = PlayerVisualTracker()
    ready = tracker.initialize(initial, (110, 90, 48, 72), sequence=1, produced_at=1.)
    result = tracker.update(current, sequence=2, produced_at=1.1)
    assert not result.valid and not tracker.active
    # Initialization is allowed to reject insufficient central texture outright.
    assert not ready or result.reason


def test_textured_body_moving_over_static_textured_background_tracks_body_not_background():
    background = np.random.default_rng(12).integers(20, 235, (320, 480, 3), dtype=np.uint8)
    sprite = scene()[90:162, 110:158].copy()
    initial = background.copy()
    initial[90:162, 110:158] = sprite
    current = background.copy()
    current[93:165, 114:162] = sprite
    tracker = PlayerVisualTracker()
    assert tracker.initialize(initial, (110, 90, 48, 72), sequence=1, produced_at=1.)
    result = tracker.update(current, sequence=2, produced_at=1.1)
    assert result.valid, result.reason
    assert result.delta == pytest.approx((4., 3.), abs=.2)


def test_textureless_center_rejects_initialization_despite_textured_periphery():
    frame = scene()
    frame[96:141, 129:139] = 120
    tracker = PlayerVisualTracker()
    assert not tracker.initialize(frame, (110, 90, 48, 72), sequence=1, produced_at=1.)


def test_motion_gate_uses_actual_sub_100ms_interval():
    tracker = initialized()
    result = tracker.update(scene(114), sequence=2, produced_at=1.01)
    assert not result.valid and '位移异常' in result.reason
    tracker = initialized()
    # The same observed displacement fits the 32px / 100ms budget at 20ms.
    assert tracker.update(scene(114), sequence=2, produced_at=1.02).valid
