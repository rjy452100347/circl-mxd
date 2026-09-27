"""Safety of persistent identity and the separately gated visual proposal."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.engine.PlayerLocalization import PlayerLocalizationTracker
from src.engine.PlayerVisualTracker import VisualObservation


FRAME = np.zeros((224, 300, 3), np.uint8)


def name(x=100, kind='full', score=.05):
    return SimpleNamespace(valid=True, player=(x, 100), match_kind=kind, score=score)


def player(x=100, confidence=.9, width=40, height=60):
    return dict(class_id=1, position=(x-width/2, 40), size=(width, height), confidence=confidence)


class FakeVisual:
    def __init__(self, dx=1):
        self.dx = dx
        self.reset()

    def reset(self):
        self.box = None

    def initialize(self, frame, box, **kwargs):
        self.box = box
        return True

    def update(self, frame, **kwargs):
        if self.box is None:
            return VisualObservation(reason='not initialized')
        x, y, w, h = self.box
        self.box = (x+self.dx, y, w, h)
        return VisualObservation(True, (self.dx, 0), self.box, 'synthetic LK', 20, 1., 0.)


def step(tracker, sequence, n=None, people=None, **kwargs):
    at = sequence*.1
    return tracker.update(n, [player()] if people is None else people, (0, 0, 300, 224),
                          sequence=sequence, produced_at=at, now=at, frame=FRAME,
                          minimap_valid=kwargs.pop('minimap_valid', True), **kwargs)


def bound(*, allow_visual=False, dx=1):
    tracker = PlayerLocalizationTracker(visual_tracker=FakeVisual(dx), allow_visual_control=allow_visual)
    for seq in range(5):
        result = step(tracker, seq, name())
        if seq < 4:
            assert result.foot is None and not result.control_allowed and not result.cache_safe
    assert result.bound and result.track_id == 1
    return tracker


@pytest.mark.parametrize('kind,score', [('left', .05), ('right', .05), ('full', .25)])
def test_weak_name_outliers_do_not_veto_identity_or_move_crop(kind, score):
    tracker = bound()
    for seq in range(5, 40):
        result = step(tracker, seq, name(200, kind, score))
        assert result.control_allowed and result.name_rejected and result.track_id == 1
        assert result.foot == (100, 100) and tracker.select_detection_anchor((200, 100)) == (100, 100)
        assert result.binding_resets == 0 and result.source_switches == 1


def test_far_other_player_is_not_an_identity_conflict():
    tracker = bound()
    result = step(tracker, 5, people=[player(240)])
    assert result.bound and result.track_id == 1 and not result.control_allowed
    assert result.shadow_only and '远处人物' in result.rejection_reason
    assert not tracker.conflict


def test_shadow_flow_cannot_move_authoritative_foot_even_with_current_body():
    tracker = bound(dx=3)
    for seq in range(5, 50):
        result = step(tracker, seq)
        assert result.visual_valid and result.source == 'body' and result.control_allowed
        assert result.foot == (100, 100) and tracker.anchor == (100, 100)


def test_independent_body_correction_bounds_visual_drift():
    tracker = bound(allow_visual=True)
    for seq in range(5, 100):
        result = step(tracker, seq)
        assert result.control_allowed and result.source == 'body'
        assert abs(tracker.anchor[0]-100) < .4


def test_shadow_proposal_never_authorizes_keys_or_changes_crop():
    tracker = bound()
    result = step(tracker, 5, people=[])
    assert result.visual_valid and result.observation_foot == (101, 100)
    assert result.shadow_only and result.foot is None and not result.control_allowed
    assert tracker.select_detection_anchor() == (100, 100)


@pytest.mark.parametrize('weak', [False, True])
def test_flow_and_low_confidence_cannot_self_renew_300ms_budget(weak):
    tracker = bound(allow_visual=True)
    for seq in range(5, 8):
        result = step(tracker, seq, people=[], player_candidates=[player(confidence=.4)] if weak else [])
        assert result.control_allowed and result.source == 'visual'
        assert result.bridge_age == pytest.approx((seq-4)*.1)
        assert tracker.last_body_at == pytest.approx(.4)
    result = step(tracker, 8, people=[], player_candidates=[player(confidence=.4)] if weak else [])
    assert not result.control_allowed and not result.bound and not result.cache_safe


def test_low_confidence_does_not_initialize_identity():
    tracker = PlayerLocalizationTracker(visual_tracker=FakeVisual(), allow_visual_control=True)
    for seq in range(10):
        result = step(tracker, seq, name(), [], player_candidates=[player(confidence=.4)])
        assert not result.bound and not result.control_allowed


def test_raw_high_candidate_beyond_public_top50_can_establish_identity():
    tracker = PlayerLocalizationTracker()
    for seq in range(5):
        result = step(tracker, seq, name(), [], player_candidates=[player()])
    assert result.bound and result.control_allowed and result.candidate_count == 1


@pytest.mark.parametrize('allow', [False, True])
def test_bbox_animation_cannot_displace_foot_or_renew_bridge(allow):
    tracker = bound(allow_visual=allow, dx=0)
    for seq in range(5, 8):
        result = step(tracker, seq, people=[player(height=90)])
        assert result.observation_foot == (100, 100)
        assert result.control_allowed == allow and result.shadow_only == (not allow)
        assert result.source == 'visual' and tracker.last_body_at == pytest.approx(.4)
    assert not step(tracker, 8, people=[player(height=90)]).control_allowed


def test_strong_name_conflict_pauses_immediately_and_requires_confirmation_to_reset():
    tracker = bound()
    first = step(tracker, 5, name(200))
    assert first.bound and first.phase == 'identity_check'
    assert not first.control_allowed and not first.cache_safe
    unresolved = step(tracker, 6)
    assert not unresolved.control_allowed
    second = step(tracker, 7, name(200))
    assert not second.bound and second.phase == 'conflict' and not second.cache_safe


def test_single_name_outlier_can_be_cleared_without_identity_reset():
    tracker = bound()
    assert not step(tracker, 5, name(200)).control_allowed
    result = step(tracker, 6, name())
    assert result.control_allowed and result.track_id == 1 and result.binding_resets == 0


def test_name_corrected_animation_does_not_recalibrate_raw_bottom_next_frame():
    tracker = bound(dx=0)
    result = step(tracker, 5, name(), people=[player(height=80)])
    assert result.control_allowed and result.foot == (100, 100)
    assert tracker.offset == (0, 0)
    result = step(tracker, 6, people=[player(height=80)])
    assert not result.control_allowed and result.observation_foot == (100, 100)
    assert result.shadow_only and tracker.anchor == (100, 100)
    assert tracker.reference_size == (40, 60)


def test_full_name_fallback_reseeds_at_current_body_region_not_old_background():
    tracker = bound(dx=0)
    result = step(tracker, 5, name(108), people=[])
    assert result.foot == (108, 100) and result.source == 'name'
    assert tracker.visual_tracker.box == (88, 40, 40, 60)
    result = step(tracker, 6, name(116), people=[])
    assert result.foot == (116, 100)
    assert tracker.visual_tracker.box == (96, 40, 40, 60)


def test_gradual_box_deformation_is_measured_against_calibration_not_last_frame():
    tracker = bound(dx=0)
    result = step(tracker, 5, people=[player(height=65)])
    assert result.control_allowed
    result = step(tracker, 6, people=[player(height=70)])
    assert not result.control_allowed and result.shadow_only
    assert tracker.reference_size == (40, 60)


def test_safety_pause_keeps_current_observations_but_never_authorizes_control():
    tracker = bound()
    for seq in range(5, 20):
        result = step(tracker, seq, safe=False, minimap_valid=False)
        assert result.bound and result.body_valid and not result.control_allowed
    result = step(tracker, 20)
    assert result.control_allowed and result.track_id == 1 and result.binding_resets == 0


def test_repeated_capture_does_not_accumulate_flow_or_binding():
    tracker = bound()
    first = step(tracker, 5, people=[])
    assert step(tracker, 5, people=[]).observation_foot == first.observation_foot
    assert tracker.visual_foot == (101, 100)


def test_changed_frame_geometry_clears_identity_and_visual_state():
    tracker = bound()
    result = tracker.update(None, [player()], (0, 0, 300, 224), sequence=5,
                            produced_at=.5, now=.5, minimap_valid=True,
                            frame=np.zeros((240, 320, 3), np.uint8))
    assert result.track_id is None and not result.bound and not result.control_allowed
    assert tracker.visual_foot is None
