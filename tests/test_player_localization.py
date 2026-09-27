from types import SimpleNamespace
import pytest
from src.engine.PlayerLocalization import PlayerLocalizationTracker


def name(x=100, kind='full'):
    return SimpleNamespace(valid=True, player=(x, 100), match_kind=kind)


def player(x=100, confidence=.9):
    return dict(class_id=1, position=(x-20, 40), size=(40, 60), confidence=confidence)


def update(t, seq, n=None, players=None, **kw):
    return t.update(n, [player()] if players is None else players, (0, 0, 300, 224),
                    sequence=seq, produced_at=seq*.1, now=seq*.1,
                    minimap_valid=kw.pop('minimap_valid', True), **kw)


def bound():
    t = PlayerLocalizationTracker()
    for i in range(5):
        s = update(t, i, name())
        assert s.bound == (i == 4)
    return t


def test_full_name_required_and_binding_needs_distinct_frames():
    t = PlayerLocalizationTracker()
    assert not update(t, 0).control_allowed
    assert not update(t, 1, name(kind='left')).bound
    for _ in range(5):
        assert update(t, 2, name()).binding_frames == 1
    assert not t.snapshot.bound


def test_complete_occlusion_can_continue_fresh_body_tracking_without_ttl():
    t = bound()
    for i in range(5, 650):
        s = update(t, i)
        assert s.source == 'body' and s.control_allowed and s.foot == (100, 100)
    assert update(t, 650, name(kind='right')).source == 'body'


def test_fragment_confirmation_gap_does_not_interrupt_body():
    t = bound()
    for i, n in enumerate([None, name(kind='left'), None, name(kind='right'), name()], 5):
        assert update(t, i, n).control_allowed


def test_body_is_primary_even_when_names_flicker_or_change_fragment():
    t = bound()
    for i, n in enumerate([name(), None, name(kind='left'), None, name(kind='right')], 5):
        s = update(t, i, n)
        assert s.source == 'body' and s.control_allowed
        assert t.select_detection_anchor((110, 100)) == (100, 100)


def test_binding_accumulates_full_names_in_one_second_window():
    t = PlayerLocalizationTracker()
    for i in range(9):
        s = update(t, i, name() if i % 2 == 0 else name(kind='left'))
    assert s.bound and s.source == 'body'


def test_binding_success_clears_old_recovery_state():
    t = PlayerLocalizationTracker()
    update(t, 0, minimap_valid=False)
    for i in range(1, 6):
        s = update(t, i, name())
    assert s.bound and s.body_valid and not t.missed and t.recovery == 0


def test_recovery_first_observation_refreshes_clock_but_not_control():
    t = bound()
    assert not update(t, 5, players=[]).control_allowed
    first = update(t, 7)
    assert first.bound and not first.control_allowed
    assert t.last_body_at == pytest.approx(.7)
    assert update(t, 8).control_allowed


def test_minimap_pause_tracks_body_without_rebinding_or_control():
    t = bound()
    for i in range(5, 20):
        s = update(t, i, minimap_valid=False)
        assert s.bound and s.body_valid and not s.control_allowed
    assert update(t, 20).source == 'body'


def test_name_fallback_needs_two_current_frames_then_body_needs_two():
    t = bound()
    update(t, 5)  # Break the name streak.
    assert not update(t, 6, name(), players=[]).control_allowed
    s = update(t, 7, name(), players=[])
    assert s.source == 'name' and s.control_allowed
    assert update(t, 8, name()).source == 'name'
    assert update(t, 9, name()).source == 'body'


def test_binding_window_does_not_accumulate_old_samples():
    t = PlayerLocalizationTracker()
    for i in range(20):
        s = update(t, i, name() if i % 3 == 0 else None)
        assert not s.bound


def test_binding_gap_over_300ms_requires_new_samples():
    t = PlayerLocalizationTracker()
    update(t, 0, name())
    update(t, 1, name())
    assert update(t, 5, name()).binding_frames == 1


def test_initial_binding_not_authorized_during_safety_pause():
    t = PlayerLocalizationTracker()
    for i in range(8):
        assert not update(t, i, name(), safe=False).bound


def test_movement_pause_counters_and_observation_permission_are_independent():
    t = bound()
    update(t, 5, safe=False)
    s = update(t, 6, safe=False)
    assert s.body_valid and s.observation_foot == (100, 100) and s.foot is None
    assert s.pause_count == 1 and s.binding_resets == 0
    s = update(t, 7)
    assert s.control_allowed and s.pause_seconds == pytest.approx(.2)
    assert s.source_switches == 1  # Initial binding is stationary; pauses don't switch.


def test_duplicate_recovery_and_safety_reenable_need_a_new_frame():
    t = bound()
    update(t, 5, players=[])
    s = update(t, 6)
    assert s.recovery_frames == 1
    assert update(t, 6).recovery_frames == 1
    assert update(t, 7).body_valid
    assert not update(t, 7, safe=False).control_allowed
    assert not update(t, 7).control_allowed
    assert update(t, 8).control_allowed


def test_preconfirmed_name_fallback_is_immediate_but_never_replayed():
    t = bound()
    assert update(t, 5, name(), players=[]).source == 'name'
    assert not update(t, 6, players=[]).control_allowed


def test_reset_forgets_binding_source_counts_and_crop_priority():
    t = bound()
    t.reset()
    assert not t.snapshot.bound and not t.snapshot.control_allowed
    assert t.select_detection_anchor() is None
    assert t.select_detection_anchor((200, 100)) == (200, 100)
    assert not update(t, 0).control_allowed


def test_binding_ambiguous_or_jumping_candidate_clears_window():
    t = PlayerLocalizationTracker()
    for i in range(3):
        update(t, i, name())
    assert update(t, 3, name(), [player(), player(105)]).binding_frames == 0
    update(t, 4, name())
    assert update(t, 5, name(200), [player(200)]).binding_frames == 0


def test_distant_name_cannot_grab_bound_crop_or_override_identity_conflict():
    t = bound()
    assert t.select_detection_anchor((200, 100)) == (100, 100)
    s = update(t, 5, name(200))
    assert not s.control_allowed and not s.cache_safe and s.bound
    assert s.phase == 'identity_check'
    assert update(t, 6, name()).control_allowed  # One outlier is not an identity reset.


@pytest.mark.parametrize('players,reason', [([], '没有输出人物框'),
    ([player(confidence=.64)], '置信度不足'),
    ([dict(class_id=1, confidence=.9, position=(0, 40), size=(100, 60))], '裁剪边缘')])
def test_rejection_diagnostics_distinguish_observed_causes(players, reason):
    s = update(bound(), 5, players=players)
    assert reason in s.rejection_reason


@pytest.mark.parametrize('players', [[player(confidence=.64)], [],
                                    [player(), player(110)], [player(220)]])
def test_bad_body_does_not_move_or_select_nearest(players):
    t = bound()
    assert not update(t, 5, players=players).control_allowed


def test_conflict_blocks_name_until_full_rebinding():
    t = bound()
    s = update(t, 5, name(125))
    assert not s.control_allowed and not s.cache_safe
    assert not update(t, 6, name(125)).bound
    for i in range(7, 11):
        assert not update(t, i, name()).control_allowed
    assert update(t, 11, name()).control_allowed


def test_short_missing_needs_two_new_frames_and_long_loss_requires_name():
    t = bound()
    assert not update(t, 5, players=[]).control_allowed
    assert not update(t, 6).control_allowed
    assert update(t, 7).control_allowed
    assert not update(t, 11).control_allowed
    assert not t.snapshot.bound


def test_minimap_and_safety_gate_release_without_old_success():
    t = bound()
    assert not update(t, 5, minimap_valid=False).control_allowed
    assert not update(t, 6, safe=False).control_allowed
    t.reset()
    assert not update(t, 7).control_allowed


def test_crop_edges_and_size_changes_rejected():
    t = bound()
    edge = dict(class_id=1, confidence=.9, position=(0, 40), size=(100, 60))
    assert not update(t, 5, players=[edge]).control_allowed
    wide = dict(class_id=1, confidence=.9, position=(10, 40), size=(180, 60))
    assert not update(t, 6, players=[wide]).control_allowed


def test_stale_duplicate_cannot_keep_control():
    t = bound()
    assert not t.update(None, [player()], (0, 0, 300, 224), sequence=4,
                        produced_at=.4, now=.8, minimap_valid=True).control_allowed


def test_healing_tracks_fresh_body_without_control_and_resumes_without_rebind():
    t = bound()
    for i in range(5, 30):
        s = update(t, i, safe=False)
        assert s.bound and not s.control_allowed
    assert update(t, 30).source == 'body'
    assert t.snapshot.control_allowed


def test_unstable_calibration_offsets_are_not_bound():
    t = PlayerLocalizationTracker()
    reasons = []
    for i, x in enumerate((80, 120, 80, 120, 80)):
        s = update(t, i, name(x))
        assert not s.bound
        reasons.append(s.reason)
    assert any('校正不稳定' in reason for reason in reasons)


def test_side_pet_misclassified_as_player_cannot_bind_by_proximity_alone():
    t = PlayerLocalizationTracker()
    for i in range(6):
        s = update(t, i, name(), players=[player(145)])
        assert not s.bound


def test_metrics_ignore_duplicate_frames_and_measure_outage():
    from src.engine.PlayerLocalization import PetTrackingMetrics
    t = bound()
    metrics = PetTrackingMetrics()
    bad = update(t, 5, players=[])
    metrics.add(5, .5, False, bad, 1)
    metrics.add(5, .6, False, bad, 1)
    metrics.observe_availability(1.5, False)
    assert metrics.frames == 1 and metrics.longest == 1.
    assert 'P95 1.00ms' in metrics.summary()
