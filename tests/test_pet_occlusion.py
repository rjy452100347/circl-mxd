import cv2
import numpy as np

from src.engine.NameTagLocalizer import NameTagLocalizer
from src.engine.NameTagTracking import NameTagTracking


def scene(side=None, x=100):
    rng = np.random.default_rng(45210)
    tag = np.full((20, 100, 3), 40, np.uint8)
    # Distinctive, deterministic synthetic glyphs; no personal samples.
    for col in range(5, 95, 10):
        bits = rng.integers(0, 2, (12, 6), dtype=np.uint8)
        tag[4:16, col:col+6] = np.where(bits[..., None], 240, 40)
    frame = np.full((200, 500, 3), 85, np.uint8)
    frame[80:100, x:x+100] = tag
    if side == 'left':
        frame[80:100, x:x+38] = (0, 200, 255)
    elif side == 'right':
        frame[80:100, x+62:x+100] = (0, 200, 255)
    elif side == 'all':
        frame[80:100, x:x+100] = (0, 200, 255)
    return tag, frame


def test_partial_requires_full_identity_and_distinct_frames():
    tag, covered = scene('left')
    loc = NameTagLocalizer([(tag, (50, -10))], max_score=.2)
    assert not loc.locate(covered, frame_sequence=1).valid
    assert loc.locate(scene()[1], frame_sequence=2).valid
    assert not loc.locate(covered, frame_sequence=3).valid
    assert not loc.locate(covered, frame_sequence=3).valid
    result = loc.locate(covered, frame_sequence=4)
    assert result.valid and result.match_kind == 'right'
    assert result.player == (150, 70)
    assert result.tag_top_left == (100, 80)


def test_both_sides_and_full_occlusion_recovery():
    tag, frame = scene()
    loc = NameTagLocalizer([(tag, (50, -10))], max_score=.2)
    assert loc.locate(frame).valid
    for covered, expected in [('right', 'left'), ('left', 'right')]:
        first = loc.locate(scene(covered)[1])
        assert first.valid == (expected == 'right')  # Confirmed fragments switch without a forced gap.
        result = loc.locate(scene(covered)[1])
        assert result.valid and result.match_kind == expected
        assert result.player == (150, 70)
    for _ in range(70):
        result = loc.locate(scene('all')[1])
        assert not result.valid and result.player is None
    assert not loc.locate(frame).valid
    assert loc.locate(frame).valid


def test_full_only_validation_cannot_be_rescued_by_partial():
    tag, frame = scene()
    loc = NameTagLocalizer([(tag, (50, -10))], max_score=.2, allow_partial=False)
    assert loc.locate(frame).valid
    for _ in range(3):
        assert not loc.locate(scene('right')[1]).valid


def test_cache_has_no_ttl_but_displacement_latches_until_name_recovers():
    track = NameTagTracking()
    track.update((150, 70), (30, 40), now=0)
    for now in (2, 5, 60, 600):
        snap = track.update(None, (31, 40), now=now)
        assert snap.anchor == (150, 70) and snap.attack_allowed
    assert not track.update(None, (33, 40), now=601).attack_allowed
    assert not track.update(None, (30, 40), now=602).attack_allowed
    track.update((151, 70), (30, 40), now=603)
    assert track.update(None, (30, 40), now=604).attack_allowed


def test_cache_minimap_loss_latches_healing_does_not_and_reset_clears():
    track = NameTagTracking()
    assert track.update(None, (30, 40), now=0).anchor is None
    track.update((150, 70), (30, 40), now=1)
    assert not track.update(None, (30, 40), now=2, inhibit='强制补血').attack_allowed
    assert track.update(None, (30, 40), now=3).attack_allowed
    assert not track.update(None, None, now=4).attack_allowed
    assert not track.update(None, (30, 40), now=5).attack_allowed
    track.reset()
    assert track.update(None, (30, 40), now=6).anchor is None


def test_identical_local_candidates_are_rejected_not_chosen_by_confidence():
    tag, full = scene()
    loc = NameTagLocalizer([(tag, (50, -10))], max_score=.2)
    loc.locate(full)
    frame = scene('right')[1]
    frame[115:135, 100:160] = tag[:, :60]
    for _ in range(3):
        assert not loc.locate(frame).valid


def test_left_right_disagreement_is_rejected():
    tag, full = scene()
    loc = NameTagLocalizer([(tag, (50, -10))], max_score=.2)
    loc.locate(full)
    frame = scene('right')[1]
    frame[115:135, 140:200] = tag[:, 40:]
    assert loc._partial_match(frame) is None


def test_switching_template_origin_uses_calibrated_foot_not_top_left():
    tag, full = scene()
    rng = np.random.default_rng(77)
    second = rng.choice([40, 240], (20, 100, 1)).astype(np.uint8).repeat(3, axis=2)
    loc = NameTagLocalizer([(tag, (50, -10)), (second, (150, -10))], max_score=.1, max_jump=20)
    assert loc.locate(full).player == (150, 70)
    frame = np.full_like(full, 85)
    frame[80:100, :100] = second
    result = loc.locate(frame)
    assert result.valid and result.player == (150, 70) and result.sample_index == 1


def test_short_template_and_outside_foot_and_reset_are_safe():
    tag, full = scene()
    short = NameTagLocalizer([(tag[:, :30], (15, -10))])
    assert short.fragments == []
    invalid = NameTagLocalizer([(tag, (50, -200))])
    assert not invalid.locate(full).valid
    loc = NameTagLocalizer([(tag, (50, -10))], max_score=.2)
    loc.locate(full)
    loc.reset()
    for _ in range(3):
        assert not loc.locate(scene('left')[1]).valid


def test_partial_search_cannot_follow_a_far_away_fragment():
    tag, full = scene()
    loc = NameTagLocalizer([(tag, (50, -10))], max_score=.2)
    loc.locate(full)
    for _ in range(3):
        assert not loc.locate(scene('left', x=350)[1]).valid
