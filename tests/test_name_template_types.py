"""Multi-template name tracking; synthetic glyphs, no private screenshots."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from src.engine.NameTagLocalizer import NameTagLocalizer
from src.engine.NameTagProfileRepository import NameTagProfileRepository, NameTagProfileError
from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.states.fixed_platform import FixedPlatformState


def templates():
    rng = np.random.default_rng(99)
    tag = np.full((20, 100, 3), 40, np.uint8)
    for col in range(4, 94, 10):
        tag[4:16, col:col+6] = np.where(rng.integers(0, 2, (12, 6, 1)), 240, 40)
    return tag, tag[:, :40].copy(), tag[:, 60:].copy()


def frame(kind='full', x=100):
    tag, left, right = templates()
    image = np.full((200, 500, 3), 85, np.uint8)
    image[80:100, x:x+100] = (0, 200, 255)
    if kind == 'full':
        image[80:100, x:x+100] = tag
    elif kind == 'left':
        image[80:100, x:x+40] = left
    elif kind == 'right':
        image[80:100, x+60:x+100] = right
    return image


def localizer():
    full, left, right = templates()
    return NameTagLocalizer([(full, (50, -10), 'full'),
                             (left, (50, -10), 'partial'),
                             (right, (-10, -10), 'partial')], max_score=.1)


def test_explicit_partial_templates_cannot_establish_initial_identity():
    loc = localizer()
    for seq in range(10):
        assert not loc.locate(frame('left' if seq%2 else 'right'), frame_sequence=seq).valid
    assert loc.locate(frame(), frame_sequence=10).valid


def test_alternating_calibrated_partial_templates_confirm_same_foot_and_stay_valid():
    loc = localizer()
    assert loc.locate(frame(), frame_sequence=0).player == (150, 70)
    assert not loc.locate(frame('left'), frame_sequence=1).valid
    # Different samples can confirm the same identity, not reset confirmation.
    result = loc.locate(frame('right'), frame_sequence=2)
    assert result.valid and result.match_kind == 'partial' and result.player == (150, 70)
    for seq in range(3, 25):
        result = loc.locate(frame('left' if seq%2 else 'right', x=100+seq), frame_sequence=seq)
        assert result.valid and result.player == (150+seq, 70)


def test_partial_duplicate_cannot_confirm_and_reset_requires_full_name_again():
    loc = localizer()
    loc.locate(frame(), frame_sequence=0)
    assert not loc.locate(frame('left'), frame_sequence=1).valid
    assert not loc.locate(frame('right'), frame_sequence=1).valid
    assert loc.locate(frame('right'), frame_sequence=2).valid
    loc.reset()
    assert not loc.locate(frame('right'), frame_sequence=3).valid


def test_partial_recovery_after_total_occlusion_requires_current_two_frames():
    loc = localizer()
    loc.locate(frame())
    loc.locate(frame('left'))
    assert loc.locate(frame('right')).valid
    for _ in range(40):
        result = loc.locate(frame('none'))
        assert not result.valid and result.player is None
    assert not loc.locate(frame('left')).valid
    assert loc.locate(frame('right')).valid


def test_partial_never_searches_far_or_selects_ambiguous_text():
    loc = localizer()
    loc.locate(frame())
    for _ in range(3):
        assert not loc.locate(frame('left', 350)).valid
    loc = localizer()
    loc.locate(frame())
    image = frame('left')
    image[120:140, 100:140] = templates()[1]
    for _ in range(3):
        assert not loc.locate(image).valid


def repo(tmp_path):
    return NameTagProfileRepository(bundled_root=tmp_path, writable_root=tmp_path)


def test_profile_types_and_offsets_roundtrip_without_modifying_images(tmp_path):
    repository = repo(tmp_path)
    stage = repository.create_stage('同一角色')
    full, left, right = templates()
    repository.stage_sample(stage, full, (50,-10))
    repository.stage_sample(stage, left, (50,-10), kind='partial')
    repository.stage_sample(stage, right, (-10,-10), kind='partial')
    path = repository.commit_profile(stage, require_enabled=True)
    data = repository.load_profile('同一角色', require_enabled=True)
    assert [e['kind'] for e in data['samples']] == ['full', 'partial', 'partial']
    assert data['samples'][2]['player_offset'] == [-10, -10]
    loc = NameTagLocalizer.from_profile(path)
    assert loc.locate(frame()).valid
    loc.locate(frame('left'))
    assert loc.locate(frame('right')).player == (150,70)


def test_old_entries_default_full_and_partial_only_cannot_apply(tmp_path):
    repository = repo(tmp_path)
    stage = repository.create_stage('兼容')
    filename = repository.stage_sample(stage, templates()[0], (50,-10))
    stage.data['samples'][0].pop('kind')
    repository.commit_profile(stage, require_enabled=True)
    assert repository.load_profile('兼容')['samples'][0]['kind'] == 'full'
    repository.stage_sample_kind(stage, filename, 'partial')
    repository.commit_profile(stage)  # Draft storage is allowed.
    with pytest.raises(NameTagProfileError, match='完整名字'):
        repository.load_profile('兼容', require_enabled=True)
    with pytest.raises(ValueError, match='完整名字'):
        NameTagLocalizer.from_profile(repository.resolve_profile('兼容'))


@pytest.mark.parametrize('kind', ['unknown', '', None])
def test_invalid_sample_type_rejected_atomically(tmp_path, kind):
    repository = repo(tmp_path)
    stage = repository.create_stage('样本')
    before = copy.deepcopy(stage.data)
    with pytest.raises(NameTagProfileError):
        repository.stage_sample(stage, templates()[1], (50,-10), kind=kind)
    assert stage.data == before and not stage.new_images


def test_partial_too_small_is_not_accepted(tmp_path):
    repository = repo(tmp_path)
    stage = repository.create_stage('样本')
    with pytest.raises(NameTagProfileError, match='24'):
        repository.stage_sample(stage, templates()[1][:,:20], (5,5), kind='partial')


def test_fixed_platform_ignores_stale_body_authority_in_both_directions():
    bot = SimpleNamespace(cfg={'fixed_platform': {'width_px': 80}},
                          current_minimap_player_valid=True, current_nametag_valid=True,
                          player_localization=SimpleNamespace(control_allowed=False))
    state = FixedPlatformState('fixed_platform', bot)
    assert state._has_current_localization()
    bot.current_nametag_valid = False
    bot.player_localization.control_allowed = True
    assert not state._has_current_localization()


def test_fixed_pipeline_uses_name_immediately_without_constructing_body_tracker():
    bot = object.__new__(MapleStoryAutoBot)
    assert bot._process_fixed_player_localization((150,70)) == (150,70)
    assert bot._process_fixed_player_localization(None) is None
    assert bot.player_localization is None
    assert not hasattr(bot, 'player_localization_tracker')
