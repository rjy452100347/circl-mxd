import copy
import cv2
import numpy as np
import pytest

from src.engine.MinimapObservation import (MinimapObserver, inspect_minimap_roi,
    player_candidates, select_player_candidate, minimap_match_mask)
from src.utils.common import get_minimap_loc_size, get_player_location_on_minimap, find_pattern_sqdiff


def frame():
    image = np.full((400, 600, 3), 90, np.uint8)
    cv2.rectangle(image, (20, 60), (240, 240), (225, 225, 225), 1)
    image[61:240, 21:240] = (45, 35, 30)
    for y in (90, 130, 170, 210):
        cv2.line(image, (35, y), (210, y), (30, 110, 50), 4)
    image[181:185, 91:95] = (0, 255, 255)
    return image


def cfg(rect=(21, 61, 219, 179)):
    return {'minimap': {'roi': [rect[0]/600, rect[1]/400, rect[2]/600, rect[3]/400],
        'player_hsv': {'lower': [20, 100, 120], 'upper': [40, 255, 255]},
        'player_color': [136, 255, 255]}}


def test_incomplete_saved_roi_reports_recommendation_without_mutation():
    config = cfg((21, 61, 180, 130))
    before = copy.deepcopy(config)
    evidence = inspect_minimap_roi(frame(), config)
    assert not evidence.valid and evidence.reason == 'roi_mismatch'
    assert evidence.rect == (21, 61, 180, 130)
    assert evidence.recommended == (21, 61, 219, 179)
    assert config == before


def test_all_consumers_share_content_rectangle_without_extra_pixel_crop():
    evidence = inspect_minimap_roi(frame(), cfg())
    assert evidence.valid
    assert get_minimap_loc_size(frame()) == evidence.rect
    observer = MinimapObserver()
    for seq in range(3):
        snapshot = observer.update(frame(), cfg(), seq, seq*.1, now=seq*.1)
    assert snapshot.valid and snapshot.roi.rect == evidence.rect
    assert snapshot.player == (72, 122)


def test_three_distinct_frames_required_and_duplicate_does_not_confirm():
    observer = MinimapObserver()
    for _ in range(4):
        assert not observer.update(frame(), cfg(), 1, .1, now=.1).valid
    assert not observer.update(frame(), cfg(), 2, .2, now=.2).valid
    assert observer.update(frame(), cfg(), 3, .3, now=.3).valid
    assert not observer.update(frame(), cfg(), 3, .3, now=.61).valid


def test_missing_or_moved_border_revokes_roi_immediately():
    observer = MinimapObserver()
    for seq in range(3):
        observer.update(frame(), cfg(), seq, seq*.1, now=seq*.1)
    damaged = frame()
    damaged[60:64, 20:241] = 90
    assert not observer.update(damaged, cfg(), 3, .3, now=.3).valid


def test_short_break_in_antialiased_border_is_tolerated():
    image = frame()
    image[60, 100:102] = 90
    assert inspect_minimap_roi(image, cfg()).valid


def test_two_maps_without_saved_roi_are_ambiguous():
    image = frame()
    image[60:241, 300:521] = image[60:241, 20:241]
    assert inspect_minimap_roi(image).reason == 'roi_ambiguous'


def test_small_point_survives_without_three_by_three_opening():
    image = np.zeros((30, 30, 3), np.uint8)
    image[8:10, 12:14] = (0, 255, 255)
    assert get_player_location_on_minimap(image, player_hsv=cfg()['minimap']['player_hsv']) == (12, 8)


def test_exact_color_disconnected_points_are_not_averaged():
    image = np.zeros((30, 40, 3), np.uint8)
    image[8:11, 5:8] = image[8:11, 25:28] = (136, 255, 255)
    assert get_player_location_on_minimap(image) is None


def test_brightness_alone_cannot_disambiguate_two_player_markers():
    hsv = np.zeros((30, 40, 3), np.uint8)
    hsv[8:11, 5:8] = (30, 255, 255)
    hsv[8:11, 25:28] = (30, 140, 140)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    candidates, _ = player_candidates(image, player_hsv=cfg()['minimap']['player_hsv'])
    assert select_player_candidate(candidates)[1] == 'dot_ambiguous'


def test_history_rejects_far_candidate_and_ambiguity_is_explained():
    candidates = [(1., (5, 5), (4, 4, 3, 3)), (1., (20, 5), (19, 4, 3, 3))]
    assert select_player_candidate(candidates)[1] == 'dot_ambiguous'
    assert select_player_candidate(candidates, (5, 5), 8)[0] == (5, 5)
    assert select_player_candidate(candidates, (100, 100), 8)[1] == 'dot_jump'


def test_matcher_local_mode_does_not_implicitly_search_global():
    rng = np.random.default_rng(12)
    base = rng.integers(0, 200, (80, 160, 3), np.uint8)
    pattern = base[20:35, 130:145].copy()
    local = find_pattern_sqdiff(base, pattern, last_result=(0, 0), search_mode='local', return_margin=True)
    glob = find_pattern_sqdiff(base, pattern, search_mode='global', return_margin=True)
    assert local[0][0] <= 50 and local[2]
    assert glob[0] == (130, 20) and not glob[2]


def test_matcher_ambiguous_repeated_platform_and_empty_mask():
    rng = np.random.default_rng(13)
    pattern = rng.integers(0, 200, (12, 15, 3), np.uint8)
    base = np.zeros((40, 90, 3), np.uint8)
    base[10:22, 10:25] = base[10:22, 60:75] = pattern
    result = find_pattern_sqdiff(base, pattern, search_mode='global', return_margin=True)
    assert result[3] < .02
    result = find_pattern_sqdiff(base, pattern, search_mode='global', return_margin=True,
                                 mask=np.zeros(pattern.shape[:2], np.uint8))
    assert not np.isfinite(result[1])


def test_matching_rejects_textureless_content_even_with_only_one_location():
    image = np.full((30, 40, 3), 60, np.uint8)
    result = find_pattern_sqdiff(image, image, mask=np.full((30, 40), 255, np.uint8),
                                 search_mode='global', return_margin=True)
    assert not np.isfinite(result[1])


def test_matching_mask_removes_player_and_border_without_mutating_image():
    image = frame()[61:240, 21:240].copy()
    before = image.copy()
    mask = minimap_match_mask(image, cfg()['minimap'], (71, 121))
    assert mask[121, 71] == 0 and not np.any(mask[:2])
    assert np.array_equal(before, image)


@pytest.mark.parametrize('point', [(22, 120), (236, 120), (110, 62), (110, 236)])
def test_small_player_at_each_content_edge(point):
    image = frame()
    image[181:185, 91:95] = (45, 35, 30)
    x, y = point
    image[y:y+2, x:x+2] = (0, 255, 255)
    observer = MinimapObserver()
    for seq in range(3):
        result = observer.update(image, cfg(), seq, seq*.1, now=seq*.1)
    assert result.valid
    assert abs(result.player[0]-(x-21)) <= 1
    assert abs(result.player[1]-(y-61)) <= 1


def test_auto_roi_movement_requires_confirmation_not_silent_tracking():
    config = cfg()
    config['minimap'].pop('roi')
    observer = MinimapObserver()
    for seq in range(3):
        observer.update(frame(), config, seq, seq*.1, now=seq*.1)
    # An intervening occlusion must not permit an automatic re-anchor.
    hidden = np.full_like(frame(), 90)
    assert not observer.update(hidden, config, 3, .3, now=.3).valid
    moved = np.full_like(frame(), 90)
    moved[:, 20:] = frame()[:, :-20]
    for seq in range(4, 8):
        result = observer.update(moved, config, seq, seq*.1, now=seq*.1)
        assert result.reason == 'roi_mismatch' and not result.valid
        assert result.roi.rect == (21, 61, 219, 179)


def test_studio_invalid_or_duplicate_frames_never_write_map_or_route():
    from types import SimpleNamespace
    from src.engine.RouteStudioSession import RouteStudioSession
    config = cfg((21, 61, 180, 130))
    config['game_window'] = {'size': [400, 600], 'title_bar_height': 0, 'resize_on_start': False}
    session = RouteStudioSession(config, map_id='example', new_map=True)
    samples = []
    session.recorder = SimpleNamespace(feed=samples.append)
    session.route_recording_active = session.map_capture_active = True
    packet = [frame(), 0., 1]
    session.capture = SimpleNamespace(get_frame_packet=lambda: packet)
    assert session.tick(now=0.) is None
    assert session.base_bgr is None and not session.map_dirty
    count = len(samples)
    assert session.tick(now=.1) is None
    assert len(samples) == count
    assert session.tick(now=.4) is None
    assert session.last_position is None
    assert not any(sample.valid for sample in samples)


def test_studio_new_map_bootstrap_uses_shared_confirmations():
    from types import SimpleNamespace
    from src.engine.RouteStudioSession import RouteStudioSession
    config = cfg()
    config['game_window'] = {'size': [400, 600], 'title_bar_height': 0, 'resize_on_start': False}
    session = RouteStudioSession(config, map_id='example', new_map=True)
    session.map_capture_active = True
    packet = [frame(), 0., 0]
    session.capture = SimpleNamespace(get_frame_packet=lambda: packet)
    for seq in range(4):
        packet[1:] = [seq*.1, seq]
        result = session.tick(now=seq*.1)
        if seq < 2:
            assert session.base_bgr is None
        if seq < 3:
            assert result is None
    assert result is not None and result['valid']
    assert session.last_position == (102, 152)
