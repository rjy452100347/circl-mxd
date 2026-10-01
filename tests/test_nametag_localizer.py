import cv2
import numpy as np
from types import SimpleNamespace

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.engine.NameTagLocalizer import NameTagLocalizer


def _tag():
    image = np.full((24, 90, 3), 45, dtype=np.uint8)
    cv2.putText(
        image, "PLAYER", (3, 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (245, 245, 245), 1, cv2.LINE_AA,
    )
    return image


def _frame(tag, point, shape=(180, 500, 3)):
    frame = np.full(shape, 80, dtype=np.uint8)
    x, y = point
    h, w = tag.shape[:2]
    frame[y:y + h, x:x + w] = tag
    return frame


def test_nametag_localizer_returns_calibrated_player_point():
    tag = _tag()
    localizer = NameTagLocalizer([(tag, (45, -12))], max_score=0.2)

    result = localizer.locate(_frame(tag, (80, 60)))

    assert result.valid
    assert result.tag_top_left == (80, 60)
    assert result.player == (125, 48)


def test_nametag_localizer_uses_local_search_for_normal_motion():
    tag = _tag()
    localizer = NameTagLocalizer(
        [(tag, (45, -12))], max_score=0.2, local_search_radius=30,
        global_refresh_frames=100,
    )
    assert localizer.locate(_frame(tag, (80, 60))).valid

    moved = localizer.locate(_frame(tag, (96, 65)))

    assert moved.valid
    assert moved.tag_top_left == (96, 65)


def test_nametag_localizer_returns_unknown_instead_of_stale_position():
    tag = _tag()
    localizer = NameTagLocalizer([(tag, (45, -12))], max_score=0.2)
    assert localizer.locate(_frame(tag, (80, 60))).valid

    missing = localizer.locate(np.full((180, 500, 3), 80, dtype=np.uint8))

    assert not missing.valid
    assert missing.player is None
    assert missing.reason == "not_found"


def test_large_position_jump_requires_two_matching_frames():
    tag = _tag()
    localizer = NameTagLocalizer(
        [(tag, (45, -12))], max_score=0.2,
        max_jump=50, jump_confirm_frames=2, global_refresh_frames=1,
    )
    assert localizer.locate(_frame(tag, (20, 60))).valid

    first = localizer.locate(_frame(tag, (300, 60)))
    second = localizer.locate(_frame(tag, (300, 60)))

    assert not first.valid
    assert first.reason == "jump_unconfirmed"
    assert second.valid
    assert second.tag_top_left == (300, 60)


def test_normal_tracking_preprocesses_only_local_roi(monkeypatch):
    tag = _tag()
    localizer = NameTagLocalizer(
        [(tag, (45, -12))], max_score=0.2, local_search_radius=30,
        global_refresh_frames=30,
    )
    assert localizer.locate(_frame(tag, (80, 60))).valid

    calls = []
    original = cv2.cvtColor

    def recording_cvt_color(image, code):
        calls.append(image.shape)
        return original(image, code)

    monkeypatch.setattr(cv2, "cvtColor", recording_cvt_color)
    result = localizer.locate(_frame(tag, (96, 65)))

    assert result.valid
    assert calls
    assert all(shape[0] < 180 and shape[1] < 500 for shape in calls)


def test_local_failure_falls_back_to_full_frame_preprocessing(monkeypatch):
    tag = _tag()
    localizer = NameTagLocalizer(
        [(tag, (45, -12))], max_score=0.2, local_search_radius=20,
        global_refresh_frames=30, max_jump=1000,
    )
    assert localizer.locate(_frame(tag, (30, 60))).valid

    calls = []
    original = cv2.cvtColor

    def recording_cvt_color(image, code):
        calls.append(image.shape)
        return original(image, code)

    monkeypatch.setattr(cv2, "cvtColor", recording_cvt_color)
    result = localizer.locate(_frame(tag, (300, 60)))

    assert result.valid
    assert any(shape[:2] == (180, 500) for shape in calls)
    assert any(shape[0] < 180 and shape[1] < 500 for shape in calls)


def test_configured_refresh_frame_uses_global_preprocessing(monkeypatch):
    tag = _tag()
    localizer = NameTagLocalizer(
        [(tag, (45, -12))], max_score=0.2,
        local_search_radius=30, global_refresh_frames=2,
    )
    assert localizer.locate(_frame(tag, (80, 60))).valid

    calls = []
    original = cv2.cvtColor

    def recording_cvt_color(image, code):
        calls.append(image.shape)
        return original(image, code)

    monkeypatch.setattr(cv2, "cvtColor", recording_cvt_color)
    assert localizer.locate(_frame(tag, (82, 60))).valid

    assert calls
    assert all(shape[:2] == (180, 500) for shape in calls)


def test_bot_uses_dedicated_nametag_search_boundary_instead_of_health_ui():
    bot = object.__new__(MapleStoryAutoBot)
    bot.img_frame = np.zeros((1200, 500, 3), dtype=np.uint8)
    bot.img_frame_debug = None
    bot.cfg = {
        "nametag": {"search_y_limit": 1040},
        "ui_coords": {"ui_y_start": 915},
    }
    bot.nametag_last_result = None
    bot.loc_nametag = None
    received = []

    class Localizer:
        def locate(self, _frame, *, y_limit):
            received.append(y_limit)
            return SimpleNamespace(valid=False)

    bot.nametag_localizer = Localizer()

    assert bot.get_player_location_by_nametag() is None
    assert received == [1040]


def test_bot_nametag_boundary_keeps_legacy_ui_fallback_and_clips_to_frame():
    bot = object.__new__(MapleStoryAutoBot)
    bot.img_frame = np.zeros((700, 500, 3), dtype=np.uint8)
    bot.img_frame_debug = None
    bot.cfg = {
        "nametag": {"search_y_limit": 0},
        "ui_coords": {"ui_y_start": 915},
    }
    bot.nametag_last_result = None
    bot.loc_nametag = None
    received = []

    class Localizer:
        def locate(self, _frame, *, y_limit):
            received.append(y_limit)
            return SimpleNamespace(valid=False)

    bot.nametag_localizer = Localizer()

    assert bot.get_player_location_by_nametag() is None
    assert received == [700]
