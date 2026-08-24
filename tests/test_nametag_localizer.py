import cv2
import numpy as np

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
