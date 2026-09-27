"""Anonymous, generated HUD fixtures: no account screenshots or private assets."""
import cv2
import numpy as np
import pytest

from src.engine.HealthMonitor import HealthBarRecognizer, HEALTH_READY, HEALTH_LOST


def connected_hud(width=1282, hp=40, mp=70):
    image = np.full((148, 1282, 3), 35, np.uint8)
    # A single full-width outer decoration, unlike the legacy 758 px panel.
    cv2.rectangle(image, (0, 78), (1281, 147), (230, 230, 230), 2)
    for x, value, color in ((436, hp, (0, 0, 220)),
                             (594, mp, (220, 80, 0)),
                             (759, 65, (0, 220, 220))):
        cv2.rectangle(image, (x, 127), (x + 156, 143), (160, 160, 160), 1)
        image[128:143, x + 1:x + 156] = 55
        fill = round(155 * value / 100)
        image[128:143, x + 1:x + 1 + fill] = color
    if width != 1282:
        image = cv2.resize(image, (width, round(148 * width / 1282)))
    return image


def calibrate(recognizer, image):
    for sequence in range(1, 4):
        result = recognizer.recognize(image, sequence, sequence / 10)
    return result


@pytest.mark.parametrize('width', [1282, 2049])
@pytest.mark.parametrize('hp,mp', [(100, 100), (40, 70), (5, 0)])
def test_connected_panel_uses_track_width_not_colored_width(width, hp, mp):
    result = calibrate(HealthBarRecognizer(), connected_hud(width, hp, mp))
    assert result.state == HEALTH_READY
    assert result.hp_percent == pytest.approx(hp, abs=4)
    assert result.mp_percent == pytest.approx(mp, abs=4)
    assert result.hp_rect[2] == pytest.approx(157 * width / 1282, abs=5)


def test_duplicate_capture_cannot_complete_calibration():
    recognizer = HealthBarRecognizer()
    image = connected_hud()
    for _ in range(5):
        assert recognizer.recognize(image, 1, .1).state != HEALTH_READY
    assert recognizer.recognize(image, 2, .2).state != HEALTH_READY
    assert recognizer.recognize(image, 3, .3).state == HEALTH_READY


def test_locked_mp_obstruction_does_not_invalidate_hp():
    recognizer = HealthBarRecognizer()
    image = connected_hud()
    calibrate(recognizer, image)
    image[124:146, 593:753] = 35
    result = recognizer.recognize(image, 4, .4)
    assert result.role_valid('hp')
    assert not result.role_valid('mp')
    assert result.mp_percent is None
    assert result.hp_percent == pytest.approx(40, abs=4)


def test_coordinate_context_change_requires_recalibration():
    recognizer = HealthBarRecognizer()
    image = connected_hud()
    calibrate(recognizer, image)
    assert recognizer.recognize(image, 4, .4, (0, 900)).state != HEALTH_READY


def test_uniform_erasure_is_not_zero_hp():
    recognizer = HealthBarRecognizer()
    image = connected_hud()
    calibrate(recognizer, image)
    image[124:146, 434:594] = 55
    result = recognizer.recognize(image, 4, .4)
    assert not result.role_valid('hp')
    assert result.hp_percent is None


def test_locked_empty_hp_requires_intact_track_and_neighbor():
    recognizer = HealthBarRecognizer()
    calibrate(recognizer, connected_hud())
    result = recognizer.recognize(connected_hud(hp=0), 4, .4)
    assert result.role_valid('hp')
    assert result.hp_percent == pytest.approx(0, abs=2)


def test_red_blue_fill_without_track_borders_is_not_hud():
    image = connected_hud()
    for x in (436, 594, 759):
        image[127, x:x+157] = 35
        image[143, x:x+157] = 35
        image[127:144, x] = 35
        image[127:144, x+156] = 35
    assert calibrate(HealthBarRecognizer(), image).state != HEALTH_READY


def test_two_equally_plausible_huds_are_not_guessed():
    image = connected_hud()
    image[25:42, 436:916] = image[127:144, 436:916]
    assert calibrate(HealthBarRecognizer(), image).state != HEALTH_READY


def test_moving_fill_does_not_move_locked_track():
    recognizer = HealthBarRecognizer()
    ready = calibrate(recognizer, connected_hud(hp=100))
    for seq, value in enumerate([80, 40, 10, 0, 100], 4):
        result = recognizer.recognize(connected_hud(hp=value), seq, seq / 10)
        assert result.hp_rect == ready.hp_rect
        assert result.hp_percent == pytest.approx(value, abs=4)


def test_channel_recovery_requires_three_distinct_good_frames():
    recognizer = HealthBarRecognizer()
    image = connected_hud()
    calibrate(recognizer, image)
    blocked = image.copy()
    blocked[124:146, 593:753] = 35
    for seq in range(4, 9):
        result = recognizer.recognize(blocked, seq, seq / 10)
        assert result.role_valid('hp') and not result.role_valid('mp')
    for seq in (9, 10):
        result = recognizer.recognize(image, seq, seq / 10)
        assert result.role_valid('hp') and not result.role_valid('mp')
    assert recognizer.recognize(image, 11, 1.1).role_valid('mp')


def test_both_channels_lost_discard_layout_and_recalibrate():
    recognizer = HealthBarRecognizer()
    calibrate(recognizer, connected_hud())
    for seq in range(4, 7):
        result = recognizer.recognize(np.full((148, 1282, 3), 55, np.uint8), seq, seq / 10)
        assert not result.role_valid('hp') and not result.role_valid('mp')
    assert recognizer.locked_rects is None
    for seq in (7, 8):
        assert not recognizer.recognize(connected_hud(), seq, seq / 10).role_valid('hp')
    assert recognizer.recognize(connected_hud(), 9, .9).role_valid('hp')


def test_empty_hp_before_initial_identity_is_not_accepted():
    assert not calibrate(HealthBarRecognizer(), connected_hud(hp=0)).role_valid('hp')


def test_roi_truncation_does_not_create_a_zero_reading():
    result = calibrate(HealthBarRecognizer(), connected_hud()[:139])
    assert not result.role_valid('hp') and result.hp_percent is None
