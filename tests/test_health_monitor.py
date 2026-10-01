from types import SimpleNamespace
import threading
import time

import cv2
import numpy as np
import pytest

from src.engine.HealthMonitor import (
    HEALTH_CALIBRATING,
    HEALTH_LOST,
    HEALTH_READY,
    HealthBarRecognizer,
    HealthMonitor,
    HealthSnapshot,
)


def _monitor():
    cfg = {
        "health_monitor": {
            "auto_hp_enabled": True,
            "auto_mp_enabled": True,
            "fps_limit": 20,
            "add_hp_percent": 40,
            "add_mp_percent": 20,
            "add_hp_cooldown": 1,
            "add_mp_cooldown": 1,
            "return_home_watch_dog_timeout": 10,
            "force_heal": False,
            "return_home_if_no_potion": False,
        },
        "key": {"add_hp": "home", "add_mp": "end", "return_home": "x"},
    }
    requests = []
    clears = []
    kb = SimpleNamespace(
        requests=requests,
        clears=clears,
        request_aux_action=lambda **request: requests.append(request),
        clear_aux_actions=lambda actions=None: clears.append(actions),
    )
    return HealthMonitor(cfg, kb)


def test_health_monitor_owns_compact_roi_and_processes_sequence_once():
    monitor = _monitor()
    source = np.zeros((12, 30, 3), dtype=np.uint8)

    monitor.update_frame(source, frame_sequence=7)
    source[:] = 255
    owned = monitor._take_latest_frame()

    assert owned.shape == source.shape
    assert not np.shares_memory(owned, source)
    assert not np.any(owned)
    assert monitor._take_latest_frame() is None


def test_health_monitor_ignores_duplicate_sequence_and_accepts_newer_frame():
    monitor = _monitor()
    monitor.update_frame(np.zeros((4, 5, 3), dtype=np.uint8), frame_sequence=3)
    first = monitor._take_latest_frame()
    monitor.update_frame(np.ones((4, 5, 3), dtype=np.uint8), frame_sequence=3)

    assert not np.any(first)
    assert monitor._take_latest_frame() is None

    monitor.update_frame(np.ones((4, 5, 3), dtype=np.uint8), frame_sequence=4)
    assert np.all(monitor._take_latest_frame() == 1)

    monitor.update_frame(np.full((4, 5, 3), 2, dtype=np.uint8), frame_sequence=2)
    assert monitor._take_latest_frame() is None


def _synthetic_hud(width=1282, include_exp=True, values=(40, 70, 90)):
    image = np.zeros((145, 1282, 3), dtype=np.uint8)
    colors = ((0, 0, 220), (220, 80, 0), (0, 220, 220))
    xs = (100, 330, 560)
    count = 3 if include_exp else 2
    for x, value, color in zip(xs[:count], values[:count], colors[:count]):
        cv2.rectangle(image, (x, 55), (x + 159, 84), (255, 255, 255), 2)
        cv2.rectangle(image, (x + 2, 57), (x + 157, 82), (55, 55, 55), -1)
        fill = round(155 * value / 100)
        cv2.rectangle(image, (x + 2, 57), (x + 2 + fill, 82), color, -1)
    if width != image.shape[1]:
        scale = width / image.shape[1]
        image = cv2.resize(
            image, (width, round(image.shape[0] * scale)),
            interpolation=cv2.INTER_NEAREST,
        )
    return image


def _synthetic_classic_combined_hud(width=1282, values=(40, 70, 90)):
    """Model the merged external contour measured on the classic CN client."""
    image = np.zeros((148, 1282, 3), dtype=np.uint8)
    panel = (262, 68, 758, 58)
    x, y, w, h = panel
    cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (245, 245, 245), 2)
    colors = ((0, 0, 220), (220, 80, 0), (0, 220, 220))
    bars = ((470, 128, 102, 18), (572, 128, 103, 18), (678, 128, 112, 18))
    for (bx, by, bw, bh), value, color in zip(bars, values, colors):
        cv2.rectangle(
            image, (bx, by), (bx + bw - 1, by + bh - 1),
            (150, 150, 150), 1,
        )
        cv2.rectangle(
            image, (bx + 1, by + 1), (bx + bw - 2, by + bh - 2),
            (55, 55, 55), -1,
        )
        fill = round((bw - 2) * value / 100)
        if fill:
            cv2.rectangle(
                image, (bx + 1, by + 1),
                (bx + fill, by + bh - 2), color, -1,
            )
    if width != image.shape[1]:
        scale = width / image.shape[1]
        image = cv2.resize(
            image, (width, round(image.shape[0] * scale)),
            interpolation=cv2.INTER_NEAREST,
        )
    return image


def _calibrate(recognizer, image):
    snapshots = [
        recognizer.recognize(image, sequence, produced_at=float(sequence))
        for sequence in range(1, 4)
    ]
    return snapshots[-1]


def test_recognizer_calibrates_reference_and_current_client_widths():
    for width in (1282, 2049):
        snapshot = _calibrate(HealthBarRecognizer(), _synthetic_hud(width))
        assert snapshot.state == HEALTH_READY
        assert snapshot.hp_percent == pytest.approx(40, abs=3)
        assert snapshot.mp_percent == pytest.approx(70, abs=3)
        assert snapshot.exp_percent == pytest.approx(90, abs=3)
        assert snapshot.confidence >= 0.8


def test_recognizer_calibrates_classic_cn_merged_panel_at_both_widths():
    for width in (1282, 2049):
        recognizer = HealthBarRecognizer()
        snapshot = _calibrate(
            recognizer,
            _synthetic_classic_combined_hud(width, values=(35, 65, 85)),
        )

        assert snapshot.state == HEALTH_READY
        assert recognizer.last_candidate_layout == "classic_combined"
        assert snapshot.hp_percent == pytest.approx(35, abs=4)
        assert snapshot.mp_percent == pytest.approx(65, abs=4)
        assert snapshot.exp_percent == pytest.approx(85, abs=4)
        assert snapshot.confidence >= 0.55


def test_classic_cn_merged_panel_requires_hp_mp_color_identity():
    image = _synthetic_classic_combined_hud(values=(35, 65, 85))
    # Turn the inferred MP track green: panel geometry alone must not unlock
    # recovery, because a wrong 0% reading could repeatedly press potion keys.
    image[130:144, 574:673] = (0, 200, 0)

    snapshot = _calibrate(HealthBarRecognizer(), image)

    assert snapshot.state != HEALTH_READY
    assert snapshot.hp_percent is None
    assert snapshot.mp_percent is None


def test_classic_cn_locked_panel_uses_color_identity_not_bright_border():
    recognizer = HealthBarRecognizer()
    image = _synthetic_classic_combined_hud(values=(35, 65, 85))
    ready = _calibrate(recognizer, image)

    assert ready.state == HEALTH_READY
    assert ready.confidence >= 0.55

    image[130:144, 472:570] = (55, 55, 55)
    lost = recognizer.recognize(image, 4, produced_at=4.0)

    assert lost.state == HEALTH_LOST
    assert lost.reason == "classic_bar_identity_invalid"
    assert lost.hp_percent is None


def test_classic_cn_empty_mp_can_trigger_safe_zero_reading():
    recognizer = HealthBarRecognizer()
    snapshot = _calibrate(
        recognizer,
        _synthetic_classic_combined_hud(values=(75, 0, 50)),
    )

    assert snapshot.state == HEALTH_READY
    assert snapshot.hp_percent == pytest.approx(75, abs=4)
    assert snapshot.mp_percent == pytest.approx(0, abs=2)


def test_recognizer_can_calibrate_when_mp_is_empty_at_startup():
    image = _synthetic_hud(values=(40, 0, 90))
    image[57:83, 332:488] = (55, 55, 55)
    snapshot = _calibrate(HealthBarRecognizer(), image)

    assert snapshot.state == HEALTH_READY
    assert snapshot.hp_percent == pytest.approx(40, abs=3)
    assert snapshot.mp_percent == pytest.approx(0, abs=2)


def test_recognizer_calibrates_when_mp_and_exp_are_both_empty():
    image = _synthetic_hud(values=(40, 0, 0))
    image[57:83, 332:488] = (55, 55, 55)
    image[57:83, 562:718] = (55, 55, 55)

    snapshot = _calibrate(HealthBarRecognizer(), image)

    assert snapshot.state == HEALTH_READY
    assert snapshot.hp_percent == pytest.approx(40, abs=3)
    assert snapshot.mp_percent == pytest.approx(0, abs=2)
    assert snapshot.exp_percent == pytest.approx(0, abs=2)


def test_empty_mp_without_third_hud_bar_is_not_accepted():
    image = _synthetic_hud(include_exp=False, values=(40, 0, 0))
    image[57:83, 332:488] = (55, 55, 55)

    snapshot = _calibrate(HealthBarRecognizer(), image)

    assert snapshot.state != HEALTH_READY


def test_non_blue_colored_bar_is_not_accepted_as_empty_mp():
    image = _synthetic_hud()
    image[57:83, 332:488] = (0, 200, 0)

    snapshot = _calibrate(HealthBarRecognizer(), image)

    assert snapshot.state != HEALTH_READY
    assert snapshot.hp_percent is None
    assert snapshot.mp_percent is None


def test_small_red_icon_inside_white_box_is_not_accepted_as_hp_bar():
    image = _synthetic_hud(values=(40, 0, 90))
    image[57:83, 102:258] = (55, 55, 55)
    image[57:83, 332:488] = (55, 55, 55)
    cv2.rectangle(image, (106, 66), (113, 73), (0, 0, 220), -1)

    snapshot = _calibrate(HealthBarRecognizer(), image)

    assert snapshot.state != HEALTH_READY
    assert snapshot.hp_percent is None
    assert snapshot.mp_percent is None


def test_invalid_negative_return_home_timeout_never_requests_return_home():
    monitor = _monitor()
    monitor.cfg["health_monitor"]["return_home_if_no_potion"] = True
    monitor.cfg["health_monitor"]["return_home_watch_dog_timeout"] = -1

    for sequence in range(1, 4):
        monitor._evaluate(HealthSnapshot(
            sequence=sequence, produced_at=float(sequence), state=HEALTH_READY,
            hp_percent=10.0, mp_percent=100.0,
        ))

    assert all(
        request["action"] != "return_home"
        for request in monitor.kb.requests
    )


def test_exp_is_optional_and_extra_white_candidate_does_not_break_hp_mp():
    image = _synthetic_hud(include_exp=False)
    cv2.rectangle(image, (900, 10), (1060, 39), (255, 255, 255), 2)
    snapshot = _calibrate(HealthBarRecognizer(), image)

    assert snapshot.state == HEALTH_READY
    assert snapshot.hp_percent == pytest.approx(40, abs=3)
    assert snapshot.mp_percent == pytest.approx(70, abs=3)


@pytest.mark.parametrize("missing_x", (100, 330))
def test_missing_hp_or_mp_never_shifts_other_bars_into_recovery_slots(missing_x):
    image = _synthetic_hud()
    image[50:90, missing_x - 5:missing_x + 170] = 0

    snapshot = _calibrate(HealthBarRecognizer(), image)

    assert snapshot.state != HEALTH_READY
    assert snapshot.hp_percent is None
    assert snapshot.mp_percent is None


def test_invalid_snapshot_clears_stale_low_health_requests():
    monitor = _monitor()
    low = HealthSnapshot(
        sequence=1, produced_at=1.0, state=HEALTH_READY,
        hp_percent=20.0, mp_percent=80.0,
    )
    monitor._evaluate(low)
    monitor._evaluate(HealthSnapshot(
        sequence=2, produced_at=1.1, state=HEALTH_READY,
        hp_percent=20.0, mp_percent=80.0,
    ))
    assert monitor.kb.requests[-1]["action"] == "add_hp"

    monitor._evaluate(HealthSnapshot(
        sequence=3, produced_at=1.2, state=HEALTH_LOST,
        reason="bar_border_invalid",
    ))

    assert monitor._low_counts == {"hp": 0, "mp": 0}
    assert monitor.kb.clears[-1] == {"add_hp", "add_mp", "return_home"}


def test_disabled_zero_threshold_never_requests_recovery():
    monitor = _monitor()
    monitor.auto_hp_enabled = False
    monitor.cfg["health_monitor"]["add_hp_percent"] = 0
    snapshot = HealthSnapshot(
        sequence=1, produced_at=1.0, state=HEALTH_READY,
        hp_percent=0.0, mp_percent=100.0,
    )
    monitor._evaluate(snapshot)
    monitor._evaluate(snapshot)
    assert not monitor.kb.requests


def test_force_heal_has_distinct_priority_state_until_hp_recovers():
    monitor = _monitor()
    monitor.cfg["health_monitor"]["force_heal"] = True

    monitor._evaluate(HealthSnapshot(
        sequence=1, produced_at=1.0, state=HEALTH_READY,
        hp_percent=20.0, mp_percent=10.0,
    ))
    assert monitor.kb.is_need_force_heal is True

    monitor._evaluate(HealthSnapshot(
        sequence=2, produced_at=1.1, state=HEALTH_READY,
        hp_percent=20.0, mp_percent=10.0,
    ))
    hp_request = next(
        request for request in monitor.kb.requests
        if request["action"] == "add_hp"
    )
    assert hp_request["priority"] == "forced"

    monitor._evaluate(HealthSnapshot(
        sequence=3, produced_at=1.2, state=HEALTH_READY,
        hp_percent=80.0, mp_percent=10.0,
    ))
    assert monitor.kb.is_need_force_heal is False


def test_recovery_request_is_cancelled_when_reading_recovers():
    pending = {}

    class Keyboard:
        def request_aux_action(self, **request):
            pending[request["action"]] = request

        def clear_aux_actions(self, actions=None):
            if actions is None:
                pending.clear()
            else:
                for action in actions:
                    pending.pop(action, None)

    monitor = _monitor()
    monitor.kb = Keyboard()
    low = HealthSnapshot(
        sequence=1, produced_at=1.0, state=HEALTH_READY,
        hp_percent=20.0, mp_percent=80.0,
    )
    monitor._evaluate(low)
    monitor._evaluate(HealthSnapshot(
        sequence=2, produced_at=1.1, state=HEALTH_READY,
        hp_percent=20.0, mp_percent=80.0,
    ))
    assert "add_hp" in pending

    monitor._evaluate(HealthSnapshot(
        sequence=3, produced_at=1.2, state=HEALTH_READY,
        hp_percent=80.0, mp_percent=80.0,
    ))

    assert "add_hp" not in pending
    assert "return_home" not in pending


def test_snapshot_becomes_lost_instead_of_exposing_stale_percentages():
    monitor = _monitor()
    monitor._publish_snapshot(HealthSnapshot(
        sequence=8, produced_at=10.0, state=HEALTH_READY,
        hp_percent=25.0, mp_percent=50.0,
    ))

    snapshot = monitor.get_snapshot(now=10.31)

    assert snapshot.state == HEALTH_LOST
    assert snapshot.reason == "frame_stale"
    assert snapshot.hp_percent is None
    assert snapshot.mp_percent is None


def test_calibration_snapshot_also_becomes_lost_when_frames_stop():
    monitor = _monitor()
    monitor._publish_snapshot(HealthSnapshot(
        sequence=2, produced_at=20.0, state=HEALTH_CALIBRATING,
        reason="bar_shapes_not_found",
    ))

    snapshot = monitor.get_snapshot(now=20.31)

    assert snapshot.state == HEALTH_LOST
    assert snapshot.reason == "frame_stale"


def test_delayed_capture_packet_cannot_trigger_recovery():
    class ReadyRecognizer:
        @staticmethod
        def recognize(_image, sequence, produced_at, roi_origin):
            return HealthSnapshot(
                sequence=sequence, produced_at=produced_at,
                state=HEALTH_READY, hp_percent=10.0, mp_percent=100.0,
                roi_origin=roi_origin,
            )

    monitor = _monitor()
    monitor.recognizer = ReadyRecognizer()
    monitor.start()
    try:
        monitor.update_frame(
            np.zeros((10, 20, 3), dtype=np.uint8),
            frame_sequence=11,
            captured_at=time.monotonic() - 1.0,
        )
        deadline = time.monotonic() + 1.0
        while monitor.get_snapshot().sequence != 11 and time.monotonic() < deadline:
            time.sleep(0.005)

        snapshot = monitor.get_snapshot()
        assert snapshot.sequence == 11
        assert snapshot.state == HEALTH_LOST
        assert snapshot.reason == "frame_stale"
        assert monitor.kb.requests == []
    finally:
        monitor.stop()


def test_stop_cannot_leave_a_request_from_an_inflight_recognizer():
    pending = {}

    class Keyboard:
        def request_aux_action(self, **request):
            pending[request["action"]] = request

        def clear_aux_actions(self, actions=None):
            if actions is None:
                pending.clear()
            else:
                for action in actions:
                    pending.pop(action, None)

    entered = threading.Event()
    release = threading.Event()

    class BlockingRecognizer:
        def recognize(self, image, sequence, produced_at, roi_origin):
            entered.set()
            assert release.wait(1.0)
            return HealthSnapshot(
                sequence=sequence, produced_at=produced_at,
                state=HEALTH_READY, hp_percent=10.0, mp_percent=100.0,
                roi_origin=roi_origin,
            )

    monitor = _monitor()
    monitor.kb = Keyboard()
    monitor.recognizer = BlockingRecognizer()
    monitor.start()
    monitor.update_frame(np.zeros((10, 20, 3), dtype=np.uint8), frame_sequence=1)
    assert entered.wait(1.0)

    stopper = threading.Thread(target=monitor.stop)
    stopper.start()
    assert monitor._stop_event.wait(1.0)
    release.set()
    stopper.join(1.0)

    assert not stopper.is_alive()
    assert pending == {}
