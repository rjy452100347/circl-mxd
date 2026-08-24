import threading

import numpy as np

import src.input.GameWindowCapturor as module
from src.input.GameWindowCapturor import GameWindowCapturor


def _capturor(frame, *, static=False):
    capture = object.__new__(GameWindowCapturor)
    capture.frame = frame
    capture.frame_received_at = 10.0
    capture.frame_stale_timeout = 0.75
    capture.is_capture_closed = False
    capture.is_static_test_frame = static
    capture.lock = threading.Lock()
    capture.capture_control = None
    return capture


def test_live_frame_is_rejected_after_freshness_deadline(monkeypatch):
    capture = _capturor(np.zeros((4, 5, 4), dtype=np.uint8))
    now = {"value": 10.5}
    monkeypatch.setattr(module.time, "monotonic", lambda: now["value"])

    assert capture.get_frame().shape == (4, 5, 3)
    now["value"] = 10.76
    assert capture.get_frame() is None


def test_closed_capture_clears_last_frame(monkeypatch):
    capture = _capturor(np.zeros((4, 5, 4), dtype=np.uint8))
    monkeypatch.setattr(module.cv2, "destroyAllWindows", lambda: None)

    capture.on_closed()

    assert capture.get_frame() is None
    assert capture.frame is None
    assert capture.frame_received_at is None
    assert capture.is_capture_closed


def test_static_three_channel_test_frame_does_not_expire(monkeypatch):
    source = np.full((4, 5, 3), 7, dtype=np.uint8)
    capture = _capturor(source, static=True)
    monkeypatch.setattr(module.time, "monotonic", lambda: 1000.0)

    result = capture.get_frame()

    assert np.array_equal(result, source)
    assert result is not source


def test_stop_invalidates_frame_even_when_capture_backend_fails():
    capture = _capturor(np.zeros((4, 5, 4), dtype=np.uint8))

    class FailedControl:
        def stop(self):
            raise RuntimeError("capture stop failed")

    capture.capture_control = FailedControl()
    try:
        capture.stop()
    except RuntimeError:
        pass
    else:
        raise AssertionError("capture backend failure should remain visible")

    assert capture.get_frame() is None
    assert capture.frame is None
    assert capture.is_capture_closed
