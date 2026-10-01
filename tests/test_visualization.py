import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.engine.HealthMonitor import HEALTH_CALIBRATING, HEALTH_LOST, HealthSnapshot
from src.runtime_policy import RUNTIME_POLICY
from src.visualization import LatestVisualizationSlot, VisualizationFrame


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _viz_frame(sequence):
    return VisualizationFrame(
        mode="game",
        sequence=sequence,
        produced_at=time.monotonic(),
        image=np.full((2, 2, 3), sequence, dtype=np.uint8),
    )


def test_latest_visualization_slot_overwrites_without_notification_backlog():
    slot = LatestVisualizationSlot()

    assert slot.publish(_viz_frame(1))
    for sequence in range(2, 101):
        assert not slot.publish(_viz_frame(sequence))

    assert slot.has_frame
    assert slot.dropped == 99
    assert slot.take().sequence == 100
    assert not slot.has_frame
    assert slot.publish(_viz_frame(101))


def test_player_exclusion_debug_draws_orange_outline_only_when_valid():
    class _Detector:
        @staticmethod
        def player_exclusion_rect(_shape, _anchor):
            return (10, 8, 30, 28)

    bot = object.__new__(MapleStoryAutoBot)
    bot.monster_detector = _Detector()
    bot.current_nametag_valid = True
    bot.loc_player = (20, 28)
    canvas = np.zeros((40, 50, 3), dtype=np.uint8)

    bot.draw_player_exclusion_debug(canvas)

    assert tuple(canvas[8, 20]) == (0, 165, 255)
    assert tuple(canvas[18, 20]) == (0, 0, 0)

    bot.current_nametag_valid = False
    hidden_canvas = np.zeros_like(canvas)
    bot.draw_player_exclusion_debug(hidden_canvas)
    assert not np.any(hidden_canvas)


def test_ladder_debug_draws_approach_and_mount_on_window_minimap():
    ladder = {
        "approach": (110, 220), "mount": (115, 220),
        "mount_direction": "right", "stage": "fine_align",
        "align_error": 3, "settle_frames": 1,
        "attempts": 1, "max_attempts": 3, "failure_reason": "",
    }
    bot = object.__new__(MapleStoryAutoBot)
    bot.route_navigator = type("Navigator", (), {"ladder_debug": ladder})()
    bot.img_minimap = np.zeros((40, 50, 3), dtype=np.uint8)
    bot.loc_minimap = (20, 30)
    bot.loc_minimap_global = (100, 200)
    bot.cfg = {"minimap": {"offset": [0, 0]}}
    canvas = np.zeros((520, 200, 3), dtype=np.uint8)

    bot.draw_ladder_execution_debug(canvas)

    # Global points map into the currently visible minimap rectangle.
    assert tuple(canvas[46, 30]) == (0, 255, 255)
    assert tuple(canvas[45, 35]) == (0, 165, 255)


def test_visualization_slot_clear_discards_pending_mode_frame():
    slot = LatestVisualizationSlot()
    slot.publish(_viz_frame(1))

    slot.clear(reset_dropped=True)

    assert slot.take() is None
    assert slot.dropped == 0
    assert slot.publish(_viz_frame(2))


def test_health_debug_uses_snapshot_rects_and_shows_unknown_as_dashes(monkeypatch):
    texts = []
    original_put_text = cv2.putText

    def record_text(image, text, *args, **kwargs):
        texts.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_text)
    snapshot = HealthSnapshot(
        sequence=4, produced_at=time.monotonic(), state=HEALTH_LOST,
        hp_rect=(10, 5, 30, 12), mp_rect=(50, 5, 30, 12),
        reason="bar_border_invalid", roi_origin=(0, 100),
    )
    monitor = type("Monitor", (), {
        "active": True,
        "get_snapshot": lambda self: snapshot,
    })()
    bot = object.__new__(MapleStoryAutoBot)
    bot.health_monitor = monitor
    bot.img_frame = np.zeros((180, 600, 3), dtype=np.uint8)
    bot.img_frame_debug = np.zeros_like(bot.img_frame)

    bot.draw_health_monitor_debug()

    assert tuple(bot.img_frame_debug[105, 10]) == (0, 0, 255)
    assert tuple(bot.img_frame_debug[105, 50]) == (255, 0, 0)
    assert "HP: --" in texts
    assert "MP: --" in texts
    assert any("Health: LOST" in text for text in texts)


def test_health_calibration_draws_orange_search_region():
    snapshot = HealthSnapshot(
        sequence=1, produced_at=time.monotonic(), state=HEALTH_CALIBRATING,
        reason="bars_not_found", roi_origin=(0, 80),
    )
    monitor = type("Monitor", (), {
        "active": True,
        "get_snapshot": lambda self: snapshot,
    })()
    bot = object.__new__(MapleStoryAutoBot)
    bot.health_monitor = monitor
    bot.img_frame = np.zeros((120, 500, 3), dtype=np.uint8)
    bot.img_frame_debug = np.zeros_like(bot.img_frame)

    bot.draw_health_monitor_debug()

    assert tuple(bot.img_frame_debug[80, 0]) == (0, 165, 255)


def test_visualization_request_is_consumed_once_per_control_frame():
    bot = object.__new__(MapleStoryAutoBot)
    bot.is_ui = True
    bot.is_need_show_debug_window = True
    bot.viz_mode = "route"

    bot.request_visualization()

    assert bot._claim_visualization_mode() == "route"
    assert bot._claim_visualization_mode() is None


def test_selected_mode_allocates_only_its_own_debug_buffer():
    bot = object.__new__(MapleStoryAutoBot)
    bot.img_frame = np.zeros((20, 30, 3), dtype=np.uint8)
    bot.img_route = np.zeros((8, 12, 3), dtype=np.uint8)
    bot.img_route[0, 0] = (255, 0, 0)  # RGB red

    bot._frame_visualization_mode = "route"
    bot._prepare_visualization_buffers()
    assert bot.img_frame_debug is None
    assert tuple(bot.img_route_debug[0, 0]) == (0, 0, 255)

    bot._frame_visualization_mode = "game"
    bot._prepare_visualization_buffers()
    assert bot.img_route_debug is None
    assert bot.img_frame_debug is not bot.img_frame
    assert not np.shares_memory(bot.img_frame_debug, bot.img_frame)

    bot._frame_visualization_mode = None
    bot._prepare_visualization_buffers()
    assert bot.img_frame_debug is None
    assert bot.img_route_debug is None


def test_preview_resize_caps_width_and_preserves_bgr_and_aspect_ratio():
    image = np.zeros((720, 2048, 3), dtype=np.uint8)
    image[:, :] = (17, 83, 211)

    resized = MapleStoryAutoBot._resize_preview(
        image, RUNTIME_POLICY.preview_max_width, cv2.INTER_AREA
    )

    assert resized.shape == (450, 1280, 3)
    assert tuple(resized[0, 0]) == (17, 83, 211)
    assert resized.flags.c_contiguous


def test_game_preview_keeps_bottom_hud_in_final_output():
    bot = object.__new__(MapleStoryAutoBot)
    bot.img_frame_debug = np.zeros((120, 200, 3), dtype=np.uint8)
    bot.img_frame_debug[90:, :] = (7, 8, 9)
    bot.is_ui = True
    bot.loc_player = None
    bot.cfg = {"bot": {"mode": "aux"}, "ui_coords": {"ui_y_start": 80}}
    bot.draw_combat_ranges_debug = lambda _canvas: None
    bot.draw_player_exclusion_debug = lambda _canvas: None
    bot.draw_fixed_platform_debug = lambda _canvas: None

    preview = bot.get_frame_debug_for_viz()

    assert preview.shape == (120, 200, 3)
    assert tuple(preview[100, 20]) == (7, 8, 9)


def test_controller_uses_200ms_timer_and_requests_first_frame_immediately(
        qt_app, monkeypatch):
    import src.ui.AutoBotController as controller_module

    class DummyBot:
        def __init__(self):
            self.requests = 0
            self.enabled_modes = []

        def set_visualization_sink(self, sink):
            self.sink = sink

        def request_visualization(self):
            self.requests += 1

        def enable_viz(self, mode):
            self.enabled_modes.append(mode)

        def disable_viz(self):
            pass

    class DummyListener:
        def __init__(self, **_kwargs):
            pass

        def register_func_key_handler(self, *_args):
            pass

    dummy_bot = DummyBot()
    monkeypatch.setattr(controller_module, "MapleStoryAutoBot", lambda _args: dummy_bot)
    monkeypatch.setattr(controller_module, "KeyBoardListener", DummyListener)
    controller = controller_module.AutoBotController()
    try:
        controller._bot_running = True
        controller.enable_bot_viz("game")

        assert controller.visualization_timer.interval() == 200
        assert controller.visualization_timer.isActive()
        assert dummy_bot.enabled_modes == ["game"]
        assert dummy_bot.requests == 1

        controller.enable_bot_viz("route")
        assert dummy_bot.enabled_modes == ["game", "route"]
        assert dummy_bot.requests == 2
        assert controller.take_latest_visualization() is None
    finally:
        controller.visualization_timer.stop()
        controller.deleteLater()


def test_f1_from_worker_thread_clicks_start_button_on_qt_main_thread(
        qt_app, monkeypatch):
    import src.ui.AutoBotController as controller_module

    class DummyBot:
        def set_visualization_sink(self, sink):
            self.sink = sink

    class DummyListener:
        def __init__(self, **_kwargs):
            self.handlers = {}

        def register_func_key_handler(self, key, handler):
            self.handlers[key] = handler

    class DummySignal:
        def connect(self, _slot):
            pass

    class DummyUi:
        def __init__(self):
            from PySide6.QtWidgets import QPushButton
            self.button_start_pause = QPushButton()
            self.request_close = DummySignal()

    dummy_bot = DummyBot()
    monkeypatch.setattr(controller_module, "MapleStoryAutoBot", lambda _args: dummy_bot)
    monkeypatch.setattr(controller_module, "KeyBoardListener", DummyListener)
    controller = controller_module.AutoBotController()
    ui = DummyUi()
    clicked_threads = []
    main_thread_id = threading.get_ident()
    ui.button_start_pause.clicked.connect(
        lambda: clicked_threads.append(threading.get_ident())
    )
    controller.update_signal(ui)
    try:
        worker = threading.Thread(target=controller.kb_listener.handlers["f1"])
        worker.start()
        worker.join()

        assert clicked_threads == []
        qt_app.processEvents()
        assert clicked_threads == [main_thread_id]
    finally:
        controller.visualization_timer.stop()
        ui.button_start_pause.deleteLater()
        controller.deleteLater()


def test_runtime_stop_signal_finishes_controller_on_qt_main_thread(
        qt_app, monkeypatch):
    import src.ui.AutoBotController as controller_module

    class DummyBot:
        def set_visualization_sink(self, sink):
            self.visualization_sink = sink

        def set_termination_sink(self, sink):
            self.termination_sink = sink

    class DummyListener:
        def __init__(self, **_kwargs):
            pass

    class DummyLock:
        def __init__(self):
            self.releases = 0

        def release(self):
            self.releases += 1

    class DummyUi:
        def __init__(self):
            self.calls = []

        def handle_runtime_stopped(self, reason):
            self.calls.append((reason, threading.get_ident()))

    dummy_bot = DummyBot()
    monkeypatch.setattr(controller_module, "MapleStoryAutoBot", lambda _args: dummy_bot)
    monkeypatch.setattr(controller_module, "KeyBoardListener", DummyListener)
    controller = controller_module.AutoBotController()
    controller.route_control_lock = DummyLock()
    controller.ui = DummyUi()
    controller._bot_running = True
    controller._active_generation = 7
    controller.visualization_timer.start()
    main_thread_id = threading.get_ident()
    try:
        worker = threading.Thread(
            target=lambda: dummy_bot.termination_sink(7, "return_home")
        )
        worker.start()
        worker.join()

        assert controller._bot_running
        qt_app.processEvents()

        assert not controller._bot_running
        assert not controller.visualization_timer.isActive()
        assert controller.route_control_lock.releases == 1
        assert controller.ui.calls == [("return_home", main_thread_id)]

        # A duplicate worker completion is harmless and cannot double-release
        # the exclusive route-control lock.
        dummy_bot.termination_sink(7, "return_home")
        qt_app.processEvents()
        assert controller.route_control_lock.releases == 1

        # A delayed notification from the prior run cannot stop or unlock a
        # newly started generation.
        controller._bot_running = True
        controller._active_generation = 8
        dummy_bot.termination_sink(7, "input_error")
        qt_app.processEvents()
        assert controller._bot_running
        assert controller.route_control_lock.releases == 1
        assert controller.ui.calls == [("return_home", main_thread_id)]
    finally:
        controller._bot_running = False
        controller.visualization_timer.stop()
        controller.deleteLater()
