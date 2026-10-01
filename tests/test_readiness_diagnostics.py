import copy
import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.engine import ReadinessDiagnostics as diag
from src.engine.HealthMonitor import HealthSnapshot
from src.engine.NameTagLocalizer import NameTagResult
from src.utils.common import load_yaml


def config(mode="continuous_attack"):
    cfg = copy.deepcopy(load_yaml("config/config_default.yaml"))
    cfg["bot"]["mode"] = mode
    cfg["game_window"].update(size=[100, 160], title_bar_height=0,
                              client_crop=None, resize_on_start=False)
    cfg["ui_coords"]["ui_y_start"] = 80
    cfg["nametag"]["search_y_limit"] = 100
    cfg["health_monitor"].update(auto_hp_enabled=True, auto_mp_enabled=False)
    return cfg


class Cancel:
    def __init__(self):
        self.cancelled = False

    def is_set(self):
        return self.cancelled

    def wait(self, seconds):
        return self.cancelled


@pytest.fixture
def fake_perception(monkeypatch):
    class Repository:
        def load_profile(self, *_a, **_kw):
            return {}

        def resolve_profile(self, _name):
            return "example"

    class Name:
        @classmethod
        def from_profile(cls, _path):
            return cls()

        def locate(self, _frame, **_kw):
            return NameTagResult((50, 40), (40, 50), (20, 8), .05, True, "ok")

    class Detector:
        def __init__(self, *_a, **_kw):
            pass

        def detect(self, *_a, **_kw):
            return []

    class Health:
        def recognize(self, frame, sequence, captured_at, **_kw):
            return HealthSnapshot(sequence=sequence, produced_at=captured_at,
                                  state="READY", hp_percent=80, mp_percent=90)

    class Capture:
        def __init__(self, cfg):
            assert cfg["game_window"]["resize_on_start"] is False
            self.stopped = False
            self.sequence = 0

        def get_frame_packet(self):
            self.sequence += 1
            return np.zeros((100, 160, 3), np.uint8), time.monotonic(), self.sequence

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(diag, "NameTagProfileRepository", Repository)
    monkeypatch.setattr(diag, "NameTagLocalizer", Name)
    monkeypatch.setattr(diag, "OpenVinoMonsterDetector", Detector)
    monkeypatch.setattr(diag, "HealthBarRecognizer", Health)
    return Capture


def test_readonly_check_has_no_map_requirement_or_input_and_releases_capture(fake_perception, monkeypatch):
    import src.input.KeyBoardController as keyboard
    monkeypatch.setattr(keyboard, "KeyBoardController", lambda *_a, **_kw: pytest.fail("检查不得创建输入控制器"))
    cfg = config()
    original = copy.deepcopy(cfg)
    capture = fake_perception(cfg)
    reports = []
    diag.ReadinessChecker(cfg, reports.append, Cancel(), capture_factory=lambda _cfg: capture).run()
    rows = {r.key: r for r in reports[-1].rows}
    assert rows["map"].status == rows["route"].status == rows["roi"].status == "unused"
    assert rows["name"].status == rows["yolo"].status == rows["hp"].status == "pass"
    assert "怪物 0" in rows["yolo"].detail
    assert rows["mp"].status == "unused"
    assert reports[-1].name_preview.startswith(b"\x89PNG")
    assert capture.stopped
    assert capture.sequence == 20
    assert cfg == original


def test_invalid_roi_does_not_hide_name_or_health_results(fake_perception, monkeypatch):
    monkeypatch.setattr(diag, "apply_map_minimap_roi_override", lambda *_a: (_ for _ in ()).throw(ValueError("ROI 配置非法")))
    reports = []
    diag.ReadinessChecker(config("fixed_platform"), reports.append, Cancel(), capture_factory=fake_perception).run()
    rows = {r.key: r for r in reports[-1].rows}
    assert rows["roi"].status == "fail"
    assert "ROI 配置非法" in rows["roi"].detail
    assert rows["name"].status == rows["hp"].status == "pass"


def test_cancel_releases_capture_and_never_claims_ready(fake_perception):
    capture = fake_perception(config())
    cancel = Cancel()
    reports = []

    def publish(report):
        reports.append(report)
        if any(r.key == "name" and r.status == "pass" for r in report.rows):
            cancel.cancelled = True

    diag.ReadinessChecker(config(), publish, cancel, capture_factory=lambda _cfg: capture).run()
    assert capture.stopped
    assert capture.sequence == 1
    assert "取消" in reports[-1].summary


def test_exception_still_stops_capture(fake_perception):
    capture = fake_perception(config())
    capture.get_frame_packet = lambda: (_ for _ in ()).throw(RuntimeError("capture failed"))
    with pytest.raises(RuntimeError):
        diag.ReadinessChecker(config(), lambda _r: None, Cancel(), capture_factory=lambda _cfg: capture).run()
    assert capture.stopped


def test_inference_exception_keeps_name_and_health_results(fake_perception, monkeypatch):
    def broken(*_a, **_kw):
        raise RuntimeError("推理输出格式错误")

    monkeypatch.setattr(diag.OpenVinoMonsterDetector, "detect", broken)
    reports = []
    diag.ReadinessChecker(config(), reports.append, Cancel(), capture_factory=fake_perception).run()
    rows = {r.key: r for r in reports[-1].rows}
    assert rows["yolo"].status == "fail"
    assert "输出格式" in rows["yolo"].detail
    assert rows["name"].status == rows["hp"].status == "pass"


def test_name_failure_clears_preview_and_outside_foot_is_failure():
    frame = np.zeros((100, 160, 3), np.uint8)
    missed = NameTagResult(None, None, None, .8, False, "score_too_high")
    row, png = diag.name_result(missed, frame)
    assert row.status == "fail" and png is None
    outside = NameTagResult((500, 40), (40, 50), (20, 8), .05, True, "ok")
    assert diag.name_result(outside, frame)[0].status == "fail"


def test_runtime_never_uses_previous_frame_pose_or_detection():
    bot = SimpleNamespace(cfg=config("normal"), run_generation=5,
        nametag_last_result=None, current_minimap_roi_valid=False,
        current_minimap_player_valid=False, _diagnostic_map_checked=False,
        minimap_pose_snapshot=SimpleNamespace(valid=True), monsters=["old"],
        _diagnostic_yolo_checked=False, img_frame=np.zeros((100, 160, 3), np.uint8))
    report = diag.runtime_report(bot, True)
    rows = {r.key: r for r in report.rows}
    assert rows["name"].status == rows["map"].status == rows["yolo"].status == "pending"
    assert report.name_preview is None
    assert report.generation == 5
    bot.cfg = config()
    bot.continuous_attack_state = SimpleNamespace(direction_locked=True)
    rows = {r.key: r for r in diag.runtime_report(bot, True).rows}
    assert rows["name"].status == rows["yolo"].status == "unused"


def test_readonly_lost_frame_clears_success_and_preview():
    checker = diag.ReadinessChecker(config(), lambda _r: None, Cancel())
    checker.put("name", "pass", "旧名字")
    checker.put("hp", "pass", "旧血量")
    checker.preview = b"old image"
    checker.missing_frame()
    assert checker.rows["name"].status == checker.rows["hp"].status == "pending"
    assert checker.rows["mp"].status == "unused"
    assert checker.preview is None


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(app, monkeypatch):
    import src.ui.AutoBotController as module

    class Bot:
        def set_visualization_sink(self, _sink):
            pass

    class Listener:
        pass

    monkeypatch.setattr(module, "MapleStoryAutoBot", lambda _args: Bot())
    monkeypatch.setattr(module, "KeyBoardListener", lambda **_kw: Listener())
    instance = module.AutoBotController()
    yield instance
    instance.diagnostics_timer.stop()
    instance.visualization_timer.stop()
    instance.deleteLater()


def pump(app, predicate):
    deadline = time.monotonic()+3
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert predicate()


def test_async_start_loads_off_gui_and_launches_on_gui(app, controller):
    main = threading.get_ident()
    calls = []
    controller._prepare_with_errors = lambda _path: calls.append(("load", threading.get_ident())) or 0
    controller._launch_bot = lambda: calls.append(("launch", threading.get_ident())) or 0
    assert controller.start_bot_async("unused")
    pump(app, lambda: not controller.preparation_busy)
    assert calls[0][0] == "load" and calls[0][1] != main
    assert calls[1] == ("launch", main)


def test_cancel_start_cannot_launch_or_accept_another_operation(app, controller):
    gate = threading.Event()
    controller._prepare_with_errors = lambda _path: gate.wait(2) and 0
    controller._launch_bot = lambda: pytest.fail("取消后不可启动")
    assert controller.start_bot_async("unused")
    assert not controller.check_readiness(config())
    controller.cancel_preparation()
    gate.set()
    pump(app, lambda: not controller.preparation_busy)
    assert "取消" in controller.last_start_error


def test_start_error_preserves_specific_resource_reason(controller):
    from src.utils.logger import logger

    def fail(_path):
        logger.error("[名字定位] 样本损坏：示例角色/sample_001.png")
        return -1

    controller._prepare_start = fail
    assert controller._prepare_with_errors("unused") == -1
    assert "样本损坏" in controller.last_start_error


def test_pause_does_not_replay_earlier_loading_report(controller):
    report = diag.ReadinessReport((), time.monotonic(), "实时定位失败", generation=2)
    controller.auto_bot.readiness_report = report
    controller.auto_bot.pause = lambda: None
    controller._check_report = diag.ReadinessReport((), time.monotonic(), "旧加载结果")
    controller._bot_running = True
    controller._active_generation = 2
    presented = []
    controller.ui = SimpleNamespace(update_readiness_report=lambda r, **_kw: presented.append(r))
    controller.pause_bot()
    controller._present_diagnostics()
    assert presented == [report]
    assert controller._check_report is None


def test_report_slot_overwrites_and_rejects_previous_generation(controller):
    for i in range(1000):
        controller._publish_check_report(diag.ReadinessReport((), time.monotonic(), str(i)))
    assert controller._check_report.summary == "999"
    controller._bot_running = True
    controller._active_generation = 3
    controller.auto_bot.readiness_report = diag.ReadinessReport((), time.monotonic(), "old", generation=2)
    controller.ui = SimpleNamespace(update_readiness_report=lambda *_a, **_kw: pytest.fail("陈旧运行结果"))
    controller._present_diagnostics()


@pytest.fixture
def window(app, monkeypatch):
    from src.ui.ui import MainWindow
    from src.utils.logger import logger
    monkeypatch.setattr(MainWindow, "load_ui_state", lambda _self: None)
    ui = MainWindow(SimpleNamespace(pause_bot=lambda: None))
    yield ui
    logger._logger.removeHandler(ui.qt_log_handler)
    ui.deleteLater()


def test_ui_readiness_failure_recovery_and_invalidated_config(window):
    report = diag.ReadinessReport((diag.CheckResult("name", "fail", "名字匹配失败"),),
                                  time.monotonic(), "请重新标定")
    window.update_readiness_report(report)
    row = [key for key, *_ in diag.CHECKS].index("name")
    window.readiness_table.selectRow(row)
    assert window.readiness_table.item(row, 1).text() == "失败"
    assert "标定" in window.readiness_recovery.text()
    assert "名字匹配失败" in window.readiness_last_issue.text()
    window.nametag_profile_combo.setEditText("新角色")
    assert window.readiness_table.item(row, 1).text() == "待重检"
    assert "失效" in window.readiness_summary.text()


def test_ui_stop_keeps_evidence_and_unlocked_recovery(window):
    window.update_debug_canvas(np.ones((40, 40, 3), np.uint8)*128)
    window.set_gbox_enabled(False)
    window.handle_runtime_stopped("route_ladder_failed")
    assert not window.debug_canvas.pixmap().isNull()
    assert "非实时" in window.game_viz_state.text()
    assert "挂梯" in window.readiness_last_issue.text()
    assert window.button_readiness_name.isEnabled()
    assert window.button_check_readiness.isEnabled()


def test_ui_busy_blocks_f1_configuration_and_recovery(window):
    window._set_readiness_busy(True)
    assert not window.button_start_pause.isEnabled()
    assert not window.button_load_config.isEnabled()
    assert not window.button_readiness_name.isEnabled()
    assert not window.bot_mode.isEnabled()
    window._set_readiness_busy(False)
    assert window.button_start_pause.isEnabled()


def test_runtime_stale_report_cannot_stay_green(window):
    report = diag.ReadinessReport((diag.CheckResult("name", "pass", "名字已匹配"),),
                                  time.monotonic(), "正常")
    window.update_readiness_report(report, live=True)
    window.mark_readiness_stale()
    row = [key for key, *_ in diag.CHECKS].index("name")
    assert window.readiness_table.item(row, 1).text() == "已过期"
    assert "非当前" in window.game_viz_state.text()
