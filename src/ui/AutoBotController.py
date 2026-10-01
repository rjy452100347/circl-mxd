# Standard Import
from argparse import Namespace
from collections import deque
import math
import copy
import logging
import sys
import threading
import time

# Pyside
from PySide6.QtCore import Signal, QObject, QTimer, Qt

#  Local Import
from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.utils.logger import logger
from src.utils.common import load_yaml
from src.config_compat import ConfigCompatibilityError, validate_custom_config
from src.input.KeyBoardListener import KeyBoardListener
from src.route_control_lock import RouteControlBusyError, RouteControlLock
from src.runtime_policy import RUNTIME_POLICY
from src.visualization import LatestVisualizationSlot, VisualizationFrame
from src.engine.ReadinessDiagnostics import ReadinessChecker, ReadinessReport, CheckResult, initial_rows

class AutoBotController(QObject):
    '''
    AutoBot Controller server as a middleman between engine and UI
    '''
    visualization_ready = Signal()
    toggle_bot_requested = Signal()
    runtime_stopped = Signal(int, str)
    preparation_finished = Signal(str, int)

    def __init__(self):
        """
        Init
        """
        super().__init__()
        self.ui = None
        self.last_start_error = ""
        self.preparation_busy = False
        self._preparation_cancel = threading.Event()
        self._check_report = None
        self._last_presented_report = None
        self._preparation_kind = None
        self.preparation_finished.connect(self._finish_preparation, Qt.QueuedConnection)
        self.diagnostics_timer = QTimer(self)
        self.diagnostics_timer.setInterval(200)
        self.diagnostics_timer.timeout.connect(self._present_diagnostics)
        self.diagnostics_timer.start()
        self.route_control_lock = RouteControlLock("autobot")
        self.visualization_slot = LatestVisualizationSlot()
        self.visualization_timer = QTimer(self)
        self.visualization_timer.setInterval(round(1000 / RUNTIME_POLICY.preview_fps))
        self.visualization_timer.timeout.connect(self._request_visualization)
        self._visualization_mode = None
        self._bot_running = False
        self._active_generation = 0
        self._visualization_sequence = 0
        self._visualization_sequence_lock = threading.Lock()
        preview_window = RUNTIME_POLICY.preview_fps * 5 * 60
        self._presentation_times = deque(maxlen=preview_window)
        self._presentation_durations = deque(maxlen=preview_window)
        self._preview_ages = deque(maxlen=preview_window)
        self._last_visualization_log_at = 0.0
        self.runtime_stopped.connect(
            self._handle_runtime_stopped, Qt.QueuedConnection
        )

        # Init Auto Bot
        try:
            # Fake args to pass to AutoBot
            args = Namespace(
                disable_control=False,
                cfg="default",
                debug=False,
                is_ui=True,
                disable_viz=True,
                test_image='',
                init_state='',
            )
            self.auto_bot = MapleStoryAutoBot(args)
            self.auto_bot.diagnostics_enabled = True
            self.auto_bot.load_status_sink = self._publish_load_status
        except Exception as e:
            logger.error(f"MapleStoryAutoBot Init Failed: {e}")
            sys.exit(1)
        else:
            logger.info("MapleStoryAutoBot Init Successfully")

        # The worker publishes into a capacity-one mailbox.  Qt only queues a
        # lightweight readiness notification, never a multi-megabyte ndarray.
        self.auto_bot.set_visualization_sink(self.publish_visualization)
        if hasattr(self.auto_bot, "set_termination_sink"):
            self.auto_bot.set_termination_sink(self.runtime_stopped.emit)

        # Monitor function keys
        self.kb_listener = KeyBoardListener(is_autobot=True)

    def toggle_enable(self):
        '''
        toggle_enable
        '''
        self.is_enable = not self.is_enable
        logger.info(f"Player pressed F1, is_enable:{self.is_enable}")

        # Make sure all key are released
        self.release_all_key()

    def update_signal(self, ui):
        '''
        Only called after UI init
        '''
        self.ui = ui
        self.visualization_ready.connect(self._present_latest_visualization)
        self.toggle_bot_requested.connect(
            ui.button_start_pause.click, Qt.QueuedConnection
        )
        # Register Function Key handler
        self.kb_listener.register_func_key_handler(
            'f1', self.toggle_bot_requested.emit
        )
        self.kb_listener.register_func_key_handler('emergency', lambda: ui.request_close.emit())

    def _prepare_start(self, cfg_path):
        '''
        Start the bot engine threads
        '''
        try:
            self.route_control_lock.acquire()
        except RouteControlBusyError as exc:
            logger.error(str(exc))
            return -1

        previous_thread = getattr(self.auto_bot, "thread_auto_bot", None)
        if previous_thread is not None and previous_thread.is_alive():
            logger.error("上一次运行尚未安全退出，请稍后再启动。")
            self.route_control_lock.release()
            return -1

        try:

            # Get config from ui
            cfg = load_yaml(cfg_path)
            self._loading_rows = initial_rows(cfg)
            self._loading_key = None
        except Exception as exc:
            logger.error(f"[start_bot] 启动预检失败：{exc}")
            self.route_control_lock.release()
            return -1
        try:
            validate_custom_config(cfg)
        except ConfigCompatibilityError as exc:
            logger.error(str(exc))
            self.route_control_lock.release()
            return -1

        # Auto bot load config
        try:
            load_result = self.auto_bot.load_config(cfg)
        except Exception as exc:
            logger.error(f"[start_bot] 加载配置或资产失败：{exc}")
            self.route_control_lock.release()
            return -1
        if load_result != 0:
            self.route_control_lock.release()
            return -1 # Load fail

        return 0

    def _publish_load_status(self, key, detail):
        previous = getattr(self, "_loading_key", None)
        if previous is not None:
            self._loading_rows[previous] = CheckResult(previous, "pass", "静态资源检查通过；当前画面尚待识别")
        self._loading_key = key
        if key is not None:
            self._loading_rows[key] = CheckResult(key, "pending", detail)
        self._check_report = ReadinessReport(tuple(self._loading_rows.values()), time.monotonic(), detail)

    def _launch_bot(self):
        """Only called on the Qt thread after resource loading has finished."""
        self.auto_bot.readiness_report = None

        # Start the bot engine
        try:
            self.auto_bot.start()
        except Exception as e:
            logger.error(f"[start_bot] {e}")
            self.last_start_error = f"游戏窗口或运行线程启动失败：{e}"
            try:
                self.auto_bot.terminate_threads()
            except Exception as cleanup_error:
                logger.error(f"[start_bot] 清理失败：{cleanup_error}")
            finally:
                self.route_control_lock.release()
            return -1 # Start fail

        self._bot_running = True
        self._active_generation = int(
            getattr(self.auto_bot, "run_generation", self._active_generation + 1)
        )
        self._start_visualization_requests()
        self._check_report = None

        return 0 # start bot success

    def _prepare_with_errors(self, cfg_path):
        self.last_start_error = ""
        self._loading_key = None
        self._loading_rows = {}
        owner = threading.get_ident()
        errors = []

        class ErrorCollector(logging.Handler):
            def emit(self, record):
                if record.thread == owner and record.levelno >= logging.ERROR:
                    errors.append(record.getMessage())

        handler = ErrorCollector()
        logger._logger.addHandler(handler)
        try:
            result = self._prepare_start(cfg_path)
            if result != 0:
                self.last_start_error = "\n".join(errors[-3:]) or "资源加载失败，请检查名字、地图和 YOLO 部署。"
                key = getattr(self, "_loading_key", None) or "window"
                rows = getattr(self, "_loading_rows", {})
                rows[key] = CheckResult(key, "fail", self.last_start_error)
                self._check_report = ReadinessReport(tuple(rows.values()), time.monotonic(), "启动检查失败，未开始运行")
            return result
        except Exception as exc:
            self.last_start_error = f"启动检查失败：{exc}"
            self.route_control_lock.release()
            return -1
        finally:
            logger._logger.removeHandler(handler)

    def start_bot(self, cfg_path):
        """Synchronous compatibility entry point (UI uses the async variant)."""
        if self.preparation_busy:
            self.last_start_error = "正在检查或加载，请稍后再启动。"
            return -1
        return self._launch_bot() if self._prepare_with_errors(cfg_path) == 0 else -1

    def _begin_preparation(self, kind, work):
        if self.preparation_busy or self._bot_running:
            return False
        previous = getattr(self.auto_bot, "thread_auto_bot", None)
        if previous is not None and previous.is_alive():
            self.last_start_error = "上一次运行尚未退出，请稍后重试。"
            return False
        self.preparation_busy = True
        self._preparation_kind = kind
        self._preparation_cancel.clear()
        self._check_report = None
        self._last_presented_report = None

        def worker():
            result = -1
            try:
                result = work()
            except Exception as exc:
                self.last_start_error = f"检查失败：{exc}"
                self._check_report = ReadinessReport(
                    (CheckResult("window", "fail", self.last_start_error),),
                    time.monotonic(), self.last_start_error)
            finally:
                self.preparation_finished.emit(kind, result)

        self._preparation_thread = threading.Thread(target=worker, daemon=True)
        self._preparation_thread.start()
        return True

    def start_bot_async(self, cfg_path):
        return self._begin_preparation("start", lambda: self._prepare_with_errors(cfg_path))

    def check_readiness(self, cfg):
        candidate = copy.deepcopy(cfg)

        def work():
            with RouteControlLock("readiness_check"):
                ReadinessChecker(candidate, self._publish_check_report,
                                 self._preparation_cancel).run()
            return 0

        return self._begin_preparation("check", work)

    def _publish_check_report(self, report):
        self._check_report = report

    def cancel_preparation(self):
        self._preparation_cancel.set()

    def _finish_preparation(self, kind, result):
        if not self.preparation_busy or kind != self._preparation_kind:
            return
        cancelled = self._preparation_cancel.is_set()
        if kind == "start":
            if cancelled:
                self.route_control_lock.release()
                result = -1
                self.last_start_error = "启动已取消，没有开始自动运行。"
            elif result == 0:
                result = self._launch_bot()
        self._present_diagnostics()
        self.preparation_busy = False
        self._preparation_kind = None
        if self.ui is not None and hasattr(self.ui, "finish_readiness_operation"):
            self.ui.finish_readiness_operation(kind, result, cancelled)

    def _present_diagnostics(self):
        report = (getattr(self.auto_bot, "readiness_report", None)
                  if self._bot_running else self._check_report)
        if report is None or report is self._last_presented_report:
            if (self._bot_running and report is not None and
                    time.monotonic() - report.produced_at > 1 and self.ui is not None and
                    hasattr(self.ui, "mark_readiness_stale")):
                self.ui.mark_readiness_stale()
            return
        if self._bot_running and report.generation != self._active_generation:
            return
        self._last_presented_report = report
        if self.ui is not None and hasattr(self.ui, "update_readiness_report"):
            self.ui.update_readiness_report(report, live=self._bot_running)

    def pause_bot(self):
        '''
        Gracefully pause in the engine
        '''
        self._present_diagnostics()
        self._check_report = None
        self._bot_running = False
        self.visualization_timer.stop()
        self.visualization_slot.clear()
        self.auto_bot.pause()
        self.route_control_lock.release()

    def terminate_bot(self):
        '''
        Called when user stop bot or close UI
        '''
        if self.preparation_busy:
            self.cancel_preparation()
            return
        self.diagnostics_timer.stop()
        # Terminate all bot threads
        self._bot_running = False
        self.visualization_timer.stop()
        self.visualization_slot.clear()
        self.auto_bot.terminate_threads()
        self.route_control_lock.release()

    def _handle_runtime_stopped(self, generation, reason):
        """Finish a worker-initiated stop on Qt's main thread."""
        if (not self._bot_running or
                int(generation) != self._active_generation):
            return
        self._present_diagnostics()
        self._check_report = None
        self._bot_running = False
        self.visualization_timer.stop()
        self.visualization_slot.clear()
        self.route_control_lock.release()
        if self.ui is not None and hasattr(self.ui, "handle_runtime_stopped"):
            self.ui.handle_runtime_stopped(reason)

    def enable_bot_viz(self, mode="game"):
        '''
        Called when user switch to viz tab
        '''
        self._visualization_mode = mode
        self.visualization_slot.clear(reset_dropped=True)
        self._presentation_times.clear()
        self._presentation_durations.clear()
        self._preview_ages.clear()
        self._last_visualization_log_at = 0.0
        self.auto_bot.enable_viz(mode)
        self._start_visualization_requests()

    def disable_bot_viz(self):
        '''
        Called when user switch from viz tab
        '''
        self._visualization_mode = None
        self.visualization_timer.stop()
        self.visualization_slot.clear()
        self.auto_bot.disable_viz()

    def _start_visualization_requests(self):
        if not self._bot_running or self._visualization_mode is None:
            return
        self.visualization_timer.start()
        self._request_visualization()

    def _request_visualization(self):
        if self._bot_running and self._visualization_mode is not None:
            self.auto_bot.request_visualization()

    def publish_visualization(self, mode, image, performance=None):
        """Worker-thread sink: replace the pending preview without queue growth."""
        if (not self._bot_running or mode != self._visualization_mode or
                image is None):
            return
        with self._visualization_sequence_lock:
            self._visualization_sequence += 1
            sequence = self._visualization_sequence
        frame = VisualizationFrame(
            mode=mode,
            sequence=sequence,
            produced_at=time.monotonic(),
            image=image,
            performance=dict(performance or {}),
        )
        if self.visualization_slot.publish(frame):
            self.visualization_ready.emit()

    def take_latest_visualization(self):
        return self.visualization_slot.take()

    @staticmethod
    def _p95(samples):
        if not samples:
            return 0.0
        ordered = sorted(float(value) for value in samples)
        index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return ordered[index]

    def _present_latest_visualization(self):
        frame = self.take_latest_visualization()
        if frame is None or self.ui is None or frame.mode != self._visualization_mode:
            return
        started_at = time.perf_counter()
        if frame.mode == "game":
            self.ui.update_debug_canvas(frame.image)
        elif frame.mode == "route":
            self.ui.update_route_map_canvas(frame.image)
        else:
            return
        presented_at = time.monotonic()
        ui_ms = (time.perf_counter() - started_at) * 1000.0
        self._presentation_times.append(presented_at)
        self._presentation_durations.append(ui_ms)
        preview_age_ms = max(0.0, (presented_at - frame.produced_at) * 1000.0)
        self._preview_ages.append(preview_age_ms)
        display_fps = 0.0
        if len(self._presentation_times) >= 2:
            duration = self._presentation_times[-1] - self._presentation_times[0]
            if duration > 0:
                display_fps = (len(self._presentation_times) - 1) / duration
        metrics = dict(frame.performance)
        metrics.update({
            "ui_ms": ui_ms,
            "ui_p95_ms": self._p95(self._presentation_durations),
            "display_fps": display_fps,
            "preview_age_ms": preview_age_ms,
            "preview_age_p95_ms": self._p95(self._preview_ages),
            "dropped": float(self.visualization_slot.dropped),
        })
        self.ui.update_visualization_metrics(frame.mode, metrics)
        if presented_at - self._last_visualization_log_at >= 5.0:
            logger.info(
                "[Performance] "
                f"mode={frame.mode} "
                f"control_p95={metrics.get('control_p95_ms', 0.0):.1f}ms "
                f"infer_p95={metrics.get('infer_p95_ms', 0.0):.1f}ms "
                f"viz_p95={metrics.get('viz_p95_ms', 0.0):.1f}ms "
                f"ui_p95={metrics['ui_p95_ms']:.1f}ms "
                f"display_fps={metrics['display_fps']:.1f} "
                f"preview_age_p95={metrics['preview_age_p95_ms']:.1f}ms "
                f"dropped={int(metrics['dropped'])}"
            )
            self._last_visualization_log_at = presented_at
