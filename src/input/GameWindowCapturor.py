'''
Execute this script:
python mapleStoryAutoLevelUp.py --map cloud_balcony --monster brown_windup_bear,pink_windup_bear
'''
# Standard import
import time
import threading

# Libarary Import
from windows_capture import WindowsCapture, Frame, InternalCaptureControl
import cv2

# local import
from src.utils.logger import logger
from src.utils.common import get_game_window_target_by_token, load_image, resize_window
from src.runtime_policy import RUNTIME_POLICY

class GameWindowCapturor:
    '''
    GameWindowCapturor
    '''
    def __init__(self, cfg, test_image_name = None):
        self.cfg = cfg
        self.frame = None
        self.frame_received_at = None
        self.frame_sequence = -1
        self.lock = threading.Lock()
        self.is_terminated = False
        self.is_capture_closed = False
        self.is_static_test_frame = test_image_name is not None
        self.frame_stale_timeout = max(
            0.1,
            RUNTIME_POLICY.frame_stale_timeout_seconds,
        )
        self.fps = 0
        self.fps_limit = RUNTIME_POLICY.capture_fps
        self.t_last_run = 0.0
        self.capture_control = None
        self.window_title = ""
        self.window_hwnd = None

        # If use test image as input, disable the whole capture thread
        if test_image_name is not None:
            self.frame = load_image(f"test/{test_image_name}.png")
            return

        # Get game window title
        target = get_game_window_target_by_token(cfg["game_window"]["title"])
        if target is None:
            raise RuntimeError(
                f"[GameWindowCapturor] Unable to find window title containing: {cfg['game_window']['title']}"
            )
        else:
            self.window_hwnd, self.window_title = target
            logger.info(f"[GameWindowCapturor] Found game window title: {self.window_title}")

        if cfg["game_window"].get("resize_on_start", True):
            resize_window(self.window_title, width=1296, height=759)

        # Create capture handler
        self.capture = WindowsCapture(window_name=self.window_title)
        self.capture.event(self.on_frame_arrived)
        self.capture.event(self.on_closed)

        # Start capturing thread
        self.capture_control = self.capture.start_free_threaded()

        logger.info("[GameWindowCapturor] Init done")

    def on_frame_arrived(self, frame: Frame,
                         capture_control: InternalCaptureControl):
        '''
        Frame arrived callback: store frame into buffer with lock.
        '''
        with self.lock:
            self.frame = frame.frame_buffer
            self.frame_received_at = time.monotonic()
            self.frame_sequence += 1
            self.is_capture_closed = False
        self.limit_fps()

    def on_closed(self):
        '''
        Capture closed callback.
        '''
        with self.lock:
            self.frame = None
            self.frame_received_at = None
            self.is_capture_closed = True
        logger.warning("[GameWindowCapturor] closed.")
        cv2.destroyAllWindows()

    def get_frame(self):
        '''
        Safely get latest game window frame.
        '''
        packet = self.get_frame_packet()
        return None if packet is None else packet[0]

    def get_frame_packet(self):
        """Atomically return an owned frame with its capture time/sequence."""
        with self.lock:
            if self.frame is None:
                return None
            if not self.is_static_test_frame:
                if self.is_capture_closed or self.frame_received_at is None:
                    return None
                if time.monotonic() - self.frame_received_at > self.frame_stale_timeout:
                    return None
            if self.frame.ndim == 3 and self.frame.shape[2] == 3:
                owned_frame = self.frame.copy()
            elif self.frame.ndim == 3 and self.frame.shape[2] == 4:
                owned_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGRA2BGR)
            else:
                logger.error(
                    f"[GameWindowCapturor] Unsupported frame shape: {self.frame.shape}"
                )
                return None
            # A static debug image intentionally has no real capture age or
            # source sequence; callers may treat each control pass as fresh.
            captured_at = (
                time.monotonic() if self.is_static_test_frame else
                self.frame_received_at
            )
            source_sequence = (
                None if self.is_static_test_frame else
                int(getattr(self, "frame_sequence", 0))
            )
            return owned_frame, captured_at, source_sequence

    def stop(self):
        '''
        Stop capturing thread
        '''
        try:
            if self.capture_control is not None:
                self.capture_control.stop()
        finally:
            with self.lock:
                self.frame = None
                self.frame_received_at = None
                self.is_capture_closed = True
        logger.info("[GameWindowCapturor] Terminated")

    def limit_fps(self):
        '''
        Limit FPS
        '''
        # If the loop finished early, sleep to maintain target FPS
        target_duration = 1.0 / self.fps_limit  # seconds per frame
        frame_duration = time.time() - self.t_last_run
        if frame_duration < target_duration:
            time.sleep(target_duration - frame_duration)

        # Update FPS
        self.fps = round(1.0 / (time.time() - self.t_last_run))
        self.t_last_run = time.time()
        # logger.info(f"FPS = {self.fps}")
