"""Live capture/localization session used by Route Studio.

The session deliberately does not own a render loop.  The Qt window calls
``tick`` at the immutable route-recorder rate, so recording cannot add a
second high-frequency capture or inference pipeline.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from src.engine.MinimapPoseTracker import MinimapPoseTracker
from src.engine.SemanticRouteRecorder import (
    RouteRecordingSample,
    SemanticRouteRecorder,
)
from src.input.GameWindowCapturor import GameWindowCapturor
from src.input.KeyBoardListener import KeyBoardListener
from src.utils.common import (
    find_pattern_sqdiff,
    get_minimap_loc_size,
    get_player_location_on_minimap,
    prepare_game_frame,
)


class RouteStudioSessionError(RuntimeError):
    pass


class RouteStudioSession:
    """Own capture, foreground input observation and minimap localization."""

    def __init__(self, cfg, *, map_id, base_bgr=None, new_map=False):
        self.cfg = cfg
        self.map_id = str(map_id)
        self.new_map = bool(new_map)
        self.base_bgr = None if base_bgr is None else base_bgr.copy()
        if not self.new_map and self.base_bgr is None:
            raise RouteStudioSessionError("现有地图必须先加载 map.png。")
        self.capture = None
        self.keyboard = None
        self.recorder = None
        self.map_capture_active = False
        self.map_dirty = False
        self.route_recording_active = False
        self.minimap_roi = None
        self.last_minimap_global = None
        self.last_position = None
        self.last_score = None
        self.last_status = "未开始捕获"
        self.pose_tracker = MinimapPoseTracker(
            lambda *args, **kwargs: find_pattern_sqdiff(*args, **kwargs),
            max_score=float(
                self.cfg.get("route", {}).get("localization_max_score", 0.20)
            ),
        )
        self.last_pose_snapshot = self.pose_tracker.last_snapshot

    def _reset_pose_tracker(self):
        self.pose_tracker.reset()
        self.last_pose_snapshot = self.pose_tracker.last_snapshot
        self.last_minimap_global = None

    @property
    def active(self):
        return self.capture is not None

    def start(self):
        if self.active:
            return
        self.capture = GameWindowCapturor(self.cfg)
        self.keyboard = KeyBoardListener(self.cfg, is_autobot=False)
        self.last_status = "捕获已启动"

    def stop(self):
        self.route_recording_active = False
        self.map_capture_active = False
        if self.keyboard is not None:
            self.keyboard.stop()
            self.keyboard = None
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        self._reset_pose_tracker()
        self.last_status = "捕获已停止"

    def begin_map_capture(self):
        if not self.new_map:
            raise RouteStudioSessionError("现有地图的 map.png 已锁定。")
        self.start()
        self.map_capture_active = True

    def end_map_capture(self):
        self.map_capture_active = False

    def discard_map_capture(self, restored_base=None):
        """Drop only unsaved stitched pixels and reset localization history."""
        self.map_capture_active = False
        self.base_bgr = (
            None if restored_base is None
            else np.ascontiguousarray(restored_base.copy())
        )
        self.map_dirty = False
        self._reset_pose_tracker()
        self.last_position = None
        self.last_score = None
        self.last_status = "已放弃未保存的地图捕获数据"

    def discard_route_recording(self):
        """Drop an unfinished recorder without changing the editor document."""
        self.route_recording_active = False
        self.recorder = None
        self.last_position = None
        self.last_score = None
        self.last_status = "已放弃未完成的路线录制数据"

    def begin_route_recording(self, route_index):
        self.start()
        if self.base_bgr is None:
            raise RouteStudioSessionError("请先捕获新地图底图。")
        height, width = self.base_bgr.shape[:2]
        route_cfg = self.cfg.get("route", {})
        self.recorder = SemanticRouteRecorder(
            map_id=self.map_id,
            route_index=int(route_index),
            canvas_size=(width, height),
            jump_key=self.cfg.get("key", {}).get("jump", ""),
            teleport_key=self.cfg.get("key", {}).get("teleport", ""),
            localization_max_score=float(
                route_cfg.get("localization_max_score", 0.20)
            ),
            ladder_combo_window_seconds=float(
                route_cfg.get("ladder_record_combo_seconds", 0.60)
            ),
            ladder_mount_timeout_seconds=float(
                route_cfg.get("ladder_record_timeout_seconds", 1.80)
            ),
            ladder_success_distance=int(
                route_cfg.get("ladder_record_success_distance", 12)
            ),
            ladder_confirm_frames=int(
                route_cfg.get("ladder_record_confirm_frames", 3)
            ),
        )
        self.route_recording_active = True

    def pause_route_recording(self):
        self.route_recording_active = False

    def apply_minimap_roi(self, roi):
        """Apply one already-validated ROI and reset localization history."""
        minimap_cfg = self.cfg.setdefault("minimap", {})
        if roi is None:
            minimap_cfg.pop("roi", None)
        else:
            minimap_cfg["roi"] = list(roi)
        self.minimap_roi = None
        self._reset_pose_tracker()
        self.last_position = None
        self.last_score = None
        self.last_status = "小地图 ROI 已更新，等待重新定位"

    def capture_calibration_frame(self):
        """Return one prepared frame while reusing this session's capturer."""
        self.start()
        raw = self.capture.get_frame() if self.capture is not None else None
        frame = prepare_game_frame(raw, self.cfg) if raw is not None else None
        if frame is None:
            return None
        return np.ascontiguousarray(frame.copy())

    @staticmethod
    def _route_mask(minimap):
        return np.any(minimap != [0, 0, 0], axis=2).astype(np.uint8) * 255

    @staticmethod
    def _hide_player_dot(minimap, player):
        result = minimap.copy()
        px, py = player
        x0, x1 = max(0, px - 5), min(result.shape[1], px + 6)
        y0, y1 = max(0, py - 5), min(result.shape[0], py + 6)
        result[y0:y1, x0:x1] = (0, 0, 0)
        return result

    def _initialize_new_map(self, minimap, player):
        padding = 30
        clean = self._hide_player_dot(minimap, player)
        self.base_bgr = cv2.copyMakeBorder(
            clean,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        self.last_minimap_global = (padding, padding)
        self.pose_tracker.reset()
        self.map_dirty = True

    def _ensure_capacity(self, x, y, width, height):
        assert self.base_bgr is not None
        padding = 30
        map_height, map_width = self.base_bgr.shape[:2]
        top = max(0, padding - y)
        left = max(0, padding - x)
        bottom = max(0, y + height + padding - map_height)
        right = max(0, x + width + padding - map_width)
        if not any((top, left, bottom, right)):
            return x, y
        self.base_bgr = cv2.copyMakeBorder(
            self.base_bgr,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        self.pose_tracker.shift_origin(left, top)
        if self.pose_tracker.last_camera is not None:
            self.last_minimap_global = self.pose_tracker.last_camera
        return x + left, y + top

    def _localize(self, minimap, player):
        if self.base_bgr is None:
            self._initialize_new_map(minimap, player)
        assert self.base_bgr is not None
        mask = self._route_mask(minimap)
        offset = self.cfg.get("minimap", {}).get("offset", [0, 0])
        snapshot = self.pose_tracker.update(
            self.base_bgr,
            minimap,
            player,
            mask=mask,
            offset=offset,
        )
        self.last_pose_snapshot = snapshot
        score = float(snapshot.score if snapshot.score is not None else 1.0)
        if not snapshot.valid or snapshot.camera_position is None:
            return None, score
        x, y = snapshot.camera_position
        if self.new_map and self.map_capture_active:
            old_x, old_y = x, y
            x, y = self._ensure_capacity(
                x, y, minimap.shape[1], minimap.shape[0]
            )
            if (x, y) != (old_x, old_y):
                snapshot = self.pose_tracker.last_snapshot
                self.last_pose_snapshot = snapshot
            clean = self._hide_player_dot(minimap, player)
            target = self.base_bgr[
                y:y + minimap.shape[0], x:x + minimap.shape[1]
            ]
            blank = np.all(target == [0, 0, 0], axis=2)
            visible = np.any(clean != [0, 0, 0], axis=2)
            copy_mask = blank & visible
            target[copy_mask] = clean[copy_mask]
            if np.any(copy_mask):
                self.map_dirty = True
        self.last_minimap_global = (x, y)
        return snapshot.stable_position, score

    def tick(self, now=None):
        now = time.monotonic() if now is None else float(now)
        if not self.active:
            return None
        raw = self.capture.get_frame()
        frame = prepare_game_frame(raw, self.cfg) if raw is not None else None
        if frame is None:
            self._feed_invalid(now, "游戏画面不可用")
            return None
        roi = get_minimap_loc_size(frame, self.cfg)
        if roi is None:
            self._feed_invalid(now, "小地图 ROI 不可用")
            return None
        x, y, width, height = (int(value) for value in roi)
        self.minimap_roi = (x, y, width, height)
        minimap = np.ascontiguousarray(frame[y:y + height, x:x + width].copy())
        player = get_player_location_on_minimap(
            minimap,
            minimap_player_color=self.cfg["minimap"]["player_color"],
            player_hsv=self.cfg["minimap"].get("player_hsv"),
        )
        if player is None:
            self._feed_invalid(now, "未识别小地图人物点")
            return None
        try:
            position, score = self._localize(minimap, player)
        except cv2.error as exc:
            self._feed_invalid(now, f"地图定位失败：{exc}")
            return None
        if position is None:
            self._feed_invalid(
                now, f"地图定位尚未稳定：{self.last_pose_snapshot.reason}"
            )
            return None
        self.last_position = position
        self.last_score = score
        max_score = float(self.cfg.get("route", {}).get("localization_max_score", 0.20))
        valid = score <= max_score
        self.last_status = (
            f"定位{'\u6709\u6548' if valid else '\u8d85\u9608\u503c'} "
            f"score={score:.4f} player={position}"
        )
        if self.route_recording_active and self.recorder is not None:
            keys = self.keyboard.key_pressing if self.keyboard is not None else ()
            self.recorder.feed(RouteRecordingSample.create(
                now, position, keys, score, valid
            ))
        return {
            "frame": frame,
            "minimap": minimap,
            "position": position,
            "score": score,
            "valid": valid,
        }

    def _feed_invalid(self, now, status):
        self.last_status = status
        self.last_position = None
        self.last_score = None
        if self.route_recording_active and self.recorder is not None:
            keys = self.keyboard.key_pressing if self.keyboard is not None else ()
            self.recorder.feed(RouteRecordingSample.create(
                now, None, keys, None, False
            ))

    def save_map_atomic(self, path):
        if self.base_bgr is None:
            raise RouteStudioSessionError("当前没有可保存的地图底图。")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(".png", self.base_bgr)
        if not success:
            raise RouteStudioSessionError("无法编码 map.png。")
        handle, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(handle)
        try:
            Path(temporary).write_bytes(encoded.tobytes())
            os.replace(temporary, target)
            self.map_dirty = False
        finally:
            try:
                Path(temporary).unlink()
            except OSError:
                pass
