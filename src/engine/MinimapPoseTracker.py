"""Continuity-guarded minimap-to-map localization shared by runtime and studio."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class MinimapPoseSnapshot:
    valid: bool
    raw_position: tuple[int, int] | None
    stable_position: tuple[int, int] | None
    camera_position: tuple[int, int] | None
    score: float | None
    source: str
    reason: str


class MinimapPoseTracker:
    """Reject discontinuous template matches before they reach route control."""

    def __init__(
        self,
        matcher: Callable,
        *,
        max_score: float = 0.20,
        max_frame_jump: int = 8,
        local_failures_before_global: int = 3,
        global_confirm_frames: int = 2,
        global_candidate_tolerance: int = 2,
        median_window: int = 3,
    ):
        self.matcher = matcher
        self.max_score = float(max_score)
        self.max_frame_jump = max(1, int(max_frame_jump))
        self.local_failures_before_global = max(
            1, int(local_failures_before_global)
        )
        self.global_confirm_frames = max(1, int(global_confirm_frames))
        self.global_candidate_tolerance = max(
            0, int(global_candidate_tolerance)
        )
        self.median_window = max(1, int(median_window))
        self.reset()

    @staticmethod
    def _distance(first, second):
        return abs(int(first[0]) - int(second[0])) + abs(
            int(first[1]) - int(second[1])
        )

    def reset(self):
        self._last_sequence = None
        self._accepted_at = None
        self.last_camera: tuple[int, int] | None = None
        self.last_raw_position: tuple[int, int] | None = None
        self._positions = deque(maxlen=self.median_window)
        self._local_failures = 0
        self._pending_global: tuple[int, int] | None = None
        self._pending_global_frames = 0
        self.last_snapshot = MinimapPoseSnapshot(
            False, None, None, None, None, "none", "reset"
        )

    def shift_origin(self, dx, dy):
        """Shift accepted coordinates after Route Studio grows map borders."""
        dx, dy = int(dx), int(dy)
        if self.last_camera is not None:
            self.last_camera = (
                self.last_camera[0] + dx,
                self.last_camera[1] + dy,
            )
        if self.last_raw_position is not None:
            self.last_raw_position = (
                self.last_raw_position[0] + dx,
                self.last_raw_position[1] + dy,
            )
        self._positions = deque(
            ((x + dx, y + dy) for x, y in self._positions),
            maxlen=self.median_window,
        )
        if self._pending_global is not None:
            self._pending_global = (
                self._pending_global[0] + dx,
                self._pending_global[1] + dy,
            )
        snapshot = self.last_snapshot
        shifted_raw = (
            None if snapshot.raw_position is None else
            (snapshot.raw_position[0] + dx, snapshot.raw_position[1] + dy)
        )
        shifted_stable = (
            None if snapshot.stable_position is None else
            (snapshot.stable_position[0] + dx, snapshot.stable_position[1] + dy)
        )
        shifted_camera = (
            None if snapshot.camera_position is None else
            (snapshot.camera_position[0] + dx, snapshot.camera_position[1] + dy)
        )
        self.last_snapshot = MinimapPoseSnapshot(
            snapshot.valid, shifted_raw, shifted_stable, shifted_camera,
            snapshot.score, snapshot.source, snapshot.reason,
        )

    def _snapshot(self, valid, raw, stable, camera, score, source, reason):
        snapshot = MinimapPoseSnapshot(
            bool(valid), raw, stable, camera,
            None if score is None else float(score), str(source), str(reason),
        )
        self.last_snapshot = snapshot
        return snapshot

    def _stable_position(self):
        xs = [point[0] for point in self._positions]
        ys = [point[1] for point in self._positions]
        return (
            int(round(statistics.median(xs))),
            int(round(statistics.median(ys))),
        )

    def _accept(self, camera, raw, score, source, *, reset_history=False):
        camera = int(camera[0]), int(camera[1])
        raw = int(raw[0]), int(raw[1])
        if reset_history:
            self._positions.clear()
        self.last_camera = camera
        self.last_raw_position = raw
        self._positions.append(raw)
        self._local_failures = 0
        self._pending_global = None
        self._pending_global_frames = 0
        return self._snapshot(
            True, raw, self._stable_position(), camera, score, source, "ok"
        )

    def invalidate_observation(self, reason='minimap_invalid'):
        """Break consecutive confirmation, keeping route coordinates/retries."""
        self._pending_global = None
        self._pending_global_frames = 0
        return self._snapshot(False, None, None, None, None, 'none', reason)

    def update(
        self, map_image, minimap, player, *, mask=None, offset=(0, 0),
        allow_large_jump=False, sequence=None, produced_at=None, now=None,
    ):
        import time
        now = time.monotonic() if now is None else now
        if produced_at is not None and not 0 <= now-produced_at <= .30:
            return self.invalidate_observation('frame_stale')
        if sequence is not None and self._last_sequence is not None and sequence <= self._last_sequence:
            return self.last_snapshot
        self._last_sequence = sequence
        global_search = self.last_camera is None or self._local_failures >= self.local_failures_before_global
        kwargs = {"mask": mask, "search_mode": "global" if global_search else "local",
                  "global_threshold": self.max_score, "return_margin": True}
        if not global_search:
            kwargs["last_result"] = self.last_camera
        result = self.matcher(map_image, minimap, **kwargs)
        camera, score, local = result[:3]
        margin = result[3] if len(result) > 3 else None
        camera = tuple(int(v) for v in camera)
        raw = (camera[0]+int(player[0])+int(offset[0]), camera[1]+int(player[1])+int(offset[1]))
        source = "global" if global_search else "local"
        if not math.isfinite(float(score)) or float(score) > self.max_score:
            self._local_failures += 1
            self._pending_global_frames = 0
            return self._snapshot(False, raw, None, camera, score, source, "score")
        if margin is not None and (math.isnan(margin) or margin < .02):
            self._local_failures += 1
            self._pending_global_frames = 0
            return self._snapshot(False, raw, None, camera, score, source, "map_ambiguous")
        jump = 0 if self.last_raw_position is None else self._distance(raw, self.last_raw_position)
        dt = .1 if produced_at is None or self._accepted_at is None else max(.1, min(.3, produced_at-self._accepted_at))
        tolerance = self.max_frame_jump * dt / .1
        if not global_search:
            if jump > tolerance and not allow_large_jump:
                self._local_failures += 1
                return self._snapshot(False, raw, None, camera, score, source, "position_jump")
            self._accepted_at = produced_at
            return self._accept(camera, raw, score, "local",
                                reset_history=allow_large_jump and jump > tolerance)
        if self.last_camera is None or jump > tolerance:
            if self._pending_global is not None and self._distance(camera, self._pending_global) <= self.global_candidate_tolerance:
                self._pending_global_frames += 1
            else:
                self._pending_global, self._pending_global_frames = camera, 1
            if self._pending_global_frames < self.global_confirm_frames:
                return self._snapshot(False, raw, None, camera, score, source, "global_unconfirmed")
        self._accepted_at = produced_at
        return self._accept(camera, raw, score, "global_confirmed", reset_history=True)
