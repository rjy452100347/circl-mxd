"""Compile foreground manual input and map positions into semantic segments."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

from src.engine.SemanticRoute import (
    SCHEMA_VERSION,
    calculate_ladder_mount_x,
    validate_route_document,
)


@dataclass(frozen=True)
class RouteRecordingSample:
    timestamp: float
    position: tuple[int, int] | None
    pressed_keys: frozenset[str]
    localization_score: float | None = None
    valid: bool = True

    @classmethod
    def create(
        cls,
        timestamp: float,
        position,
        pressed_keys: Iterable[str],
        localization_score: float | None = None,
        valid: bool = True,
    ):
        point = None if position is None else (int(position[0]), int(position[1]))
        return cls(
            float(timestamp),
            point,
            frozenset(str(key).lower() for key in pressed_keys),
            localization_score,
            bool(valid),
        )


class SemanticRouteRecorder:
    """Stateful 10 FPS route compiler independent of any GUI or capture API."""

    def __init__(
        self,
        *,
        map_id: str,
        route_index: int,
        canvas_size: tuple[int, int],
        jump_key: str,
        teleport_key: str = "",
        localization_max_score: float = 0.20,
        ladder_combo_window_seconds: float = 0.60,
        ladder_mount_timeout_seconds: float = 1.80,
        ladder_success_distance: int = 12,
        ladder_confirm_frames: int = 3,
        gap_break_seconds: float = 0.50,
        jump_break_distance: int = 8,
    ):
        if not jump_key:
            raise ValueError("录制器必须配置跳跃键。")
        self.map_id = str(map_id)
        self.route_index = int(route_index)
        self.canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self.jump_key = str(jump_key).lower()
        self.teleport_key = str(teleport_key).lower()
        self.localization_max_score = float(localization_max_score)
        self.ladder_combo_window_seconds = float(ladder_combo_window_seconds)
        self.ladder_mount_timeout_seconds = float(ladder_mount_timeout_seconds)
        self.ladder_success_distance = int(ladder_success_distance)
        self.ladder_confirm_frames = max(1, int(ladder_confirm_frames))
        self.gap_break_seconds = float(gap_break_seconds)
        self.jump_break_distance = int(jump_break_distance)
        self.segments: list[dict] = []
        self._redo_stack: list[list[dict]] = []
        self._serial = 1
        self._previous_keys = frozenset()
        self._previous_position: tuple[int, int] | None = None
        self._last_valid_at: float | None = None
        self._gap_started_at: float | None = None
        self._walk_segment: dict | None = None
        self._pending_jump: dict | None = None
        self._ladder: dict | None = None
        self._drop: dict | None = None
        self._pending_discontinuous_position: tuple[int, int] | None = None

    @property
    def dirty(self):
        return bool(
            self.segments or self._walk_segment or self._pending_jump
            or self._ladder or self._drop
        )

    def _identifier(self, prefix):
        identifier = f"{prefix}_{self._serial:03d}"
        self._serial += 1
        return identifier

    @staticmethod
    def _distance(first, second):
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _push_segment(self, segment):
        self.segments.append(copy.deepcopy(segment))
        self._redo_stack.clear()

    def _finish_walk(self):
        if self._walk_segment is None:
            return
        if len(self._walk_segment["points"]) >= 2:
            self._push_segment(self._walk_segment)
        self._walk_segment = None

    def _finish_pending_jump(self):
        if self._pending_jump is None:
            return
        pending = self._pending_jump
        self._push_segment({
            "id": self._identifier("jump"),
            "type": "jump",
            "anchor": list(pending["anchor"]),
            "direction": pending["direction"],
        })
        self._pending_jump = None

    def _finish_ladder(self, position, exit_direction):
        ladder = self._ladder
        if ladder is None:
            return
        points = ladder["points"]
        if points[-1] != list(position):
            points.append(list(position))
        if len(points) < 2:
            points.append(list(position))
        approach = list(ladder["approach"])
        segment = {
            "id": self._identifier("ladder"),
            "type": "ladder",
            "approach": approach,
            "mount": [int(approach[0]), int(approach[1])],
            "mount_direction": ladder["mount_direction"],
            "points": points,
            "exit": list(position),
            "exit_direction": exit_direction,
        }
        mount_x = calculate_ladder_mount_x(
            segment, minimum_rise=self.ladder_success_distance
        )
        if mount_x is not None:
            segment["mount"][0] = mount_x
        self._push_segment(segment)
        self._ladder = None

    def _finish_drop(self, position):
        drop = self._drop
        if drop is None:
            return
        self._push_segment({
            "id": self._identifier("drop"),
            "type": "drop",
            "anchor": list(drop["anchor"]),
            "landing": list(position),
        })
        self._drop = None

    def break_continuity(self):
        self._finish_walk()
        self._pending_jump = None
        self._ladder = None
        self._drop = None
        self._previous_position = None
        self._pending_discontinuous_position = None

    def _valid_sample(self, sample):
        return (
            sample.valid
            and sample.position is not None
            and (
                sample.localization_score is None
                or sample.localization_score <= self.localization_max_score
            )
        )

    def _horizontal_direction(self, keys):
        left = "left" in keys
        right = "right" in keys
        if left == right:
            return "none"
        return "left" if left else "right"

    def _record_walk(self, position, direction):
        if direction not in {"left", "right"}:
            self._finish_walk()
            return
        if (
            self._walk_segment is None
            or self._walk_segment["direction"] != direction
        ):
            self._finish_walk()
            self._walk_segment = {
                "id": self._identifier("walk"),
                "type": "walk",
                "direction": direction,
                "points": [list(position)],
            }
            return
        last = tuple(self._walk_segment["points"][-1])
        if self._distance(last, position) < 1:
            return
        delta_x = position[0] - last[0]
        if direction == "right" and delta_x < 0:
            return
        if direction == "left" and delta_x > 0:
            return
        self._walk_segment["points"].append(list(position))

    def _update_pending_jump(self, sample, direction):
        pending = self._pending_jump
        if pending is None:
            return False
        elapsed = sample.timestamp - pending["timestamp"]
        if "up" in sample.pressed_keys and elapsed <= self.ladder_combo_window_seconds:
            pending["up_seen"] = True
        last_position = pending["candidate_points"][-1]
        if pending["up_seen"] and list(sample.position) != last_position:
            pending["candidate_points"].append(list(sample.position))
        if pending["up_seen"]:
            if int(sample.position[1]) < int(pending["last_y"]):
                pending["upward_frames"] += 1
            elif int(sample.position[1]) > int(pending["last_y"]):
                pending["upward_frames"] = 0
            pending["last_y"] = int(sample.position[1])
        if (
            pending["up_seen"]
            and sample.position[1] <= pending["anchor"][1] - self.ladder_success_distance
            and pending["upward_frames"] >= self.ladder_confirm_frames
        ):
            self._ladder = {
                "approach": pending["anchor"],
                "mount_direction": pending["direction"],
                "points": copy.deepcopy(pending["candidate_points"]),
                "exit_armed": pending["direction"] == "none",
            }
            self._pending_jump = None
            return True
        if (
            (not pending["up_seen"] and elapsed > self.ladder_combo_window_seconds)
            or (pending["up_seen"] and "up" not in sample.pressed_keys)
            or elapsed > self.ladder_mount_timeout_seconds
        ):
            self._finish_pending_jump()
        return False

    def feed(self, sample: RouteRecordingSample):
        """Consume one already-localized frame; never performs capture or input."""
        keys = sample.pressed_keys
        if not self._valid_sample(sample):
            if self._gap_started_at is None:
                self._gap_started_at = sample.timestamp
            self._finish_walk()
            self._previous_keys = keys
            return

        position = sample.position
        assert position is not None
        if self._gap_started_at is not None:
            gap = sample.timestamp - self._gap_started_at
            distance = (
                self._distance(position, self._previous_position)
                if self._previous_position is not None else 0
            )
            if gap > self.gap_break_seconds or distance > self.jump_break_distance:
                self.break_continuity()
            self._gap_started_at = None
        elif (
            self._previous_position is not None
            and self._pending_jump is None
            and self._ladder is None
            and self._drop is None
            and self._distance(position, self._previous_position)
            > self.jump_break_distance
        ):
            # Reject an isolated abnormal pose. A new distant location is
            # accepted only after a second consistent frame and starts a new
            # continuity island; the first candidate itself is never saved.
            if (
                self._pending_discontinuous_position is not None
                and self._distance(position, self._pending_discontinuous_position)
                <= self.jump_break_distance
            ):
                self._finish_walk()
                self._previous_position = None
                self._pending_discontinuous_position = None
            else:
                self._pending_discontinuous_position = position
                self._finish_walk()
                self._previous_keys = keys
                return
        else:
            self._pending_discontinuous_position = None

        direction = self._horizontal_direction(keys)
        jump_rising = (
            self.jump_key in keys and self.jump_key not in self._previous_keys
        )
        teleport_rising = bool(
            self.teleport_key
            and self.teleport_key in keys
            and self.teleport_key not in self._previous_keys
        )

        if self._ladder is not None:
            points = self._ladder["points"]
            if self._distance(tuple(points[-1]), position) >= 1:
                points.append(list(position))
            if "up" in keys and direction == "none":
                self._ladder["exit_armed"] = True
            horizontal_exit = direction in {"left", "right"} and (
                self._ladder["exit_armed"]
                or direction != self._ladder["mount_direction"]
            )
            if "up" not in keys or horizontal_exit:
                self._finish_ladder(position, direction)
            self._previous_keys = keys
            self._previous_position = position
            self._last_valid_at = sample.timestamp
            return

        if self._drop is not None:
            if (
                self.jump_key not in keys
                and "down" not in keys
                and position[1] >= self._drop["anchor"][1] + 4
            ):
                self._finish_drop(position)
            self._previous_keys = keys
            self._previous_position = position
            self._last_valid_at = sample.timestamp
            return

        self._update_pending_jump(sample, direction)
        if self._ladder is not None:
            self._previous_keys = keys
            self._previous_position = position
            self._last_valid_at = sample.timestamp
            return

        if jump_rising:
            self._finish_walk()
            if "down" in keys:
                self._drop = {"anchor": position}
            else:
                self._pending_jump = {
                    "anchor": position,
                    "direction": direction,
                    "timestamp": sample.timestamp,
                    "up_seen": "up" in keys,
                    "candidate_points": [list(position)],
                    "last_y": int(position[1]),
                    "upward_frames": 0,
                }
        elif teleport_rising:
            self._finish_walk()
            teleport_direction = (
                direction if direction != "none"
                else "up" if "up" in keys
                else "down" if "down" in keys
                else None
            )
            if teleport_direction is not None:
                self._push_segment({
                    "id": self._identifier("teleport"),
                    "type": "teleport",
                    "anchor": list(position),
                    "direction": teleport_direction,
                })
        elif self._pending_jump is None:
            self._record_walk(position, direction)

        self._previous_keys = keys
        self._previous_position = position
        self._last_valid_at = sample.timestamp

    def undo(self):
        self._finish_walk()
        if not self.segments:
            return False
        self._redo_stack.append(copy.deepcopy(self.segments))
        self.segments.pop()
        return True

    def redo(self):
        if not self._redo_stack:
            return False
        self.segments = self._redo_stack.pop()
        return True

    def finish_goal(self, position):
        point = (int(position[0]), int(position[1]))
        self._finish_walk()
        self._finish_pending_jump()
        if self._ladder is not None:
            self._finish_ladder(point, "none")
        if self._drop is not None:
            self._finish_drop(point)
        self.segments = [
            segment for segment in self.segments if segment["type"] != "goal"
        ]
        self._push_segment({
            "id": self._identifier("goal"),
            "type": "goal",
            "position": list(point),
            "radius": 6,
        })
        return self.document()

    def document(self, *, draft=False):
        document = {
            "schema_version": SCHEMA_VERSION,
            "map_id": self.map_id,
            "route_index": self.route_index,
            "canvas_size": [self.canvas_size[0], self.canvas_size[1]],
            "loop": True,
            "segments": copy.deepcopy(self.segments),
        }
        if draft:
            document["draft"] = True
        return validate_route_document(document, allow_draft=draft)
