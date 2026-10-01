"""Ordered semantic route follower without path finding."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from src.engine.RouteNavigator import RouteDecision
from src.engine.SemanticRoute import SemanticRouteError, validate_route_document


@dataclass(frozen=True)
class SemanticRouteDiagnostics:
    route_index: int
    segment_index: int
    point_index: int
    traversal_state: str
    reason: str
    mount_direction: str = "none"
    align_error: int | None = None
    settle_frames: int = 0
    ladder_attempts: int = 0
    candidate_center: tuple[int, int] | None = None
    candidate_offset: int = 0
    max_rise: int = 0
    pulse_count: int = 0
    cached_offset: int | None = None


class SemanticRouteNavigator:
    """Replay recorded segments in order with bounded local reacquisition."""

    LADDER_CANDIDATE_OFFSETS = (0, -2, 2, -4, 4)

    def __init__(
        self,
        documents: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        lookahead_points: int = 6,
        local_back_points: int = 5,
        local_forward_points: int = 30,
        walk_tolerance: int = 3,
        direction_hysteresis: int = 3,
        reacquire_radius: int = 18,
        ambiguity_margin: int = 4,
        goal_radius: int = 6,
        ladder_align_tolerance: int = 5,
        ladder_retry_frames: int = 5,
        ladder_success_distance: int = 12,
        ladder_max_attempts: int = 5,
        ladder_precision_tolerance: int = 1,
        ladder_settle_frames: int = 2,
        ladder_forward_hold_frames: int = 4,
        ladder_retry_runup_distance: int = 8,
        ladder_attempt_timeout: float = 1.5,
        ladder_confirm_frames: int = 3,
        event_commit_seconds: float = 0.45,
    ):
        if not documents:
            raise SemanticRouteError("语义路线至少需要一个文档。")
        self.routes = tuple(validate_route_document(document) for document in documents)
        self.lookahead_points = max(1, int(lookahead_points))
        self.local_back_points = max(0, int(local_back_points))
        self.local_forward_points = max(1, int(local_forward_points))
        self.walk_tolerance = max(1, int(walk_tolerance))
        self.direction_hysteresis = max(0, int(direction_hysteresis))
        self.reacquire_radius = max(1, int(reacquire_radius))
        self.ambiguity_margin = max(1, int(ambiguity_margin))
        self.goal_radius = max(1, int(goal_radius))
        self.ladder_align_tolerance = max(1, int(ladder_align_tolerance))
        self.ladder_retry_frames = max(1, int(ladder_retry_frames))
        self.ladder_success_distance = max(1, int(ladder_success_distance))
        self.ladder_max_attempts = max(1, int(ladder_max_attempts))
        self.ladder_precision_tolerance = max(
            0, min(int(ladder_precision_tolerance), self.ladder_align_tolerance)
        )
        self.ladder_settle_frames = max(1, int(ladder_settle_frames))
        self.ladder_forward_hold_frames = max(1, int(ladder_forward_hold_frames))
        self.ladder_retry_runup_distance = max(1, int(ladder_retry_runup_distance))
        self.ladder_attempt_timeout = max(0.2, float(ladder_attempt_timeout))
        self.ladder_confirm_frames = max(1, int(ladder_confirm_frames))
        self.event_commit_seconds = max(0.05, float(event_commit_seconds))
        self.route_position = 0
        self.segment_position = 0
        self.point_position = 0
        self._last_direction: str | None = None
        self._traversal_state = ""
        self._traversal_started_at = 0.0
        self._traversal_start = (0, 0)
        self._ladder_frames = 0
        self._ladder_attempts = 0
        self._ladder_exit_frames = 0
        self._ladder_exit_start = (0, 0)
        self._ladder_settle_count = 0
        self._ladder_last_player: tuple[int, int] | None = None
        self._ladder_fine_move_frame = True
        self._ladder_attempt_started_at = 0.0
        self._ladder_attempt_start = (0, 0)
        self._ladder_confirm_count = 0
        self._ladder_peak_rise = 0
        self._ladder_forward_frames = 0
        self._ladder_retry_settle_count = 0
        self._ladder_suspended_at: float | None = None
        self._ladder_failure_reason = ""
        self._ladder_candidate_index = 0
        self._ladder_active_offset = 0
        self._ladder_pulse_count = 0
        self._ladder_last_pulse_frame = 0
        self._ladder_success_offsets: dict[tuple[int, str], int] = {}
        self._stop_emitted = False
        self._reacquire_requested = False
        self._local_reacquire_failures = 0
        self._route_completed = False
        self._last_reason = "semantic_ready"

    @property
    def route_index(self):
        return self.route_position

    @property
    def has_pending_traversal(self):
        return self._traversal_state not in {
            "", "coarse_align", "fine_align", "settle", "forward_runup",
            "forward_reset", "forward_reset_settle",
        }

    @property
    def requires_route_exclusive_control(self):
        """A semantic ladder owns motion from approach through exit."""
        return not self._route_completed and self._segment.get("type") == "ladder"

    @property
    def diagnostics(self):
        segment = self._segment
        mount_direction = (
            segment.get("mount_direction", "none")
            if segment.get("type") == "ladder" else "none"
        )
        align_error = None
        if segment.get("type") == "ladder" and self._ladder_last_player is not None:
            align_error = self._candidate_mount()[0] - self._ladder_last_player[0]
        return SemanticRouteDiagnostics(
            self.route_position,
            self.segment_position,
            self.point_position,
            self._traversal_state or "none",
            self._last_reason,
            mount_direction,
            align_error,
            self._ladder_settle_count,
            self._ladder_attempts,
            self._candidate_mount() if segment.get("type") == "ladder" else None,
            self._ladder_active_offset,
            self._ladder_peak_rise,
            self._ladder_pulse_count,
            self._ladder_success_offsets.get(self._ladder_cache_key())
            if segment.get("type") == "ladder" else None,
        )

    @property
    def ladder_debug(self):
        """Return an immutable-friendly snapshot for both preview canvases."""
        segment = self._segment
        if segment.get("type") != "ladder":
            return None
        return {
            "approach": tuple(segment["approach"]),
            "mount": tuple(segment["mount"]),
            "candidate_mount": self._candidate_mount(),
            "candidate_approach": self._candidate_approach(),
            "candidate_offset": self._ladder_active_offset,
            "mount_direction": segment.get("mount_direction", "none"),
            "stage": self._traversal_state or "ready",
            "align_error": (
                None if self._ladder_last_player is None else
                self._candidate_mount()[0] - self._ladder_last_player[0]
            ),
            "settle_frames": self._ladder_settle_count,
            "attempts": self._ladder_attempts,
            "max_attempts": self.ladder_max_attempts,
            "pulse_count": self._ladder_pulse_count,
            "max_rise": self._ladder_peak_rise,
            "cached_offset": self._ladder_success_offsets.get(
                self._ladder_cache_key()
            ),
            "failure_reason": self._ladder_failure_reason,
        }

    def reset_events(self):
        self._clear_traversal()
        self._stop_emitted = False

    def reset_progress(self, route_index=0):
        self.route_position = int(route_index) % len(self.routes)
        self.segment_position = 0
        self.point_position = 0
        self._last_direction = None
        self.reset_events()
        self._ladder_success_offsets.clear()
        self._route_completed = False
        self._last_reason = "semantic_reset"

    def request_reacquire(self, reason="semantic_resume_reacquire"):
        """Discard stale cruise progress before the next fresh route frame."""
        if self._traversal_state in {
            "coarse_align", "fine_align", "settle", "forward_runup",
            "forward_reset", "forward_reset_settle",
        }:
            self._clear_traversal()
        if self.has_pending_traversal:
            return False
        self._reacquire_requested = True
        self._last_direction = None
        self._last_reason = str(reason)
        return True

    def _clear_traversal(self):
        self._traversal_state = ""
        self._traversal_started_at = 0.0
        self._ladder_frames = 0
        self._ladder_attempts = 0
        self._ladder_exit_frames = 0
        self._ladder_settle_count = 0
        self._ladder_last_player = None
        self._ladder_fine_move_frame = True
        self._ladder_attempt_started_at = 0.0
        self._ladder_confirm_count = 0
        self._ladder_peak_rise = 0
        self._ladder_forward_frames = 0
        self._ladder_retry_settle_count = 0
        self._ladder_suspended_at = None
        self._ladder_failure_reason = ""
        self._ladder_candidate_index = 0
        self._ladder_active_offset = 0
        self._ladder_pulse_count = 0
        self._ladder_last_pulse_frame = 0

    def suspend_traversal(self, now=None):
        """Freeze a committed ladder attempt across perception/healing gaps."""
        if self._segment.get("type") != "ladder":
            return False
        if self._ladder_suspended_at is None:
            self._ladder_suspended_at = (
                time.monotonic() if now is None else float(now)
            )
        return True

    def _resume_suspended_traversal(self, now):
        if self._ladder_suspended_at is None:
            return
        suspended_for = max(0.0, float(now) - self._ladder_suspended_at)
        if self._ladder_attempt_started_at:
            self._ladder_attempt_started_at += suspended_for
        self._ladder_suspended_at = None

    @staticmethod
    def _distance(first, second):
        return abs(int(first[0]) - int(second[0])) + abs(
            int(first[1]) - int(second[1])
        )

    @property
    def _document(self):
        return self.routes[self.route_position]

    @property
    def _segments(self):
        return self._document["segments"]

    @property
    def _segment(self):
        return self._segments[self.segment_position]

    def _decision(self, command, point=None, reason="semantic_route"):
        self._last_reason = reason
        segment = self._segment
        return RouteDecision(
            command,
            self.route_position,
            tuple(point) if point is not None else None,
            (self.route_position, segment["id"]),
            reason,
        )

    def _advance_segment(self):
        self._clear_traversal()
        self._stop_emitted = False
        self._last_direction = None
        self.point_position = 0
        self._local_reacquire_failures = 0
        self.segment_position += 1
        if self.segment_position >= len(self._segments):
            raise SemanticRouteError("路线在 Goal 之前意外结束。")

    def _candidate_points(self, segment):
        kind = segment["type"]
        if kind == "walk":
            return segment["points"]
        if kind == "ladder":
            return [segment["approach"], segment["mount"], *segment["points"], segment["exit"]]
        return []

    def _local_reacquire(self, player):
        segment = self._segment
        points = self._candidate_points(segment)
        if not points:
            return True
        if segment["type"] == "walk":
            start = max(0, self.point_position - self.local_back_points)
            stop = min(len(points), self.point_position + self.local_forward_points + 1)
        else:
            start, stop = 0, len(points)
        candidates = [
            (self._distance(player, points[index]), index)
            for index in range(start, stop)
        ]
        distance, index = min(candidates)
        if distance > self.reacquire_radius:
            return False
        if segment["type"] == "walk":
            self.point_position = index
        return True

    def _global_reacquire(self, player):
        per_segment = []
        for segment_index, segment in enumerate(self._segments):
            if segment["type"] not in {"walk", "ladder"}:
                continue
            points = self._candidate_points(segment)
            candidates = [
                (self._distance(player, point), point_index)
                for point_index, point in enumerate(points)
            ]
            if not candidates:
                continue
            distance, point_index = min(candidates)
            per_segment.append((distance, segment_index, point_index))
        if not per_segment:
            return False
        per_segment.sort()
        best = per_segment[0]
        if best[0] > self.reacquire_radius:
            return False
        if (
            len(per_segment) > 1
            and per_segment[1][0] - best[0] < self.ambiguity_margin
        ):
            self._last_reason = "semantic_reacquire_ambiguous"
            return False
        self.segment_position = best[1]
        segment = self._segment
        self.point_position = best[2] if segment["type"] == "walk" else 0
        self._clear_traversal()
        self._last_reason = "semantic_reacquired_global"
        return True

    def _adjacent_reacquire(self, player):
        candidates = []
        for segment_index in range(
            max(0, self.segment_position - 1),
            min(len(self._segments), self.segment_position + 2),
        ):
            segment = self._segments[segment_index]
            points = self._candidate_points(segment)
            for point_index, point in enumerate(points):
                candidates.append((
                    self._distance(player, point), segment_index, point_index
                ))
        if not candidates:
            return False
        candidates.sort()
        best = candidates[0]
        if best[0] > self.reacquire_radius:
            return False
        competing_segments = [
            item for item in candidates
            if item[1] != best[1]
        ]
        if (
            competing_segments
            and competing_segments[0][0] - best[0] < self.ambiguity_margin
        ):
            self._last_reason = "semantic_reacquire_ambiguous"
            return False
        self.segment_position = best[1]
        segment = self._segment
        self.point_position = best[2] if segment["type"] == "walk" else 0
        self._clear_traversal()
        self._last_reason = "semantic_reacquired_adjacent"
        return True

    def _ensure_reacquired(self, player):
        if self._traversal_state:
            return True
        if self._reacquire_requested:
            found = self._adjacent_reacquire(player)
            if found:
                self._reacquire_requested = False
                self._local_reacquire_failures = 0
                return True
            self._local_reacquire_failures += 1
            if self._local_reacquire_failures < 3:
                self._last_reason = "semantic_reacquire_wait"
                return False
            found = self._global_reacquire(player)
            if found:
                self._reacquire_requested = False
                self._local_reacquire_failures = 0
            return found
        if self._local_reacquire(player):
            self._local_reacquire_failures = 0
            return True
        if self._adjacent_reacquire(player):
            self._local_reacquire_failures = 0
            return True
        self._local_reacquire_failures += 1
        if self._local_reacquire_failures < 3:
            self._last_reason = "semantic_reacquire_wait"
            return False
        found = self._global_reacquire(player)
        if found:
            self._local_reacquire_failures = 0
        return found

    def _horizontal_approach(self, player, target, reason):
        dx = int(target[0]) - int(player[0])
        if abs(dx) <= self.walk_tolerance:
            return None
        direction = "right" if dx > 0 else "left"
        self._last_direction = direction
        return self._decision(
            f"{direction} none none", target, reason
        )

    def _walk(self, player):
        segment = self._segment
        points = segment["points"]
        candidates = []
        start = max(0, self.point_position - self.local_back_points)
        stop = min(len(points), self.point_position + self.local_forward_points + 1)
        for index in range(start, stop):
            candidates.append((self._distance(player, points[index]), index))
        distance, nearest = min(candidates)
        if distance > self.reacquire_radius:
            if not self._global_reacquire(player):
                return self._decision(
                    "stop stop none", player, self._last_reason or "semantic_route_lost"
                )
            return None
        self.point_position = nearest
        endpoint = points[-1]
        if (
            nearest == len(points) - 1
            or self._distance(player, endpoint) <= self.walk_tolerance
        ):
            self._advance_segment()
            return None
        target_index = min(len(points) - 1, nearest + self.lookahead_points)
        target = points[target_index]
        dx = int(target[0]) - int(player[0])
        recorded = segment["direction"]
        if abs(dx) <= self.direction_hysteresis:
            direction = self._last_direction or recorded
        else:
            direction = "right" if dx > 0 else "left"
        # A one-frame localization reversal must not flip a stable recorded
        # direction until its error exceeds the configured hysteresis.
        if direction != recorded and abs(dx) <= self.direction_hysteresis * 2:
            direction = recorded
        self._last_direction = direction
        return self._decision(
            f"{direction} none none", target, "semantic_walk"
        )

    def _event_approach(self, player, anchor, reason):
        if self._distance(player, anchor) <= self.walk_tolerance:
            return None
        return self._horizontal_approach(player, anchor, reason) or self._decision(
            "stop none none", anchor, reason
        )

    def _jump(self, player, now):
        segment = self._segment
        anchor = segment["anchor"]
        if not self._traversal_state:
            approach = self._event_approach(player, anchor, "semantic_jump_approach")
            if approach is not None:
                return approach
            self._traversal_state = "jump"
            self._traversal_started_at = now
            self._traversal_start = tuple(player)
        moved = abs(int(player[1]) - self._traversal_start[1]) >= 4
        if moved or now - self._traversal_started_at >= self.event_commit_seconds:
            self._advance_segment()
            return None
        direction = segment.get("direction", "none")
        return self._decision(
            f"{direction} none jump", anchor, "semantic_jump"
        )

    def _drop(self, player, now):
        segment = self._segment
        anchor = segment["anchor"]
        if not self._traversal_state:
            approach = self._event_approach(player, anchor, "semantic_drop_approach")
            if approach is not None:
                return approach
            self._traversal_state = "drop"
            self._traversal_started_at = now
            self._traversal_start = tuple(player)
        landing_y = int(segment["landing"][1])
        if int(player[1]) >= landing_y - self.walk_tolerance:
            self._advance_segment()
            return None
        return self._decision("none down jump", anchor, "semantic_drop")

    def _teleport(self, player, now):
        segment = self._segment
        anchor = segment["anchor"]
        if not self._traversal_state:
            approach = self._event_approach(player, anchor, "semantic_teleport_approach")
            if approach is not None:
                return approach
            self._traversal_state = "teleport"
            self._traversal_started_at = now
            self._traversal_start = tuple(player)
        moved = self._distance(player, self._traversal_start) >= 10
        if moved or now - self._traversal_started_at >= self.event_commit_seconds:
            self._advance_segment()
            return None
        direction = segment["direction"]
        command = (
            f"{direction} none teleport"
            if direction in {"left", "right"}
            else f"none {direction} teleport"
        )
        return self._decision(command, anchor, "semantic_teleport")

    @staticmethod
    def _horizontal_command_toward(player_x, target_x):
        return "right" if int(target_x) > int(player_x) else "left"

    def _ladder_cache_key(self):
        return self.route_position, str(self._segment.get("id", self.segment_position))

    def _candidate_offsets(self):
        learned = self._ladder_success_offsets.get(self._ladder_cache_key())
        offsets = list(self.LADDER_CANDIDATE_OFFSETS)
        if learned in offsets:
            offsets.remove(learned)
            offsets.insert(0, learned)
        return tuple(offsets[:self.ladder_max_attempts])

    def _select_ladder_candidate(self, index):
        offsets = self._candidate_offsets()
        self._ladder_candidate_index = max(0, int(index))
        if self._ladder_candidate_index >= len(offsets):
            return False
        self._ladder_active_offset = int(offsets[self._ladder_candidate_index])
        return True

    def _candidate_approach(self):
        point = self._segment["approach"]
        return int(point[0]) + self._ladder_active_offset, int(point[1])

    def _candidate_mount(self):
        point = self._segment["mount"]
        return int(point[0]) + self._ladder_active_offset, int(point[1])

    def _ladder_success_threshold(self):
        approach_y = int(self._segment["approach"][1])
        exit_y = int(self._segment["exit"][1])
        return min(
            self.ladder_success_distance,
            max(5, approach_y - exit_y - 2),
        )

    def _ladder_position_stable(self, player, *, required_frames):
        stable = (
            self._ladder_last_player is not None
            and abs(int(player[0]) - self._ladder_last_player[0]) <= 1
            and abs(int(player[1]) - self._ladder_last_player[1]) <= 1
        )
        self._ladder_settle_count = self._ladder_settle_count + 1 if stable else 1
        self._ladder_last_player = tuple(player)
        return self._ladder_settle_count >= int(required_frames)

    def _start_ladder_attempt(self, player, now, mount_direction):
        self._traversal_state = "confirm"
        self._traversal_start = tuple(player)
        self._ladder_attempt_start = tuple(player)
        self._ladder_attempt_started_at = float(now)
        self._ladder_attempts += 1
        self._ladder_confirm_count = 0
        self._ladder_peak_rise = 0
        self._ladder_forward_frames = 1 if mount_direction in {"left", "right"} else 0
        self._ladder_frames = 0
        self._ladder_last_pulse_frame = 0
        self._ladder_pulse_count += 1
        command = (
            "none up mount"
            if mount_direction == "none"
            else f"{mount_direction} up mount"
        )
        return self._decision(command, self._candidate_mount(), "semantic_ladder_mount")

    def _fail_ladder_attempt(self, player, reason):
        self._ladder_failure_reason = str(reason)
        self._ladder_settle_count = 0
        self._ladder_last_player = tuple(player)
        self._ladder_confirm_count = 0
        self._ladder_peak_rise = 0
        self._ladder_attempt_started_at = 0.0
        next_candidate = self._ladder_candidate_index + 1
        if not self._select_ladder_candidate(next_candidate):
            self._traversal_state = "failed"
            return self._decision(
                "stop stop none", self._candidate_mount(),
                "semantic_ladder_failed",
            )
        mount_direction = self._segment.get("mount_direction", "none")
        self._traversal_state = (
            "forward_reset"
            if mount_direction in {"left", "right"}
            else "coarse_align"
        )
        return self._decision(
            "stop stop none", self._candidate_mount(),
            f"semantic_ladder_retry_{reason}",
        )

    def _vertical_ladder_alignment(self, player, now):
        segment = self._segment
        approach = self._candidate_approach()
        mount = self._candidate_mount()
        dx = int(mount[0]) - int(player[0])
        dy = int(approach[1]) - int(player[1])
        self._ladder_last_player = tuple(player) \
            if self._ladder_last_player is None else self._ladder_last_player

        if abs(dy) > self.walk_tolerance:
            self._ladder_settle_count = 0
            self._ladder_last_player = tuple(player)
            return self._decision(
                "stop stop none", approach, "semantic_ladder_wrong_height"
            )
        if abs(dx) > self.ladder_align_tolerance:
            self._traversal_state = "coarse_align"
            self._ladder_settle_count = 0
            self._ladder_last_player = tuple(player)
            direction = self._horizontal_command_toward(player[0], mount[0])
            return self._decision(
                f"{direction} none none", mount, "semantic_ladder_coarse_align"
            )
        if abs(dx) > self.ladder_precision_tolerance:
            self._traversal_state = "fine_align"
            self._ladder_settle_count = 0
            self._ladder_last_player = tuple(player)
            move_frame = self._ladder_fine_move_frame
            self._ladder_fine_move_frame = not self._ladder_fine_move_frame
            if move_frame:
                direction = self._horizontal_command_toward(player[0], mount[0])
                return self._decision(
                    f"{direction} none none", mount, "semantic_ladder_fine_align"
                )
            return self._decision(
                "none none none", mount, "semantic_ladder_fine_brake"
            )

        self._traversal_state = "settle"
        if not self._ladder_position_stable(
            player, required_frames=self.ladder_settle_frames
        ):
            return self._decision(
                "none none none", mount, "semantic_ladder_settle"
            )
        return self._start_ladder_attempt(player, now, "none")

    def _forward_reset_target(self, mount_direction):
        approach_x = self._candidate_approach()[0]
        sign = 1 if mount_direction == "right" else -1
        width = int(self._document["canvas_size"][0])
        return min(
            max(0, approach_x - sign * self.ladder_retry_runup_distance),
            width - 1,
        )

    def _forward_ladder_approach(self, player, now, mount_direction):
        segment = self._segment
        approach = self._candidate_approach()
        mount = self._candidate_mount()
        sign = 1 if mount_direction == "right" else -1
        dy = int(approach[1]) - int(player[1])
        if abs(dy) > self.walk_tolerance:
            return self._decision(
                "stop stop none", approach, "semantic_ladder_wrong_height"
            )

        if self._traversal_state in {"forward_reset", "forward_reset_settle"}:
            reset_x = self._forward_reset_target(mount_direction)
            reset_error = reset_x - int(player[0])
            if abs(reset_error) > self.ladder_precision_tolerance:
                self._traversal_state = "forward_reset"
                self._ladder_retry_settle_count = 0
                direction = self._horizontal_command_toward(player[0], reset_x)
                self._ladder_last_player = tuple(player)
                return self._decision(
                    f"{direction} none none", (reset_x, approach[1]),
                    "semantic_ladder_forward_reset",
                )
            self._traversal_state = "forward_reset_settle"
            stable = (
                self._ladder_last_player is not None
                and abs(int(player[0]) - self._ladder_last_player[0]) <= 1
                and abs(int(player[1]) - self._ladder_last_player[1]) <= 1
            )
            self._ladder_retry_settle_count = (
                self._ladder_retry_settle_count + 1 if stable else 1
            )
            self._ladder_last_player = tuple(player)
            if self._ladder_retry_settle_count < 2:
                return self._decision(
                    "none none none", (reset_x, approach[1]),
                    "semantic_ladder_forward_reset_settle",
                )
            self._traversal_state = "forward_runup"

        progress = sign * (int(player[0]) - int(approach[0]))
        wrong_side = (
            sign * (int(player[0]) - int(mount[0]))
            > 0
        )
        if wrong_side or progress > self.ladder_precision_tolerance:
            self._traversal_state = "forward_reset"
            return self._forward_ladder_approach(player, now, mount_direction)
        if progress >= -self.ladder_precision_tolerance:
            return self._start_ladder_attempt(player, now, mount_direction)
        self._traversal_state = "forward_runup"
        return self._decision(
            f"{mount_direction} none none", approach,
            "semantic_ladder_forward_runup",
        )

    def _confirm_ladder_mount(self, player, is_on_ladder, now):
        mount = self._candidate_mount()
        mount_direction = self._segment.get("mount_direction", "none")
        rise = int(self._ladder_attempt_start[1]) - int(player[1])
        self._ladder_peak_rise = max(self._ladder_peak_rise, rise)
        centered = abs(int(player[0]) - int(mount[0])) <= 3
        previous_player = self._ladder_last_player
        rising_or_stable = (
            self._ladder_last_player is None
            or int(player[1]) <= self._ladder_last_player[1]
        )
        success_distance = self._ladder_success_threshold()
        if rise >= success_distance and centered and rising_or_stable:
            self._ladder_confirm_count += 1
        else:
            self._ladder_confirm_count = 0
        self._ladder_last_player = tuple(player)
        if self._ladder_confirm_count >= self.ladder_confirm_frames:
            self._ladder_success_offsets[self._ladder_cache_key()] = \
                self._ladder_active_offset
            self._traversal_state = "climbing"
            return self._decision(
                "none up none", self._segment["exit"],
                "semantic_ladder_climb",
            )

        elapsed = float(now) - self._ladder_attempt_started_at
        returned_to_platform = (
            self._ladder_peak_rise >= 2
            and rise <= 1
            and elapsed >= 0.30
        )
        if returned_to_platform:
            return self._fail_ladder_attempt(player, "returned")
        if (
            previous_player is not None
            and int(player[1]) > int(previous_player[1])
            and self._ladder_peak_rise >= 2
        ):
            return self._fail_ladder_attempt(player, "descending")
        if elapsed >= self.ladder_attempt_timeout:
            return self._fail_ladder_attempt(player, "timeout")

        horizontal = "none"
        reason = "semantic_ladder_mount_observed" if is_on_ladder else \
            "semantic_ladder_mount_wait"
        if mount_direction in {"left", "right"}:
            sign = 1 if mount_direction == "right" else -1
            distance_to_center = sign * (int(mount[0]) - int(player[0]))
            should_hold = (
                distance_to_center > self.ladder_precision_tolerance
                and self._ladder_forward_frames < self.ladder_forward_hold_frames
            )
            if should_hold:
                horizontal = mount_direction
                self._ladder_forward_frames += 1
                reason = "semantic_ladder_forward_hold"
            elif abs(int(player[0]) - int(mount[0])) > 3:
                return self._fail_ladder_attempt(player, "horizontal_drift")
        elif abs(int(player[0]) - int(mount[0])) > 3:
            return self._fail_ladder_attempt(player, "horizontal_drift")

        self._ladder_frames += 1
        action = "none"
        if self._ladder_frames - self._ladder_last_pulse_frame >= self.ladder_retry_frames:
            action = "mount"
            self._ladder_last_pulse_frame = self._ladder_frames
            self._ladder_pulse_count += 1
        # is_on_ladder is deliberately diagnostic-only here: an ordinary
        # vertical jump can satisfy the legacy viewport heuristic.
        return self._decision(f"{horizontal} up {action}", mount, reason)

    def _ladder(self, player, is_on_ladder, now):
        segment = self._segment
        mount_direction = segment.get("mount_direction", "none")
        exit_point = segment["exit"]
        self._ladder_last_player = tuple(player) \
            if self._ladder_last_player is None else self._ladder_last_player

        if self._traversal_state == "failed":
            return self._decision(
                "stop stop none", self._candidate_mount(), "semantic_ladder_failed"
            )
        if self._traversal_state == "confirm":
            return self._confirm_ladder_mount(player, is_on_ladder, now)
        if self._traversal_state == "climbing":
            if int(player[1]) <= int(exit_point[1]) + self.walk_tolerance:
                self._traversal_state = "exiting"
                self._ladder_exit_start = tuple(player)
                self._ladder_exit_frames = 0
            else:
                self._ladder_last_player = tuple(player)
                return self._decision(
                    "none up none", exit_point, "semantic_ladder_climb"
                )
        if self._traversal_state == "exiting":
            moved_horizontally = (
                abs(int(player[0]) - self._ladder_exit_start[0]) >= 3
            )
            if not is_on_ladder:
                self._ladder_exit_frames += 1
            else:
                self._ladder_exit_frames = 0
            if moved_horizontally or self._ladder_exit_frames >= 2:
                self._advance_segment()
                return None
            direction = segment.get("exit_direction", "none")
            command = (
                f"{direction} none none"
                if direction in {"left", "right"}
                else "none up none"
            )
            return self._decision(command, exit_point, "semantic_ladder_exit")

        if not self._traversal_state:
            self._ladder_attempts = 0
            self._ladder_pulse_count = 0
            self._select_ladder_candidate(0)
            self._traversal_state = (
                "forward_runup"
                if mount_direction in {"left", "right"}
                else "coarse_align"
            )
        if mount_direction in {"left", "right"}:
            return self._forward_ladder_approach(player, now, mount_direction)
        return self._vertical_ladder_alignment(player, now)

    def _stop(self, player):
        position = self._segment["position"]
        approach = self._event_approach(player, position, "semantic_stop_approach")
        if approach is not None:
            return approach
        if not self._stop_emitted:
            self._stop_emitted = True
            return self._decision("stop stop none", position, "semantic_stop")
        self._advance_segment()
        return None

    def _goal(self, player):
        segment = self._segment
        position = segment["position"]
        radius = int(segment.get("radius", self.goal_radius))
        if self._distance(player, position) > radius:
            approach = self._horizontal_approach(
                player, position, "semantic_goal_approach"
            )
            return approach or self._decision(
                "stop none none", position, "semantic_goal_wait"
            )
        original_route = self.route_position
        event_id = (original_route, segment["id"])
        is_last = self.route_position == len(self.routes) - 1
        should_loop = bool(self._document.get("loop", True))
        if is_last and not should_loop:
            self._route_completed = True
        else:
            self.route_position = (self.route_position + 1) % len(self.routes)
        self.segment_position = 0
        self.point_position = 0
        self._last_direction = None
        self.reset_events()
        self._last_reason = "semantic_goal_advanced"
        return RouteDecision(
            "none none goal",
            original_route,
            tuple(position),
            event_id,
            "semantic_goal_advanced",
        )

    def decide(
        self,
        player,
        *,
        is_on_ladder=False,
        suspend_traversal=False,
        now=None,
    ):
        now = time.monotonic() if now is None else float(now)
        player = int(player[0]), int(player[1])
        if suspend_traversal and self.suspend_traversal(now):
            return self._decision(
                "stop stop none", player, "semantic_ladder_suspended"
            )
        self._resume_suspended_traversal(now)
        if self._route_completed:
            return self._decision(
                "stop stop none", player, "semantic_route_complete"
            )
        for _ in range(4):
            if not self._ensure_reacquired(player):
                reason = self._last_reason
                if reason not in {"semantic_reacquire_ambiguous"}:
                    reason = "semantic_route_lost"
                return self._decision("stop stop none", player, reason)
            kind = self._segment["type"]
            if kind == "walk":
                decision = self._walk(player)
            elif kind == "jump":
                decision = self._jump(player, now)
            elif kind == "drop":
                decision = self._drop(player, now)
            elif kind == "ladder":
                decision = self._ladder(player, bool(is_on_ladder), now)
            elif kind == "teleport":
                decision = self._teleport(player, now)
            elif kind == "stop":
                decision = self._stop(player)
            elif kind == "goal":
                decision = self._goal(player)
            else:
                raise SemanticRouteError(f"不支持的路线线段：{kind!r}。")
            if decision is not None:
                return decision
        return self._decision(
            "stop stop none", player, "semantic_transition_overflow"
        )
