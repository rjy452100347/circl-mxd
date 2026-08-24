from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RouteDecision:
    command: str
    route_index: int
    point: tuple[int, int] | None = None
    event_id: tuple | None = None
    reason: str = "route_hit"


class RouteNavigator:
    """Stateful nearest-pixel route follower with edge-triggered events."""

    def __init__(self, routes, color_code, color_code_up_down, search_range=10,
                 event_rearm_distance=5, mount_search_range=10,
                 mount_retry_frames=5, mount_success_distance=12,
                 mount_horizontal_tolerance=6):
        if not routes:
            raise ValueError("At least one route image is required")
        self.routes = routes
        self.palette = list(color_code.items()) + list(color_code_up_down.items())
        self.search_range = int(search_range)
        self.event_rearm_distance = int(event_rearm_distance)
        self.mount_search_range = int(mount_search_range)
        self.mount_retry_frames = max(1, int(mount_retry_frames))
        self.mount_success_distance = int(mount_success_distance)
        self.mount_horizontal_tolerance = int(mount_horizontal_tolerance)
        self.route_position = 0
        self._latched_event = None
        self._latched_anchor = None
        self._suppressed_after_goal = None
        self._mount_pending = None
        self._mount_frames = 0
        self._event_labels = self._build_event_labels()

    @property
    def route_index(self):
        return self.route_position

    @property
    def has_pending_traversal(self):
        """Whether a visually-confirmed ladder mount is still in progress."""
        return self._mount_pending is not None

    def reset_events(self):
        self._latched_event = None
        self._latched_anchor = None
        self._suppressed_after_goal = None
        self._mount_pending = None
        self._mount_frames = 0

    def _build_event_labels(self):
        labels = {}
        for route_index, image in enumerate(self.routes):
            for color, command in self.palette:
                if command.split()[2] not in {"jump", "teleport", "goal"}:
                    continue
                mask = np.all(image == np.asarray(color, dtype=np.uint8), axis=2).astype(np.uint8)
                _count, component_labels = cv2.connectedComponents(mask, connectivity=8)
                labels[(route_index, color)] = component_labels
        return labels

    def _event_for(self, color, point, command):
        if command.split()[2] not in {"jump", "teleport", "goal"}:
            return None
        component = int(self._event_labels[(self.route_position, color)][point[1], point[0]])
        return self.route_position, color, component

    def _has_up_near(self, point):
        route = self.routes[self.route_position]
        px, py = point
        radius = self.mount_search_range
        y0, y1 = max(0, py - radius), min(route.shape[0], py + radius + 1)
        x0, x1 = max(0, px - radius), min(route.shape[1], px + radius + 1)
        for color, command in self.palette:
            if command != "none up none":
                continue
            coords = np.argwhere(np.all(route[y0:y1, x0:x1] == color, axis=2))
            for row, col in coords:
                if abs((x0 + int(col)) - px) + abs((y0 + int(row)) - py) <= radius:
                    return True
        return False

    def decide(self, player):
        px, py = int(player[0]), int(player[1])
        if self._mount_pending is not None:
            anchor_x, anchor_y = self._mount_pending
            if py <= anchor_y - self.mount_success_distance:
                self._mount_pending = None
                self._mount_frames = 0
            elif (abs(px - anchor_x) > self.mount_horizontal_tolerance or
                  py > anchor_y + self.event_rearm_distance):
                # A hit or a failed approach moved the player away from the
                # ladder. Re-enable the horizontal route so it can approach
                # the marker again.
                self.reset_events()
            else:
                # Mounting is exclusive: never merge the jump with a nearby
                # left/right route pixel. Hold Up and periodically retry the
                # Space+Up edge until visual feedback proves upward movement.
                self._mount_frames += 1
                if self._mount_frames >= self.mount_retry_frames:
                    self._mount_frames = 0
                    return RouteDecision(
                        "none up mount", self.route_position,
                        self._mount_pending, self._latched_event, "mount_retry"
                    )
                return RouteDecision(
                    "none up none", self.route_position,
                    self._mount_pending, self._latched_event, "mount_pending"
                )

        if self._latched_anchor is not None:
            distance = abs(px - self._latched_anchor[0]) + abs(py - self._latched_anchor[1])
            if distance >= self.event_rearm_distance:
                self.reset_events()

        route = self.routes[self.route_position]
        radius = self.search_range
        y0, y1 = max(0, py - radius), min(route.shape[0], py + radius + 1)
        x0, x1 = max(0, px - radius), min(route.shape[1], px + radius + 1)
        hits = []
        for palette_index, (color, command) in enumerate(self.palette):
            if command == self._suppressed_after_goal:
                continue
            coords = np.argwhere(np.all(route[y0:y1, x0:x1] == color, axis=2))
            for row, col in coords:
                point = x0 + int(col), y0 + int(row)
                distance = abs(point[0] - px) + abs(point[1] - py)
                if distance > radius:
                    continue
                event_id = self._event_for(color, point, command)
                if event_id is not None and event_id == self._latched_event:
                    continue
                action = command.split()[2]
                priority = 0 if action == "goal" else 1 if action in {"jump", "teleport"} else 2
                hits.append((distance, priority, palette_index, point, command, event_id))

        if not hits:
            return RouteDecision("stop stop none", self.route_position, reason="route_lost")
        _distance, _priority, _palette, point, command, event_id = min(hits)
        original_route = self.route_position
        if event_id is not None:
            self._latched_event = event_id
            self._latched_anchor = point
        if command == "none none jump" and self._has_up_near(point):
            command = "none up mount"
            self._mount_pending = point
            self._mount_frames = 0
        if command.split()[2] == "goal":
            inbound = [hit for hit in hits if hit[4].split()[2] != "goal"]
            self._suppressed_after_goal = min(inbound)[4] if inbound else None
            self.route_position = (self.route_position + 1) % len(self.routes)
            return RouteDecision(command, original_route, point, event_id, "goal_route_advanced")
        if command == "stop stop stop":
            command = "stop stop none"
        return RouteDecision(command, original_route, point, event_id)
