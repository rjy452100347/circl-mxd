"""Bounded cross-thread transport for diagnostic visualization frames."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class VisualizationFrame:
    """One owned preview image plus the metrics captured when it was produced."""

    mode: str
    sequence: int
    produced_at: float
    image: Any
    performance: Mapping[str, float] = field(default_factory=dict)


class LatestVisualizationSlot:
    """A capacity-one mailbox that coalesces queued UI notifications."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._notification_pending = False
        self._dropped = 0

    def publish(self, frame: VisualizationFrame) -> bool:
        """Store *frame* and return whether a lightweight notification is needed."""
        with self._lock:
            if self._frame is not None:
                self._dropped += 1
            self._frame = frame
            if self._notification_pending:
                return False
            self._notification_pending = True
            return True

    def take(self):
        """Atomically consume the newest frame and re-arm notifications."""
        with self._lock:
            frame = self._frame
            self._frame = None
            self._notification_pending = False
            return frame

    def clear(self, *, reset_dropped=False):
        with self._lock:
            self._frame = None
            self._notification_pending = False
            if reset_dropped:
                self._dropped = 0

    @property
    def dropped(self):
        with self._lock:
            return self._dropped

    @property
    def has_frame(self):
        with self._lock:
            return self._frame is not None
