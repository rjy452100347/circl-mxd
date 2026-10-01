"""Scale-aware HP/MP recognition and safe recovery request production."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import threading
import time

import cv2
import numpy as np

from src.utils.logger import logger


HEALTH_DISABLED = "DISABLED"
HEALTH_CALIBRATING = "CALIBRATING"
HEALTH_READY = "READY"
HEALTH_LOST = "LOST"
HEALTH_SNAPSHOT_MAX_AGE_SECONDS = 0.30


@dataclass(frozen=True)
class HealthSnapshot:
    sequence: int = -1
    produced_at: float = 0.0
    state: str = HEALTH_DISABLED
    hp_percent: float | None = None
    mp_percent: float | None = None
    exp_percent: float | None = None
    hp_rect: tuple[int, int, int, int] | None = None
    mp_rect: tuple[int, int, int, int] | None = None
    exp_rect: tuple[int, int, int, int] | None = None
    confidence: float = 0.0
    reason: str = "disabled"
    roi_origin: tuple[int, int] = (0, 0)

    @property
    def valid(self):
        return self.state == HEALTH_READY


@dataclass(frozen=True)
class _BarRoleEvidence:
    hp_purity: float = 0.0
    mp_purity: float = 0.0
    exp_purity: float = 0.0
    colored_coverage: float = 0.0
    hp_starts_left: bool = False
    mp_starts_left: bool = False
    exp_starts_left: bool = False


class HealthBarRecognizer:
    """Calibrate on the HUD, then read only stable HP/MP bar rectangles."""

    def __init__(self, reference_width=1282, calibration_frames=3):
        self.reference_width = max(1, int(reference_width))
        self.calibration_frames = max(1, int(calibration_frames))
        self.locked_rects = None
        self.locked_layout = None
        self._pending_rects = None
        self._stable_frames = 0
        self._locked_failures = 0
        self.last_candidate_reason = "bars_not_found"
        self.last_candidate_layout = None

    @staticmethod
    def _bright_neutral_mask(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(
            hsv,
            np.array((0, 0, 210), dtype=np.uint8),
            np.array((179, 70, 255), dtype=np.uint8),
        )

    def _candidate_rects(self, image):
        height, width = image.shape[:2]
        scale = self.reference_width / float(width)
        if abs(scale - 1.0) > 0.01:
            normalized = cv2.resize(
                image,
                (self.reference_width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
        else:
            normalized = image
            scale = 1.0
        mask = self._bright_neutral_mask(normalized)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ratio = w / max(1.0, float(h))
            if 4.0 < ratio < 12.0 and 1800 < area < 8000:
                candidates.append((x, y, w, h))
        candidates.sort(key=lambda rect: rect[0])
        selected = (
            self._select_aligned_group(candidates, normalized)
            if len(candidates) >= 2 else None
        )
        candidate_layout = "separate_bars" if selected is not None else None
        if selected is None:
            selected = self._select_classic_combined_group(
                contours, normalized
            )
            if selected is not None:
                candidate_layout = "classic_combined"
        if selected is None:
            self.last_candidate_layout = None
            self.last_candidate_reason = (
                "bar_shapes_not_found" if len(candidates) < 2 else
                "bar_identity_unconfirmed"
            )
            return None
        self.last_candidate_layout = candidate_layout
        self.last_candidate_reason = "bars_found"
        inverse = 1.0 / scale
        result = []
        for x, y, w, h in selected:
            result.append((
                int(round(x * inverse)), int(round(y * inverse)),
                max(1, int(round(w * inverse))),
                max(1, int(round(h * inverse))),
            ))
        return tuple(result)

    @classmethod
    def _select_classic_combined_group(cls, contours, image):
        """Recover the three embedded bars from the classic-client HUD panel.

        The classic CN client joins the level/name/status decoration into one
        wide contour.  HP, MP and EXP sit immediately below its right half and
        have dark (not bright-neutral) outlines, so they are not independent
        contours.  Ratios below come from a live 2049x1152 client capture;
        color/left-fill validation remains mandatory before calibration.
        """
        image_height, image_width = image.shape[:2]
        panels = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / max(1.0, float(h))
            if not (
                10.0 <= ratio <= 15.5 and
                image_width * 0.45 <= w <= image_width * 0.75 and
                35 <= h <= max(100, round(image_height * 0.60)) and
                y >= round(image_height * 0.20)
            ):
                continue
            panels.append((x, y, w, h))

        best = None
        for x, y, w, h in panels:
            bar_y = y + h + max(1, round(h * (2.0 / 58.0)))
            inferred = (
                (
                    x + round(w * (208.0 / 758.0)), bar_y,
                    max(8, round(w * (102.0 / 758.0))),
                    max(8, round(h * (18.0 / 58.0))),
                ),
                (
                    x + round(w * (310.0 / 758.0)), bar_y,
                    max(8, round(w * (103.0 / 758.0))),
                    max(8, round(h * (18.0 / 58.0))),
                ),
                (
                    x + round(w * (416.0 / 758.0)), bar_y,
                    max(8, round(w * (112.0 / 758.0))),
                    max(8, round(h * (18.0 / 58.0))),
                ),
            )
            if any(
                bx < 0 or by < 0 or
                bx + bw > image_width or by + bh > image_height
                for bx, by, bw, bh in inferred
            ):
                continue
            selected = cls._select_aligned_group(inferred, image)
            if selected is None or len(selected) < 2:
                continue
            hp_role = cls._role_confidences(image, selected[0])
            mp_role = cls._role_confidences(image, selected[1])
            identity_score = hp_role.hp_purity + mp_role.mp_purity
            if best is None or identity_score > best[0]:
                best = (identity_score, selected)
        return None if best is None else best[1]

    @staticmethod
    def _role_confidences(image, rect):
        """Return color purity, absolute coverage and left-fill evidence."""
        x, y, w, h = rect
        pad_x = max(2, int(round(w * 0.02)))
        pad_y = max(2, int(round(h * 0.18)))
        interior = image[
            y + pad_y:y + h - pad_y,
            x + pad_x:x + w - pad_x,
        ]
        if interior.size == 0:
            return _BarRoleEvidence()
        hsv = cv2.cvtColor(interior, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        colored = (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 35)
        colored_count = int(np.count_nonzero(colored))
        colored_coverage = float(colored_count) / max(1, colored.size)
        if colored_count < max(12, round(colored.size * 0.005)):
            return _BarRoleEvidence(colored_coverage=colored_coverage)
        role_masks = (
            colored & ((hue <= 14) | (hue >= 165)),
            colored & (hue >= 85) & (hue <= 140),
            colored & (hue >= 15) & (hue <= 45),
        )

        def starts_at_left(role_mask):
            occupied_columns = np.flatnonzero(
                np.mean(role_mask, axis=0) >= 0.60
            )
            if occupied_columns.size == 0:
                return False
            left_tolerance = max(2, round(role_mask.shape[1] * 0.06))
            return int(occupied_columns[0]) <= left_tolerance

        purities = tuple(
            float(np.count_nonzero(mask)) / colored_count
            for mask in role_masks
        )
        return _BarRoleEvidence(
            hp_purity=purities[0], mp_purity=purities[1],
            exp_purity=purities[2], colored_coverage=colored_coverage,
            hp_starts_left=starts_at_left(role_masks[0]),
            mp_starts_left=starts_at_left(role_masks[1]),
            exp_starts_left=starts_at_left(role_masks[2]),
        )

    @staticmethod
    def _aligned_pair(first, second):
        _x1, y1, w1, h1 = first
        x2, y2, w2, h2 = second
        x1 = first[0]
        mean_w = (w1 + w2) / 2.0
        mean_h = (h1 + h2) / 2.0
        if abs((y1 + h1 / 2.0) - (y2 + h2 / 2.0)) > max(6.0, mean_h * 0.45):
            return False
        if abs(w1 - w2) > mean_w * 0.35 or abs(h1 - h2) > mean_h * 0.35:
            return False
        gap = x2 - (x1 + w1)
        return 0 <= gap <= mean_w * 2.0

    @classmethod
    def _select_aligned_group(cls, candidates, image=None):
        if len(candidates) < 2:
            return None
        ordered_candidates = sorted(candidates, key=lambda rect: rect[0])
        if image is None:
            # Compatibility for callers that only exercise the geometry
            # helper. Runtime calibration always supplies the source image.
            roles = {
                rect: _BarRoleEvidence(
                    hp_purity=1.0, mp_purity=1.0,
                    colored_coverage=1.0,
                    hp_starts_left=True, mp_starts_left=True,
                )
                for rect in ordered_candidates
            }
        else:
            roles = {
                rect: cls._role_confidences(image, rect)
                for rect in ordered_candidates
            }
        best_pair = None
        for first, second in combinations(ordered_candidates, 2):
            if not cls._aligned_pair(first, second):
                continue
            first_role = roles[first]
            second_role = roles[second]
            hp_confidence = first_role.hp_purity
            mp_confidence = second_role.mp_purity
            # Geometry alone is not safe: a random pair of empty white boxes
            # would read as 0% and immediately request potions.
            # MP may be completely empty at startup.  An empty/neutral second
            # bar is safe only when a confirmed red HP bar anchors the pair;
            # a yellow second bar is EXP and means MP is actually missing.
            empty_mp = (
                mp_confidence < 0.55 and
                second_role.exp_purity < 0.45 and
                second_role.colored_coverage <= 0.02
            )
            hp_confirmed = (
                hp_confidence >= 0.55 and first_role.hp_starts_left
            )
            mp_confirmed = (
                mp_confidence >= 0.55 and second_role.mp_starts_left
            )
            if not hp_confirmed or (not mp_confirmed and not empty_mp):
                continue
            size_variation = abs(first[2] - second[2]) + abs(first[3] - second[3])
            row_variation = abs(
                (first[1] + first[3] / 2.0) -
                (second[1] + second[3] / 2.0)
            )
            mp_score = mp_confidence if mp_confidence >= 0.55 else 0.25
            score = 100.0 * (hp_confidence + mp_score) - size_variation - row_variation
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, first, second, empty_mp)
        if best_pair is None:
            return None

        _score, hp_rect, mp_rect, empty_mp = best_pair
        exp_rect = None
        exp_score = 0.0
        for candidate in ordered_candidates:
            if candidate[0] <= mp_rect[0] or not cls._aligned_pair(mp_rect, candidate):
                continue
            role = roles[candidate]
            confidence = role.exp_purity
            if (confidence >= 0.45 and role.exp_starts_left and
                    confidence > exp_score):
                exp_rect = candidate
                exp_score = confidence
        # An actually empty MP bar has no color identity of its own. Require
        # a third adjacent bar to complete the HUD signature; otherwise two
        # unrelated neutral boxes beside a red icon could trigger recovery.
        # EXP can itself be exactly 0%, so a strictly consecutive, neutral and
        # equally spaced third box is also a valid identity signal.
        if empty_mp and exp_rect is None:
            hp_index = ordered_candidates.index(hp_rect)
            mp_index = ordered_candidates.index(mp_rect)
            if (mp_index != hp_index + 1 or
                    mp_index + 1 >= len(ordered_candidates)):
                return None
            neutral_exp = ordered_candidates[mp_index + 1]
            neutral_role = roles[neutral_exp]
            gap_hp_mp = mp_rect[0] - (hp_rect[0] + hp_rect[2])
            gap_mp_exp = neutral_exp[0] - (mp_rect[0] + mp_rect[2])
            gap_tolerance = max(
                6.0,
                (hp_rect[2] + mp_rect[2] + neutral_exp[2]) / 3.0 * 0.25,
            )
            if (neutral_role.colored_coverage > 0.02 or
                    not cls._aligned_pair(mp_rect, neutral_exp) or
                    abs(gap_hp_mp - gap_mp_exp) > gap_tolerance):
                return None
            exp_rect = neutral_exp
        if exp_rect is None:
            return (hp_rect, mp_rect)
        return (hp_rect, mp_rect, exp_rect)

    @staticmethod
    def _rects_stable(previous, current):
        if previous is None or len(previous) != len(current):
            return False
        for old, new in zip(previous, current):
            tolerance = max(4, round(max(old[2], new[2]) * 0.04))
            if any(abs(a - b) > tolerance for a, b in zip(old, new)):
                return False
        return True

    @classmethod
    def _border_confidence(cls, crop):
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 5:
            return 0.0
        mask = cls._bright_neutral_mask(crop)
        border = np.concatenate((
            mask[:2, :].ravel(), mask[-2:, :].ravel(),
            mask[:, :2].ravel(), mask[:, -2:].ravel(),
        ))
        return float(np.count_nonzero(border)) / max(1, border.size)

    @classmethod
    def _classic_identity_confidence(cls, image, rects):
        """Validate locked classic bars by role, not their dark outlines."""
        selected = cls._select_aligned_group(rects, image)
        if selected is None or tuple(selected[:2]) != tuple(rects[:2]):
            return 0.0
        hp_role = cls._role_confidences(image, rects[0])
        mp_role = cls._role_confidences(image, rects[1])
        hp_confidence = (
            hp_role.hp_purity if hp_role.hp_starts_left else 0.0
        )
        if mp_role.mp_starts_left:
            mp_confidence = mp_role.mp_purity
        elif mp_role.colored_coverage <= 0.02 and len(rects) >= 3:
            # A completely empty MP track has no blue pixels.  It is safe only
            # because the confirmed HP plus consecutive third track completed
            # the HUD identity check in _select_aligned_group.
            mp_confidence = 0.55
        else:
            mp_confidence = 0.0
        return min(hp_confidence, mp_confidence)

    @staticmethod
    def _read_fill_percent(crop):
        height, width = crop.shape[:2]
        pad_x = max(2, int(round(width * 0.02)))
        pad_y = max(2, int(round(height * 0.20)))
        interior = crop[pad_y:height - pad_y, pad_x:width - pad_x]
        if interior.size == 0 or interior.shape[1] < 4:
            return None
        hsv = cv2.cvtColor(interior, cv2.COLOR_BGR2HSV)
        colored = ((hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 35)).astype(np.uint8)
        columns = (np.mean(colored, axis=0) >= 0.30).astype(np.uint8)
        kernel_width = max(3, int(round(columns.size * 0.015)))
        if kernel_width % 2 == 0:
            kernel_width += 1
        columns = cv2.morphologyEx(
            columns.reshape(1, -1), cv2.MORPH_CLOSE,
            np.ones((1, kernel_width), dtype=np.uint8),
        ).ravel()
        gap_required = max(3, int(round(columns.size * 0.025)))
        gap = 0
        boundary = columns.size
        for index, filled in enumerate(columns):
            if filled:
                gap = 0
            else:
                gap += 1
                if gap >= gap_required:
                    boundary = index - gap + 1
                    break
        return round(100.0 * boundary / max(1, columns.size), 2)

    @staticmethod
    def _crop(image, rect):
        x, y, w, h = rect
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            return None
        return image[y:y + h, x:x + w]

    def recognize(self, image, sequence, produced_at=None, roi_origin=(0, 0)):
        produced_at = time.monotonic() if produced_at is None else float(produced_at)
        if image is None or image.ndim != 3 or image.size == 0:
            return HealthSnapshot(
                sequence, produced_at, HEALTH_LOST, reason="frame_invalid",
                roi_origin=roi_origin,
            )
        if self.locked_rects is None:
            rects = self._candidate_rects(image)
            if rects is None:
                self._pending_rects = None
                self._stable_frames = 0
                return HealthSnapshot(
                    sequence, produced_at, HEALTH_CALIBRATING,
                    reason=self.last_candidate_reason, roi_origin=roi_origin,
                )
            if self._rects_stable(self._pending_rects, rects):
                self._stable_frames += 1
            else:
                self._pending_rects = rects
                self._stable_frames = 1
            if self._stable_frames < self.calibration_frames:
                return HealthSnapshot(
                    sequence, produced_at, HEALTH_CALIBRATING,
                    hp_rect=rects[0], mp_rect=rects[1],
                    exp_rect=rects[2] if len(rects) > 2 else None,
                    reason=f"stabilizing_{self._stable_frames}",
                    roi_origin=roi_origin,
                )
            self.locked_rects = rects
            self.locked_layout = self.last_candidate_layout
            self._locked_failures = 0
            logger.info(
                "[Health Monitor] Calibrated bars "
                f"layout={self.last_candidate_layout}: {rects}"
            )

        locked_rects = self.locked_rects
        crops = [self._crop(image, rect) for rect in locked_rects]
        if self.locked_layout == "classic_combined":
            identity_confidence = self._classic_identity_confidence(
                image, locked_rects
            )
            confidences = [identity_confidence] * len(crops)
            invalid_identity = identity_confidence < 0.55
            invalid_reason = "classic_bar_identity_invalid"
        else:
            confidences = [self._border_confidence(crop) for crop in crops]
            invalid_identity = (
                len(confidences) < 2 or min(confidences[:2]) < 0.25
            )
            invalid_reason = "bar_border_invalid"
        if invalid_identity:
            self._locked_failures += 1
            displayed_rects = locked_rects
            if self._locked_failures >= 3:
                self.locked_rects = None
                self.locked_layout = None
                self._pending_rects = None
                self._stable_frames = 0
            return HealthSnapshot(
                sequence, produced_at, HEALTH_LOST,
                hp_rect=displayed_rects[0],
                mp_rect=displayed_rects[1],
                exp_rect=displayed_rects[2] if len(displayed_rects) > 2 else None,
                confidence=min(confidences[:2]) if confidences else 0.0,
                reason=invalid_reason, roi_origin=roi_origin,
            )
        self._locked_failures = 0
        values = [self._read_fill_percent(crop) for crop in crops]
        return HealthSnapshot(
            sequence=sequence, produced_at=produced_at, state=HEALTH_READY,
            hp_percent=values[0], mp_percent=values[1],
            exp_percent=values[2] if len(values) > 2 else None,
            hp_rect=locked_rects[0], mp_rect=locked_rects[1],
            exp_rect=locked_rects[2] if len(locked_rects) > 2 else None,
            confidence=min(confidences[:2]), reason="ready",
            roi_origin=roi_origin,
        )


class HealthMonitor:
    """Latest-frame health worker; it never injects keyboard input directly."""

    def __init__(self, cfg, kb_controller, recognizer=None):
        self.cfg = cfg
        self.kb = kb_controller
        health = cfg["health_monitor"]
        legacy_enabled = bool(health.get("enable", True))
        self.auto_hp_enabled = bool(health.get(
            "auto_hp_enabled",
            legacy_enabled and float(health.get("add_hp_percent", 0)) > 0,
        ))
        self.auto_mp_enabled = bool(health.get(
            "auto_mp_enabled",
            legacy_enabled and float(health.get("add_mp_percent", 0)) > 0,
        ))
        self.active = self.auto_hp_enabled or self.auto_mp_enabled
        self.recognizer = recognizer or HealthBarRecognizer()
        self._stop_event = threading.Event()
        self._frame_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._frame_packet = None
        self._frame_sequence = -1
        self._processed_frame_sequence = -1
        self.thread = None
        self._low_counts = {"hp": 0, "mp": 0}
        self._hp_low_since = None
        initial_state = HEALTH_CALIBRATING if self.active else HEALTH_DISABLED
        self._snapshot = HealthSnapshot(state=initial_state, reason=initial_state.lower())
        self.hp_percent = None
        self.mp_percent = None
        self.exp_percent = None
        self.loc_size_bars = [(0, 0, 0, 0)] * 3
        logger.info(f"[Health Monitor] Init done active={self.active}")

    def start(self):
        if not self.active or (self.thread is not None and self.thread.is_alive()):
            return
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("[Health Monitor] Started")

    def stop(self):
        self._stop_event.set()
        self._frame_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                logger.warning("[Health Monitor] Stop timed out.")
            else:
                logger.info("[Health Monitor] Terminated")
        # Clear after joining as well: a recognizer that was already in flight
        # when stop began must not leave one final recovery request behind.
        self._clear_recovery_state()

    def update_frame(
        self, img_frame, frame_sequence=None, roi_origin=(0, 0), captured_at=None
    ):
        if (not self.active or self._stop_event.is_set() or
                img_frame is None or img_frame.size == 0):
            return
        with self._frame_lock:
            if frame_sequence is None:
                frame_sequence = self._frame_sequence + 1
            frame_sequence = int(frame_sequence)
            if frame_sequence <= self._frame_sequence:
                return
            captured_at = (
                time.monotonic() if captured_at is None else float(captured_at)
            )
            self._frame_packet = (
                img_frame.copy(order="C"), frame_sequence,
                (int(roi_origin[0]), int(roi_origin[1])),
                captured_at,
            )
            self._frame_sequence = frame_sequence
        self._frame_event.set()

    def _take_latest_packet(self):
        with self._frame_lock:
            if (self._frame_packet is None or
                    self._frame_sequence <= self._processed_frame_sequence):
                return None
            packet = self._frame_packet
            self._processed_frame_sequence = packet[1]
            return packet

    def _take_latest_frame(self):
        """Compatibility helper retained for focused frame-ownership tests."""
        packet = self._take_latest_packet()
        return None if packet is None else packet[0]

    def get_snapshot(self, now=None):
        with self._snapshot_lock:
            snapshot = self._snapshot
        now = time.monotonic() if now is None else float(now)
        if (snapshot.state in {HEALTH_READY, HEALTH_CALIBRATING} and
                snapshot.produced_at > 0 and
                now - snapshot.produced_at > HEALTH_SNAPSHOT_MAX_AGE_SECONDS):
            return replace(
                snapshot, state=HEALTH_LOST,
                hp_percent=None, mp_percent=None, exp_percent=None,
                confidence=0.0, reason="frame_stale",
            )
        return snapshot

    def _publish_snapshot(self, snapshot):
        with self._snapshot_lock:
            previous_state = self._snapshot.state
            self._snapshot = snapshot
        self.hp_percent = snapshot.hp_percent
        self.mp_percent = snapshot.mp_percent
        self.exp_percent = snapshot.exp_percent
        rects = (snapshot.hp_rect, snapshot.mp_rect, snapshot.exp_rect)
        self.loc_size_bars = [rect or (0, 0, 0, 0) for rect in rects]
        if previous_state != snapshot.state:
            logger.info(
                f"[Health Monitor] state={snapshot.state} reason={snapshot.reason}"
            )

    def _clear_recovery_state(self):
        self._low_counts = {"hp": 0, "mp": 0}
        self._hp_low_since = None
        self._set_force_heal(False)
        if hasattr(self.kb, "clear_aux_actions"):
            self.kb.clear_aux_actions({"add_hp", "add_mp", "return_home"})

    def _set_force_heal(self, enabled):
        if hasattr(self.kb, "set_force_heal"):
            self.kb.set_force_heal(enabled)
        else:
            # Lightweight test/fallback controllers retain the same state
            # contract without being allowed to inject keys directly.
            self.kb.is_need_force_heal = bool(enabled)

    def _request_recovery(self, action, snapshot, priority="normal"):
        if (not self._stop_event.is_set() and
                hasattr(self.kb, "request_aux_action")):
            self.kb.request_aux_action(
                action=action, source_sequence=snapshot.sequence,
                requested_at=snapshot.produced_at, priority=priority,
            )

    def _evaluate(self, snapshot):
        if self._stop_event.is_set() or not snapshot.valid:
            self._clear_recovery_state()
            return
        health = self.cfg["health_monitor"]
        values = {"hp": snapshot.hp_percent, "mp": snapshot.mp_percent}
        enabled = {"hp": self.auto_hp_enabled, "mp": self.auto_mp_enabled}
        hp_threshold = float(health["add_hp_percent"])
        force_heal_active = (
            bool(health.get("force_heal", False)) and
            self.auto_hp_enabled and snapshot.hp_percent is not None and
            hp_threshold > 0 and snapshot.hp_percent <= hp_threshold
        )
        self._set_force_heal(force_heal_active)
        for kind in ("hp", "mp"):
            value = values[kind]
            threshold = float(health[f"add_{kind}_percent"])
            if not enabled[kind] or threshold <= 0 or value is None:
                self._low_counts[kind] = 0
                if hasattr(self.kb, "clear_aux_actions"):
                    self.kb.clear_aux_actions({f"add_{kind}"})
                continue
            if value <= threshold:
                self._low_counts[kind] += 1
                if self._low_counts[kind] >= 2:
                    priority = "forced" if kind == "hp" and health.get(
                        "force_heal", False
                    ) else "normal"
                    self._request_recovery(f"add_{kind}", snapshot, priority)
            else:
                self._low_counts[kind] = 0
                if hasattr(self.kb, "clear_aux_actions"):
                    self.kb.clear_aux_actions({f"add_{kind}"})

        hp_low = (
            self.auto_hp_enabled and snapshot.hp_percent is not None and
            snapshot.hp_percent <= float(health["add_hp_percent"])
        )
        if hp_low:
            if self._hp_low_since is None:
                self._hp_low_since = snapshot.produced_at
            timeout = float(health["return_home_watch_dog_timeout"])
            if (health.get("return_home_if_no_potion", False) and
                    timeout > 0 and
                    snapshot.produced_at - self._hp_low_since >= timeout):
                self._request_recovery("return_home", snapshot, "emergency")
        else:
            self._hp_low_since = None
            if hasattr(self.kb, "clear_aux_actions"):
                self.kb.clear_aux_actions({"return_home"})

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            self._frame_event.wait(timeout=0.2)
            self._frame_event.clear()
            if self._stop_event.is_set():
                break
            packet = self._take_latest_packet()
            if packet is None:
                current = self.get_snapshot()
                if (current.state == HEALTH_LOST and
                        current.reason == "frame_stale"):
                    with self._snapshot_lock:
                        already_stale = self._snapshot.reason == "frame_stale"
                    if not already_stale:
                        self._publish_snapshot(current)
                        self._clear_recovery_state()
                continue
            image, sequence, roi_origin, captured_at = packet
            try:
                snapshot = self.recognizer.recognize(
                    image, sequence, captured_at, roi_origin
                )
                if time.monotonic() - captured_at > HEALTH_SNAPSHOT_MAX_AGE_SECONDS:
                    snapshot = replace(
                        snapshot, state=HEALTH_LOST,
                        hp_percent=None, mp_percent=None, exp_percent=None,
                        confidence=0.0, reason="frame_stale",
                    )
                if self._stop_event.is_set():
                    break
                self._publish_snapshot(snapshot)
                self._evaluate(snapshot)
            except Exception as exc:
                logger.error(f"[Health Monitor] {exc}")
                snapshot = HealthSnapshot(
                    sequence=sequence, produced_at=time.monotonic(),
                    state=HEALTH_LOST, reason="recognizer_error",
                    roi_origin=roi_origin,
                )
                self._publish_snapshot(snapshot)
                self._clear_recovery_state()
        self._clear_recovery_state()
