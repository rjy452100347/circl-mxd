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
    hp_state: str | None = None
    mp_state: str | None = None
    hp_confidence: float = 0.0
    mp_confidence: float = 0.0
    hp_reason: str | None = None
    mp_reason: str | None = None
    hp_action: str = ""
    mp_action: str = ""

    def role_state(self, role):
        return getattr(self, f"{role}_state") or self.state

    def role_reason(self, role):
        return getattr(self, f"{role}_reason") or self.reason

    def role_valid(self, role):
        value = getattr(self, f"{role}_percent")
        return (self.role_state(role) == HEALTH_READY and value is not None
                and np.isfinite(value) and 0 <= value <= 100)

    def stale(self):
        return replace(self, state=HEALTH_LOST, hp_state=HEALTH_LOST,
            mp_state=HEALTH_LOST, hp_percent=None, mp_percent=None, exp_percent=None,
            confidence=0.0, hp_confidence=0.0, mp_confidence=0.0,
            reason="frame_stale", hp_reason="frame_stale", mp_reason="frame_stale",
            hp_action="画面过期，禁止补血", mp_action="画面过期，禁止补蓝")

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
        self.last_candidate_reason = "bars_not_found"
        self.last_candidate_layout = None
        self._context = None
        self._last_sequence = None
        self._last_snapshot = None
        self._role_failures = [0, 0]
        self._role_stable = [0, 0]

    def reset(self):
        self.locked_rects = self.locked_layout = self._pending_rects = None
        self._stable_frames = 0
        self._role_failures = [0, 0]
        self._role_stable = [0, 0]
        self._context = self._last_sequence = self._last_snapshot = None

    @staticmethod
    def _track_mask(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return ((hsv[:, :, 1] <= 90) & (hsv[:, :, 2] >= 100)).astype(np.uint8)

    @classmethod
    def _track_confidence(cls, crop):
        """Require all four sides, not a colored component's bounding box."""
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 8:
            return 0.0
        mask = cls._track_mask(crop)
        # Tolerate antialiasing / rounded corners, but not a missing endpoint.
        top = float(np.max(np.mean(mask[:3, 3:-3], axis=1)))
        bottom = float(np.max(np.mean(mask[-3:, 3:-3], axis=1)))
        left = float(np.max(np.mean(mask[3:-3, :3], axis=0)))
        right = float(np.max(np.mean(mask[3:-3, -3:], axis=0)))
        return min(top, bottom, left, right)

    @classmethod
    def _track_candidates(cls, image):
        """Pair complete neutral top/bottom rails inside a connected HUD.

        Color determines role only AFTER both full-width rails and endpoints
        have been found. In particular, depleted fill must never resize a bar.
        The search is bounded to the caller's normalized bottom ROI.
        """
        mask = cls._track_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            if (60 <= w <= 320 and 8 <= h <= 46 and 4 <= w / h <= 18
                    and cls._track_confidence(cls._crop(image, rect)) >= .55):
                candidates.append(rect)
        rows = []
        for row in mask:
            changes = np.diff(np.pad(row.astype(np.int8), (1, 1)))
            starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
            rows.append([(int(a), int(b)) for a, b in zip(starts, ends)
                         if 60 <= b - a <= 320])
        for y, spans in enumerate(rows):
            for x0, x1 in spans:
                for bottom in range(y + 7, min(len(rows), y + 46)):
                    for bx0, bx1 in rows[bottom]:
                        if abs(x0 - bx0) > 3 or abs(x1 - bx1) > 3:
                            continue
                        x, end = min(x0, bx0), max(x1, bx1)
                        w, h = end - x, bottom - y + 1
                        if not 4 <= w / h <= 18:
                            continue
                        # Rounded rails can start one pixel inside the side.
                        for dx in (0, -1, -2):
                            expanded = (x + dx, y, w - dx, h)
                            if cls._track_confidence(cls._crop(image, expanded)) >= .55:
                                candidates.append(expanded)
                                break
        # Nested rail rows describe the same track: retain the complete box.
        unique = []
        for rect in sorted(candidates, key=lambda r: r[2] * r[3], reverse=True):
            if not any(abs(rect[0]-r[0]) <= 4 and abs(rect[1]-r[1]) <= 4
                       and abs(rect[2]-r[2]) <= 8 and abs(rect[3]-r[3]) <= 6
                       for r in unique):
                unique.append(rect)
        # A fill/neutral-background boundary may close a smaller contour inside
        # a real track. Prefer the enclosing same-role track, never its fill.
        roles = {r: cls._role_confidences(image, r) for r in unique}
        def same_role(a, b):
            return any(getattr(roles[a], key) >= .55 and getattr(roles[b], key) >= .55
                       for key in ("hp_purity", "mp_purity", "exp_purity"))
        return [r for r in unique if not any(
            s != r and abs(s[0]-r[0]) <= 3 and s[0]+s[2] >= r[0]+r[2]-2
            and r[2]+8 < s[2] < r[2]*1.8 and abs(s[1]-r[1]) <= 3 and abs(s[3]-r[3]) <= 4
            and same_role(r, s) for s in unique)]

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
                # Legacy ratios seed the search only. Snap to actual rails so
                # rounding / morphology cannot place the top outside a bar.
                tracks = self._track_candidates(normalized)
                refined = []
                for rect in selected:
                    nearby = [r for r in tracks if all(abs(a-b) <= 4
                              for a, b in zip(rect, r))]
                    refined.append(min(nearby, key=lambda r: sum(abs(a-b)
                                   for a, b in zip(rect, r))) if nearby else rect)
                selected = tuple(refined)
        if selected is None:
            tracks = self._track_candidates(normalized)
            selected = self._select_aligned_group(tracks, normalized)
            if selected is not None:
                # A second disjoint plausible HUD is unsafe to guess between.
                remaining = [r for r in tracks if not any(
                    abs(r[0]-s[0]) < max(r[2], s[2]) / 2 and
                    abs(r[1]-s[1]) < max(r[3], s[3]) for s in selected[:2])]
                if self._select_aligned_group(remaining, normalized) is not None:
                    self.last_candidate_reason = "bar_candidates_ambiguous"
                    self.last_candidate_layout = None
                    return None
                candidate_layout = "connected_tracks"
        if selected is None:
            self.last_candidate_layout = None
            hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
            has_color = np.count_nonzero((hsv[:, :, 1] > 65) &
                ((hsv[:, :, 0] < 15) | ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 140)))) > 12
            self.last_candidate_reason = ("bar_identity_unconfirmed" if len(tracks) >= 2
                else "bar_tracks_incomplete" if has_color else "bar_shapes_not_found")
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
        ranked_pairs = []
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
            if empty_mp and ordered_candidates.index(second) != ordered_candidates.index(first) + 1:
                continue
            size_variation = abs(first[2] - second[2]) + abs(first[3] - second[3])
            row_variation = abs(
                (first[1] + first[3] / 2.0) -
                (second[1] + second[3] / 2.0)
            )
            mp_score = mp_confidence if mp_confidence >= 0.55 else 0.25
            score = 100.0 * (hp_confidence + mp_score) - size_variation - row_variation
            ranked_pairs.append((score, first, second, empty_mp))
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, first, second, empty_mp)
        if best_pair is None:
            return None
        if image is not None:
            for alternative in ranked_pairs:
                if best_pair[0] - alternative[0] > 15:
                    continue
                if not cls._rects_stable(best_pair[1:3], alternative[1:3]):
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

    @staticmethod
    def _read_fill_percent(crop, role=None):
        height, width = crop.shape[:2]
        pad_x = max(2, int(round(width * 0.02)))
        pad_y = max(2, int(round(height * 0.20)))
        interior = crop[pad_y:height - pad_y, pad_x:width - pad_x]
        if interior.size == 0 or interior.shape[1] < 4:
            return None
        hsv = cv2.cvtColor(interior, cv2.COLOR_BGR2HSV)
        colored = ((hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 35)).astype(np.uint8)
        hue = hsv[:, :, 0]
        if role == "hp":
            colored &= ((hue <= 14) | (hue >= 165)).astype(np.uint8)
        elif role == "mp":
            colored &= ((hue >= 85) & (hue <= 140)).astype(np.uint8)
        elif role == "exp":
            colored &= ((hue >= 15) & (hue <= 45)).astype(np.uint8)
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
        context = (None if image is None else image.shape, tuple(roi_origin))
        if context != self._context:
            self.reset()
            self._context = context
        if self._last_sequence is not None and sequence <= self._last_sequence:
            return self._last_snapshot
        snapshot = self._recognize(image, sequence, produced_at, roi_origin)
        self._last_sequence, self._last_snapshot = sequence, snapshot
        return snapshot

    def _recognize(self, image, sequence, produced_at=None, roi_origin=(0, 0)):
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
            self._role_stable = [self.calibration_frames] * 2
            logger.info(
                "[Health Monitor] Calibrated bars "
                f"layout={self.last_candidate_layout}: {rects}"
            )

        locked_rects = self.locked_rects
        crops = [self._crop(image, rect) for rect in locked_rects]
        evidence = [self._role_confidences(image, rect) for rect in locked_rects]
        structure = [self._track_confidence(crop) for crop in crops]
        identities = [getattr(e, f"{role}_purity") >= .55 and
                      getattr(e, f"{role}_starts_left")
                      for role, e in zip(("hp", "mp", "exp"), evidence)]
        values, states, reasons, confidences = [], [], [], []
        for index, role in enumerate(("hp", "mp")):
            valid_structure = structure[index] >= .55
            # A zero has no color identity: require an intact neutral interior
            # AND another still identified HUD track, never geometry alone.
            empty = evidence[index].colored_coverage <= .02
            neighbor = any(j != index and identities[j] and structure[j] >= .55
                           for j in range(len(crops)))
            identity = identities[index] or (empty and neighbor)
            good = valid_structure and identity
            if good:
                self._role_failures[index] = 0
                self._role_stable[index] += 1
            else:
                self._role_failures[index] += 1
                self._role_stable[index] = 0
            ready = good and self._role_stable[index] >= self.calibration_frames
            states.append(HEALTH_READY if ready else HEALTH_CALIBRATING if good else HEALTH_LOST)
            reasons.append("ready" if ready else
                f"stabilizing_{self._role_stable[index]}" if good else
                "bar_border_invalid" if not valid_structure else "bar_identity_unconfirmed")
            values.append(self._read_fill_percent(crops[index], role) if ready else None)
            confidences.append(structure[index] if good else 0.0)
        if min(self._role_failures) >= 3:
            self.locked_rects = self.locked_layout = self._pending_rects = None
            self._stable_frames = 0
        elif max(self._role_failures) >= 3 and max(self._role_failures) % 3 == 0:
            # Retry discovery without discarding the other valid channel.
            candidate = self._candidate_rects(image)
            if candidate is not None and not self._rects_stable(locked_rects, candidate):
                self.locked_rects = self.locked_layout = self._pending_rects = None
                self._stable_frames = 0
        summary_ready = all(state == HEALTH_READY for state in states)
        exp_value = (self._read_fill_percent(crops[2], "exp")
                     if len(crops) > 2 and structure[2] >= .55 and
                     (identities[2] or evidence[2].colored_coverage <= .02) else None)
        return HealthSnapshot(
            sequence=sequence, produced_at=produced_at,
            state=HEALTH_READY if summary_ready else HEALTH_LOST,
            hp_percent=values[0], mp_percent=values[1],
            exp_percent=exp_value,
            hp_rect=locked_rects[0], mp_rect=locked_rects[1],
            exp_rect=locked_rects[2] if len(locked_rects) > 2 else None,
            confidence=min(confidences), reason="ready" if summary_ready else "partial_health_invalid",
            roi_origin=roi_origin,
            hp_state=states[0], mp_state=states[1],
            hp_reason=reasons[0], mp_reason=reasons[1],
            hp_confidence=confidences[0], mp_confidence=confidences[1],
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
        self._frame_context = None
        self._processed_context = None
        self._snapshot_context = None
        self.thread = None
        self._low_counts = {"hp": 0, "mp": 0}
        self._hp_low_since = None
        self._last_evaluated_sequence = -1
        self._request_log_times = {}
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
        self, img_frame, frame_sequence=None, roi_origin=(0, 0), captured_at=None,
        frame_context=None,
    ):
        if not self.active or self._stop_event.is_set():
            return
        context = (None if img_frame is None else img_frame.shape,
                   tuple(roi_origin), frame_context)
        with self._frame_lock:
            if context != self._frame_context:
                self._frame_context = context
                self._frame_sequence = self._processed_frame_sequence = -1
                self._clear_recovery_state()
            if frame_sequence is None:
                frame_sequence = self._frame_sequence + 1
            frame_sequence = int(frame_sequence)
            if frame_sequence <= self._frame_sequence:
                return
            captured_at = (
                time.monotonic() if captured_at is None else float(captured_at)
            )
            self._frame_packet = (
                None if img_frame is None else img_frame.copy(order="C"), frame_sequence,
                (int(roi_origin[0]), int(roi_origin[1])),
                captured_at, context,
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
            snapshot_context = self._snapshot_context
        if self._frame_context != snapshot_context:
            return replace(snapshot.stale(), reason="health_context_changed",
                           hp_reason="health_context_changed", mp_reason="health_context_changed")
        now = time.monotonic() if now is None else float(now)
        if ((snapshot.state in {HEALTH_READY, HEALTH_CALIBRATING} or
                snapshot.role_valid("hp") or snapshot.role_valid("mp")) and
                snapshot.produced_at > 0 and
                now - snapshot.produced_at > HEALTH_SNAPSHOT_MAX_AGE_SECONDS):
            return snapshot.stale()
        return replace(snapshot, hp_action=self._action_status("hp", snapshot, now),
                       mp_action=self._action_status("mp", snapshot, now))

    def _action_status(self, role, snapshot, now):
        if not getattr(self, f"auto_{role}_enabled"):
            return "未启用"
        if not snapshot.role_valid(role):
            return "识别无效，禁止补药"
        threshold = float(self.cfg["health_monitor"][f"add_{role}_percent"])
        if threshold <= 0:
            return "阈值为 0，补药关闭"
        if getattr(snapshot, f"{role}_percent") > threshold:
            return "未达到补药阈值"
        if self._stop_event.is_set() or not getattr(self.kb, "is_enable", True):
            return "输入已暂停"
        if self._low_counts[role] < 2:
            return "低值确认中，等待第二个有效帧"
        if hasattr(self.kb, "recovery_status"):
            return self.kb.recovery_status(f"add_{role}", now)
        return "已产生补药请求，等待输入控制器"

    def _publish_snapshot(self, snapshot):
        with self._snapshot_lock:
            previous = self._snapshot
            self._snapshot = snapshot
            self._snapshot_context = self._processed_context
        self.hp_percent = snapshot.hp_percent
        self.mp_percent = snapshot.mp_percent
        self.exp_percent = snapshot.exp_percent
        rects = (snapshot.hp_rect, snapshot.mp_rect, snapshot.exp_rect)
        self.loc_size_bars = [rect or (0, 0, 0, 0) for rect in rects]
        signature = lambda s: tuple((s.role_state(r), s.role_reason(r)) for r in ("hp", "mp"))
        if signature(previous) != signature(snapshot):
            logger.info(
                f"[Health Monitor] HP={snapshot.role_state('hp')} reason={snapshot.role_reason('hp')} "
                f"MP={snapshot.role_state('mp')} reason={snapshot.role_reason('mp')}"
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
            accepted = self.kb.request_aux_action(
                action=action, source_sequence=snapshot.sequence,
                requested_at=snapshot.produced_at, priority=priority,
            )
            if accepted is False:
                return
            if snapshot.produced_at - self._request_log_times.get(action, float('-inf')) >= 2:
                logger.info(f"[RecoveryRequest] action={action} sequence={snapshot.sequence} "
                            f"priority={priority} 请求已产生；不表示游戏已喝药")
                self._request_log_times[action] = snapshot.produced_at

    def _evaluate(self, snapshot, now=None):
        if now is not None and now - snapshot.produced_at > HEALTH_SNAPSHOT_MAX_AGE_SECONDS:
            snapshot = snapshot.stale()
        if (self._stop_event.is_set() or
                not any(snapshot.role_valid(r) for r in ("hp", "mp"))):
            self._clear_recovery_state()
            return
        if snapshot.sequence <= self._last_evaluated_sequence:
            return
        self._last_evaluated_sequence = snapshot.sequence
        health = self.cfg["health_monitor"]
        values = {"hp": snapshot.hp_percent, "mp": snapshot.mp_percent}
        enabled = {"hp": self.auto_hp_enabled, "mp": self.auto_mp_enabled}
        hp_threshold = float(health["add_hp_percent"])
        force_heal_active = (
            bool(health.get("force_heal", False)) and
            self.auto_hp_enabled and snapshot.role_valid("hp") and
            hp_threshold > 0 and snapshot.hp_percent <= hp_threshold
        )
        self._set_force_heal(force_heal_active)
        for kind in ("hp", "mp"):
            value = values[kind]
            threshold = float(health[f"add_{kind}_percent"])
            if not enabled[kind] or threshold <= 0 or not snapshot.role_valid(kind):
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
            self.auto_hp_enabled and snapshot.role_valid("hp") and
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
            image, sequence, roi_origin, captured_at, context = packet
            try:
                if context != self._processed_context:
                    if hasattr(self.recognizer, "reset"):
                        self.recognizer.reset()
                    self._clear_recovery_state()
                    self._last_evaluated_sequence = -1
                    self._processed_context = context
                snapshot = self.recognizer.recognize(
                    image, sequence, captured_at, roi_origin
                )
                if time.monotonic() - captured_at > HEALTH_SNAPSHOT_MAX_AGE_SECONDS:
                    snapshot = snapshot.stale()
                if self._stop_event.is_set():
                    break
                # A capture-context change while inference was in flight must
                # not publish or dispatch the old window's recovery reading.
                with self._frame_lock:
                    if context != self._frame_context:
                        self._clear_recovery_state()
                        continue
                    self._evaluate(snapshot, now=time.monotonic())
                    self._publish_snapshot(snapshot)
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
