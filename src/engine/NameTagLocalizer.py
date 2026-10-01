from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.engine.NameTagProfileRepository import load_bgr_image


@dataclass(frozen=True)
class NameTagResult:
    player: tuple[int, int] | None
    tag_top_left: tuple[int, int] | None
    tag_size: tuple[int, int] | None
    score: float
    valid: bool
    reason: str
    sample_index: int | None = None


@dataclass(frozen=True)
class _Sample:
    gray: np.ndarray
    text: np.ndarray
    player_offset: tuple[int, int]


class NameTagLocalizer:
    """Locate a unique player name using local-first multi-template matching."""

    def __init__(
        self,
        samples,
        max_score=0.30,
        local_search_radius=140,
        global_refresh_frames=30,
        max_jump=250,
        jump_confirm_frames=2,
        jump_confirm_radius=25,
        max_missed_frames=5,
        edge_weight=0.65,
    ):
        if not samples:
            raise ValueError("At least one name-tag sample is required")
        self.samples = [self._prepare_sample(image, offset) for image, offset in samples]
        self.max_score = float(max_score)
        self.local_search_radius = int(local_search_radius)
        self.global_refresh_frames = max(1, int(global_refresh_frames))
        self.max_jump = int(max_jump)
        self.jump_confirm_frames = max(1, int(jump_confirm_frames))
        self.jump_confirm_radius = int(jump_confirm_radius)
        self.max_missed_frames = max(0, int(max_missed_frames))
        self.edge_weight = min(1.0, max(0.0, float(edge_weight)))
        self.last_tag = None
        self.last_sample_index = None
        self.missed_frames = 0
        self.frame_index = 0
        self._pending_jump = None
        self._pending_jump_frames = 0

    @classmethod
    def from_profile(cls, profile_dir, **overrides):
        profile_dir = Path(profile_dir)
        with (profile_dir / "profile.yaml").open("r", encoding="utf-8") as stream:
            profile = yaml.safe_load(stream) or {}
        samples = []
        for entry in profile.get("samples", []):
            if not entry.get("enabled", True):
                continue
            image = load_bgr_image(profile_dir / entry["file"])
            samples.append((image, tuple(int(v) for v in entry["player_offset"])))
        settings = dict(profile.get("settings", {}))
        settings.update(overrides)
        return cls(samples, **settings)

    @staticmethod
    def _prepare_sample(image, player_offset):
        if image is None or image.ndim != 3 or image.size == 0:
            raise ValueError("Name-tag sample must be a non-empty BGR image")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        text = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([179, 90, 255]))
        if int(np.count_nonzero(text)) < 4:
            raise ValueError("Name-tag sample does not contain enough light text pixels")
        return _Sample(gray, text, tuple(int(v) for v in player_offset))

    @staticmethod
    def _correlation_error(search, template):
        response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        response = np.nan_to_num(response, nan=-1.0, posinf=1.0, neginf=-1.0)
        _min_value, max_value, _min_location, max_location = cv2.minMaxLoc(response)
        return max_location, 1.0 - float(max_value)

    def _match_sample(self, gray, text, sample, origin=(0, 0)):
        h, w = sample.gray.shape[:2]
        if gray.shape[0] < h or gray.shape[1] < w:
            return None
        gray_loc, gray_error = self._correlation_error(gray, sample.gray)
        text_loc, text_error = self._correlation_error(text, sample.text)
        # The current client renders the name with light, low-saturation
        # glyphs. This mask rejects most changing map colors; grayscale keeps
        # unrelated bright UI/map details from winning by themselves.
        if abs(gray_loc[0] - text_loc[0]) + abs(gray_loc[1] - text_loc[1]) <= 4:
            point = text_loc
            score = self.edge_weight * text_error + (1.0 - self.edge_weight) * gray_error
        else:
            point = text_loc
            score = text_error + 0.10
        return (origin[0] + point[0], origin[1] + point[1]), score

    def _best_match(self, gray, text, local_only):
        matches = []
        for index, sample in enumerate(self.samples):
            search_gray, search_text = gray, text
            origin = (0, 0)
            if local_only and self.last_tag is not None:
                h, w = sample.gray.shape[:2]
                x, y = self.last_tag
                radius = self.local_search_radius
                x0, y0 = max(0, x - radius), max(0, y - radius)
                x1 = min(gray.shape[1], x + radius + w)
                y1 = min(gray.shape[0], y + radius + h)
                search_gray = gray[y0:y1, x0:x1]
                search_text = text[y0:y1, x0:x1]
                origin = (x0, y0)
            match = self._match_sample(search_gray, search_text, sample, origin)
            if match is not None:
                point, score = match
                matches.append((score, index, point))
        return min(matches) if matches else None

    @staticmethod
    def _prepare_search(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        text = cv2.inRange(
            hsv, np.array([0, 0, 120]), np.array([179, 90, 255])
        )
        return gray, text

    def _best_local_match(self, frame):
        """Preprocess only the union of the existing per-template search ROIs."""
        if self.last_tag is None:
            return None
        x, y = self.last_tag
        radius = self.local_search_radius
        max_h = max(sample.gray.shape[0] for sample in self.samples)
        max_w = max(sample.gray.shape[1] for sample in self.samples)
        x0, y0 = max(0, x - radius), max(0, y - radius)
        x1 = min(frame.shape[1], x + radius + max_w)
        y1 = min(frame.shape[0], y + radius + max_h)
        gray, text = self._prepare_search(frame[y0:y1, x0:x1])

        matches = []
        for index, sample in enumerate(self.samples):
            h, w = sample.gray.shape[:2]
            sample_x1 = min(frame.shape[1], x + radius + w) - x0
            sample_y1 = min(frame.shape[0], y + radius + h) - y0
            match = self._match_sample(
                gray[:sample_y1, :sample_x1],
                text[:sample_y1, :sample_x1],
                sample,
                origin=(x0, y0),
            )
            if match is not None:
                point, score = match
                matches.append((score, index, point))
        return min(matches) if matches else None

    def _invalid(self, reason, score=1.0):
        self.missed_frames += 1
        if self.missed_frames > self.max_missed_frames:
            self.last_tag = None
            self.last_sample_index = None
        return NameTagResult(None, None, None, float(score), False, reason)

    def locate(self, frame, y_limit=None):
        self.frame_index += 1
        if frame is None or frame.ndim != 3 or frame.size == 0:
            return self._invalid("frame_invalid")
        if y_limit is not None:
            frame = frame[:max(1, min(frame.shape[0], int(y_limit))), :]

        use_local = (self.last_tag is not None and
                     self.frame_index % self.global_refresh_frames != 0)
        if use_local:
            best = self._best_local_match(frame)
        else:
            gray, text = self._prepare_search(frame)
            best = self._best_match(gray, text, local_only=False)
        if best is None or best[0] > self.max_score:
            if use_local:
                gray, text = self._prepare_search(frame)
                best = self._best_match(gray, text, local_only=False)
            if best is None or best[0] > self.max_score:
                return self._invalid("not_found", 1.0 if best is None else best[0])

        score, sample_index, point = best
        if self.last_tag is not None:
            jump = abs(point[0] - self.last_tag[0]) + abs(point[1] - self.last_tag[1])
            if jump > self.max_jump:
                if (self._pending_jump is not None and
                        abs(point[0] - self._pending_jump[0]) +
                        abs(point[1] - self._pending_jump[1]) <= self.jump_confirm_radius):
                    self._pending_jump_frames += 1
                else:
                    self._pending_jump = point
                    self._pending_jump_frames = 1
                if self._pending_jump_frames < self.jump_confirm_frames:
                    return self._invalid("jump_unconfirmed", score)

        self.last_tag = point
        self.last_sample_index = sample_index
        self.missed_frames = 0
        self._pending_jump = None
        self._pending_jump_frames = 0
        sample = self.samples[sample_index]
        player = (
            point[0] + sample.player_offset[0],
            point[1] + sample.player_offset[1],
        )
        h, w = sample.gray.shape[:2]
        return NameTagResult(player, point, (w, h), score, True, "valid", sample_index)
