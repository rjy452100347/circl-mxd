from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.engine.NameTagProfileRepository import load_bgr_image


def name_search_y_limit(cfg, frame_height):
    """0 searches the full game frame, independently of the health-bar ROI."""
    configured = int(cfg.get("nametag", {}).get("search_y_limit", 0) or 0)
    return min(int(frame_height), configured) if configured > 0 else int(frame_height)


@dataclass(frozen=True)
class NameTagResult:
    player: tuple[int, int] | None
    tag_top_left: tuple[int, int] | None
    tag_size: tuple[int, int] | None
    score: float
    valid: bool
    reason: str
    sample_index: int | None = None
    match_kind: str = 'full'
    fragment_rect: tuple[int, int, int, int] | None = None
    candidate_margin: float | None = None


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
        allow_partial=True,
    ):
        if not samples:
            raise ValueError("At least one name-tag sample is required")
        self.sample_kinds = [sample[2] if len(sample) > 2 else 'full' for sample in samples]
        if any(kind not in ('full', 'partial') for kind in self.sample_kinds):
            raise ValueError('名字样本类型必须为 full 或 partial')
        self.samples = [self._prepare_sample(sample[0], sample[1]) for sample in samples]
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
        self.allow_partial = bool(allow_partial)
        self.last_player = None
        self.last_kind = 'full'
        self.partial_confirmed = False
        self._confirmation = None
        self._sequence = None
        self._sequence_result = None
        self.fragments = []
        for index, sample in enumerate(self.samples):
            h, w = sample.gray.shape
            if self.sample_kinds[index] == 'partial':
                if w < 24 or h < 5 or np.count_nonzero(sample.text) < 32 or np.std(sample.gray) <= 1:
                    raise ValueError('局部名字样本至少 24×5 px，且需要足够的浅色文字')
                self.fragments.append((index, 'partial', 0, sample))
                continue
            fw = int(np.ceil(w * .6))
            for kind, dx in (('left', 0), ('right', w-fw)):
                gray, text = sample.gray[:, dx:dx+fw], sample.text[:, dx:dx+fw]
                if fw >= 24 and h >= 5 and np.count_nonzero(text) >= 32 and np.std(gray) > 1:
                    self.fragments.append((index, kind, dx, _Sample(gray, text, sample.player_offset)))

    def reset(self):
        self._prepared_local = None
        self.last_tag = self.last_player = self.last_sample_index = None
        self.last_kind = 'full'
        self.partial_confirmed = False
        self.missed_frames = 0
        self._pending_jump = self._confirmation = None
        self._pending_jump_frames = 0
        self._sequence = self._sequence_result = None

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
            samples.append((image, tuple(int(v) for v in entry["player_offset"]), entry.get('kind', 'full')))
        if not any(s[2] == 'full' for s in samples):
            raise ValueError('至少需要一个已启用的完整名字样本；局部样本不能独立确认身份')
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
            if self.sample_kinds[index] != 'full':
                continue
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
        # Different profiles/templates may use different rectangles and offsets.
        points = [(self.last_player[0]-s.player_offset[0],
                   self.last_player[1]-s.player_offset[1]) for s in self.samples]
        radius = self.local_search_radius
        # Occlusion has already stopped travel. Check for the full name in
        # the same bounded recovery ROI before matching either fragment.
        # Distant full-name reacquisition still uses the periodic global pass.
        if self.last_kind != 'full' or self.missed_frames:
            radius = min(radius, 64)
        x0, y0 = max(0, min(p[0] for p in points)-radius), max(0, min(p[1] for p in points)-radius)
        x1 = min(frame.shape[1], max(p[0]+s.gray.shape[1] for p, s in zip(points, self.samples))+radius)
        y1 = min(frame.shape[0], max(p[1]+s.gray.shape[0] for p, s in zip(points, self.samples))+radius)
        if x1 <= x0 or y1 <= y0:
            return None
        gray, text = self._prepare_search(frame[y0:y1, x0:x1])
        self._prepared_local = (gray, text, x0, y0)

        matches = []
        for index, sample in enumerate(self.samples):
            if self.sample_kinds[index] != 'full':
                continue
            x, y = points[index]
            h, w = sample.gray.shape[:2]
            sx0, sy0 = max(0, x-radius)-x0, max(0, y-radius)-y0
            sample_x1 = min(frame.shape[1], x + radius + w) - x0
            sample_y1 = min(frame.shape[0], y + radius + h) - y0
            match = self._match_sample(
                gray[sy0:sample_y1, sx0:sample_x1],
                text[sy0:sample_y1, sx0:sample_x1],
                sample,
                origin=(x0+sx0, y0+sy0),
            )
            if match is not None:
                point, score = match
                matches.append((score, index, point))
        return min(matches) if matches else None

    def _invalid(self, reason, score=1.0):
        self.missed_frames += 1
        return NameTagResult(None, None, None, float(score), False, reason)

    def locate(self, frame, y_limit=None, *, frame_sequence=None):
        if frame_sequence is not None and frame_sequence == self._sequence:
            return self._sequence_result
        result = self._locate(frame, y_limit)
        self._prepared_local = None
        self._sequence, self._sequence_result = frame_sequence, result
        return result

    def _partial_match(self, frame):
        candidates = []
        radius = min(64, self.local_search_radius)
        for index, kind, dx, fragment in self.fragments:
            sample = self.samples[index]
            x = self.last_player[0]-sample.player_offset[0]+dx
            y = self.last_player[1]-sample.player_offset[1]
            h, w = fragment.gray.shape
            x0, y0 = max(0, x-radius), max(0, y-radius)
            x1, y1 = min(frame.shape[1], x+radius+w), min(frame.shape[0], y+radius+h)
            if x1-x0 < w or y1-y0 < h:
                continue
            prepared = getattr(self, '_prepared_local', None)
            if (prepared is not None and x0 >= prepared[2] and y0 >= prepared[3]
                    and x1 <= prepared[2]+prepared[0].shape[1]
                    and y1 <= prepared[3]+prepared[0].shape[0]):
                gray = prepared[0][y0-prepared[3]:y1-prepared[3], x0-prepared[2]:x1-prepared[2]]
                text = prepared[1][y0-prepared[3]:y1-prepared[3], x0-prepared[2]:x1-prepared[2]]
            else:
                gray, text = self._prepare_search(frame[y0:y1, x0:x1])
            gr = cv2.matchTemplate(gray, fragment.gray, cv2.TM_CCOEFF_NORMED)
            tr = cv2.matchTemplate(text, fragment.text, cv2.TM_CCOEFF_NORMED)
            errors = 1 - (self.edge_weight*tr + (1-self.edge_weight)*gr)
            errors = np.nan_to_num(errors, nan=2.0, posinf=2.0, neginf=2.0)
            for _ in range(2):
                flat = int(np.argmin(errors))
                py, px = np.unravel_index(flat, errors.shape)
                score = float(errors[py, px])
                point = (x0+int(px)-dx, y0+int(py))
                player = (point[0]+sample.player_offset[0], point[1]+sample.player_offset[1])
                candidates.append((score, index, point, kind, (point[0]+dx, point[1], w, h), player))
                errors[max(0, py-4):py+5, max(0, px-4):px+5] = 2.0
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])
        best = candidates[0]
        if best[0] > min(self.max_score, .20):
            return None
        other = [c for c in candidates if max(abs(a-b) for a, b in zip(c[5], best[5])) > 4]
        margin = (other[0][0]-best[0]) if other else 2.0
        if margin < .08:
            return None
        for candidate in candidates:
            if (candidate[3] != best[3] and candidate[0] <= min(self.max_score, .20)
                    and max(abs(a-b) for a, b in zip(candidate[5], best[5])) > 3):
                return None
        return (*best[:5], margin)

    def _locate(self, frame, y_limit=None):
        self.frame_index += 1
        self._prepared_local = None
        if frame is None or frame.ndim != 3 or frame.size == 0:
            self.reset()
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
        kind, rect, margin = 'full', None, None
        if (best is None or best[0] > self.max_score) and self.allow_partial and self.last_player is not None:
            partial = self._partial_match(frame)
            if partial:
                score, index, point, kind, rect, margin = partial
                best = (score, index, point)
        if best is None or best[0] > self.max_score:
            # During sustained occlusion keep trying nearby fragments, but do
            # not preprocess the full game window on every missed frame.
            if use_local and (self.missed_frames == 0 or
                              (self._confirmation is not None and self._confirmation[0] == 'full')):
                gray, text = self._prepare_search(frame)
                best = self._best_match(gray, text, local_only=False)
            if best is None or best[0] > self.max_score:
                self._confirmation = None
                self._pending_jump = None
                self._pending_jump_frames = 0
                return self._invalid("not_found", 1.0 if best is None else best[0])

        score, sample_index, point = best
        sample = self.samples[sample_index]
        player = (point[0]+sample.player_offset[0], point[1]+sample.player_offset[1])
        h, w = sample.gray.shape
        if not (0 <= point[0] <= frame.shape[1]-w and 0 <= point[1] <= frame.shape[0]-h
                and 0 <= player[0] < frame.shape[1] and 0 <= player[1] < frame.shape[0]):
            self._confirmation = None
            return self._invalid('point_outside', score)
        # Identity is shared by all calibrated templates. Once local evidence
        # has been confirmed, changing template labels does not invalidate it.
        required = 2 if self.last_player is not None and (self.missed_frames or
                    (kind != 'full' and not self.partial_confirmed)) else 1
        reason = 'recovery_unconfirmed'
        if self.last_player is not None:
            jump = abs(player[0] - self.last_player[0]) + abs(player[1] - self.last_player[1])
            if jump > self.max_jump:
                required = max(required, self.jump_confirm_frames)
                reason = 'jump_unconfirmed'
        if required > 1:
            previous = self._confirmation
            count = (previous[2]+1 if previous
                     and max(abs(a-b) for a, b in zip(previous[1], player)) <= 24 else 1)
            self._confirmation = (kind, player, count)
            if count < required:
                return self._invalid(reason, score)

        self.last_tag = point
        self.last_sample_index = sample_index
        self.last_player, self.last_kind = player, kind
        if kind != 'full':
            self.partial_confirmed = True
        self._confirmation = None
        self.missed_frames = 0
        self._pending_jump = None
        self._pending_jump_frames = 0
        return NameTagResult(player, point, (w, h), score, True, "valid", sample_index,
                             kind, rect, margin)
