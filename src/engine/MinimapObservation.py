"""Shared, frame-sequenced minimap perception. Never changes saved coordinates."""
from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from src.engine.MapProjectConfig import normalized_roi_to_pixels


REASONS = {
    'roi_not_found': '未找到完整小地图边界，请展开小地图或校准 ROI',
    'roi_mismatch': '已保存 ROI 与当前完整小地图不一致，请暂停后确认校准',
    'roi_ambiguous': '多个小地图区域候选，无法安全确定，请手动校准',
    'roi_confirming': '小地图边界连续确认中（需要 3 个新帧）',
    'frame_stale': '画面过期或不可用，等待新捕获帧',
    'dot_no_color': '未发现人物颜色像素，请检查人物颜色与遮挡',
    'dot_shape': '颜色像素形状不符合人物点，未采用',
    'dot_ambiguous': '多个相似人物点候选，等待消除歧义',
    'dot_jump': '人物点位移异常，暂停并等待重新确认',
    'dot_confirming': '人物点连续确认中（需要 2 个新帧）',
    'ok': '本帧小地图区域与人物点有效',
}


@dataclass(frozen=True)
class RoiEvidence:
    rect: tuple | None
    recommended: tuple | None
    valid: bool
    reason: str
    source: str


@dataclass(frozen=True)
class MinimapObservationSnapshot:
    sequence: int
    produced_at: float
    roi: RoiEvidence
    player: tuple | None = None
    candidate_count: int = 0
    reason: str = 'dot_confirming'
    candidates: tuple = ()
    pose: object | None = None

    @property
    def valid(self):
        return self.roi.valid and self.player is not None and self.reason == 'ok'

    @property
    def recovery(self):
        if self.reason.startswith('roi_'):
            return '暂停后点击“校准小地图 ROI”，确认完整范围并重新启动'
        if self.reason.startswith('dot_'):
            return '检查人物颜色、小地图遮挡与黄点，等待连续新帧确认'
        return '检查游戏窗口和捕获画面'


def _near(a, b, tolerance=3):
    return a is not None and b is not None and all(abs(x-y) <= tolerance for x, y in zip(a, b))


def find_minimap_candidates(frame):
    """Find bounded map CONTENT interiors, excluding headings and outer chrome."""
    if frame is None or frame.size == 0:
        return []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    neutral = cv2.inRange(hsv, np.array((0, 0, 155)), np.array((179, 85, 255)))
    closed = cv2.morphologyEx(neutral, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (60 <= w and 40 <= h and .35 <= w/h <= 5
                and w*h <= frame.shape[0]*frame.shape[1]*.98):
            continue
        crop = closed[y:y+h, x:x+w] > 0
        border = min(np.mean(np.any(crop[:3], axis=0)), np.mean(np.any(crop[-3:], axis=0)),
                     np.mean(np.any(crop[:, :3], axis=1)), np.mean(np.any(crop[:, -3:], axis=1)))
        if border < .80:
            continue
        inside = hsv[y+3:y+h-3, x+3:x+w-3]
        dark = float(np.mean(inside[:, :, 2] < 150))
        colored = float(np.mean((inside[:, :, 1] > 70) & (inside[:, :, 2] > 50)))
        if dark < .40 or colored < .015 or float(np.std(inside[:, :, 2])) < 12:
            continue
        # Hole contours already include the adjacent rail pixel. Remove it
        # once here; every consumer uses exactly this same content rectangle.
        rect = (x+1, y+1, w-2, h-2)
        score = .7*border + .3*dark
        candidates.append((score, rect))
    unique = []
    for score, rect in sorted(candidates, reverse=True):
        if not any(_near(rect, other, 5) for _, other in unique):
            unique.append((score, rect))
    # Nested outer window / heading frames are not the content rectangle.
    return [(s, r) for s, r in unique if not any(
        r != t and r[0] <= t[0] and r[1] <= t[1] and
        r[0]+r[2] >= t[0]+t[2] and r[1]+r[3] >= t[1]+t[3]
        and t[2]*t[3] > r[2]*r[3]*.40 for _, t in unique)]


def inspect_minimap_roi(frame, cfg=None, source=None):
    if frame is None or frame.size == 0:
        return RoiEvidence(None, None, False, 'frame_stale', source or 'automatic')
    normalized = (cfg or {}).get('minimap', {}).get('roi')
    rect = normalized_roi_to_pixels(normalized, frame.shape[1], frame.shape[0]) if normalized else None
    source = source or ('global' if rect else 'automatic')
    # Configured maps are searched nearby, never silently replaced elsewhere.
    if rect:
        x, y, w, h = rect
        margin = max(64, round(max(w, h)*.75))
        x0, y0 = max(0, x-margin), max(0, y-margin)
        crop = frame[y0:min(frame.shape[0], y+h+margin), x0:min(frame.shape[1], x+w+margin)]
        candidates = [(s, (r[0]+x0, r[1]+y0, r[2], r[3])) for s, r in find_minimap_candidates(crop)]
    else:
        candidates = find_minimap_candidates(frame)
    if not candidates:
        return RoiEvidence(rect, None, False, 'roi_not_found', source)
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0]-candidates[1][0] < .08:
        return RoiEvidence(rect, None, False, 'roi_ambiguous', source)
    recommended = candidates[0][1]
    if rect and not _near(rect, recommended, 3):
        return RoiEvidence(rect, recommended, False, 'roi_mismatch', source)
    return RoiEvidence(rect or recommended, recommended, True, 'ok', source)


def player_candidates(image, color=(136, 255, 255), player_hsv=None):
    if image is None or image.size == 0:
        return [], 'dot_no_color'
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, np.asarray(player_hsv['lower'], np.uint8), np.asarray(player_hsv['upper'], np.uint8))
            if player_hsv else cv2.inRange(image, np.asarray(color, np.uint8), np.asarray(color, np.uint8)))
    if not np.any(mask):
        return [], 'dot_no_color'
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if not (3 <= area <= 160 and max(w, h) <= 16 and .5 <= w/h <= 2):
            continue
        pixels = hsv[y:y+h, x:x+w][labels[y:y+h, x:x+w] == i]
        hue_score = 1.0
        if player_hsv:
            low, high = player_hsv['lower'][0], player_hsv['upper'][0]
            hue_score = max(0., 1-abs(float(pixels[:, 0].mean())-(low+high)/2)/max(1, (high-low)/2))
        compact = area/(w*h)
        # Saturation/value already gate color membership. Brightness alone
        # must not break a tie between two equally plausible player markers.
        score = .45*min(w/h, h/w)+.25*compact+.30*hue_score
        candidates.append((score, tuple(int(round(v)) for v in centers[i]), (x, y, w, h)))
    return sorted(candidates, reverse=True), 'ok' if candidates else 'dot_shape'


def select_player_candidate(candidates, previous=None, radius=8):
    nearby = [c for c in candidates if previous is None or math.dist(c[1], previous) <= radius]
    if not nearby:
        return None, 'dot_jump' if candidates else 'dot_shape'
    nearby.sort(reverse=True)
    if len(nearby) > 1 and nearby[0][0]-nearby[1][0] < .02:
        return None, 'dot_ambiguous'
    return nearby[0][1], 'ok'


class MinimapObserver:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sequence = None
        self.snapshot = None
        self.context = None
        self.roi_pending = None
        self.roi_anchor = None
        self.roi_frames = 0
        self.player = self.pending_player = None
        self.player_at = None
        self.player_frames = 0
        self.player_live = False

    def update(self, frame, cfg, sequence, produced_at, *, now=None, source=None):
        now = time.monotonic() if now is None else now
        context = (None if frame is None else frame.shape, str(cfg.get('game_window', {})),
                   str(cfg.get('minimap', {})), source)
        if context != self.context:
            self.reset()
            self.context = context
        if frame is None or frame.size == 0 or not 0 <= now-produced_at <= .30:
            self.roi_frames = self.player_frames = 0
            self.player_live = False
            self.snapshot = MinimapObservationSnapshot(sequence, produced_at,
                RoiEvidence(None, None, False, 'frame_stale', source or 'automatic'), reason='frame_stale')
            return self.snapshot
        if self.sequence is not None and sequence <= self.sequence:
            return self.snapshot
        self.sequence = sequence
        roi = inspect_minimap_roi(frame, cfg, source)
        if (roi.valid and not cfg.get('minimap', {}).get('roi')
                and self.roi_anchor is not None and roi.rect != self.roi_anchor):
            # Automatic discovery also anchors the coordinate system. Never
            # follow a moved minimap while routes/platform bounds remain old.
            roi = RoiEvidence(self.roi_anchor, roi.rect, False, 'roi_mismatch', roi.source)
        if not roi.valid:
            if roi.reason != 'roi_mismatch':
                self.roi_frames = 0
            self.player_frames = 0
            self.player_live = False
            self.snapshot = MinimapObservationSnapshot(sequence, produced_at, roi, reason=roi.reason)
            return self.snapshot
        self.roi_frames = self.roi_frames+1 if _near(self.roi_pending, roi.rect) else 1
        self.roi_pending = roi.rect
        if self.roi_frames < 3:
            roi = RoiEvidence(roi.rect, roi.recommended, False, 'roi_confirming', roi.source)
        elif self.roi_anchor is None:
            self.roi_anchor = roi.rect
        x, y, w, h = roi.rect
        candidates, reason = player_candidates(frame[y:y+h, x:x+w],
            cfg['minimap'].get('player_color', (136, 255, 255)), cfg['minimap'].get('player_hsv'))
        dt = .1 if self.player_at is None else max(.1, min(.3, produced_at-self.player_at))
        point, reason = select_player_candidate(candidates, self.player if self.player_live else None, 8*dt/.1) if candidates else (None, reason)
        if point is None:
            self.player_live = False
            self.player_frames = 0
        elif not self.player_live:
            self.player_frames = self.player_frames+1 if self.pending_player and math.dist(point, self.pending_player) <= 8*dt/.1 else 1
            self.pending_player = point
            if self.player_frames < 2:
                point, reason = None, 'dot_confirming'
        if point is not None:
            self.player, self.player_at, self.player_live = point, produced_at, True
        self.snapshot = MinimapObservationSnapshot(sequence, produced_at, roi,
            point if roi.valid else None, len(candidates), reason if roi.valid else roi.reason,
            candidates=tuple(candidate[1] for candidate in candidates))
        return self.snapshot


def minimap_match_mask(image, cfg, player=None):
    mask = np.any(image != 0, axis=2).astype(np.uint8)*255
    mask[:2] = mask[-2:] = 0
    mask[:, :2] = mask[:, -2:] = 0
    candidates, _ = player_candidates(image, cfg.get('player_color', (136, 255, 255)), cfg.get('player_hsv'))
    for _, _, (x, y, w, h) in candidates:
        mask[max(0, y-2):y+h+2, max(0, x-2):x+w+2] = 0
    if player is not None:
        x, y = player
        mask[max(0, y-6):y+7, max(0, x-6):x+7] = 0
    other = cfg.get('other_player_color', (0, 0, 255))
    dots = cv2.inRange(image, np.asarray(other, np.uint8), np.asarray(other, np.uint8))
    mask[cv2.dilate(dots, np.ones((5, 5), np.uint8)) > 0] = 0
    return mask
