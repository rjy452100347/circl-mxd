"""Bounded, evidence-only optical flow for an already identified player.

This component does not establish identity and does not extend the identity
lease.  The caller must enforce its budget since the last independent, strong
observation.  A successful update is current image evidence, never a prediction.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualObservation:
    valid: bool = False
    delta: tuple = (0., 0.)
    box: tuple | None = None
    reason: str = ''
    points: int = 0
    inlier_ratio: float = 0.
    fb_error: float | None = None


class PlayerVisualTracker:
    """Sparse LK with spatial, forward/backward and fixed-appearance gates.

    Only a bounded local grey image, initial torso appearance and feature points
    survive a call.  New features/reference appearance are acquired exclusively
    by ``initialize`` after independent identity validation by the caller.
    """
    MAX_ROI_SIDE = 384
    MAX_GAP = .3
    MIN_POINTS = 8
    MAX_POINTS = 64
    MIN_INLIER_RATIO = .65
    MAX_FB_ERROR = 1.25
    MAX_POINT_RESIDUAL = 2.
    MIN_APPEARANCE = .60
    MIN_CELL_APPEARANCE = .35
    MIN_CENTER_APPEARANCE = .75
    MIN_CENTER_CELL_APPEARANCE = .60
    MIN_CENTER_TEXTURE = 6.

    def __init__(self):
        self.reset()

    def reset(self):
        self._gray = None
        self._reference = None
        self._points = None
        self._roi = None
        self._core = None
        self._box = None
        self._shape = None
        self._sequence = None
        self._produced_at = None

    @property
    def active(self):
        return self._gray is not None

    @staticmethod
    def _valid_frame(frame):
        return (isinstance(frame, np.ndarray) and frame.dtype == np.uint8
                and frame.ndim == 3 and frame.shape[2] == 3
                and min(frame.shape[:2]) > 0)

    @staticmethod
    def _inside(box, shape):
        x, y, width, height = box
        return (all(math.isfinite(v) for v in box) and width >= 12 and height >= 20
                and x >= 0 and y >= 0
                and x + width <= shape[1] and y + height <= shape[0])

    def _roi_for(self, box, shape):
        x, y, width, height = box
        rw = min(shape[1], self.MAX_ROI_SIDE, int(math.ceil(width)) + 192)
        rh = min(shape[0], self.MAX_ROI_SIDE, int(math.ceil(height)) + 192)
        rx = max(0, min(shape[1] - rw, int(round(x + width / 2 - rw / 2))))
        ry = max(0, min(shape[0] - rh, int(round(y + height / 2 - rh / 2))))
        return rx, ry, rw, rh

    @staticmethod
    def _gray_roi(frame, roi):
        x, y, width, height = roi
        # cvtColor allocates its own local output; no parent full-frame view is kept.
        return cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _spatial_support(points, core):
        """Require both sides, multiple vertical regions, not one tracked patch."""
        cx, cy, cw, ch = core
        values = np.asarray(points).reshape(-1, 2)
        if not len(values):
            return False
        rel = values - (cx, cy)
        if np.ptp(rel[:, 0]) < cw * .25 or np.ptp(rel[:, 1]) < ch * .30:
            return False
        cells = {(min(1, max(0, int(px / cw * 2))),
                  min(2, max(0, int(py / ch * 3)))) for px, py in rel}
        return len(cells) >= 4 and len({p[0] for p in cells}) == 2

    @staticmethod
    def _correlation(reference, current):
        a = reference.astype(np.float32).reshape(-1)
        b = current.astype(np.float32).reshape(-1)
        a -= a.mean()
        b -= b.mean()
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denominator) if denominator > 1e-5 else -1.

    @staticmethod
    def _center_support(points, core):
        """Peripheral background points cannot substitute for central evidence.

        This is a conservative geometry/texture gate, not a segmentation mask or
        proof of foreground identity.  Pure visual observations still need the
        caller's independent identity and control policy.
        """
        cx, cy, cw, ch = core
        values = np.asarray(points).reshape(-1, 2) - (cx, cy)
        center = values[(values[:, 0] >= cw // 3)
                        & (values[:, 0] < cw - cw // 3)
                        & (values[:, 1] >= 0) & (values[:, 1] < ch)]
        if len(center) < 6:
            return False
        rows = np.minimum(2, (center[:, 1] * 3 / ch).astype(int))
        return all(np.count_nonzero(rows == row) >= 2 for row in range(3))

    @staticmethod
    def _center_texture_valid(reference):
        height, width = reference.shape
        center = reference[:, width // 3:width - width // 3]
        return all(float(np.std(center[row * height // 3:(row + 1) * height // 3]))
                   >= PlayerVisualTracker.MIN_CENTER_TEXTURE for row in range(3))

    def _appearance_valid(self, current):
        if self._correlation(self._reference, current) < self.MIN_APPEARANCE:
            return False
        height, width = self._reference.shape
        # A textured background surrounding a low-texture body can keep the
        # aggregate NCC high after the body disappears.  Demand independent
        # evidence from the central column and all three of its vertical bands.
        central = slice(width // 3, width - width // 3)
        if self._correlation(self._reference[:, central], current[:, central]) < self.MIN_CENTER_APPEARANCE:
            return False
        for row in range(3):
            ys = slice(row * height // 3, (row + 1) * height // 3)
            if self._correlation(self._reference[ys, central], current[ys, central]) < self.MIN_CENTER_CELL_APPEARANCE:
                return False
        supported = set()
        for row in range(3):
            for col in range(2):
                ys = slice(row * height // 3, (row + 1) * height // 3)
                xs = slice(col * width // 2, (col + 1) * width // 2)
                if self._correlation(self._reference[ys, xs], current[ys, xs]) >= self.MIN_CELL_APPEARANCE:
                    supported.add((col, row))
        return (len(supported) >= 4 and len({v[0] for v in supported}) == 2
                and len({v[1] for v in supported}) >= 2)

    def initialize(self, frame, box, *, sequence, produced_at):
        self.reset()
        try:
            box = tuple(float(v) for v in box)
            produced_at = float(produced_at)
        except (TypeError, ValueError):
            return False
        if (not self._valid_frame(frame) or len(box) != 4
                or not math.isfinite(produced_at) or not self._inside(box, frame.shape)
                or max(box[2:]) > self.MAX_ROI_SIDE - 24):
            return False
        x, y, width, height = box
        # Exclude peripheral weapons, pet/name labels, and lower HUD/background.
        left, right = int(math.ceil(x + width * .20)), int(math.floor(x + width * .80))
        top, bottom = int(math.ceil(y + height * .08)), int(math.floor(y + height * .72))
        if right - left < 8 or bottom - top < 10:
            return False
        roi = self._roi_for(box, frame.shape)
        gray = self._gray_roi(frame, roi)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cx, cy, cw, ch = left - roi[0], top - roi[1], right - left, bottom - top
        mask[cy:cy + ch, cx:cx + cw] = 255
        try:
            points = cv2.goodFeaturesToTrack(gray, maxCorners=self.MAX_POINTS,
                                             qualityLevel=.015, minDistance=3,
                                             mask=mask, blockSize=3)
        except cv2.error:
            return False
        core = (cx, cy, cw, ch)
        reference = gray[cy:cy + ch, cx:cx + cw].copy()
        if (points is None or len(points) < self.MIN_POINTS
                or not self._spatial_support(points, core)
                or not self._center_support(points, core)
                or not self._center_texture_valid(reference)):
            return False
        self._gray = gray
        self._reference = reference
        self._points = points
        self._roi = roi
        # Fixed core geometry relative to the original body box, including subpixels.
        self._core = (left - x, top - y, cw, ch)
        self._box = box
        self._shape = frame.shape
        self._sequence = sequence
        self._produced_at = produced_at
        return True

    def _fail(self, reason, *, points=0, ratio=0., fb_error=None):
        self.reset()
        return VisualObservation(reason=reason, points=points,
                                 inlier_ratio=ratio, fb_error=fb_error)

    def update(self, frame, *, sequence, produced_at):
        if not self.active:
            return VisualObservation(reason='视觉跟踪未初始化')
        if not self._valid_frame(frame) or frame.shape != self._shape:
            return self._fail('画面坐标系变化或画面无效')
        try:
            produced_at = float(produced_at)
        except (TypeError, ValueError):
            return self._fail('视觉帧时间无效')
        if not math.isfinite(produced_at):
            return self._fail('视觉帧时间无效')
        out_of_order = (isinstance(sequence, (int, np.integer))
                        and isinstance(self._sequence, (int, np.integer))
                        and sequence < self._sequence)
        if sequence == self._sequence or out_of_order or produced_at <= self._produced_at:
            return VisualObservation(reason='重复或非新捕获帧')
        dt = produced_at - self._produced_at
        if dt > self.MAX_GAP + 1e-9:
            return self._fail('视觉画面间隔超过 300 ms')

        current = self._gray_roi(frame, self._roi)
        try:
            # At very coarse levels a small sprite becomes a few pixels and the
            # surrounding background/pet dominates LK's window.  Keep at least
            # eight pixels of identified core in the coarsest pyramid level.
            levels = max(0, min(3, int(math.log2(max(8, min(self._core[2:])) / 8))))
            core_x = self._box[0] + self._core[0] - self._roi[0]
            core_y = self._box[1] + self._core[1] - self._roi[1]
            # Smaller windows avoid synthetic border pixels dominating a core
            # very close to the actual image edge.
            core_right = self._gray.shape[1] - core_x - self._core[2]
            core_bottom = self._gray.shape[0] - core_y - self._core[3]
            window = 15 if min(core_x, core_y, core_right, core_bottom) < 12 else 21
            lk = dict(winSize=(window, window), maxLevel=levels,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
            forward, status_f, error_f = cv2.calcOpticalFlowPyrLK(self._gray, current, self._points, None, **lk)
            if forward is None or status_f is None or not np.isfinite(forward).all():
                return self._fail('视觉特征未找到可靠前向匹配')
            backward, status_b, _ = cv2.calcOpticalFlowPyrLK(current, self._gray, forward, None, **lk)
            if backward is None or status_b is None:
                return self._fail('视觉特征未找到可靠反向匹配')
        except cv2.error:
            return self._fail('光流计算失败')
        old = self._points.reshape(-1, 2)
        new = forward.reshape(-1, 2)
        fb = np.linalg.norm(backward.reshape(-1, 2) - old, axis=1)
        height, width = current.shape
        good = (status_f.reshape(-1).astype(bool) & status_b.reshape(-1).astype(bool)
                & np.isfinite(fb) & (fb <= self.MAX_FB_ERROR)
                & np.isfinite(new).all(axis=1)
                & (new[:, 0] >= 1) & (new[:, 0] < width - 1)
                & (new[:, 1] >= 1) & (new[:, 1] < height - 1))
        if error_f is not None:
            good &= np.isfinite(error_f.reshape(-1)) & (error_f.reshape(-1) <= 35.)
        if np.count_nonzero(good) < self.MIN_POINTS:
            return self._fail('前后向一致的身体特征不足', points=int(good.sum()))
        deltas = new - old
        median = np.median(deltas[good], axis=0)
        good &= np.linalg.norm(deltas - median, axis=1) <= self.MAX_POINT_RESIDUAL
        count = int(good.sum())
        ratio = count / len(old)
        fb_error = float(np.median(fb[good])) if count else None
        if count < self.MIN_POINTS or ratio < self.MIN_INLIER_RATIO:
            return self._fail('身体特征位移不一致或遮挡过多', points=count, ratio=ratio, fb_error=fb_error)
        median = np.median(deltas[good], axis=0)
        dx, dy = float(median[0]), float(median[1])
        if math.hypot(dx, dy) > 32. * min(.3, max(1e-6, dt)) / .1:
            return self._fail('身体光流位移异常', points=count, ratio=ratio, fb_error=fb_error)
        x, y, bw, bh = self._box
        box = x + dx, y + dy, bw, bh
        if not self._inside(box, frame.shape):
            return self._fail('身体光流位置越出画面', points=count, ratio=ratio, fb_error=fb_error)
        offx, offy, cw, ch = self._core
        cx, cy = box[0] + offx - self._roi[0], box[1] + offy - self._roi[1]
        if cx < 0 or cy < 0 or cx + cw > width or cy + ch > height:
            return self._fail('身体特征越出局部搜索区域', points=count, ratio=ratio, fb_error=fb_error)
        if not self._spatial_support(new[good], (cx, cy, cw, ch)):
            return self._fail('身体特征集中在单一局部区域', points=count, ratio=ratio, fb_error=fb_error)
        if not self._center_support(new[good], (cx, cy, cw, ch)):
            return self._fail('中央身体特征支持不足，不能用外围纹理代替', points=count, ratio=ratio, fb_error=fb_error)
        appearance = cv2.getRectSubPix(current, (cw, ch), (cx + (cw - 1) / 2, cy + (ch - 1) / 2))
        if not self._appearance_valid(appearance):
            return self._fail('人物外观与初始参考不一致', points=count, ratio=ratio, fb_error=fb_error)

        new_roi = self._roi_for(box, frame.shape)
        surviving = new[good].copy()
        surviving += np.array((self._roi[0] - new_roi[0], self._roi[1] - new_roi[1]), dtype=np.float32)
        self._gray = self._gray_roi(frame, new_roi)
        self._points = surviving.reshape(-1, 1, 2)
        self._roi = new_roi
        self._box = box
        self._sequence = sequence
        self._produced_at = produced_at
        return VisualObservation(True, (dx, dy), box, '身体视觉跟踪有效', count, ratio, fb_error)
