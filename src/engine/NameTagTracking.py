"""Separate current visual evidence from an indefinitely retained crop anchor."""
from dataclasses import dataclass
from collections import Counter, deque
import math


@dataclass(frozen=True)
class NameTagTrackingSnapshot:
    live: bool = False
    anchor: tuple | None = None
    age: float = 0.0
    attack_allowed: bool = False
    reason: str = '等待完整名字定位'
    source: str = 'full'


class NameTagTracking:
    def __init__(self):
        self.reset()

    def reset(self):
        self.anchor = None
        self.minimap = None
        self.observed_at = 0.0
        self.blocked_reason = ''
        self.source = 'full'
        self.snapshot = NameTagTrackingSnapshot()

    def update(self, player, minimap, *, now, inhibit='', source='full'):
        if player is not None:
            self.anchor = tuple(player)
            self.minimap = tuple(minimap) if minimap is not None else None
            self.observed_at = now
            self.blocked_reason = ''
            self.source = source
            self.snapshot = NameTagTrackingSnapshot(True, self.anchor,
                                                     reason=inhibit or '实时名字定位', source=source)
            return self.snapshot
        if self.anchor is None:
            self.snapshot = NameTagTrackingSnapshot()
            return self.snapshot
        if minimap is None or self.minimap is None:
            self.blocked_reason = '小地图人物定位失效，需重新识别名字'
        elif any(abs(a-b) > 2 for a, b in zip(minimap, self.minimap)):
            self.blocked_reason = '人物累计位移超过 2 px，需重新识别名字'
        reason = self.blocked_reason or inhibit
        self.snapshot = NameTagTrackingSnapshot(
            False, self.anchor, max(0.0, now-self.observed_at), not bool(reason),
            reason or '已停止移动，缓存攻击可用',
            self.source,
        )
        return self.snapshot


class NameTemplateMetrics:
    """Image-free read-only metrics; no body binding or YOLO player authority."""
    def __init__(self):
        self.frames = self.names = self.available = 0
        self.sequence = None
        self.kinds = Counter()
        self.times = deque(maxlen=400)
        self.outage_at = None
        self.longest = 0.

    def observe_availability(self, at, available):
        if not available and self.outage_at is None:
            self.outage_at = at
        if self.outage_at is not None:
            self.longest = max(self.longest, at-self.outage_at)
        if available:
            self.outage_at = None

    def add(self, sequence, at, result, allowed, elapsed_ms):
        if sequence == self.sequence:
            return
        self.sequence = sequence
        self.frames += 1
        valid = result is not None and result.valid
        self.names += bool(valid)
        self.available += bool(allowed)
        self.kinds[getattr(result, 'match_kind', 'full') if valid else 'missing'] += 1
        self.times.append(elapsed_ms)
        self.observe_availability(at, allowed)

    def summary(self):
        times = sorted(self.times)
        p95 = times[max(0, math.ceil(len(times)*.95)-1)] if times else 0.
        count = max(1, self.frames)
        return (f'新帧 {self.frames}；名字有效 {self.names/count:.0%}；'
                f'定位及小地图有效 {self.available/count:.0%}；'
                f'完整 {self.kinds["full"]}，局部 {sum(self.kinds[k] for k in ("left", "right", "partial"))}；'
                f'最长中断 {self.longest:.1f}s；名字匹配 P95 {p95:.2f}ms；YOLO 不参与人物定位')
