"""Name-authorized body tracking; never predicts a missing visual position."""
from collections import deque
from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class PlayerLocalizationSnapshot:
    sequence: int | None
    produced_at: float
    source: str = 'none'
    foot: tuple | None = None
    box: tuple | None = None
    bound: bool = False
    binding_frames: int = 0
    control_allowed: bool = False
    reason: str = '等待完整名字定位'
    cache_safe: bool = True
    body_valid: bool = False
    name_valid: bool = False
    observation_foot: tuple | None = None
    recovery_frames: int = 0
    rejection_reason: str = ''
    switch_reason: str = ''
    binding_resets: int = 0
    source_switches: int = 0
    pause_count: int = 0
    pause_seconds: float = 0.
    track_id: int | None = None
    phase: str = 'waiting_identity'
    visual_valid: bool = False
    visual_reason: str = ''
    bridge_age: float = 0.
    name_rejected: bool = False
    shadow_only: bool = False
    candidate_count: int = 0

    @property
    def diagnostic_detail(self):
        phase = {'waiting_identity': '等待身份绑定', 'binding': '身份绑定中',
                 'tracking': '本人持续跟踪', 'identity_check': '身份核对中',
                 'visual_bridge': '短时视觉桥接', 'name_observation': '名字备用',
                 'recovering': '恢复确认中', 'reacquiring': '等待重新关联',
                 'lost': '定位失效', 'conflict': '身份冲突'}.get(self.phase, self.phase)
        return (f'本人 ID={self.track_id or "未建立"}；{phase}；{self.reason}；'
                f'控制={"允许" if self.control_allowed else "禁止"}；'
                f'LK={self.visual_reason or "未初始化"}；桥接 {self.bridge_age:.2f}s；'
                f'绑定重置 {self.binding_resets}；来源切换 {self.source_switches}；'
                f'控制暂停 {self.pause_count} 次/{self.pause_seconds:.1f}s')


class PlayerLocalizationTracker:
    CONFIDENCE = .65
    BIND_FRAMES = 5
    BIND_WINDOW = 1.
    OBSERVATION_TTL = .3

    def __init__(self, *, visual_tracker=None, allow_visual_control=False):
        # Runtime remains in shadow mode until a real-frame read-only acceptance.
        # Readiness replay may evaluate proposed control without creating an input chain.
        self.visual_tracker = visual_tracker
        self.allow_visual_control = allow_visual_control
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=self.BIND_FRAMES)
        self.offset = self.reference_size = None
        self.last_foot = self.last_body_at = None
        self.last_sequence = None
        self.missed = False
        self.recovery = 0
        self.conflict = False
        self.identity_unsafe = False
        self.snapshot = PlayerLocalizationSnapshot(None, 0.)
        self.sample_times = deque(maxlen=self.BIND_FRAMES)
        self.pending_bottom = self.pending_at = None
        self.last_name_foot = self.last_name_at = None
        self.name_frames = 0
        self.ever_bound = False
        self.last_control_at = None
        self.binding_resets = self.source_switches = self.pause_count = 0
        self.pause_at = None
        self.pause_seconds = 0.
        self.last_selected_source = 'none'
        self.track_id = None
        self.track_serial = 0
        self.reference_box = None
        self.pending_name_conflict = None
        self.full_conflict_frames = 0
        self.last_frame_shape = None
        self.last_frame_at = None
        self.visual_foot = None
        if self.visual_tracker is not None:
            self.visual_tracker.reset()

    @property
    def anchor(self):
        return self.last_foot if self.offset is not None else self.snapshot.foot or self.last_foot

    def select_detection_anchor(self, name_foot=None, cached_foot=None):
        """One shared pre-inference choice; never let a flickering name move a bound crop."""
        if self.offset is not None and self.last_foot is not None:
            return self.last_foot
        return name_foot or self.anchor or cached_foot

    def _clear_pending(self):
        if self.samples:
            self.binding_resets += 1
        self.samples.clear()
        self.sample_times.clear()
        self.pending_bottom = self.pending_at = None

    def _unbind(self, *, conflict=False):
        had_binding = self.offset is not None
        self._clear_pending()
        if had_binding:
            self.binding_resets += 1
        self.offset = self.reference_size = None
        self.missed = False
        self.recovery = 0
        self.identity_unsafe = True
        self.conflict = self.conflict or conflict
        self.name_frames = 0
        self.last_name_foot = self.last_name_at = None
        self.pending_name_conflict = None
        self.full_conflict_frames = 0
        if self.visual_tracker is not None:
            self.visual_tracker.reset()
        self.visual_foot = None

    def _seed_visual(self, frame, box, sequence, produced_at):
        if frame is None:
            return
        if self.visual_tracker is None:
            from src.engine.PlayerVisualTracker import PlayerVisualTracker
            self.visual_tracker = PlayerVisualTracker()
        self.visual_tracker.initialize(frame, box, sequence=sequence, produced_at=produced_at)
        self.visual_foot = self.last_foot

    @staticmethod
    def _overlap(a, b):
        x, y = max(a[0], b[0]), max(a[1], b[1])
        w = max(0., min(a[0]+a[2], b[0]+b[2])-x)
        h = max(0., min(a[1]+a[3], b[1]+b[3])-y)
        area = w*h
        return area/max(1., a[2]*a[3]+b[2]*b[3]-area)

    @staticmethod
    def _box(item):
        x, y = item['position']
        w, h = item['size']
        return (x, y, w, h), (x+w/2, y+h)

    def update(self, name, players, visible_rect, *, sequence, produced_at, now,
               minimap_valid, safe=True, frame=None, player_candidates=()):
        name_foot = tuple(name.player) if name is not None and name.valid else None
        rejection = ''
        visual = None
        name_rejected = False
        candidate_count = len(players)
        phase = 'tracking' if self.offset is not None else 'waiting_identity'
        bridge_age = max(0., produced_at-self.last_body_at) if self.last_body_at is not None else 0.

        def finish(reason, foot=None, box=None, body=False, *, visual_bridge=False):
            shadow = visual_bridge and not self.allow_visual_control
            allowed = foot is not None and safe and minimap_valid and not self.conflict and not shadow
            source = 'visual' if visual_bridge else 'body' if body else ('name' if foot is not None else 'none')
            switch_reason = ''
            if source != self.last_selected_source:
                switch_reason = f'{self.last_selected_source}->{source}：{reason}'
                self.source_switches += 1
                self.last_selected_source = source
            if self.snapshot.control_allowed and not allowed:
                self.pause_count += 1
                self.pause_at = now
            if allowed:
                self.last_control_at = now
                if self.pause_at is not None:
                    self.pause_seconds += max(0., now-self.pause_at)
                    self.pause_at = None
            if not safe:
                reason = '输入安全暂停；仅跟踪人物，不执行控制；'+reason
            elif not minimap_valid:
                reason = '小地图失效，已停止控制；'+reason
            if shadow:
                reason = 'LK 仅观察，尚未实机验收，不授权按键；'+reason
            self.snapshot = PlayerLocalizationSnapshot(
                sequence=sequence, produced_at=produced_at, source=source,
                foot=foot if allowed else None, box=box, bound=self.offset is not None,
                binding_frames=len(self.samples), control_allowed=allowed, reason=reason,
                cache_safe=self.ever_bound and not self.identity_unsafe,
                body_valid=body and box is not None and self.offset is not None and not self.missed,
                name_valid=name_foot is not None, observation_foot=foot,
                recovery_frames=self.recovery, rejection_reason=rejection,
                switch_reason=switch_reason, binding_resets=self.binding_resets,
                source_switches=self.source_switches, pause_count=self.pause_count,
                pause_seconds=self.pause_seconds+(max(0., now-self.pause_at) if self.pause_at is not None else 0.),
                track_id=self.track_id, phase=phase,
                visual_valid=bool(visual is not None and visual.valid),
                visual_reason=visual.reason if visual is not None else '', bridge_age=bridge_age,
                name_rejected=name_rejected, shadow_only=shadow, candidate_count=candidate_count)
            return self.snapshot

        if not 0 <= now-produced_at <= self.OBSERVATION_TTL+1e-9:
            self._unbind()
            name_foot = None
            rejection = '画面过期'
            phase = 'lost'
            return finish('人物画面过期，等待完整名字重新绑定')
        if sequence is not None and sequence == self.last_sequence:
            if not safe or not minimap_valid:
                previous = self.snapshot
                return finish('安全条件失效，已停止控制', previous.observation_foot,
                              previous.box, previous.body_valid, visual_bridge=previous.source == 'visual')
            return self.snapshot
        if self.last_sequence is not None and sequence is not None and sequence < self.last_sequence:
            self._unbind()
            name_foot = None
            rejection = '捕获帧序号倒退'
            return finish('定位失效；捕获帧序号倒退，等待重新绑定')
        self.last_sequence = sequence
        if self.last_frame_at is not None and produced_at <= self.last_frame_at:
            self._unbind()
            phase, rejection = 'lost', '捕获时间未递增'
            return finish('定位失效；捕获时间未递增')
        self.last_frame_at = produced_at
        if frame is not None:
            shape = tuple(frame.shape)
            if self.last_frame_shape is not None and shape != self.last_frame_shape:
                self.reset()
                self.last_frame_shape = shape
                phase, rejection = 'lost', '游戏画面尺寸改变'
                return finish('坐标系变化，等待完整名字重新绑定')
            self.last_frame_shape = shape
        if self.last_body_at is not None and produced_at-self.last_body_at > self.OBSERVATION_TTL+1e-9 and self.offset is not None:
            self._unbind()
            phase = 'reacquiring'
        if self.offset is not None and frame is not None and self.visual_tracker is not None:
            visual = self.visual_tracker.update(frame, sequence=sequence, produced_at=produced_at)
            if visual.valid and self.visual_foot is not None:
                self.visual_foot = tuple(a+b for a, b in zip(self.visual_foot, visual.delta))
        if name_foot is not None:
            continuous = (self.last_name_at is not None and
                0 < produced_at-self.last_name_at <= self.OBSERVATION_TTL+1e-9 and
                math.dist(name_foot, self.last_name_foot) <= 24)
            self.name_frames = min(2, self.name_frames+1) if continuous else 1
            self.last_name_foot, self.last_name_at = name_foot, produced_at
        else:
            self.name_frames = 0
            self.last_name_foot = self.last_name_at = None
        fallback = name_foot if self.name_frames >= 2 else None
        # Initial name authorizes binding, not movement. Do not patrol while
        # collecting calibration frames (which makes pet occlusion self-perpetuating).
        initial_name = None
        full = name_foot is not None and getattr(name, 'match_kind', 'full') == 'full'
        strong_name = full and math.isfinite(getattr(name, 'score', 0.)) and getattr(name, 'score', 0.) <= .20
        candidates = []
        rejected = set()
        player_count = 0
        x0, y0, x1, y1 = visible_rect
        # Public top-50 output must stay unchanged, but raw high-confidence
        # people beyond that cap still belong to the identity association pool.
        merged = {}
        for item in (*players, *player_candidates):
            key = (item.get('class_id', 1), tuple(item['position']), tuple(item['size']))
            if key not in merged or item['confidence'] > merged[key]['confidence']:
                merged[key] = item
        candidate_count = len(merged)
        for item in merged.values():
            if item.get('class_id', 1) != 1:
                continue
            player_count += 1
            if not math.isfinite(item['confidence']) or item['confidence'] < self.CONFIDENCE:
                rejected.add('置信度不足')
                continue
            box, bottom = self._box(item)
            x, y, w, h = box
            if not all(math.isfinite(v) for v in box) or w <= 0 or h <= 0:
                rejected.add('尺寸不符')
                continue
            if x <= x0 or y <= y0 or x+w >= x1 or y+h >= y1:
                rejected.add('触及裁剪边缘')
                continue
            candidates.append((box, bottom))
        rejection = '没有输出人物框' if not player_count else '、'.join(sorted(rejected)) if not candidates else ''
        if self.offset is None:
            while self.sample_times and produced_at-self.sample_times[0] > self.BIND_WINDOW+1e-9:
                self.sample_times.popleft()
                self.samples.popleft()
            if self.pending_at is not None and produced_at-self.pending_at > self.OBSERVATION_TTL+1e-9:
                self._clear_pending()
            if not safe or not minimap_valid:
                self._clear_pending()
                return finish('等待安全条件及完整名字绑定人物身体', initial_name)
            if full:
                nearby = [(box, foot) for box, foot in candidates
                          if math.dist(foot, name_foot) <= 48
                          and box[0]-8 <= name_foot[0] <= box[0]+box[2]+8]
            elif self.pending_bottom is not None:
                dt = max(.1, min(.3, produced_at-self.pending_at))
                nearby = [(box, foot) for box, foot in candidates
                          if math.dist(foot, self.pending_bottom) <= 32*dt/.1]
            else:
                return finish('等待完整名字绑定人物身体', initial_name)
            if len(nearby) > 1:
                self._clear_pending()
                rejection = '人物候选歧义'
                return finish('人物候选歧义，未绑定', initial_name)
            if not nearby:
                if candidates:
                    rejection = '距离不符'
                    self._clear_pending()
                return finish('人物身体绑定等待；'+rejection, initial_name)
            box, bottom = nearby[0]
            if self.pending_bottom is not None:
                dt = max(.1, min(.3, produced_at-self.pending_at))
                if math.dist(bottom, self.pending_bottom) > 32*dt/.1:
                    self._clear_pending()
                    rejection = '人物位置跳变'
                    return finish('人物位置跳变，重新绑定', initial_name)
            self.pending_bottom, self.pending_at = bottom, produced_at
            if not full:
                return finish(f'等待完整名字；人物身体绑定中 {len(self.samples)}/5', initial_name)
            self.samples.append((name_foot[0]-bottom[0], name_foot[1]-bottom[1], box[2], box[3]))
            self.sample_times.append(produced_at)
            offset = tuple(statistics.median(s[i] for s in self.samples) for i in (0, 1))
            if any(math.dist(s[:2], offset) > 8 for s in self.samples):
                self._clear_pending()
                rejection = '身体脚点校正不稳定'
                return finish('身体脚点校正不稳定，重新绑定', initial_name)
            if len(self.samples) < self.BIND_FRAMES:
                phase = 'binding'
                return finish(f'人物身体绑定中 {len(self.samples)}/5', initial_name, box)
            self.offset = offset
            self.reference_size = tuple(statistics.median(s[i] for s in self.samples) for i in (2, 3))
            self.last_foot, self.last_body_at = (bottom[0]+offset[0], bottom[1]+offset[1]), produced_at
            self.reference_box = box
            if self.track_id is None or self.conflict:
                self.track_serial += 1
                self.track_id = self.track_serial
            self.conflict = self.identity_unsafe = False
            self.missed, self.recovery, self.ever_bound = False, 0, True
            self.samples.clear()
            self.sample_times.clear()
            self.pending_bottom = self.pending_at = None
            self._seed_visual(frame, box, sequence, produced_at)
            phase, bridge_age = 'tracking', 0.
            return finish('身体主定位；人物身体已绑定', tuple(round(v) for v in self.last_foot), box, True)

        dt = max(.1, min(.3, produced_at-self.last_body_at))
        visual_observed_foot = self.visual_foot if visual is not None and visual.valid else None
        # Shadow observations cannot change association, authority, crop or foot.
        visual_foot = visual_observed_foot if self.allow_visual_control else None
        expected_box = visual.box if visual_foot is not None else self.reference_box
        sized = [(box, (bottom[0]+self.offset[0], bottom[1]+self.offset[1]))
                 for box, bottom in candidates
                 if all(.5*r <= v <= 2*r for v, r in zip(box[2:], self.reference_size))]
        nearby = [(box, foot) for box, foot in sized
                  if math.dist(foot, visual_foot or self.last_foot) <= 32*dt/.1 or
                  (expected_box is not None and self._overlap(box, expected_box) >= .5)]
        if len(nearby) > 1:
            self._unbind(conflict=True)
            rejection, phase = '人物候选歧义', 'conflict'
            return finish(rejection+'；定位失效，等待完整名字重新绑定')

        # Partial-name observations never get a veto over a coherent body track.
        deformed = bool(nearby and any(abs(v-r) > .15*r
                        for v, r in zip(nearby[0][0][2:], self.reference_size)))
        comparison_foot = visual_foot or (nearby[0][1] if nearby and not deformed else self.last_foot)
        if name_foot is not None and math.dist(name_foot, comparison_foot) > 12:
            if not strong_name:
                name_rejected, fallback = True, None
                self.name_frames = 0
            else:
                coherent_conflict = self.pending_name_conflict is not None and math.dist(name_foot, self.pending_name_conflict) <= 24
                self.full_conflict_frames = self.full_conflict_frames+1 if coherent_conflict else 1
                self.pending_name_conflict = name_foot
                if self.full_conflict_frames >= 2:
                    self._unbind(conflict=True)
                    phase, rejection = 'conflict', '完整名字与身体持续冲突'
                    return finish('完整名字与身体持续冲突，已停止控制并取消绑定')
                phase, rejection = 'identity_check', '完整名字冲突确认中'
                # No cache attack while the identity is under investigation.
                self.identity_unsafe = True
                return finish('完整名字冲突确认中 1/2，暂停控制')
        else:
            if strong_name and name_foot is not None:
                self.full_conflict_frames = 0
                self.pending_name_conflict = None
                self.identity_unsafe = False
        if self.pending_name_conflict is not None:
            phase = 'identity_check'
            return finish('等待完整名字消除身份冲突，暂停控制')

        if not nearby:
            self.missed = True
            self.recovery = 0
            rejection = '原目标未匹配；忽略远处人物' if sized else rejection or '尺寸不符'
            weak = []
            if visual_observed_foot is not None:
                for item in player_candidates:
                    if item.get('class_id') != 1 or not .25 <= item.get('confidence', 0.) < self.CONFIDENCE:
                        continue
                    box, _ = self._box(item)
                    if all(math.isfinite(v) for v in box) and box[2] > 0 and box[3] > 0 and self._overlap(box, visual.box) >= .5:
                        weak.append(box)
                if len(weak) > 1 and self.allow_visual_control:
                    self._unbind(conflict=True)
                    phase, rejection = 'conflict', '低分人物候选歧义'
                    return finish('低分人物候选歧义，已停止控制')
            if fallback is not None:
                self.last_foot = fallback
                # Only a complete corroborated name can extend independent evidence.
                if strong_name:
                    self.last_body_at = produced_at
                    # The body may have moved while detections were absent.
                    # Reinitializing at the previous screen rectangle would seed
                    # background with the new name foot attached to it.
                    w, h = self.reference_size
                    expected_box = (fallback[0]-self.offset[0]-w/2,
                                    fallback[1]-self.offset[1]-h, w, h)
                    self.reference_box = expected_box
                    self._seed_visual(frame, expected_box, sequence, produced_at)
                phase = 'name_observation'
                return finish('名字备用；'+rejection, fallback, expected_box)
            if visual_observed_foot is not None and bridge_age <= self.OBSERVATION_TTL+1e-9:
                if self.allow_visual_control:
                    self.last_foot = visual_observed_foot
                    self.reference_box = visual.box
                phase = 'visual_bridge'
                return finish('当前画面视觉桥接'+('；低分人物候选支持' if weak else ''),
                              tuple(round(v) for v in visual_observed_foot), visual.box, visual_bridge=True)
            phase = 'lost'
            return finish('定位失效；'+rejection)
        box, foot = nearby[0]
        # Calibration offset belongs to the independently calibrated silhouette.
        # Never accept a deformed bottom as the new zero just because last frame
        # used a name fallback; that would shift the foot on the following frame.
        size_changed = any(abs(v-r) > .15*r for v, r in zip(box[2:], self.reference_size))
        if size_changed:
            if strong_name and fallback is not None and math.dist(fallback, self.last_foot) <= 32*dt/.1:
                foot = fallback
            elif visual_observed_foot is not None and bridge_age <= self.OBSERVATION_TTL+1e-9:
                # A deformed outline cannot renew an independent *position*
                # measurement. It only supports the same bounded visual bridge.
                if self.allow_visual_control:
                    self.last_foot = visual_observed_foot
                self.missed, self.recovery = True, 0
                phase = 'visual_bridge'
                return finish('身体框形变；短时视觉桥接', tuple(round(v) for v in visual_observed_foot),
                              visual.box, visual_bridge=True)
            else:
                self.missed, self.recovery = True, 0
                phase, rejection = 'lost', '身体框形变，缺少稳定脚点证据'
                return finish(rejection)
        elif visual_foot is not None:
            if math.dist(foot, visual_foot) > 12:
                self.missed, self.recovery = True, 0
                phase, rejection = 'lost', '身体与视觉位移不一致'
                return finish(rejection)
            # Independent YOLO position corrects optical drift, not the reverse.
            foot = tuple(.75*d+.25*v for d, v in zip(foot, visual_foot))
        # Observation freshness is independent of permission and recovery completion.
        self.last_foot, self.last_body_at = foot, produced_at
        self.reference_box = box
        bridge_age = 0.
        if self.missed:
            self.recovery += 1
            if self.recovery < 2:
                phase = 'recovering'
                return finish(('名字备用；' if fallback is not None else '')+'身体恢复确认中 1/2', fallback, box)
        self.missed = False
        self.recovery = 0
        # Never teach the visual tracker using an unverified/weak-only observation.
        if not size_changed or (strong_name and not name_rejected):
            self._seed_visual(frame, self.reference_box, sequence, produced_at)
        phase = 'tracking'
        if size_changed:
            phase = 'name_observation'
            return finish('名字修正身体形变；原脚点偏移保持不变', tuple(round(v) for v in foot), self.reference_box)
        return finish('身体主定位；忽略局部名字异常' if name_rejected else
                      '身体主定位；名字校验一致' if name_foot is not None else
                      '身体主定位；名字遮挡', tuple(round(v) for v in foot), self.reference_box, body=True)


class PetTrackingMetrics:
    """Bounded, image-free diagnostics; not a claim of identity correctness."""
    def __init__(self):
        self.frames = self.names = self.bodies = self.available = self.conflicts = 0
        self.visuals = self.shadow_frames = 0
        self.times = deque(maxlen=400)
        self.sequence = None
        self.outage_at = None
        self.longest = 0.
        self.previous_conflict = False
        self.binding_resets = self.source_switches = self.pause_count = 0
        self.pause_seconds = 0.

    def add(self, sequence, at, name_valid, snapshot, elapsed_ms):
        if sequence == self.sequence:
            return
        self.sequence = sequence
        self.frames += 1
        self.names += bool(name_valid)
        self.bodies += snapshot.body_valid
        self.visuals += snapshot.visual_valid
        self.shadow_frames += snapshot.shadow_only
        self.available += snapshot.control_allowed
        self.binding_resets = snapshot.binding_resets
        self.source_switches = snapshot.source_switches
        self.pause_count = snapshot.pause_count
        self.pause_seconds = snapshot.pause_seconds
        conflict = '冲突' in snapshot.reason or '歧义' in snapshot.reason
        self.conflicts += conflict and not self.previous_conflict
        self.previous_conflict = conflict
        self.times.append(elapsed_ms)
        self.observe_availability(at, snapshot.control_allowed)

    def observe_availability(self, at, available):
        if not available and self.outage_at is None:
            self.outage_at = at
        if self.outage_at is not None:
            self.longest = max(self.longest, at-self.outage_at)
        if available:
            self.outage_at = None

    def summary(self):
        count = max(1, self.frames)
        times = sorted(self.times)
        p95 = times[min(len(times)-1, math.ceil(len(times)*.95)-1)] if times else 0.
        return (f'新帧 {self.frames}；名字 {self.names/count:.0%}；身体可接管 {self.bodies/count:.0%}；'
                f'总定位可用 {self.available/count:.0%}；最长中断 {self.longest:.1f}s；'
                f'LK 画面证据 {self.visuals/count:.0%}（仅观察 {self.shadow_frames} 帧，不代表允许控制）；'
                f'绑定重置 {self.binding_resets}；来源切换 {self.source_switches}；'
                f'控制暂停 {self.pause_count} 次/{self.pause_seconds:.1f}s；'
                f'候选/身份冲突 {self.conflicts}；融合 P95 {p95:.2f}ms（身份正确性需人工核对）')
