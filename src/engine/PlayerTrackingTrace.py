"""Bounded, image-free diagnostics for the single identified player.

This is an in-memory trace, not a recorder of game pictures or user profiles.
Only explicit scalar/status fields are copied; no input object is retained.
Export is caller-controlled and never writes to disk automatically.
"""
from collections import Counter, deque
from collections.abc import Mapping
from copy import deepcopy
from itertools import chain, islice
import math
from numbers import Integral, Real
import re


_WINDOWS_PATH = re.compile(r'(?i)(?:[a-z]:[\\/]|\\\\)[^\r\n;；]*')


def _get(value, field, default=None):
    return value.get(field, default) if isinstance(value, Mapping) else getattr(value, field, default)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def _flag(value):
    # Never call bool() on an image/array or arbitrary user object.
    return bool(value) if isinstance(value, (bool, Integral)) else False


def _coordinates(value, count):
    try:
        if len(value) != count:
            return None
        result = [_number(value[index]) for index in range(count)]
    except (TypeError, KeyError, IndexError):
        return None
    return result if all(item is not None for item in result) else None


def _text(value):
    if not isinstance(value, str):
        return ''
    value = _WINDOWS_PATH.sub('[路径已省略]', value)
    return ''.join(char for char in value[:240] if char.isprintable())


class PlayerTrackingTrace:
    MAX_FRAMES = 300
    MAX_CANDIDATES = 100

    def __init__(self):
        self._frames = deque(maxlen=self.MAX_FRAMES)
        self._last_sequence = None
        self._accepted = 0
        self._rejected_sequence = 0
        self._rejected_invalid = 0

    def record(self, name, players, snapshot, *, sequence, produced_at,
               elapsed_ms, player_candidates=()):
        """Copy this fresh frame's whitelisted values; return whether accepted.

        Sequences must be nonnegative integers and strictly increase. A new
        capture session should create a new trace, not append backwards frames.
        Candidate storage and inspection are bounded independently of input.
        """
        seq = _integer(sequence)
        at = _number(produced_at)
        if seq is None or seq < 0 or at is None:
            self._rejected_invalid += 1
            return False
        if self._last_sequence is not None and seq <= self._last_sequence:
            self._rejected_sequence += 1
            return False
        candidates, seen = [], set()
        invalid_candidates = 0
        # Model output is 300 rows plus at most 50 public detections. The scan
        # cap additionally prevents arbitrary caller iterables from growing work.
        inputs = chain(players if players is not None else (),
                       player_candidates if player_candidates is not None else ())
        for item in islice(inputs, 400):
            if not isinstance(item, Mapping):
                invalid_candidates += 1
                continue
            class_id = _integer(item.get('class_id'))
            confidence = _number(item.get('confidence'))
            position = _coordinates(item.get('position'), 2)
            size = _coordinates(item.get('size'), 2)
            if class_id is None or confidence is None or position is None or size is None:
                invalid_candidates += 1
                continue
            bbox = position + size
            key = (class_id, confidence, *bbox)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({'class_id': class_id, 'confidence': confidence, 'bbox': bbox})
            if len(candidates) >= self.MAX_CANDIDATES:
                break
        state = {
            'track_id': _integer(_get(snapshot, 'track_id')),
            'phase': _text(_get(snapshot, 'phase')),
            'source': _text(_get(snapshot, 'source')),
            'foot': _coordinates(_get(snapshot, 'foot'), 2),
            'observation_foot': _coordinates(_get(snapshot, 'observation_foot'), 2),
            'control_allowed': _flag(_get(snapshot, 'control_allowed')),
            'cache_safe': _flag(_get(snapshot, 'cache_safe')),
            'rejection_reason': _text(_get(snapshot, 'rejection_reason')),
            'visual_valid': _flag(_get(snapshot, 'visual_valid')),
            'visual_reason': _text(_get(snapshot, 'visual_reason')),
            'bridge_age': _number(_get(snapshot, 'bridge_age')),
            'shadow_only': _flag(_get(snapshot, 'shadow_only')),
        }
        self._frames.append({
            'sequence': seq, 'produced_at': at, 'elapsed_ms': _number(elapsed_ms),
            'name': {
                'valid': _flag(_get(name, 'valid')),
                'kind': _text(_get(name, 'match_kind')),
                'score': _number(_get(name, 'score')),
                'foot': _coordinates(_get(name, 'player'), 2),
            },
            'candidates': candidates, 'invalid_candidate_count': invalid_candidates,
            'snapshot': state,
        })
        self._last_sequence = seq
        self._accepted += 1
        return True

    def summary(self):
        """Statistics are over retained frames; counters cover this trace's life."""
        frames = self._frames
        timings = sorted(frame['elapsed_ms'] for frame in frames
                         if frame['elapsed_ms'] is not None and frame['elapsed_ms'] >= 0)
        p95 = timings[math.ceil(len(timings)*.95)-1] if timings else None
        available = sum(frame['snapshot']['control_allowed'] for frame in frames)
        return {
            'accepted_frames': self._accepted,
            'retained_frames': len(frames),
            'rejected_sequence': self._rejected_sequence,
            'rejected_invalid': self._rejected_invalid,
            'first_sequence': frames[0]['sequence'] if frames else None,
            'last_sequence': frames[-1]['sequence'] if frames else None,
            'control_available_frames': available,
            'control_available_ratio': available/len(frames) if frames else None,
            'sources': dict(Counter(frame['snapshot']['source'] for frame in frames)),
            'elapsed_p95_ms': p95,
        }

    def to_dict(self):
        """Return independent strict-JSON data, without owning any live inputs."""
        return {
            'schema_version': 1, 'capacity': self.MAX_FRAMES,
            'frames': deepcopy(list(self._frames)), 'summary': self.summary(),
        }
