import gc
import json
from types import SimpleNamespace
import weakref

import numpy as np

from src.engine.PlayerLocalization import PlayerLocalizationSnapshot
from src.engine.PlayerTrackingTrace import PlayerTrackingTrace


class _Observation:
    def __init__(self, **values):
        self.__dict__.update(values)


def _record(trace, sequence, **kwargs):
    return trace.record(
        kwargs.pop('name', None), kwargs.pop('players', ()),
        kwargs.pop('snapshot', PlayerLocalizationSnapshot(sequence, sequence*.1)),
        sequence=sequence, produced_at=kwargs.pop('produced_at', sequence*.1),
        elapsed_ms=kwargs.pop('elapsed_ms', 2.), **kwargs)


def _player(index=0, confidence=.9):
    return {'class_id': 1, 'confidence': confidence, 'position': (index, 20), 'size': (30, 60)}


def test_trace_keeps_latest_300_frames_only():
    trace = PlayerTrackingTrace()
    for sequence in range(350):
        assert _record(trace, sequence)
    exported = trace.to_dict()
    assert exported['capacity'] == 300
    assert len(exported['frames']) == 300
    assert exported['frames'][0]['sequence'] == 50
    assert exported['frames'][-1]['sequence'] == 349
    assert exported['summary']['accepted_frames'] == 350
    assert exported['summary']['retained_frames'] == 300


def test_duplicate_backwards_and_invalid_sequences_never_record_a_sample():
    trace = PlayerTrackingTrace()
    assert _record(trace, 10)
    assert not _record(trace, 10)
    assert not _record(trace, 9)
    assert not _record(trace, -1)
    assert not _record(trace, 10.5)
    assert not _record(trace, True)
    assert not _record(trace, 11, produced_at=float('nan'))
    assert _record(trace, np.int64(11))
    assert [item['sequence'] for item in trace.to_dict()['frames']] == [10, 11]
    assert trace.summary()['rejected_sequence'] == 2
    assert trace.summary()['rejected_invalid'] == 4


def test_candidate_pools_are_deduplicated_and_storage_is_bounded():
    trace = PlayerTrackingTrace()
    high = [_player(index) for index in range(50)]
    raw = high + [_player(index, .4) for index in range(50, 350)]
    assert _record(trace, 1, players=high, player_candidates=raw)
    candidates = trace.to_dict()['frames'][0]['candidates']
    assert len(candidates) == 100
    assert candidates[0] == {'class_id': 1, 'confidence': .9, 'bbox': [0., 20., 30., 60.]}
    assert candidates[50]['confidence'] == .4
    assert len({tuple(item['bbox']) for item in candidates}) == 100


def test_trace_copies_only_scalars_and_does_not_retain_images_or_input_objects():
    trace = PlayerTrackingTrace()
    image = np.zeros((300, 400, 3), np.uint8)
    image_ref = weakref.ref(image)
    name = _Observation(valid=True, match_kind='left', score=np.float32(.1),
                           player=np.array([100, 200]), frame=image,
                           profile_name='私人角色', private_path='C:/Users/private/sample.png')
    name_ref = weakref.ref(name)
    snapshot = _Observation(track_id=1, phase='tracking', source='body',
                               foot=np.array([100, 200]), observation_foot=(100, 200),
                               control_allowed=True, cache_safe=True, visual_valid=True,
                               visual_reason='身体视觉跟踪有效', bridge_age=.1,
                               shadow_only=False, rejection_reason='', image=image,
                               license_key='private-license')
    snapshot_ref = weakref.ref(snapshot)
    player = _player()
    player['image'] = image
    assert _record(trace, 1, name=name, players=[player], snapshot=snapshot)
    name.player[:] = 0
    snapshot.foot[:] = 0
    player['position'] = (999, 999)
    del image, name, snapshot, player
    gc.collect()
    assert image_ref() is None and name_ref() is None and snapshot_ref() is None
    exported = trace.to_dict()
    saved = exported['frames'][0]
    assert saved['name']['foot'] == saved['snapshot']['foot'] == [100., 200.]
    assert saved['candidates'][0]['bbox'][:2] == [0., 20.]
    encoded = json.dumps(exported, ensure_ascii=False, allow_nan=False)
    assert '私人角色' not in encoded and 'private' not in encoded
    assert 'image' not in encoded and 'profile_name' not in encoded


def test_export_and_summary_are_independent_and_bad_numbers_are_json_safe():
    trace = PlayerTrackingTrace()
    name = SimpleNamespace(valid=False, match_kind='full', score=float('inf'),
                           player=(float('nan'), 5))
    snapshot = SimpleNamespace(source='none', phase='lost', bridge_age=float('nan'),
                               rejection_reason='输入无效 C:\\Users\\private\\game.png',
                               visual_reason=np.zeros((5, 5)), foot=np.zeros((20, 20)))
    assert _record(trace, 1, name=name, snapshot=snapshot, elapsed_ms=float('inf'),
                   players=[_player(confidence=float('nan')), np.zeros((10, 10))])
    exported = trace.to_dict()
    saved = exported['frames'][0]
    assert saved['name']['score'] is saved['name']['foot'] is None
    assert saved['snapshot']['bridge_age'] is saved['snapshot']['foot'] is None
    assert saved['snapshot']['visual_reason'] == ''
    assert 'private' not in saved['snapshot']['rejection_reason']
    assert saved['invalid_candidate_count'] == 2
    assert saved['candidates'] == []
    json.dumps(exported, allow_nan=False)
    exported['frames'][0]['snapshot']['source'] = 'mutated'
    exported['summary']['sources']['none'] = 999
    assert trace.to_dict()['frames'][0]['snapshot']['source'] == 'none'
    assert trace.summary()['sources'] == {'none': 1}


def test_summary_reports_retained_availability_and_p95_without_io(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trace = PlayerTrackingTrace()
    assert trace.summary()['elapsed_p95_ms'] is None
    for sequence in range(20):
        snapshot = PlayerLocalizationSnapshot(sequence, sequence*.1,
                                              source='body', control_allowed=sequence < 10)
        assert _record(trace, sequence, snapshot=snapshot, elapsed_ms=sequence+1.)
    summary = trace.summary()
    assert summary['elapsed_p95_ms'] == 19.
    assert summary['control_available_frames'] == 10
    assert summary['control_available_ratio'] == .5
    assert summary['sources'] == {'body': 20}
    json.dumps(trace.to_dict(), allow_nan=False)
    assert list(tmp_path.iterdir()) == []
