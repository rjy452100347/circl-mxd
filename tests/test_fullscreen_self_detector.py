import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.engine.FullscreenSelfDetector import (
    CLASS_NAMES, MODEL_PATH, FullscreenSelfDetector, SelfLocalizationGate,
    OpenVinoDeploymentError,
)


@pytest.fixture
def deployment(tmp_path):
    xml = tmp_path / MODEL_PATH
    xml.parent.mkdir(parents=True)
    xml.write_bytes(b"test model")
    xml.with_suffix(".bin").write_bytes(b"test weights")
    (tmp_path / "ACTIVE_MODEL.txt").write_text(MODEL_PATH, encoding="utf-8")
    (xml.parent / "metadata.yaml").write_text(yaml.safe_dump(dict(
        names=CLASS_NAMES, imgsz=[736, 1280], end2end=True, args=dict(nms=False))), encoding="utf-8")
    (xml.parent / "quantization.json").write_text(json.dumps(dict(model=dict(
        xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        bin_sha256=hashlib.sha256(xml.with_suffix('.bin').read_bytes()).hexdigest(),
        input_shape=[1, 3, 736, 1280], output_shape=[1, 300, 6],
    ))), encoding="utf-8")
    return tmp_path


def detector(root, rows=(), **kwargs):
    raw = np.zeros((1, 300, 6), np.float32)
    raw[0, :len(rows)] = rows
    class Runtime:
        def __init__(self, xml):
            self.xml = xml
            self.calls = 0
        def predict(self, tensor):
            assert tensor.shape == (1, 3, 736, 1280)
            assert tensor.dtype == np.float32
            self.calls += 1
            return raw.copy()
    return FullscreenSelfDetector(root, runtime_factory=Runtime, **kwargs)


def test_centered_letterbox_color_layout_and_roundtrip(deployment):
    frame = np.full((1152, 2049, 3), (10, 20, 30), np.uint8)
    tensor, (sx, sy, tx, ty) = FullscreenSelfDetector.prepare(frame)
    assert sx == 1280/2049 and sy == 720/1152 and tx == 0 and ty == 8
    np.testing.assert_allclose(tensor[0, :, 100, 100], np.array([30, 20, 10])/255)
    np.testing.assert_allclose(tensor[0, :, 0, 0], np.full(3, 114/255))
    box = np.array([800, 400, 850, 500], dtype=float)
    scaled = box*[sx, sy, sx, sy] + [tx, ty, tx, ty]
    d = detector(deployment, [[*scaled, .9, 2]])
    result = d.detect_scene(frame, frame_token=12)
    assert result.frame_token == 12 and result.visible_rect == (0, 0, 2049, 1152)
    x, y = result.selves[0]['position']
    w, h = result.selves[0]['size']
    np.testing.assert_allclose([x, y, x+w, y+h], box, atol=1)
    assert not result.players and not result.monsters


def test_v3_game_view_crop_70pct_input_and_full_frame_coordinates(deployment):
    frame = np.full((1152, 2049, 3), (200, 0, 0), np.uint8)
    frame[:1035, 4:2044] = (10, 20, 30)
    box = np.array([800, 400, 850, 500], dtype=float)
    sx = 1280/2040 * 896/1280
    sy = 649/1035 * 515/736
    tx, ty = 192., 43*515/736+110
    relative = box - [4, 0, 4, 0]
    scaled = relative*[sx, sy, sx, sy]+[tx, ty, tx, ty]
    d = detector(deployment, [[*scaled, .9, 2]],
                 view_crop=(4, 0, 5, 117), content_scale=.70)
    scene = d.detect_scene(frame)
    assert scene.visible_rect == (4, 0, 2044, 1035)
    assert scene.frame_shape == frame.shape
    item = scene.selves[0]
    x, y = item['position']; w, h = item['size']
    np.testing.assert_allclose([x, y, x+w, y+h], box, atol=1)

    tensor, transform = d.prepare(frame[:1035, 4:2044], .70)
    np.testing.assert_allclose(transform, (sx, sy, tx, ty))
    np.testing.assert_allclose(tensor[0, :, 370, 640], [30/255, 20/255, 10/255])
    np.testing.assert_allclose(tensor[0, :, 0, 0], [114/255]*3)
    assert not np.any(np.all(tensor[0].transpose(1, 2, 0) == [0, 0, 200/255], axis=2))


def test_hundreds_of_monsters_cannot_evict_self(deployment):
    rows = [[20, 30, 60, 80, .99, 0]]*298 + [[600, 300, 640, 400, .9, 1], [800, 300, 840, 400, .51, 2]]
    d = detector(deployment, rows)
    result = d.detect_scene(np.zeros((736, 1280, 3), np.uint8))
    assert (len(result.monsters), len(result.players), len(result.selves)) == (298, 1, 1)
    assert d.last_timing.detections == 300


def test_independent_thresholds_and_exclusion_only_affect_monsters(deployment):
    rows = [[20, 20, 40, 40, .69, 0], [20, 20, 40, 40, .26, 1],
            [20, 20, 40, 40, .49, 2], [25, 20, 50, 60, .8, 0], [20, 20, 40, 40, .8, 2]]
    d = detector(deployment, rows)
    scene = d.detect_scene(np.zeros((736, 1280, 3), np.uint8))
    assert tuple(map(len, (scene.monsters, scene.players, scene.selves))) == (1, 1, 1)
    assert d.finalize_scene(scene, (35, 80)) == []
    assert len(scene.monsters) == 1  # Shared observations remain immutable.
    with pytest.raises(TypeError):
        scene.selves[0]['name'] = 'player'


@pytest.mark.parametrize('failure', ['hash', 'missing_bin', 'classes', 'active', 'shape'])
def test_rejects_bad_deployment_without_constructing_runtime(deployment, failure):
    xml = deployment / MODEL_PATH
    if failure == 'hash':
        xml.write_bytes(b'changed')
    elif failure == 'missing_bin':
        xml.with_suffix('.bin').unlink()
    elif failure == 'active':
        (deployment/'ACTIVE_MODEL.txt').write_text('wrong.xml')
    elif failure == 'classes':
        metadata = xml.parent/'metadata.yaml'
        content = yaml.safe_load(metadata.read_text())
        content['names'][2] = 'player'
        metadata.write_text(yaml.safe_dump(content))
    else:
        record = xml.parent/'quantization.json'
        content = json.loads(record.read_text())
        content['model']['input_shape'] = [1, 224, 1280, 3]
        record.write_text(json.dumps(content))
    with pytest.raises(OpenVinoDeploymentError):
        FullscreenSelfDetector(deployment, runtime_factory=lambda *_: pytest.fail('Invalid deployment must not compile'))


def test_self_requires_two_new_frames_and_cannot_fall_back_to_player(deployment):
    d = detector(deployment, [[10, 20, 60, 100, .9, 2], [400, 20, 460, 100, .99, 1]])
    scene = d.detect_scene(np.zeros((736, 1280, 3), np.uint8))
    gate = SelfLocalizationGate()
    assert not gate.update(scene, 1).valid
    assert not gate.update(scene, 1).valid
    assert gate.update(scene, 2).foot == (35, 100)
    assert gate.snapshot.valid
    assert not gate.update(replace(scene, selves=()), 3).valid
    assert gate.snapshot.reason == 'self_missing'
    assert not gate.update(scene, 4).valid
    assert gate.update(scene, 5).valid
    assert not gate.update(replace(scene, selves=scene.selves*2), 6).valid
    assert gate.snapshot.reason == 'self_ambiguous'
    assert not gate.update(scene, 7).valid
    gate.invalidate('frame_stale')
    assert not gate.update(scene, 7).valid
    assert not gate.update(scene, 8).valid
    assert gate.update(scene, 9).valid


def test_decode_discards_bad_geometry_nonfinite_rows_and_fails_unknown_class(deployment):
    d = detector(deployment, [[0, 0, 0, 0, .9, 2], [20, 20, np.nan, 50, .9, 2]])
    assert not d.detect_scene(np.zeros((736, 1280, 3), np.uint8)).selves
    d = detector(deployment, [[20, 20, 50, 50, .9, 3]])
    with pytest.raises(ValueError, match='类别'):
        d.detect_scene(np.zeros((736, 1280, 3), np.uint8))
