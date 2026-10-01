import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pytest

from src.engine.OpenVinoMonsterDetector import (
    MODEL_HEIGHT,
    MODEL_WIDTH,
    OpenVinoDeploymentError,
    OpenVinoMonsterDetector,
)


@dataclass
class _Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class _Runtime:
    def __init__(self, predictions):
        self.predictions = predictions
        self.images = []

    def predict(self, image):
        self.images.append(image.copy())
        return list(self.predictions)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _deployment(tmp_path, *, classes=None, active=None):
    root = tmp_path / "deployment"
    model_dir = root / "models" / "fp16"
    model_dir.mkdir(parents=True)
    xml = b"two-class-fp16-xml"
    binary = b"two-class-fp16-bin"
    (model_dir / "model.xml").write_bytes(xml)
    (model_dir / "model.bin").write_bytes(binary)
    int8_dir = root / "models" / "int8"
    int8_dir.mkdir(parents=True)
    int8_xml = b"two-class-int8-xml"
    int8_binary = b"two-class-int8-bin"
    (int8_dir / "model.xml").write_bytes(int8_xml)
    (int8_dir / "model.bin").write_bytes(int8_binary)
    int8_v2_dir = root / "int8_v2" / "models" / "mixed_head_fp"
    int8_v2_dir.mkdir(parents=True)
    int8_v2_xml = b"two-class-int8-v2-xml"
    int8_v2_binary = b"two-class-int8-v2-bin"
    (int8_v2_dir / "model.xml").write_bytes(int8_v2_xml)
    (int8_v2_dir / "model.bin").write_bytes(int8_v2_binary)
    (root / "ACTIVE_MODEL.txt").write_text(
        active or "int8_v2\nint8_v2/models/mixed_head_fp/model.xml\n",
        encoding="utf-8",
    )
    manifest = {
        "class_names": classes or {"0": "monster", "1": "player"},
        "preprocessing": {
            "external_input": {
                "shape": [1, 224, 1280, 3],
                "dtype": "uint8",
                "layout": "NHWC",
                "color": "BGR",
            }
        },
        "output": {"shape": [1, 300, 6], "nms": False},
        "selection": {
            "active_variant": "int8_v2",
            "model_path": "int8_v2/models/mixed_head_fp/model.xml",
        },
        "candidates": {
            "fp16": {
                "model_path": "models/fp16/model.xml",
                "sha256_xml": _sha256(xml),
                "sha256_bin": _sha256(binary),
            },
            "int8": {
                "model_path": "models/int8/model.xml",
                "sha256_xml": _sha256(int8_xml),
                "sha256_bin": _sha256(int8_binary),
            },
            "int8_v2": {
                "model_path": "int8_v2/models/mixed_head_fp/model.xml",
                "sha256_xml": _sha256(int8_v2_xml),
                "sha256_bin": _sha256(int8_v2_binary),
            },
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


def _detector(tmp_path, predictions=(), **detector_kwargs):
    runtime = _Runtime(predictions)
    detector = OpenVinoMonsterDetector(
        _deployment(tmp_path),
        runtime_factory=lambda *_args, **_kwargs: runtime,
        **detector_kwargs,
    )
    return detector, runtime


def test_deployment_contract_accepts_two_class_int8_v2(tmp_path):
    detector, _ = _detector(tmp_path)
    assert detector.manifest["class_names"] == {"0": "monster", "1": "player"}
    assert detector.variant == "int8_v2"


def test_explicit_int8_v1_is_allowed_but_never_selected_by_default(tmp_path):
    calls = []

    def factory(*args, **kwargs):
        calls.append(kwargs)
        return _Runtime([])

    detector = OpenVinoMonsterDetector(
        _deployment(tmp_path), variant="int8", runtime_factory=factory
    )
    assert detector.variant == "int8"
    assert calls[0]["variant"] == "int8"


def test_unknown_variant_is_rejected(tmp_path):
    with pytest.raises(OpenVinoDeploymentError, match="auto、fp16、int8 或 int8_v2"):
        OpenVinoMonsterDetector(
            _deployment(tmp_path), variant="fp32",
            runtime_factory=lambda *_args, **_kwargs: _Runtime([]),
        )


@pytest.mark.parametrize(
    "classes",
    [
        {"0": "monster"},
        {"0": "monster", "1": "other"},
        {"0": "monster", "1": "player", "2": "item"},
    ],
)
def test_deployment_contract_rejects_non_two_class_contract(tmp_path, classes):
    root = _deployment(tmp_path, classes=classes)
    with pytest.raises(OpenVinoDeploymentError, match="类别必须严格"):
        OpenVinoMonsterDetector(
            root, runtime_factory=lambda *_args, **_kwargs: _Runtime([])
        )


def test_deployment_contract_rejects_changed_active_model(tmp_path):
    root = _deployment(tmp_path, active="int8\nmodels/int8/model.xml\n")
    with pytest.raises(OpenVinoDeploymentError, match="ACTIVE_MODEL"):
        OpenVinoMonsterDetector(
            root, runtime_factory=lambda *_args, **_kwargs: _Runtime([])
        )


def test_deployment_contract_rejects_hash_mismatch(tmp_path):
    root = _deployment(tmp_path)
    (root / "int8_v2" / "models" / "mixed_head_fp" / "model.bin").write_bytes(
        b"changed"
    )
    with pytest.raises(OpenVinoDeploymentError, match="哈希校验失败"):
        OpenVinoMonsterDetector(
            root, runtime_factory=lambda *_args, **_kwargs: _Runtime([])
        )


def test_confidence_range_and_fixed_max_det_are_enforced(tmp_path):
    root = _deployment(tmp_path)
    factory = lambda *_args, **_kwargs: _Runtime([])
    with pytest.raises(OpenVinoDeploymentError, match="0.05–0.95"):
        OpenVinoMonsterDetector(root, confidence=0.01, runtime_factory=factory)
    with pytest.raises(OpenVinoDeploymentError, match="固定为 50"):
        OpenVinoMonsterDetector(root, max_det=49, runtime_factory=factory)


@pytest.mark.parametrize("value", [-1, 225, 10.5, "not-an-integer"])
def test_min_monster_box_side_range_and_integer_are_enforced(tmp_path, value):
    root = _deployment(tmp_path)
    factory = lambda *_args, **_kwargs: _Runtime([])
    with pytest.raises(OpenVinoDeploymentError, match="0–224"):
        OpenVinoMonsterDetector(
            root, min_monster_box_side=value, runtime_factory=factory
        )




def test_centered_crop_is_native_size_and_contiguous(tmp_path):
    detector, runtime = _detector(tmp_path)
    frame = np.full((500, 1600, 3), 17, dtype=np.uint8)
    detector.detect(frame, (800, 250))

    assert runtime.images[0].shape == (MODEL_HEIGHT, MODEL_WIDTH, 3)
    assert runtime.images[0].dtype == np.uint8
    assert runtime.images[0].flags.c_contiguous
    assert np.all(runtime.images[0] == 17)
    assert detector.last_visible_rect == (160, 138, 1440, 362)


@pytest.mark.parametrize(
    "player,visible,filled_slice",
    [
        ((0, 0), (0, 0, 640, 112), (slice(112, 224), slice(640, 1280))),
        ((1599, 0), (959, 0, 1600, 112), (slice(112, 224), slice(0, 641))),
        ((0, 499), (0, 387, 640, 500), (slice(0, 113), slice(640, 1280))),
        ((1599, 499), (959, 387, 1600, 500), (slice(0, 113), slice(0, 641))),
    ],
)
def test_edge_crops_use_black_padding(tmp_path, player, visible, filled_slice):
    detector, runtime = _detector(tmp_path)
    frame = np.full((500, 1600, 3), 29, dtype=np.uint8)
    detector.detect(frame, player)
    crop = runtime.images[0]

    assert detector.last_visible_rect == visible
    assert np.all(crop[filled_slice] == 29)
    assert np.count_nonzero(np.all(crop == 29, axis=2)) == (
        (visible[2] - visible[0]) * (visible[3] - visible[1])
    )


def test_detection_conversion_maps_to_frame_and_preserves_score_semantics(tmp_path):
    predictions = [
        _Detection(600.2, 100.4, 650.1, 150.6, 0.9, 0),
        _Detection(0, 0, 10, 10, 0.8, 1),
        _Detection(5, 5, 6, 6, 0.1, 0),
        _Detection(float("nan"), 0, 10, 10, 0.8, 0),
        _Detection(20, 20, 20, 30, 0.8, 0),
    ]
    detector, _ = _detector(tmp_path, predictions)
    frame = np.zeros((500, 1600, 3), dtype=np.uint8)

    monsters = detector.detect(frame, (800, 250))

    assert monsters == [{
        "name": "monster",
        "position": (760, 238),
        "size": (51, 51),
        "confidence": 0.9,
        "score": pytest.approx(0.1),
        "class_id": 0,
        "detector": "openvino_yolo",
    }]
    assert detector.last_timing.detections == 2
    assert detector.last_timing.monsters == 1
    assert detector.last_timing.players == 1
    assert detector.last_players == [{
        "name": "player",
        "position": (160, 138),
        "size": (10, 10),
        "confidence": 0.8,
        "score": pytest.approx(0.2),
        "class_id": 1,
        "detector": "openvino_yolo",
    }]
    assert detector.last_player_foot == (165, 148)


def test_small_monster_boxes_are_filtered_after_conversion_only(tmp_path):
    predictions = [
        _Detection(20, 20, 29, 70, 0.9, 0),
        _Detection(40, 20, 70, 29, 0.9, 0),
        _Detection(80, 20, 90, 50, 0.9, 0),
        _Detection(100, 20, 105, 25, 0.9, 1),
    ]
    detector, _ = _detector(
        tmp_path, predictions, min_monster_box_side=10
    )
    frame = np.zeros((500, 1600, 3), dtype=np.uint8)

    monsters = detector.detect(frame, (800, 250))

    assert [item["size"] for item in monsters] == [(10, 30)]
    assert [item["size"] for item in detector.last_players] == [(5, 5)]
    assert detector.last_timing.detections == 2
    assert detector.last_timing.monsters == 1
    assert detector.last_timing.players == 1


def test_zero_min_monster_box_side_disables_filter(tmp_path):
    detector, _ = _detector(
        tmp_path,
        [_Detection(5, 5, 6, 6, 0.9, 0)],
        min_monster_box_side=0,
    )
    frame = np.zeros((500, 1600, 3), dtype=np.uint8)

    monsters = detector.detect(frame, (800, 250))

    assert [item["size"] for item in monsters] == [(1, 1)]
    assert detector.last_timing.detections == 1
    assert detector.last_timing.monsters == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("player_exclusion_width", -1, "0–1280"),
        ("player_exclusion_width", 1281, "0–1280"),
        ("player_exclusion_width", 80.5, "0–1280"),
        ("player_exclusion_height", -1, "0–224"),
        ("player_exclusion_height", 225, "0–224"),
        ("player_exclusion_height", "bad", "0–224"),
    ],
)
def test_player_exclusion_dimensions_are_validated(tmp_path, field, value, message):
    with pytest.raises(OpenVinoDeploymentError, match=message):
        _detector(tmp_path, **{field: value})


def test_player_exclusion_filters_monster_centers_including_boundary(tmp_path):
    predictions = [
        _Detection(610, 42, 650, 82, 0.95, 0),  # center inside
        _Detection(590, 42, 610, 82, 0.94, 0),  # center on left boundary
        _Detection(670, 42, 710, 82, 0.93, 0),  # overlaps, center outside
        _Detection(610, 42, 650, 82, 0.92, 1),  # player is never filtered
    ]
    detector, _ = _detector(
        tmp_path,
        predictions,
        player_exclusion_width=80,
        player_exclusion_height=100,
    )
    frame = np.zeros((500, 1600, 3), dtype=np.uint8)

    monsters = detector.detect(
        frame, (800, 250), player_exclusion_anchor=(800, 250)
    )

    assert [item["position"] for item in monsters] == [(830, 180)]
    assert len(detector.last_players) == 1
    assert detector.last_timing.detections == 2
    assert detector.last_timing.monsters == 1
    assert detector.last_timing.players == 1


@pytest.mark.parametrize(
    "dimensions", [(0, 100), (80, 0)]
)
def test_zero_player_exclusion_dimension_disables_filter(tmp_path, dimensions):
    detector, _ = _detector(
        tmp_path,
        [_Detection(610, 42, 650, 82, 0.95, 0)],
        player_exclusion_width=dimensions[0],
        player_exclusion_height=dimensions[1],
    )
    frame = np.zeros((500, 1600, 3), dtype=np.uint8)

    monsters = detector.detect(
        frame, (800, 250), player_exclusion_anchor=(800, 250)
    )

    assert len(monsters) == 1


def test_missing_player_exclusion_anchor_does_not_use_stale_location(tmp_path):
    detector, _ = _detector(
        tmp_path, [_Detection(610, 42, 650, 82, 0.95, 0)]
    )
    frame = np.zeros((500, 1600, 3), dtype=np.uint8)

    monsters = detector.detect(frame, (800, 250))

    assert len(monsters) == 1


def test_player_exclusion_rect_is_clipped_to_frame_edges(tmp_path):
    detector, _ = _detector(
        tmp_path, player_exclusion_width=80, player_exclusion_height=100
    )

    assert detector.player_exclusion_rect((50, 60, 3), (5, 10)) == (
        0, 0, 45, 10
    )


def test_nearest_player_to_anchor_is_selected_for_diagnostic_foot(tmp_path):
    detector, _ = _detector(tmp_path, [
        _Detection(50, 50, 90, 100, 0.99, 1),
        _Detection(620, 80, 660, 112, 0.75, 1),
    ])
    frame = np.zeros((500, 1600, 3), dtype=np.uint8)
    assert detector.detect(frame, (800, 250)) == []
    assert len(detector.last_players) == 2
    assert detector.last_player_foot == (800, 250)


def test_detection_at_padded_edge_is_clipped_to_frame(tmp_path):
    detector, _ = _detector(
        tmp_path,
        [_Detection(630, 100, 690, 150, 0.75, 0)],
    )
    frame = np.zeros((300, 400, 3), dtype=np.uint8)

    monsters = detector.detect(frame, (0, 0))

    assert monsters[0]["position"] == (0, 0)
    assert monsters[0]["size"] == (50, 38)
