import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import src.engine.SemanticRoute as module
from src.engine.SemanticRoute import (
    SemanticRouteError,
    convert_legacy_png_to_draft,
    load_route_document,
    load_semantic_route_set,
    render_route_png,
    save_route_pair_atomic,
    validate_route_document,
)


COLORS = {
    "left none none": (255, 0, 0),
    "right none none": (0, 0, 255),
    "left none jump": (255, 127, 0),
    "right none jump": (0, 255, 255),
    "none down jump": (127, 255, 0),
    "none none jump": (255, 0, 255),
    "stop stop stop": (0, 255, 127),
    "none none goal": (255, 255, 0),
    "none up teleport": (255, 0, 127),
    "none down teleport": (127, 0, 255),
    "left none teleport": (0, 127, 0),
    "right none teleport": (139, 69, 19),
    "none up none": (127, 127, 127),
    "none down none": (255, 255, 127),
}


def _document(index=1):
    return {
        "schema_version": 1,
        "map_id": "training",
        "route_index": index,
        "canvas_size": [80, 50],
        "loop": True,
        "segments": [
            {
                "id": "walk_001",
                "type": "walk",
                "direction": "right",
                "points": [[4, 30], [15, 30], [25, 30]],
            },
            {
                "id": "ladder_001",
                "type": "ladder",
                "approach": [30, 30],
                "mount": [30, 30],
                "points": [[30, 30], [30, 20], [30, 10]],
                "exit": [36, 10],
                "exit_direction": "right",
            },
            {
                "id": "goal_001",
                "type": "goal",
                "position": [70, 10],
                "radius": 6,
            },
        ],
    }


def test_document_contract_and_expected_identity():
    validated = validate_route_document(
        _document(),
        expected_map_id="training",
        expected_canvas_size=(80, 50),
        expected_route_index=1,
    )
    assert validated == _document()
    validated["segments"][0]["points"][0][0] = 99
    assert _document()["segments"][0]["points"][0] == [4, 30]


def test_ladder_mount_direction_is_optional_and_strictly_validated():
    legacy = validate_route_document(_document())
    assert "mount_direction" not in legacy["segments"][1]

    forward = _document()
    forward["segments"][1]["mount_direction"] = "right"
    assert validate_route_document(forward)["segments"][1]["mount_direction"] == "right"

    forward["segments"][1]["mount_direction"] = "up"
    with pytest.raises(SemanticRouteError, match="mount_direction"):
        validate_route_document(forward)


@pytest.mark.parametrize(
    ("mount_direction", "command"),
    (("none", "none none jump"), ("left", "left none jump"),
     ("right", "right none jump")),
)
def test_ladder_mount_marker_uses_mount_direction_color(
    mount_direction, command
):
    document = _document()
    document["segments"][1]["mount_direction"] = mount_direction
    canvas = render_route_png(
        np.zeros((50, 80, 3), dtype=np.uint8), document, COLORS
    )
    expected_bgr = tuple(reversed(COLORS[command]))
    assert tuple(canvas[30, 30]) == expected_bgr


@pytest.mark.parametrize(
    "mutator,message",
    [
        (lambda doc: doc.update(schema_version=2), "schema_version"),
        (lambda doc: doc.update(canvas_size=[0, 80]), "canvas_size"),
        (lambda doc: doc["segments"].append({"id": "bad", "type": "warp"}), "不受支持"),
        (lambda doc: doc["segments"].append({
            "id": "goal_2", "type": "goal", "position": [1, 1], "radius": 2,
        }), "只能包含一个"),
        (lambda doc: doc["segments"][0]["points"].append([999, 1]), "超出地图范围"),
    ],
)
def test_invalid_route_is_rejected(mutator, message):
    document = _document()
    mutator(document)
    with pytest.raises(SemanticRouteError, match=message):
        validate_route_document(document)


def test_draft_cannot_be_executed():
    document = _document()
    document["draft"] = True
    validate_route_document(document, allow_draft=True)
    with pytest.raises(SemanticRouteError, match="草稿路线"):
        validate_route_document(document)


def test_rendered_continuous_trace_is_one_pixel_wide():
    base = np.zeros((50, 80, 3), dtype=np.uint8)
    rendered = render_route_png(base, _document(), COLORS)
    blue_bgr = (255, 0, 0)
    mask = np.all(rendered == blue_bgr, axis=2)
    rows = np.where(mask & (np.indices(mask.shape)[0] > 25))[0]
    assert set(rows) == {30}
    assert mask[30, 4:26].all()


def test_atomic_save_load_and_semantic_precedence(tmp_path):
    base = np.zeros((50, 80, 3), dtype=np.uint8)
    json_path, png_path = save_route_pair_atomic(tmp_path, _document(), base, COLORS)

    assert json_path.name == "route1.json"
    assert png_path.name == "route1.png"
    assert load_route_document(json_path) == _document()
    loaded = load_semantic_route_set(
        tmp_path, map_id="training", canvas_size=(80, 50)
    )
    assert loaded.source == "semantic"
    assert loaded.documents[0]["route_index"] == 1


def test_missing_json_keeps_legacy_mode(tmp_path):
    cv2.imwrite(str(tmp_path / "route1.png"), np.zeros((10, 10, 3), dtype=np.uint8))
    loaded = load_semantic_route_set(
        tmp_path, map_id="training", canvas_size=(10, 10)
    )
    assert loaded.source == "legacy"
    assert loaded.documents == ()


def test_invalid_json_does_not_fall_back_to_png(tmp_path):
    cv2.imwrite(str(tmp_path / "route1.png"), np.zeros((50, 80, 3), dtype=np.uint8))
    (tmp_path / "route1.json").write_text("{bad json", encoding="utf-8")
    with pytest.raises(SemanticRouteError, match="无法读取路线 JSON"):
        load_semantic_route_set(
            tmp_path, map_id="training", canvas_size=(80, 50)
        )


def test_non_contiguous_semantic_routes_are_rejected(tmp_path):
    base = np.zeros((50, 80, 3), dtype=np.uint8)
    save_route_pair_atomic(tmp_path, _document(index=1), base, COLORS)
    save_route_pair_atomic(tmp_path, _document(index=3), base, COLORS)
    with pytest.raises(SemanticRouteError, match="编号必须连续"):
        load_semantic_route_set(
            tmp_path, map_id="training", canvas_size=(80, 50)
        )


def test_semantic_set_rejects_paired_png_with_wrong_canvas_size(tmp_path):
    document = _document()
    (tmp_path / "route1.json").write_text(json.dumps(document), encoding="utf-8")
    cv2.imwrite(str(tmp_path / "route1.png"), np.zeros((10, 10, 3), dtype=np.uint8))

    with pytest.raises(SemanticRouteError, match="尺寸不匹配"):
        load_semantic_route_set(
            tmp_path, map_id="training", canvas_size=(80, 50)
        )


def test_atomic_pair_failure_restores_previous_files(tmp_path, monkeypatch):
    base = np.zeros((50, 80, 3), dtype=np.uint8)
    json_path, png_path = save_route_pair_atomic(tmp_path, _document(), base, COLORS)
    old_json = json_path.read_bytes()
    old_png = png_path.read_bytes()
    changed = _document()
    changed["segments"][0]["points"][1] = [20, 30]

    real_replace = module.os.replace
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected pair failure")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected"):
        save_route_pair_atomic(tmp_path, changed, base, COLORS)

    assert json_path.read_bytes() == old_json
    assert png_path.read_bytes() == old_png
    assert not list(tmp_path.glob("*.tmp"))


def test_legacy_conversion_creates_draft_without_overwriting_png(tmp_path):
    base = np.zeros((30, 40, 3), dtype=np.uint8)
    route = base.copy()
    cv2.line(route, (3, 20), (20, 20), (255, 0, 0), 1)
    cv2.circle(route, (30, 20), 2, (0, 255, 255), -1)
    path = tmp_path / "route1.png"
    cv2.imwrite(str(path), route)
    original = path.read_bytes()

    draft = convert_legacy_png_to_draft(
        path, map_id="training", route_index=1, command_colors=COLORS
    )

    data = json.loads(draft.read_text(encoding="utf-8"))
    assert data["draft"] is True
    assert data["segments"][-1]["type"] == "goal"
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError):
        convert_legacy_png_to_draft(
            path, map_id="training", route_index=1, command_colors=COLORS
        )
