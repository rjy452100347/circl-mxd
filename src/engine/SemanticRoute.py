"""Versioned semantic route assets and deterministic PNG rendering."""

from __future__ import annotations

import copy
import json
import os
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA_VERSION = 1
SEGMENT_TYPES = frozenset({
    "walk", "jump", "drop", "ladder", "teleport", "stop", "goal",
})
ROUTE_JSON_RE = re.compile(r"^route([1-9][0-9]*)\.json$")
ROUTE_PNG_RE = re.compile(r"^route([1-9][0-9]*)\.png$")


class SemanticRouteError(ValueError):
    """Raised when a semantic route asset violates the runtime contract."""


def calculate_ladder_mount_x(
    segment: dict[str, Any], *, minimum_rise: int = 12
) -> int | None:
    """Estimate the real ladder center from stable post-mount samples only."""
    approach = segment.get("approach")
    points = segment.get("points") or []
    if not approach or len(points) < 2:
        return None
    threshold_y = int(approach[1]) - max(1, int(minimum_rise))
    candidates: list[int] = []
    for index, point in enumerate(points):
        if int(point[1]) > threshold_y:
            continue
        neighbours = []
        if index:
            neighbours.append(points[index - 1])
        if index + 1 < len(points):
            neighbours.append(points[index + 1])
        if neighbours and any(abs(int(point[0]) - int(other[0])) <= 1 for other in neighbours):
            candidates.append(int(point[0]))
    if not candidates:
        candidates = [int(point[0]) for point in points if int(point[1]) <= threshold_y]
    if not candidates:
        return None
    center = float(statistics.median(candidates))
    inliers = [value for value in candidates if abs(value - center) <= 2]
    return int(round(statistics.median(inliers or candidates)))


def _read_bgr(path: str | Path):
    """Read through NumPy so Windows paths containing Chinese remain valid."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


@dataclass(frozen=True)
class SemanticRouteSet:
    documents: tuple[dict[str, Any], ...]
    source: str  # "semantic" or "legacy"


def _point(value: Any, path: str, width: int, height: int) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
        or not isinstance(value[0], int)
        or not isinstance(value[1], int)
    ):
        raise SemanticRouteError(f"{path} 必须是两个整数坐标。")
    x, y = int(value[0]), int(value[1])
    if not (0 <= x < width and 0 <= y < height):
        raise SemanticRouteError(
            f"{path}={value!r} 超出地图范围 0..{width - 1}, 0..{height - 1}。"
        )
    return x, y


def _direction(value: Any, path: str, allowed: set[str]) -> str:
    if value not in allowed:
        choices = "/".join(sorted(allowed))
        raise SemanticRouteError(f"{path} 必须是 {choices}。")
    return str(value)


def validate_route_document(
    document: dict[str, Any],
    *,
    expected_map_id: str | None = None,
    expected_canvas_size: tuple[int, int] | None = None,
    expected_route_index: int | None = None,
    allow_draft: bool = False,
) -> dict[str, Any]:
    """Validate and return a defensive deep copy of one route document."""
    if not isinstance(document, dict):
        raise SemanticRouteError("路线 JSON 根节点必须是对象。")
    doc = copy.deepcopy(document)
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise SemanticRouteError(
            f"不支持的 schema_version：{doc.get('schema_version')!r}。"
        )
    map_id = doc.get("map_id")
    if not isinstance(map_id, str) or not map_id.strip():
        raise SemanticRouteError("map_id 必须是非空字符串。")
    if expected_map_id is not None and map_id != expected_map_id:
        raise SemanticRouteError(
            f"map_id 不匹配：期望 {expected_map_id!r}，实际 {map_id!r}。"
        )
    route_index = doc.get("route_index")
    if isinstance(route_index, bool) or not isinstance(route_index, int) or route_index < 1:
        raise SemanticRouteError("route_index 必须是从 1 开始的整数。")
    if expected_route_index is not None and route_index != expected_route_index:
        raise SemanticRouteError(
            f"route_index 不匹配：期望 {expected_route_index}，实际 {route_index}。"
        )
    canvas = doc.get("canvas_size")
    if (
        not isinstance(canvas, (list, tuple))
        or len(canvas) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in canvas)
        or int(canvas[0]) <= 0
        or int(canvas[1]) <= 0
    ):
        raise SemanticRouteError("canvas_size 必须是 [width, height] 正整数。")
    width, height = int(canvas[0]), int(canvas[1])
    if expected_canvas_size is not None and (width, height) != tuple(expected_canvas_size):
        raise SemanticRouteError(
            f"canvas_size 不匹配：期望 {tuple(expected_canvas_size)}，"
            f"实际 {(width, height)}。"
        )
    if not isinstance(doc.get("loop"), bool):
        raise SemanticRouteError("loop 必须是布尔值。")
    if doc.get("draft", False) and not allow_draft:
        raise SemanticRouteError("草稿路线必须在 Route Studio 校正后才能执行。")

    segments = doc.get("segments")
    if not isinstance(segments, list) or not segments:
        raise SemanticRouteError("segments 必须是非空列表。")
    identifiers: set[str] = set()
    goal_count = 0
    for index, segment in enumerate(segments):
        path = f"segments[{index}]"
        if not isinstance(segment, dict):
            raise SemanticRouteError(f"{path} 必须是对象。")
        identifier = segment.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise SemanticRouteError(f"{path}.id 必须是非空字符串。")
        if identifier in identifiers:
            raise SemanticRouteError(f"线段 id 重复：{identifier!r}。")
        identifiers.add(identifier)
        kind = segment.get("type")
        if kind not in SEGMENT_TYPES:
            raise SemanticRouteError(f"{path}.type 不受支持：{kind!r}。")

        if kind == "walk":
            _direction(segment.get("direction"), f"{path}.direction", {"left", "right"})
            points = segment.get("points")
            if not isinstance(points, list) or len(points) < 2:
                raise SemanticRouteError(f"{path}.points 至少需要两个坐标。")
            for point_index, value in enumerate(points):
                _point(value, f"{path}.points[{point_index}]", width, height)
        elif kind == "jump":
            _point(segment.get("anchor"), f"{path}.anchor", width, height)
            _direction(
                segment.get("direction", "none"),
                f"{path}.direction",
                {"left", "right", "none"},
            )
            if "landing" in segment:
                _point(segment["landing"], f"{path}.landing", width, height)
        elif kind == "drop":
            _point(segment.get("anchor"), f"{path}.anchor", width, height)
            _point(segment.get("landing"), f"{path}.landing", width, height)
        elif kind == "ladder":
            _point(segment.get("approach"), f"{path}.approach", width, height)
            _point(segment.get("mount"), f"{path}.mount", width, height)
            points = segment.get("points")
            if not isinstance(points, list) or len(points) < 2:
                raise SemanticRouteError(f"{path}.points 至少需要两个攀爬坐标。")
            for point_index, value in enumerate(points):
                _point(value, f"{path}.points[{point_index}]", width, height)
            _point(segment.get("exit"), f"{path}.exit", width, height)
            _direction(
                segment.get("mount_direction", "none"),
                f"{path}.mount_direction",
                {"left", "right", "none"},
            )
            _direction(
                segment.get("exit_direction", "none"),
                f"{path}.exit_direction",
                {"left", "right", "none"},
            )
        elif kind == "teleport":
            _point(segment.get("anchor"), f"{path}.anchor", width, height)
            _direction(
                segment.get("direction"),
                f"{path}.direction",
                {"left", "right", "up", "down"},
            )
            if "landing" in segment:
                _point(segment["landing"], f"{path}.landing", width, height)
        elif kind == "stop":
            _point(segment.get("position"), f"{path}.position", width, height)
        elif kind == "goal":
            goal_count += 1
            _point(segment.get("position"), f"{path}.position", width, height)
            radius = segment.get("radius", 6)
            if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
                raise SemanticRouteError(f"{path}.radius 必须是正整数。")
    if goal_count != 1:
        raise SemanticRouteError(f"每条路线必须且只能包含一个 Goal，当前为 {goal_count}。")
    if segments[-1].get("type") != "goal":
        raise SemanticRouteError("Goal 必须是路线的最后一个线段。")
    return doc


def load_route_document(
    path: str | Path,
    *,
    expected_map_id: str | None = None,
    expected_canvas_size: tuple[int, int] | None = None,
    expected_route_index: int | None = None,
    allow_draft: bool = False,
) -> dict[str, Any]:
    route_path = Path(path)
    try:
        document = json.loads(route_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticRouteError(f"无法读取路线 JSON {route_path}：{exc}") from exc
    return validate_route_document(
        document,
        expected_map_id=expected_map_id,
        expected_canvas_size=expected_canvas_size,
        expected_route_index=expected_route_index,
        allow_draft=allow_draft,
    )


def discover_route_indices(directory: str | Path, suffix: str) -> list[int]:
    pattern = ROUTE_JSON_RE if suffix == ".json" else ROUTE_PNG_RE
    indices = []
    for path in Path(directory).iterdir():
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def load_semantic_route_set(
    directory: str | Path,
    *,
    map_id: str,
    canvas_size: tuple[int, int],
) -> SemanticRouteSet:
    """Choose semantic routes when present, otherwise declare legacy mode."""
    route_dir = Path(directory)
    json_indices = discover_route_indices(route_dir, ".json")
    if not json_indices:
        return SemanticRouteSet((), "legacy")
    expected = list(range(1, json_indices[-1] + 1))
    if json_indices != expected:
        raise SemanticRouteError(
            f"语义路线编号必须连续：期望 {expected}，实际 {json_indices}。"
        )
    png_indices = discover_route_indices(route_dir, ".png")
    if png_indices != json_indices:
        raise SemanticRouteError(
            "语义模式要求 routeN.json 与 routeN.png 编号完全一致："
            f"JSON={json_indices}，PNG={png_indices}。"
        )
    missing_png = [index for index in json_indices if index not in png_indices]
    if missing_png:
        raise SemanticRouteError(
            f"语义路线缺少配对 PNG：{missing_png}。"
        )
    expected_width, expected_height = canvas_size
    for index in png_indices:
        image = _read_bgr(route_dir / f"route{index}.png")
        if image is None:
            raise SemanticRouteError(f"无法读取配对 route{index}.png。")
        if image.shape[:2] != (expected_height, expected_width):
            raise SemanticRouteError(
                f"route{index}.png 尺寸不匹配：期望 "
                f"{expected_width}×{expected_height}，实际 "
                f"{image.shape[1]}×{image.shape[0]}。"
            )
    documents = tuple(
        load_route_document(
            route_dir / f"route{index}.json",
            expected_map_id=map_id,
            expected_canvas_size=canvas_size,
            expected_route_index=index,
        )
        for index in json_indices
    )
    return SemanticRouteSet(documents, "semantic")


def _rgb_for_command(command: str, command_colors: dict[str, tuple[int, int, int]]):
    color = command_colors.get(command)
    if color is None:
        raise SemanticRouteError(f"配置中缺少动作颜色：{command!r}。")
    return tuple(int(value) for value in color)


def render_route_png(
    base_bgr: np.ndarray,
    document: dict[str, Any],
    command_colors: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    """Render the human-facing legacy-compatible PNG from semantic data."""
    if base_bgr is None or base_bgr.ndim != 3 or base_bgr.shape[2] != 3:
        raise SemanticRouteError("map.png 必须是三通道 BGR 图像。")
    height, width = base_bgr.shape[:2]
    doc = validate_route_document(
        document,
        expected_canvas_size=(width, height),
        allow_draft=True,
    )
    canvas = np.ascontiguousarray(base_bgr.copy())

    def bgr(command: str):
        rgb = _rgb_for_command(command, command_colors)
        return rgb[2], rgb[1], rgb[0]

    for segment in doc["segments"]:
        kind = segment["type"]
        if kind == "walk":
            command = f"{segment['direction']} none none"
            points = np.asarray(segment["points"], dtype=np.int32)
            cv2.polylines(canvas, [points], False, bgr(command), 1, cv2.LINE_8)
        elif kind == "ladder":
            points = np.asarray(segment["points"], dtype=np.int32)
            cv2.polylines(canvas, [points], False, bgr("none up none"), 1, cv2.LINE_8)
            mount_direction = segment.get("mount_direction", "none")
            mount_command = (
                "none none jump"
                if mount_direction == "none"
                else f"{mount_direction} none jump"
            )
            cv2.circle(canvas, tuple(segment["mount"]), 2, bgr(mount_command), -1)
            exit_direction = segment.get("exit_direction", "none")
            if exit_direction in {"left", "right"}:
                cv2.line(
                    canvas,
                    tuple(segment["points"][-1]),
                    tuple(segment["exit"]),
                    bgr(f"{exit_direction} none none"),
                    1,
                    cv2.LINE_8,
                )
        elif kind == "jump":
            direction = segment.get("direction", "none")
            command = (
                "none none jump" if direction == "none"
                else f"{direction} none jump"
            )
            cv2.circle(canvas, tuple(segment["anchor"]), 2, bgr(command), -1)
        elif kind == "drop":
            cv2.circle(canvas, tuple(segment["anchor"]), 2, bgr("none down jump"), -1)
        elif kind == "teleport":
            direction = segment["direction"]
            command = f"none {direction} teleport" if direction in {"up", "down"} else (
                f"{direction} none teleport"
            )
            cv2.circle(canvas, tuple(segment["anchor"]), 2, bgr(command), -1)
        elif kind == "stop":
            cv2.circle(canvas, tuple(segment["position"]), 2, bgr("stop stop stop"), -1)
        elif kind == "goal":
            cv2.circle(canvas, tuple(segment["position"]), 2, bgr("none none goal"), -1)
    return canvas


def save_route_pair_atomic(
    directory: str | Path,
    document: dict[str, Any],
    base_bgr: np.ndarray,
    command_colors: dict[str, tuple[int, int, int]],
) -> tuple[Path, Path]:
    """Validate, stage and replace a JSON/PNG pair with rollback on failure."""
    route_dir = Path(directory)
    route_dir.mkdir(parents=True, exist_ok=True)
    height, width = base_bgr.shape[:2]
    doc = validate_route_document(
        document,
        expected_canvas_size=(width, height),
    )
    index = int(doc["route_index"])
    json_path = route_dir / f"route{index}.json"
    png_path = route_dir / f"route{index}.png"
    rendered = render_route_png(base_bgr, doc, command_colors)
    success, encoded = cv2.imencode(".png", rendered)
    if not success:
        raise SemanticRouteError("OpenCV 无法编码路线 PNG。")
    json_bytes = (
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")
    staged: list[Path] = []
    backups: dict[Path, bytes | None] = {}
    try:
        for target, payload in ((json_path, json_bytes), (png_path, encoded.tobytes())):
            backups[target] = target.read_bytes() if target.exists() else None
            handle, temporary = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=route_dir
            )
            os.close(handle)
            temp_path = Path(temporary)
            temp_path.write_bytes(payload)
            staged.append(temp_path)
        os.replace(staged[0], json_path)
        staged.pop(0)
        os.replace(staged[0], png_path)
        staged.pop(0)
    except Exception:
        for target, previous in backups.items():
            try:
                if previous is None:
                    if target.exists():
                        target.unlink()
                else:
                    target.write_bytes(previous)
            except OSError:
                pass
        raise
    finally:
        for path in staged:
            try:
                path.unlink()
            except OSError:
                pass
    return json_path, png_path


def convert_legacy_png_to_draft(
    route_png: str | Path,
    *,
    map_id: str,
    route_index: int,
    command_colors: dict[str, tuple[int, int, int]],
) -> Path:
    """Create a non-executable draft without changing the legacy route PNG."""
    path = Path(route_png)
    image = _read_bgr(path)
    if image is None:
        raise SemanticRouteError(f"无法读取旧路线图：{path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    pieces: list[tuple[tuple[int, int], dict[str, Any]]] = []
    serial = 1
    goal_present = False
    for command, color in command_colors.items():
        mask = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2).astype(np.uint8)
        count, labels = cv2.connectedComponents(mask, connectivity=8)
        for component in range(1, count):
            coords = np.argwhere(labels == component)
            if coords.size == 0:
                continue
            points = [(int(x), int(y)) for y, x in coords]
            center = (
                int(round(sum(x for x, _ in points) / len(points))),
                int(round(sum(y for _, y in points) / len(points))),
            )
            move_x, move_y, action = command.split()
            identifier = f"draft_{serial:03d}"
            serial += 1
            if action == "goal":
                segment = {
                    "id": identifier, "type": "goal",
                    "position": list(center), "radius": 6,
                }
                goal_present = True
            elif action == "jump" and move_y == "down":
                segment = {
                    "id": identifier, "type": "drop",
                    "anchor": list(center), "landing": list(center),
                }
            elif action == "jump":
                segment = {
                    "id": identifier, "type": "jump",
                    "anchor": list(center), "direction": move_x,
                }
            elif action == "teleport":
                direction = move_x if move_x in {"left", "right"} else move_y
                segment = {
                    "id": identifier, "type": "teleport",
                    "anchor": list(center), "direction": direction,
                }
            elif move_y == "up":
                ordered = sorted(points, key=lambda point: (-point[1], point[0]))
                segment = {
                    "id": identifier, "type": "ladder",
                    "approach": list(ordered[-1]),
                    "mount": list(ordered[-1]),
                    "points": [list(point) for point in ordered],
                    "exit": list(ordered[0]),
                    "exit_direction": "none",
                }
            elif move_x in {"left", "right"}:
                ordered = sorted(points, key=lambda point: point[0], reverse=move_x == "left")
                if len(ordered) < 2:
                    ordered = [ordered[0], ordered[0]]
                segment = {
                    "id": identifier, "type": "walk",
                    "direction": move_x, "points": [list(point) for point in ordered],
                }
            elif move_x == "stop" or move_y == "stop":
                segment = {"id": identifier, "type": "stop", "position": list(center)}
            else:
                continue
            pieces.append((center, segment))
    pieces.sort(key=lambda item: (item[0][1], item[0][0]))
    segments = [segment for _center, segment in pieces if segment["type"] != "goal"]
    segments.extend(segment for _center, segment in pieces if segment["type"] == "goal")
    if not goal_present:
        fallback = segments[-1] if segments else None
        if fallback is None:
            raise SemanticRouteError("旧 PNG 中没有可转换的路线颜色。")
        if fallback["type"] == "walk":
            position = fallback["points"][-1]
        else:
            position = fallback.get("position") or fallback.get("anchor") or fallback.get("exit")
        segments.append({
            "id": f"draft_{serial:03d}",
            "type": "goal",
            "position": list(position),
            "radius": 6,
        })
    document = {
        "schema_version": SCHEMA_VERSION,
        "map_id": map_id,
        "route_index": int(route_index),
        "canvas_size": [width, height],
        "loop": True,
        "draft": True,
        "warnings": [
            "由旧 PNG 自动推断；线段顺序、梯子、落点和交叉处必须人工校正。"
        ],
        "segments": segments,
    }
    validate_route_document(document, allow_draft=True)
    draft_path = path.with_name(f"route{route_index}.draft.json")
    if draft_path.exists():
        raise FileExistsError(f"不会覆盖现有草稿：{draft_path}")
    draft_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return draft_path
