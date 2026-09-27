"""Strict per-map settings shared by Route Studio and the runtime."""

from __future__ import annotations

import math
import os
import tempfile
from numbers import Real
from pathlib import Path

import yaml


MAP_CONFIG_SCHEMA_VERSION = 1
MAP_CONFIG_FILENAME = "map_config.yaml"


class MapProjectConfigError(ValueError):
    """Raised when a map-level configuration cannot be trusted."""


def validate_normalized_roi(value) -> tuple[float, float, float, float]:
    """Validate ``[x, y, width, height]`` in prepared-frame coordinates."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise MapProjectConfigError("小地图 ROI 必须包含 x、y、宽度和高度四个数值。")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise MapProjectConfigError("小地图 ROI 的四个值必须是有限数字。")
    roi = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in roi):
        raise MapProjectConfigError("小地图 ROI 的四个值必须是有限数字。")
    x, y, width, height = roi
    epsilon = 1e-9
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise MapProjectConfigError("小地图 ROI 起点不能为负，宽度和高度必须大于 0。")
    if x > 1 or y > 1 or x + width > 1 + epsilon or y + height > 1 + epsilon:
        raise MapProjectConfigError("小地图 ROI 必须完全位于归一化游戏画面 0–1 范围内。")
    return roi


def normalized_roi_to_pixels(roi, frame_width: int, frame_height: int):
    """Convert a normalized ROI with the same rounding used at runtime."""
    x, y, width, height = validate_normalized_roi(roi)
    frame_width = int(frame_width)
    frame_height = int(frame_height)
    if frame_width <= 0 or frame_height <= 0:
        raise MapProjectConfigError("游戏画面尺寸必须大于 0。")
    left = max(0, min(round(x * frame_width), frame_width))
    top = max(0, min(round(y * frame_height), frame_height))
    right = max(left, min(round((x + width) * frame_width), frame_width))
    bottom = max(top, min(round((y + height) * frame_height), frame_height))
    if right <= left or bottom <= top:
        raise MapProjectConfigError("小地图 ROI 换算后的像素区域为空。")
    return left, top, right - left, bottom - top


def pixels_to_normalized_roi(rect, frame_width: int, frame_height: int):
    """Convert an integer pixel rectangle into a stable normalized ROI."""
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        raise MapProjectConfigError("像素 ROI 必须包含 x、y、宽度和高度。")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in rect):
        raise MapProjectConfigError("像素 ROI 必须是数值。")
    frame_width = int(frame_width)
    frame_height = int(frame_height)
    if frame_width <= 0 or frame_height <= 0:
        raise MapProjectConfigError("游戏画面尺寸必须大于 0。")
    x, y, width, height = (int(round(float(item))) for item in rect)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise MapProjectConfigError("像素 ROI 起点不能为负，宽度和高度必须大于 0。")
    if x + width > frame_width or y + height > frame_height:
        raise MapProjectConfigError("像素 ROI 超出游戏画面边界。")
    return validate_normalized_roi((
        round(x / frame_width, 10),
        round(y / frame_height, 10),
        round(width / frame_width, 10),
        round(height / frame_height, 10),
    ))


def _config_path(directory) -> Path:
    return Path(directory) / MAP_CONFIG_FILENAME


def load_map_minimap_roi(directory):
    """Return a map override, or ``None`` when the map inherits its profile."""
    path = _config_path(directory)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise MapProjectConfigError(f"无法读取地图配置 {path}：{exc}") from exc
    if not isinstance(document, dict):
        raise MapProjectConfigError(f"地图配置根节点必须是对象：{path}")
    if document.get("schema_version") != MAP_CONFIG_SCHEMA_VERSION:
        raise MapProjectConfigError(
            f"地图配置 schema_version 必须为 {MAP_CONFIG_SCHEMA_VERSION}：{path}"
        )
    minimap = document.get("minimap", {})
    if not isinstance(minimap, dict):
        raise MapProjectConfigError(f"地图配置 minimap 必须是对象：{path}")
    if "roi" not in minimap:
        return None
    try:
        return validate_normalized_roi(minimap["roi"])
    except MapProjectConfigError as exc:
        raise MapProjectConfigError(f"地图配置 {path} 无效：{exc}") from exc


def apply_map_minimap_roi_override(cfg, map_id, project_root="minimaps"):
    """Apply one map override to a runtime config and return its source label."""
    map_id = str(map_id or "").strip()
    roi = load_map_minimap_roi(Path(project_root) / map_id) if map_id else None
    if roi is not None:
        cfg.setdefault("minimap", {})["roi"] = list(roi)
        return f"map:{map_id}"
    if cfg.get("minimap", {}).get("roi"):
        return "global"
    return "automatic"


def save_map_minimap_roi_atomic(directory, roi):
    """Atomically update only the map-level minimap ROI."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = _config_path(directory)
    document = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise MapProjectConfigError(f"无法读取地图配置 {path}：{exc}") from exc
        if not isinstance(document, dict):
            raise MapProjectConfigError(f"地图配置根节点必须是对象：{path}")
    document["schema_version"] = MAP_CONFIG_SCHEMA_VERSION
    minimap = document.setdefault("minimap", {})
    if not isinstance(minimap, dict):
        raise MapProjectConfigError(f"地图配置 minimap 必须是对象：{path}")
    if roi is None:
        minimap.pop("roi", None)
    else:
        minimap["roi"] = list(validate_normalized_roi(roi))

    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=directory
    )
    os.close(handle)
    try:
        with open(temporary, "w", encoding="utf-8") as output:
            yaml.safe_dump(
                document, output, allow_unicode=True, sort_keys=False,
                default_flow_style=False,
            )
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink()
        except OSError:
            pass
    return path
