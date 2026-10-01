"""Strict, writable name-tag profile storage shared by UI and runtime."""

from __future__ import annotations

import copy
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.app_paths import writable_path


DEFAULT_PROFILE_SETTINGS = {
    "max_score": 0.30,
    "local_search_radius": 140,
    "global_refresh_frames": 30,
    "max_jump": 250,
    "jump_confirm_frames": 2,
    "jump_confirm_radius": 25,
    "max_missed_frames": 5,
    "edge_weight": 0.65,
}

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_SAMPLE_PATTERN = re.compile(r"^sample_(\d+)\.png$", re.IGNORECASE)


class NameTagProfileError(RuntimeError):
    """A user-facing profile validation or persistence error."""


@dataclass
class StagedNameTagProfile:
    name: str
    data: dict
    source_dir: Path | None = None
    new_images: dict[str, np.ndarray] = field(default_factory=dict)


def validate_profile_name(name: str) -> str:
    raw = str(name)
    normalized = raw.strip()
    if not normalized:
        raise NameTagProfileError("人物名字配置名称不能为空。")
    if len(normalized) > 64:
        raise NameTagProfileError("人物名字配置名称不能超过 64 个字符。")
    if normalized in {".", ".."}:
        raise NameTagProfileError("人物名字配置名称无效。")
    if raw.endswith((" ", ".")) or normalized.endswith("."):
        raise NameTagProfileError("人物名字配置名称不能以空格或句点结尾。")
    if any(character in normalized for character in '<>:"/\\|?*'):
        raise NameTagProfileError("人物名字配置名称不能包含路径或文件名非法字符。")
    if any(ord(character) < 32 for character in normalized):
        raise NameTagProfileError("人物名字配置名称不能包含控制字符。")
    stem = normalized.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED:
        raise NameTagProfileError("人物名字配置名称不能使用 Windows 保留名称。")
    return normalized


def load_bgr_image(path: Path) -> np.ndarray:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise NameTagProfileError(f"无法读取名字样本：{path.name}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None or image.ndim != 3 or image.size == 0:
        raise NameTagProfileError(f"名字样本图片损坏或格式无效：{path.name}")
    return image


class NameTagProfileRepository:
    def __init__(self, *, bundled_root=None, writable_root=None):
        self.bundled_root = Path(bundled_root or "nametag")
        self.writable_root = Path(writable_root or writable_path("nametag"))

    @staticmethod
    def _same_path(left: Path, right: Path) -> bool:
        try:
            return left.resolve() == right.resolve()
        except OSError:
            return os.path.abspath(left) == os.path.abspath(right)

    def list_profiles(self) -> list[str]:
        names = set()
        for root in (self.bundled_root, self.writable_root):
            if not root.is_dir():
                continue
            for path in root.iterdir():
                if path.is_dir() and (path / "profile.yaml").is_file():
                    try:
                        names.add(validate_profile_name(path.name))
                    except NameTagProfileError:
                        continue
        return sorted(names, key=lambda value: value.casefold())

    def resolve_profile(self, name: str) -> Path:
        name = validate_profile_name(name)
        writable = self.writable_root / name
        if (writable / "profile.yaml").is_file():
            return writable
        bundled = self.bundled_root / name
        if (bundled / "profile.yaml").is_file():
            return bundled
        raise NameTagProfileError(f"找不到人物名字配置：{name}")

    @staticmethod
    def _validate_image_content(image: np.ndarray, filename: str):
        if image is None or image.ndim != 3 or image.size == 0:
            raise NameTagProfileError(f"名字样本图片无效：{filename}")
        if image.shape[0] < 5 or image.shape[1] < 5:
            raise NameTagProfileError(f"名字样本至少需要 5×5 px：{filename}")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        text = cv2.inRange(
            hsv, np.array([0, 0, 120]), np.array([179, 90, 255])
        )
        if int(np.count_nonzero(text)) < 4:
            raise NameTagProfileError(f"名字样本缺少足够的浅色文字像素：{filename}")

    def _validate_profile_data(
        self,
        data,
        *,
        profile_dir: Path | None,
        new_images: dict[str, np.ndarray] | None = None,
        require_enabled=False,
    ) -> dict:
        if not isinstance(data, dict):
            raise NameTagProfileError("profile.yaml 顶层必须是映射。")
        if data.get("schema_version", 1) != 1:
            raise NameTagProfileError("不支持的人物名字配置版本。")
        settings = data.get("settings", {})
        samples = data.get("samples", [])
        if not isinstance(settings, dict):
            raise NameTagProfileError("人物名字配置 settings 必须是映射。")
        if not isinstance(samples, list):
            raise NameTagProfileError("人物名字配置 samples 必须是列表。")

        normalized = {
            "schema_version": 1,
            "settings": copy.deepcopy(DEFAULT_PROFILE_SETTINGS),
            "samples": [],
        }
        normalized["settings"].update(copy.deepcopy(settings))
        unknown_settings = set(normalized["settings"]) - set(DEFAULT_PROFILE_SETTINGS)
        if unknown_settings:
            raise NameTagProfileError(
                "人物名字配置包含未知设置：" + ", ".join(sorted(unknown_settings))
            )
        numeric_rules = {
            "max_score": (float, 0.0, 1.0),
            "local_search_radius": (int, 0, None),
            "global_refresh_frames": (int, 1, None),
            "max_jump": (int, 0, None),
            "jump_confirm_frames": (int, 1, None),
            "jump_confirm_radius": (int, 0, None),
            "max_missed_frames": (int, 0, None),
            "edge_weight": (float, 0.0, 1.0),
        }
        for key, (kind, minimum, maximum) in numeric_rules.items():
            value = normalized["settings"][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise NameTagProfileError(f"人物名字配置 settings.{key} 必须是数字。")
            if kind is int and not isinstance(value, int):
                raise NameTagProfileError(f"人物名字配置 settings.{key} 必须是整数。")
            value = kind(value)
            if not np.isfinite(value) or value < minimum or (
                maximum is not None and value > maximum
            ):
                raise NameTagProfileError(f"人物名字配置 settings.{key} 超出允许范围。")
            normalized["settings"][key] = value
        seen = set()
        enabled_count = 0
        pending = new_images or {}
        for index, entry in enumerate(samples, start=1):
            if not isinstance(entry, dict):
                raise NameTagProfileError(f"第 {index} 个名字样本配置无效。")
            filename = str(entry.get("file", ""))
            if (not _SAMPLE_PATTERN.fullmatch(filename)
                    or Path(filename).name != filename):
                raise NameTagProfileError(f"第 {index} 个名字样本文件名无效。")
            if filename.casefold() in seen:
                raise NameTagProfileError(f"名字样本文件重复：{filename}")
            seen.add(filename.casefold())
            offset = entry.get("player_offset")
            if (not isinstance(offset, (list, tuple)) or len(offset) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           for value in offset)):
                raise NameTagProfileError(f"名字样本脚点偏移无效：{filename}")
            enabled = entry.get("enabled", True)
            if not isinstance(enabled, bool):
                raise NameTagProfileError(f"名字样本启用状态无效：{filename}")
            image = pending.get(filename)
            if image is None:
                if profile_dir is None or not (profile_dir / filename).is_file():
                    raise NameTagProfileError(f"缺少名字样本图片：{filename}")
                image = load_bgr_image(profile_dir / filename)
            self._validate_image_content(image, filename)
            normalized["samples"].append({
                "file": filename,
                "player_offset": [int(offset[0]), int(offset[1])],
                "enabled": enabled,
            })
            enabled_count += int(enabled)
        if require_enabled and enabled_count == 0:
            raise NameTagProfileError("人物名字配置至少需要一个已启用样本。")
        return normalized

    def load_profile(self, name: str, *, require_enabled=False) -> dict:
        profile_dir = self.resolve_profile(name)
        try:
            with (profile_dir / "profile.yaml").open("r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as exc:
            raise NameTagProfileError(f"无法加载人物名字配置：{name}") from exc
        return self._validate_profile_data(
            data, profile_dir=profile_dir, require_enabled=require_enabled
        )

    def load_sample_image(self, name: str, filename: str) -> np.ndarray:
        profile_dir = self.resolve_profile(name)
        if Path(filename).name != filename:
            raise NameTagProfileError("名字样本文件名无效。")
        return load_bgr_image(profile_dir / filename)

    def create_stage(self, name: str, *, existing=False) -> StagedNameTagProfile:
        name = validate_profile_name(name)
        if existing:
            source = self.resolve_profile(name)
            data = self.load_profile(name, require_enabled=False)
        else:
            try:
                self.resolve_profile(name)
            except NameTagProfileError:
                pass
            else:
                raise NameTagProfileError(f"人物名字配置已存在：{name}")
            source = None
            data = {
                "schema_version": 1,
                "settings": copy.deepcopy(DEFAULT_PROFILE_SETTINGS),
                "samples": [],
            }
        return StagedNameTagProfile(name=name, data=data, source_dir=source)

    @staticmethod
    def _next_sample_name(stage: StagedNameTagProfile) -> str:
        highest = 0
        for entry in stage.data.get("samples", []):
            match = _SAMPLE_PATTERN.fullmatch(str(entry.get("file", "")))
            if match:
                highest = max(highest, int(match.group(1)))
        return f"sample_{highest + 1:03d}.png"

    def stage_sample(self, stage, image, player_offset, *, enabled=True) -> str:
        filename = self._next_sample_name(stage)
        image = np.ascontiguousarray(image.copy())
        self._validate_image_content(image, filename)
        offset = [int(player_offset[0]), int(player_offset[1])]
        stage.data.setdefault("samples", []).append({
            "file": filename,
            "player_offset": offset,
            "enabled": bool(enabled),
        })
        stage.new_images[filename] = image
        return filename

    @staticmethod
    def _sample_entry(stage, filename):
        for entry in stage.data.get("samples", []):
            if entry.get("file") == filename:
                return entry
        raise NameTagProfileError(f"找不到名字样本：{filename}")

    def stage_sample_enabled(self, stage, filename, enabled):
        self._sample_entry(stage, filename)["enabled"] = bool(enabled)

    def stage_player_offset(self, stage, filename, player_offset):
        entry = self._sample_entry(stage, filename)
        entry["player_offset"] = [int(player_offset[0]), int(player_offset[1])]

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def commit_profile(self, stage: StagedNameTagProfile, *, require_enabled=False) -> Path:
        stage.name = validate_profile_name(stage.name)
        normalized = self._validate_profile_data(
            stage.data,
            profile_dir=stage.source_dir,
            new_images=stage.new_images,
            require_enabled=require_enabled,
        )
        target = self.writable_root / stage.name
        target.mkdir(parents=True, exist_ok=True)

        source = stage.source_dir
        for entry in normalized["samples"]:
            filename = entry["file"]
            destination = target / filename
            if filename in stage.new_images:
                ok, encoded = cv2.imencode(".png", stage.new_images[filename])
                if not ok:
                    raise NameTagProfileError(f"无法编码名字样本：{filename}")
                self._atomic_write_bytes(destination, encoded.tobytes())
            elif not destination.is_file():
                if source is None or not (source / filename).is_file():
                    raise NameTagProfileError(f"缺少名字样本图片：{filename}")
                temporary = destination.with_name(
                    f".{destination.name}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    shutil.copy2(source / filename, temporary)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)

        payload = yaml.safe_dump(
            normalized, allow_unicode=True, sort_keys=False
        ).encode("utf-8")
        self._atomic_write_bytes(target / "profile.yaml", payload)
        stage.data = copy.deepcopy(normalized)
        stage.source_dir = target
        stage.new_images.clear()
        return target
