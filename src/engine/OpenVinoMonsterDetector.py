from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_WIDTH = 1280
MODEL_HEIGHT = 224
MODEL_CLASS_NAMES = {0: "monster", 1: "player"}
ACTIVE_VARIANT = "int8_v2"
ACTIVE_MODEL_PATH = "int8_v2/models/mixed_head_fp/model.xml"


class OpenVinoDeploymentError(RuntimeError):
    """Raised when the immutable OpenVINO deployment contract is invalid."""


@dataclass(frozen=True)
class MonsterDetectionTiming:
    crop_ms: float = 0.0
    infer_ms: float = 0.0
    convert_ms: float = 0.0
    total_ms: float = 0.0
    detections: int = 0
    monsters: int = 0
    players: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_runtime_class(deployment_root: Path):
    """Load the maintained runtime package without changing global sys.path."""
    package_dir = deployment_root / "yolo26_openvino"
    init_path = package_dir / "__init__.py"
    if not init_path.is_file():
        raise OpenVinoDeploymentError(f"找不到 OpenVINO 运行时包：{init_path}")

    package_id = hashlib.sha1(
        str(deployment_root).encode("utf-8")
    ).hexdigest()[:12]
    module_name = f"_maple_yolo26_openvino_{package_id}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_path,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise OpenVinoDeploymentError(f"无法加载 OpenVINO 运行时包：{init_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            sys.modules.pop(module_name, None)
            if exc.name == "openvino":
                raise OpenVinoDeploymentError(
                    "当前 Python 环境缺少 openvino==2026.3.0，已阻止启动。"
                ) from exc
            raise OpenVinoDeploymentError(
                f"加载 OpenVINO 运行时失败：缺少模块 {exc.name}"
            ) from exc
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise OpenVinoDeploymentError(
                f"加载 OpenVINO 运行时失败：{exc}"
            ) from exc
    return module.OpenVINOYolo26


class OpenVinoMonsterDetector:
    """Adapter from the fixed monster/player YOLO deployment to bot boxes."""

    def __init__(
        self,
        deployment_root,
        *,
        confidence=0.25,
        max_det=50,
        variant="auto",
        runtime_factory=None,
    ):
        self.root = Path(deployment_root).expanduser().resolve()
        self.confidence = float(confidence)
        self.max_det = int(max_det)
        requested_variant = str(variant).strip().lower()
        if requested_variant not in {"auto", "fp16", "int8", "int8_v2"}:
            raise OpenVinoDeploymentError(
                "YOLO 模型版本必须是 auto、fp16、int8 或 int8_v2。"
            )
        if not 0.05 <= self.confidence <= 0.95:
            raise OpenVinoDeploymentError("YOLO 置信度必须在 0.05–0.95 之间。")
        if self.max_det != 50:
            raise OpenVinoDeploymentError("当前双类部署要求 yolo_max_det 固定为 50。")

        self.manifest = self._validate_deployment(requested_variant)
        factory = runtime_factory or _load_runtime_class(self.root)
        try:
            self.runtime = factory(
                self.root,
                confidence=self.confidence,
                max_det=self.max_det,
                variant=self.variant,
            )
        except OpenVinoDeploymentError:
            raise
        except Exception as exc:
            raise OpenVinoDeploymentError(
                f"初始化双类 YOLO OpenVINO 模型失败：{exc}"
            ) from exc
        self.last_timing = MonsterDetectionTiming()
        self.last_visible_rect = (0, 0, 0, 0)
        self.last_players = []
        self.last_player_foot = None

    def _validate_deployment(self, requested_variant="auto"):
        manifest_path = self.root / "manifest.json"
        active_path = self.root / "ACTIVE_MODEL.txt"
        if not manifest_path.is_file():
            raise OpenVinoDeploymentError(f"找不到部署清单：{manifest_path}")
        if not active_path.is_file():
            raise OpenVinoDeploymentError(f"找不到活动模型声明：{active_path}")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OpenVinoDeploymentError(f"部署清单无法读取：{exc}") from exc

        active_lines = [
            line.strip()
            for line in active_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if active_lines != [ACTIVE_VARIANT, ACTIVE_MODEL_PATH]:
            raise OpenVinoDeploymentError(
                "ACTIVE_MODEL.txt 必须指向已验收的双类 INT8 v2 模型。"
            )

        classes = {
            int(class_id): str(name)
            for class_id, name in manifest.get("class_names", {}).items()
        }
        if classes != MODEL_CLASS_NAMES:
            raise OpenVinoDeploymentError(
                "模型类别必须严格为 {0: 'monster', 1: 'player'}，"
                f"实际为：{classes}"
            )

        selection = manifest.get("selection", {})
        if (selection.get("active_variant") != ACTIVE_VARIANT or
                selection.get("model_path") != ACTIVE_MODEL_PATH):
            raise OpenVinoDeploymentError(
                "manifest 未选择已验收的双类 INT8 v2 模型。"
            )

        external = manifest.get("preprocessing", {}).get("external_input", {})
        expected_input = {
            "shape": [1, MODEL_HEIGHT, MODEL_WIDTH, 3],
            "dtype": "uint8",
            "layout": "NHWC",
            "color": "BGR",
        }
        if any(external.get(key) != value for key, value in expected_input.items()):
            raise OpenVinoDeploymentError("部署输入契约不是 uint8 BGR NHWC [1,224,1280,3]。")
        output = manifest.get("output", {})
        if output.get("shape") != [1, 300, 6] or output.get("nms") is not False:
            raise OpenVinoDeploymentError("部署输出契约不是内置 NMS 的 [1,300,6]。")

        selected_variant = (
            selection.get("active_variant")
            if requested_variant == "auto" else requested_variant
        )
        candidate = manifest.get("candidates", {}).get(selected_variant, {})
        model_path = candidate.get("model_path")
        if selected_variant == ACTIVE_VARIANT and model_path != ACTIVE_MODEL_PATH:
            raise OpenVinoDeploymentError(
                "INT8 v2 候选模型路径与活动模型不一致。"
            )
        if not model_path:
            raise OpenVinoDeploymentError(f"部署包不包含模型版本：{selected_variant}")
        self.variant = selected_variant
        model_xml = self.root / model_path
        model_bin = model_xml.with_suffix(".bin")
        for path, hash_key in (
            (model_xml, "sha256_xml"),
            (model_bin, "sha256_bin"),
        ):
            if not path.is_file():
                raise OpenVinoDeploymentError(f"找不到活动模型文件：{path}")
            expected_hash = str(candidate.get(hash_key, "")).lower()
            if not expected_hash or _sha256(path).lower() != expected_hash:
                raise OpenVinoDeploymentError(f"活动模型哈希校验失败：{path}")
        return manifest

    @staticmethod
    def input_rect(player_foot):
        player_x, player_y = map(int, player_foot)
        x0 = player_x - MODEL_WIDTH // 2
        y0 = player_y - MODEL_HEIGHT // 2
        return x0, y0, x0 + MODEL_WIDTH, y0 + MODEL_HEIGHT

    @classmethod
    def visible_input_rect(cls, frame_shape, player_foot):
        height, width = frame_shape[:2]
        x0, y0, x1, y1 = cls.input_rect(player_foot)
        return (
            max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        )

    @classmethod
    def crop_input(cls, frame, player_foot):
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("游戏画面必须是三通道 BGR ndarray。")
        if frame.dtype != np.uint8:
            raise TypeError("游戏画面必须是 uint8。")

        desired_x0, desired_y0, desired_x1, desired_y1 = cls.input_rect(player_foot)
        visible = cls.visible_input_rect(frame.shape, player_foot)
        src_x0, src_y0, src_x1, src_y1 = visible
        crop = np.zeros((MODEL_HEIGHT, MODEL_WIDTH, 3), dtype=np.uint8)
        if src_x1 > src_x0 and src_y1 > src_y0:
            dst_x0 = src_x0 - desired_x0
            dst_y0 = src_y0 - desired_y0
            dst_x1 = dst_x0 + src_x1 - src_x0
            dst_y1 = dst_y0 + src_y1 - src_y0
            crop[dst_y0:dst_y1, dst_x0:dst_x1] = \
                frame[src_y0:src_y1, src_x0:src_x1]
        return np.ascontiguousarray(crop), (desired_x0, desired_y0), visible

    def detect(self, frame, player_foot):
        total_start = time.perf_counter()
        crop_start = total_start
        image, origin, visible = self.crop_input(frame, player_foot)
        crop_end = time.perf_counter()

        infer_start = crop_end
        predictions = self.runtime.predict(image)
        infer_end = time.perf_counter()

        convert_start = infer_end
        frame_height, frame_width = frame.shape[:2]
        origin_x, origin_y = origin
        monsters = []
        players = []
        for detection in predictions:
            class_id = int(getattr(detection, "class_id"))
            confidence = float(getattr(detection, "confidence"))
            if class_id not in MODEL_CLASS_NAMES or confidence < self.confidence:
                continue
            values = [
                float(getattr(detection, key))
                for key in ("x1", "y1", "x2", "y2")
            ]
            if not all(math.isfinite(value) for value in values + [confidence]):
                continue
            x1 = max(0, min(frame_width, math.floor(values[0] + origin_x)))
            y1 = max(0, min(frame_height, math.floor(values[1] + origin_y)))
            x2 = max(0, min(frame_width, math.ceil(values[2] + origin_x)))
            y2 = max(0, min(frame_height, math.ceil(values[3] + origin_y)))
            if x2 <= x1 or y2 <= y1:
                continue
            item = {
                "name": MODEL_CLASS_NAMES[class_id],
                "position": (x1, y1),
                "size": (x2 - x1, y2 - y1),
                "confidence": confidence,
                "score": 1.0 - confidence,
                "class_id": class_id,
                "detector": "openvino_yolo",
            }
            if class_id == 0:
                monsters.append(item)
            else:
                players.append(item)
        convert_end = time.perf_counter()
        self.last_visible_rect = visible
        self.last_players = players
        if players:
            anchor_x, anchor_y = map(int, player_foot)
            selected = min(
                players,
                key=lambda item: (
                    (item["position"][0] + item["size"][0] // 2 - anchor_x) ** 2 +
                    (item["position"][1] + item["size"][1] - anchor_y) ** 2,
                    -item["confidence"],
                ),
            )
            self.last_player_foot = (
                selected["position"][0] + selected["size"][0] // 2,
                selected["position"][1] + selected["size"][1],
            )
        else:
            self.last_player_foot = None
        self.last_timing = MonsterDetectionTiming(
            crop_ms=(crop_end - crop_start) * 1000.0,
            infer_ms=(infer_end - infer_start) * 1000.0,
            convert_ms=(convert_end - convert_start) * 1000.0,
            total_ms=(convert_end - total_start) * 1000.0,
            detections=len(monsters) + len(players),
            monsters=len(monsters),
            players=len(players),
        )
        return monsters
