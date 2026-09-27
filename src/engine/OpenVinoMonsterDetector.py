from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np


MODEL_WIDTH = 1280
MODEL_HEIGHT = 224
MODEL_CLASS_NAMES = {0: "monster", 1: "player"}
PLAYER_CANDIDATE_CONFIDENCE = 0.25
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


@dataclass(frozen=True)
class DetectionScene:
    monsters: tuple
    players: tuple
    visible_rect: tuple
    frame_shape: tuple
    timing: MonsterDetectionTiming
    # A separate association-only pool: never counted or used as monsters, and
    # never allowed to displace the runtime's existing top-50 detections.
    player_candidates: tuple = ()


@dataclass(frozen=True)
class _PlayerCandidateDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 1


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
        min_monster_box_side=10,
        player_exclusion_width=80,
        player_exclusion_height=100,
        max_det=50,
        variant="auto",
        runtime_factory=None,
        player_confidence=None,
    ):
        self.root = Path(deployment_root).expanduser().resolve()
        self.confidence = float(confidence)
        self._player_candidates_enabled = player_confidence is not None
        self.player_confidence = self.confidence if player_confidence is None else float(player_confidence)
        if not .05 <= self.player_confidence <= .95:
            raise OpenVinoDeploymentError('人物检测置信度必须在 0.05–0.95 之间')
        try:
            min_box_side_value = float(min_monster_box_side)
        except (TypeError, ValueError) as exc:
            raise OpenVinoDeploymentError(
                f"YOLO 怪物框最小边长必须是 0–{MODEL_HEIGHT} 之间的整数。"
            ) from exc
        if (not math.isfinite(min_box_side_value) or
                not min_box_side_value.is_integer()):
            raise OpenVinoDeploymentError(
                f"YOLO 怪物框最小边长必须是 0–{MODEL_HEIGHT} 之间的整数。"
            )
        self.min_monster_box_side = int(min_box_side_value)
        self.player_exclusion_width = self._parse_exclusion_dimension(
            player_exclusion_width, MODEL_WIDTH, "宽度"
        )
        self.player_exclusion_height = self._parse_exclusion_dimension(
            player_exclusion_height, MODEL_HEIGHT, "高度"
        )
        self.max_det = int(max_det)
        requested_variant = str(variant).strip().lower()
        if requested_variant not in {"auto", "fp16", "int8", "int8_v2"}:
            raise OpenVinoDeploymentError(
                "YOLO 模型版本必须是 auto、fp16、int8 或 int8_v2。"
            )
        if not 0.05 <= self.confidence <= 0.95:
            raise OpenVinoDeploymentError("YOLO 置信度必须在 0.05–0.95 之间。")
        if not 0 <= self.min_monster_box_side <= MODEL_HEIGHT:
            raise OpenVinoDeploymentError(
                f"YOLO 怪物框最小边长必须是 0–{MODEL_HEIGHT} 之间的整数。"
            )
        if self.max_det != 50:
            raise OpenVinoDeploymentError("当前双类部署要求 yolo_max_det 固定为 50。")

        self.manifest = self._validate_deployment(requested_variant)
        factory = runtime_factory or _load_runtime_class(self.root)
        try:
            kwargs = {"confidence": min(self.confidence, self.player_confidence), "max_det": self.max_det}
            kwargs["variant"] = self.variant
            self.runtime = factory(self.root, **kwargs)
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

    @staticmethod
    def _parse_exclusion_dimension(value, maximum, label):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise OpenVinoDeploymentError(
                f"YOLO 人物排除区{label}必须是 0–{maximum} px 之间的整数。"
            ) from exc
        if (not math.isfinite(number) or not number.is_integer() or
                not 0 <= number <= maximum):
            raise OpenVinoDeploymentError(
                f"YOLO 人物排除区{label}必须是 0–{maximum} px 之间的整数。"
            )
        return int(number)

    def player_exclusion_rect(self, frame_shape, anchor):
        """Return the clipped player exclusion rectangle in frame coordinates."""
        if (anchor is None or self.player_exclusion_width == 0 or
                self.player_exclusion_height == 0 or len(frame_shape) < 2):
            return None
        frame_height, frame_width = map(int, frame_shape[:2])
        if frame_width <= 0 or frame_height <= 0:
            return None
        try:
            player_x, player_y = map(float, anchor)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(player_x) or not math.isfinite(player_y):
            return None
        half_width = self.player_exclusion_width / 2.0
        x0 = max(0, min(frame_width - 1, math.floor(player_x - half_width)))
        x1 = max(0, min(frame_width - 1, math.ceil(player_x + half_width)))
        y0 = max(
            0,
            min(frame_height - 1, math.floor(
                player_y - self.player_exclusion_height
            )),
        )
        y1 = max(0, min(frame_height - 1, math.ceil(player_y)))
        return (x0, y0, x1, y1)

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

    def detect(self, frame, player_foot, *, player_exclusion_anchor=None):
        scene = self.detect_scene(frame, player_foot)
        return self.finalize_scene(scene, player_exclusion_anchor)

    def finalize_scene(self, scene, player_exclusion_anchor=None):
        exclusion = self.player_exclusion_rect(scene.frame_shape, player_exclusion_anchor)
        monsters = []
        for item in scene.monsters:
            x, y = item['position']
            w, h = item['size']
            if exclusion is not None and exclusion[0] <= x+w/2 <= exclusion[2] and exclusion[1] <= y+h/2 <= exclusion[3]:
                continue
            monsters.append(dict(item))
        self.last_timing = replace(scene.timing, monsters=len(monsters),
                                   detections=len(monsters)+len(scene.players))
        return monsters

    def _current_player_candidates(self, predictions):
        """Read the output of the just-completed synchronous prediction only.

        Both deployed and protected runtimes reuse one InferRequest. Reading
        its result immediately after predict() does not run inference again or
        change the confidence/max_det filtering of its public result. Copying
        prevents the next infer() from overwriting this scene's candidates.
        Runtimes without that contract can only expose their current public
        detections; no cached fallback or second prediction is permitted.
        """
        if not self._player_candidates_enabled:
            return ()
        request = getattr(self.runtime, "infer_request", None)
        if request is None or not callable(getattr(request, "get_output_tensor", None)):
            return predictions
        raw = np.array(request.get_output_tensor().data, copy=True)
        if raw.shape != (1, 300, 6):
            return predictions
        rows = raw[0]
        rows = rows[(rows[:, 5] == 1) &
                    (rows[:, 4] >= PLAYER_CANDIDATE_CONFIDENCE) &
                    np.isfinite(rows).all(axis=1)]
        return tuple(_PlayerCandidateDetection(
            x1=float(np.clip(row[0], 0, MODEL_WIDTH)),
            y1=float(np.clip(row[1], 0, MODEL_HEIGHT)),
            x2=float(np.clip(row[2], 0, MODEL_WIDTH)),
            y2=float(np.clip(row[3], 0, MODEL_HEIGHT)),
            confidence=float(row[4]),
        ) for row in rows)

    def _convert_detection(self, detection, frame_shape, origin, threshold):
        class_id = int(getattr(detection, "class_id"))
        confidence = float(getattr(detection, "confidence"))
        if class_id not in MODEL_CLASS_NAMES or confidence < threshold:
            return None
        values = [float(getattr(detection, key))
                  for key in ("x1", "y1", "x2", "y2")]
        if not all(math.isfinite(value) for value in values + [confidence]):
            return None
        frame_height, frame_width = frame_shape[:2]
        origin_x, origin_y = origin
        x1 = max(0, min(frame_width, math.floor(values[0] + origin_x)))
        y1 = max(0, min(frame_height, math.floor(values[1] + origin_y)))
        x2 = max(0, min(frame_width, math.ceil(values[2] + origin_x)))
        y2 = max(0, min(frame_height, math.ceil(values[3] + origin_y)))
        if x2 <= x1 or y2 <= y1:
            return None
        width, height = x2 - x1, y2 - y1
        if class_id == 0 and min(width, height) < self.min_monster_box_side:
            return None
        return {
            "name": MODEL_CLASS_NAMES[class_id],
            "position": (x1, y1), "size": (width, height),
            "confidence": confidence, "score": 1.0 - confidence,
            "class_id": class_id, "detector": "openvino_yolo",
        }

    def detect_scene(self, frame, player_foot):
        total_start = time.perf_counter()
        crop_start = total_start
        image, origin, visible = self.crop_input(frame, player_foot)
        crop_end = time.perf_counter()

        infer_start = crop_end
        predictions = self.runtime.predict(image)
        infer_end = time.perf_counter()

        convert_start = infer_end
        candidate_predictions = self._current_player_candidates(predictions)
        monsters = []
        players = []
        for detection in predictions:
            class_id = int(getattr(detection, "class_id"))
            threshold = self.player_confidence if class_id == 1 else self.confidence
            item = self._convert_detection(detection, frame.shape, origin, threshold)
            if item is None:
                continue
            if class_id == 0:
                monsters.append(item)
            else:
                players.append(item)
        player_candidates = []
        for detection in candidate_predictions:
            if getattr(detection, "class_id") != 1:
                continue
            item = self._convert_detection(
                detection, frame.shape, origin, PLAYER_CANDIDATE_CONFIDENCE)
            if item is not None:
                player_candidates.append(MappingProxyType(item))
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
        return DetectionScene(tuple(MappingProxyType(dict(item)) for item in monsters),
                              tuple(MappingProxyType(dict(item)) for item in players),
                              visible, frame.shape, self.last_timing,
                              tuple(player_candidates))
