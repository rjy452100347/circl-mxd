"""Three-class full-frame perception; independent of legacy strip inference."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np
import yaml

from src.engine.OpenVinoMonsterDetector import (
    MonsterDetectionTiming, OpenVinoDeploymentError, OpenVinoMonsterDetector, _sha256,
)

WIDTH, HEIGHT = 1280, 736
CLASS_NAMES = {0: "monster", 1: "player", 2: "self"}
VARIANT = "int8_head_fp"
MODEL_PATH = "models/int8_head_fp_openvino_model/model.xml"


def uses_fullscreen_self(cfg):
    return cfg.get("monster_detect", {}).get("yolo_variant") == VARIANT


def detector_from_config(cfg, *, legacy_factory=OpenVinoMonsterDetector):
    dc = cfg["monster_detect"]
    kwargs = dict(
        confidence=dc.get("yolo_confidence", .25),
        min_monster_box_side=dc.get("yolo_min_monster_box_side", 10),
        player_exclusion_width=dc.get("yolo_player_exclusion_width", 80),
        player_exclusion_height=dc.get("yolo_player_exclusion_height", 100),
        max_det=dc.get("yolo_max_det", 300 if uses_fullscreen_self(cfg) else 50),
        variant=dc.get("yolo_variant", "int8_v2"),
    )
    if uses_fullscreen_self(cfg):
        return FullscreenSelfDetector(dc["openvino_deployment_root"],
            player_confidence=dc.get("yolo_player_confidence", .25),
            self_confidence=dc.get("yolo_self_confidence", .50),
            view_crop=dc.get("yolo_view_crop", (0, 0, 0, 0)),
            content_scale=dc.get("yolo_content_scale", 1.0), **kwargs)
    return legacy_factory(dc["openvino_deployment_root"], player_confidence=None, **kwargs)


@dataclass(frozen=True)
class SceneTiming(MonsterDetectionTiming):
    selves: int = 0


@dataclass(frozen=True)
class FullscreenScene:
    monsters: tuple
    players: tuple
    selves: tuple
    visible_rect: tuple
    frame_shape: tuple
    timing: SceneTiming
    frame_token: object = None


class FullscreenOpenVinoRuntime:
    def __init__(self, xml):
        import openvino as ov
        core = ov.Core()
        self.compiled = core.compile_model(str(xml), "CPU", {
            "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1",
        })
        if (tuple(self.compiled.input(0).shape) != (1, 3, HEIGHT, WIDTH)
                or self.compiled.input(0).element_type != ov.Type.f32
                or tuple(self.compiled.output(0).shape) != (1, 300, 6)
                or self.compiled.output(0).element_type != ov.Type.f32):
            raise OpenVinoDeploymentError("三类模型输入输出契约不匹配。")
        self.request = self.compiled.create_infer_request()

    def predict(self, tensor):
        self.request.infer({0: tensor})
        return np.array(self.request.get_output_tensor().data, copy=True)


class FullscreenSelfDetector:
    fullscreen = True
    player_exclusion_rect = OpenVinoMonsterDetector.player_exclusion_rect

    def __init__(self, root, *, confidence=.7, player_confidence=.25,
                 self_confidence=.5, min_monster_box_side=15,
                 player_exclusion_width=80, player_exclusion_height=100,
                 max_det=300, variant=VARIANT, runtime_factory=None,
                 view_crop=(0, 0, 0, 0), content_scale=1.0):
        self.root = Path(root).resolve()
        self.variant = variant
        if variant != VARIANT or max_det != 300:
            raise OpenVinoDeploymentError("三类模型必须使用 int8_head_fp，最大检测数为 300。")
        self.max_det = 300
        if (not isinstance(view_crop, (list, tuple)) or len(view_crop) != 4
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                       for v in view_crop)):
            raise OpenVinoDeploymentError("yolo_view_crop 必须是四个非负整数 [左, 上, 右, 下]。")
        self.view_crop = tuple(view_crop)
        self.content_scale = float(content_scale)
        if not math.isfinite(self.content_scale) or not 0 < self.content_scale <= 1:
            raise OpenVinoDeploymentError("yolo_content_scale 必须在 (0, 1] 之间。")
        for key, value in (("confidence", confidence), ("player_confidence", player_confidence),
                           ("self_confidence", self_confidence)):
            value = float(value)
            if not math.isfinite(value) or not .05 <= value <= .95:
                raise OpenVinoDeploymentError(f"{key} 必须在 0.05–0.95 之间。")
            setattr(self, key, value)
        for key, value, maximum in (
            ("min_monster_box_side", min_monster_box_side, HEIGHT),
            ("player_exclusion_width", player_exclusion_width, WIDTH),
            ("player_exclusion_height", player_exclusion_height, HEIGHT),
        ):
            number = float(value)
            if not math.isfinite(number) or not number.is_integer() or not 0 <= number <= maximum:
                raise OpenVinoDeploymentError(f"{key} 必须为 0–{maximum} 之间的整数。")
            setattr(self, key, int(number))
        try:
            xml = self._validate_deployment()
            self.runtime = (runtime_factory or FullscreenOpenVinoRuntime)(xml)
        except OpenVinoDeploymentError:
            raise
        except Exception as exc:
            raise OpenVinoDeploymentError(f"三类全屏模型加载失败：{exc}") from exc
        self.last_timing = SceneTiming()
        self.last_visible_rect = (0, 0, 0, 0)
        self.last_players = []
        self.last_player_foot = None

    def _validate_deployment(self):
        if (self.root / "ACTIVE_MODEL.txt").read_text(encoding="utf-8").strip() != MODEL_PATH:
            raise OpenVinoDeploymentError("ACTIVE_MODEL.txt 未指向三类 int8_head_fp 模型。")
        xml = self.root / MODEL_PATH
        metadata = yaml.safe_load((xml.parent / "metadata.yaml").read_text(encoding="utf-8"))
        names = {int(k): v for k, v in metadata.get("names", {}).items()}
        if (names != CLASS_NAMES or metadata.get("imgsz") != [HEIGHT, WIDTH]
                or metadata.get("end2end") is not True
                or metadata.get("args", {}).get("nms") is not False):
            raise OpenVinoDeploymentError("三类模型 metadata 类别或输入输出说明不匹配。")
        record = json.loads((xml.parent / "quantization.json").read_text(encoding="utf-8"))["model"]
        if record.get("input_shape") != [1, 3, HEIGHT, WIDTH] or record.get("output_shape") != [1, 300, 6]:
            raise OpenVinoDeploymentError("量化记录中的三类模型契约不匹配。")
        for path, key in ((xml, "xml_sha256"), (xml.with_suffix(".bin"), "bin_sha256")):
            if not path.is_file() or _sha256(path) != record.get(key):
                raise OpenVinoDeploymentError(f"三类模型文件缺失或哈希不匹配：{path}")
        return xml

    @staticmethod
    def prepare(frame, content_scale=1.0):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0:
            raise ValueError("游戏画面必须是非空 uint8 BGR 三通道数组。")
        if not math.isfinite(content_scale) or not 0 < content_scale <= 1:
            raise ValueError("content_scale 必须在 (0, 1] 之间。")
        height, width = frame.shape[:2]
        ratio = min(WIDTH / width, HEIGHT / height)
        new_w, new_h = round(width * ratio), round(height * ratio)
        base_left, base_top = (WIDTH-new_w)//2, (HEIGHT-new_h)//2
        base = np.full((HEIGHT, WIDTH, 3), 114, dtype=np.uint8)
        base[base_top:base_top+new_h, base_left:base_left+new_w] = cv2.resize(
            frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        final_w, final_h = round(WIDTH*content_scale), round(HEIGHT*content_scale)
        left, top = (WIDTH-final_w)//2, (HEIGHT-final_h)//2
        canvas = np.full((HEIGHT, WIDTH, 3), 114, dtype=np.uint8)
        canvas[top:top+final_h, left:left+final_w] = cv2.resize(
            base, (final_w, final_h), interpolation=cv2.INTER_AREA)
        sx = new_w/width * final_w/WIDTH
        sy = new_h/height * final_h/HEIGHT
        tx = base_left*final_w/WIDTH + left
        ty = base_top*final_h/HEIGHT + top
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255.0
        return tensor, (sx, sy, tx, ty)

    def detect_scene(self, frame, player_foot=None, *, frame_token=None):
        started = time.perf_counter()
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("游戏画面必须是三通道数组。")
        h, w = frame.shape[:2]
        crop_left, crop_top, crop_right, crop_bottom = self.view_crop
        x0, y0 = crop_left, crop_top
        view_right, view_bottom = w-crop_right, h-crop_bottom
        if x0 >= view_right or y0 >= view_bottom:
            raise ValueError(f"YOLO 游戏区域超出当前画面：{self.view_crop}, {w}x{h}")
        tensor, (sx, sy, tx, ty) = self.prepare(
            frame[y0:view_bottom, x0:view_right], self.content_scale)
        prepared = time.perf_counter()
        raw = np.asarray(self.runtime.predict(tensor))
        inferred = time.perf_counter()
        if raw.shape != (1, 300, 6):
            raise ValueError(f"三类模型输出应为 [1,300,6]，实际 {raw.shape}")
        groups = [[], [], []]
        thresholds = [self.confidence, self.player_confidence, self.self_confidence]
        for row in raw[0]:
            if not np.isfinite(row).all():
                continue
            x1, y1, x2, y2, score, raw_class = map(float, row)
            cid = int(raw_class)
            if cid not in CLASS_NAMES or abs(cid-raw_class) > 1e-3:
                raise ValueError(f"三类模型输出未知类别：{raw_class}")
            if score < thresholds[cid]:
                continue
            bx1 = max(x0, min(view_right, math.floor((x1-tx)/sx)+x0))
            bx2 = max(x0, min(view_right, math.ceil((x2-tx)/sx)+x0))
            by1 = max(y0, min(view_bottom, math.floor((y1-ty)/sy)+y0))
            by2 = max(y0, min(view_bottom, math.ceil((y2-ty)/sy)+y0))
            if bx2 <= bx1 or by2 <= by1 or (cid == 0 and min(bx2-bx1, by2-by1) < self.min_monster_box_side):
                continue
            groups[cid].append(MappingProxyType(dict(
                name=CLASS_NAMES[cid], class_id=cid, position=(bx1, by1), size=(bx2-bx1, by2-by1),
                confidence=score, score=1.0-score, detector="openvino_fullscreen",
            )))
        ended = time.perf_counter()
        self.last_timing = SceneTiming(crop_ms=(prepared-started)*1000,
            infer_ms=(inferred-prepared)*1000, convert_ms=(ended-inferred)*1000,
            total_ms=(ended-started)*1000, detections=sum(map(len, groups)),
            monsters=len(groups[0]), players=len(groups[1]), selves=len(groups[2]))
        self.last_visible_rect = (x0, y0, view_right, view_bottom)
        self.last_players = list(groups[1])
        return FullscreenScene(*(tuple(g) for g in groups), self.last_visible_rect,
                               frame.shape, self.last_timing, frame_token)

    def finalize_scene(self, scene, player_exclusion_anchor=None):
        exclusion = self.player_exclusion_rect(scene.frame_shape, player_exclusion_anchor)
        result = []
        for item in scene.monsters:
            x, y = item["position"]
            w, h = item["size"]
            if exclusion and exclusion[0] <= x+w/2 <= exclusion[2] and exclusion[1] <= y+h/2 <= exclusion[3]:
                continue
            result.append(dict(item))
        return result


@dataclass(frozen=True)
class SelfLocation:
    valid: bool = False
    foot: tuple | None = None
    reason: str = "self_missing"
    confirmations: int = 0


class SelfLocalizationGate:
    """Only two distinct fresh, unambiguous observations grant control."""
    def __init__(self):
        self.last_token = None
        self.snapshot = SelfLocation()

    def invalidate(self, reason="self_missing"):
        self.snapshot = SelfLocation(reason=reason)
        return self.snapshot

    def update(self, scene, token):
        if token is None:
            return self.invalidate("self_frame_unknown")
        if token == self.last_token:
            return self.snapshot
        self.last_token = token
        if len(scene.selves) != 1:
            return self.invalidate("self_missing" if not scene.selves else "self_ambiguous")
        candidate = scene.selves[0]
        x, y = candidate["position"]
        w, h = candidate["size"]
        foot = (x+w//2, min(scene.frame_shape[0]-1, y+h))
        count = min(2, self.snapshot.confirmations+1)
        self.snapshot = SelfLocation(count == 2, foot, "ok" if count == 2 else "self_recovery_unconfirmed", count)
        return self.snapshot
