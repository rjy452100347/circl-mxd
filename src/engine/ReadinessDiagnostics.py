"""Read-only readiness checks. No keyboard controller or bot loop is created here."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.engine.HealthMonitor import HealthBarRecognizer
from src.engine.MapProjectConfig import apply_map_minimap_roi_override
from src.engine.MinimapPoseTracker import MinimapPoseTracker
from src.engine.NameTagLocalizer import NameTagLocalizer
from src.engine.NameTagProfileRepository import NameTagProfileRepository
from src.engine.OpenVinoMonsterDetector import OpenVinoMonsterDetector
from src.engine.SemanticRoute import load_semantic_route_set
from src.utils.common import (
    find_pattern_sqdiff, get_minimap_loc_size, get_player_location_on_minimap,
    load_image, prepare_game_frame,
)


# Keys also select the existing recovery UI; user configuration cannot replace them.
CHECKS = (
    ("window", "游戏画面", "检查窗口标题、裁剪和尺寸；保持游戏画面可见"),
    ("profile", "名字资料", "选择正确角色，或打开人物名字标定"),
    ("name", "当前名字 / 脚点", "保持名字可见；追加样本或重新标定脚点"),
    ("roi", "小地图 ROI", "展开小地图；在路线编辑器中校准 ROI"),
    ("dot", "小地图人物点", "检查 ROI、人物颜色及小地图遮挡"),
    ("map", "地图资源 / 定位", "确认地图选择；检查底图或重新捕获"),
    ("route", "路线资源", "打开路线编辑器检查并保存路线"),
    ("yolo", "YOLO 怪物检测", "检查模型部署；名字有效后才能验证当前检测"),
    ("hp", "自动补血识别", "保持底部 HP 状态栏可见；检查底部 UI 裁剪"),
    ("mp", "自动补蓝识别", "保持底部 MP 状态栏可见；检查底部 UI 裁剪"),
)
REASONS = {
    "ok": "定位有效",
    "capture_invalid": "游戏画面不可用，请检查窗口、尺寸和遮挡",
    "minimap_roi_invalid": "小地图区域无效，请展开小地图或重新校准 ROI",
    "player_not_visible": "未找到小地图人物点，请检查颜色与遮挡",
    "map_localization_invalid": "地图定位无效，请检查底图或等待重新定位",
    "position_jump": "人物坐标跳变被拒绝，正在等待可靠位置",
    "score": "地图匹配分数未达标，请检查底图和 ROI",
    "local_recovery_wait": "正在尝试恢复局部地图定位",
    "global_unconfirmed": "正在确认全局地图位置",
    "bars_not_found": "未找到完整状态条，请检查底部 UI 是否可见",
    "bar_shapes_not_found": "未找到完整状态条形状，请检查底部 UI 裁剪和遮挡",
    "bar_identity_unconfirmed": "尚未确认 HP/MP 状态条身份，请保持底部状态栏可见",
    "frame_stale": "识别画面已过期，等待新的有效画面",
    "frame_invalid": "画面为空或裁剪无效",
    "classic_bar_identity_invalid": "状态条身份校验失败，请检查底部 UI 区域",
    "bar_border_invalid": "状态条边框校验失败，请检查遮挡与尺寸",
    "score_too_high": "名字匹配分数未达标，请追加当前背景样本",
    "not_found": "没有找到名字，请检查角色资料、名字遮挡及搜索下边界",
    "jump_unconfirmed": "名字位置跳变，等待连续确认；请检查是否匹配到其他人物",
}


@dataclass(frozen=True)
class CheckResult:
    key: str
    status: str
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    rows: tuple[CheckResult, ...]
    produced_at: float
    summary: str
    # A small independently owned PNG, not a reference to a live capture ndarray.
    name_preview: bytes | None = None
    generation: int = 0


def initial_rows(cfg):
    rows = {key: CheckResult(key, "pending", "尚未检查") for key, _, _ in CHECKS}
    mode = cfg.get("bot", {}).get("mode", "normal")
    unused = {"map", "route"} if mode != "normal" else set()
    if mode == "continuous_attack":
        unused |= {"roi", "dot"}
    health = cfg.get("health_monitor", {})
    for role in ("hp", "mp"):
        enabled = health.get(f"auto_{role}_enabled", health.get("enable", True)
                             and health.get(f"add_{role}_percent", 0) > 0)
        if not enabled:
            unused.add(role)
    for key in unused:
        rows[key] = CheckResult(key, "unused", "当前模式不需要 / 未启用")
    return rows


def name_result(result, frame=None):
    if result is None:
        return CheckResult("name", "pending", "本帧尚未检查，不能使用旧位置"), None
    if not result.valid:
        return CheckResult("name", "fail", f"未识别；分数 {result.score:.3f}；"
                           f"{REASONS.get(result.reason, result.reason)}"), None
    x, y = result.player
    if frame is not None and not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
        return CheckResult("name", "fail", "名字已匹配，但脚点超出画面；请重新标定脚点"), None
    preview = None
    if frame is not None:
        tx, ty = result.tag_top_left
        w, h = result.tag_size
        # Bound allocation even for an incorrect, far-away foot offset.
        x0, y0 = max(0, min(tx, x) - 20), max(0, min(ty, y) - 40)
        x1, y1 = min(frame.shape[1], max(tx + w, x) + 20), min(frame.shape[0], max(ty + h, y) + 20)
        if x1 > x0 and y1 > y0:
            scale = min(1.0, 480 / (x1-x0), 200 / (y1-y0))
            crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=scale, fy=scale)
            point = lambda px, py: (round((px-x0)*scale), round((py-y0)*scale))
            cv2.rectangle(crop, point(tx, ty), point(tx+w, ty+h), (0, 255, 0), 1)
            cv2.circle(crop, point(x, y), 4, (255, 0, 255), -1)
            ok, encoded = cv2.imencode(".png", crop)
            if ok:
                preview = encoded.tobytes()
    return CheckResult("name", "pass", f"已匹配；分数 {result.score:.3f}；脚点 ({x}, {y})。"
                       "绿色框为名字、紫点为脚点，请人工确认位置。"), preview


def health_rows(rows, snapshot):
    for role in ("hp", "mp"):
        if rows[role].status == "unused":
            continue
        if snapshot is None:
            rows[role] = CheckResult(role, "pending", "等待健康识别线程的新结果")
        elif snapshot.valid:
            value = getattr(snapshot, f"{role}_percent")
            rows[role] = CheckResult(role, "pass", f"READY；{value}%（仅表示识别有效，不表示已喝药）")
        else:
            reason = snapshot.reason
            hint = "状态条正在连续校准" if reason.startswith("stabilizing") else REASONS.get(reason, reason)
            rows[role] = CheckResult(role, "warn", f"{snapshot.state}；{hint}")


def runtime_report(bot, frame_valid):
    """Called by the control thread after a frame, never reads it from the Qt thread."""
    cfg = bot.cfg
    rows = initial_rows(cfg)
    put = lambda key, status, detail: rows.update({key: CheckResult(key, status, detail)})
    put("window", "pass" if frame_valid else "fail",
        "收到当前游戏画面" if frame_valid else REASONS["capture_invalid"])
    put("profile", "pass", f"已加载：{cfg['nametag']['name']}")
    preview = None
    locked = (cfg["bot"]["mode"] == "continuous_attack" and
              getattr(getattr(bot, "continuous_attack_state", None), "direction_locked", False))
    if locked:
        put("name", "unused", "方向已锁定，本次运行不再进行名字识别")
        put("yolo", "unused", "方向已锁定，本次运行不再进行 YOLO 推理")
    elif frame_valid:
        # Preview generation is throttled independently of control and monitoring pages.
        now = time.monotonic()
        make_preview = now - getattr(bot, "_diagnostic_preview_at", 0) >= .5
        rows["name"], preview = name_result(getattr(bot, "nametag_last_result", None),
                                          bot.img_frame if make_preview else None)
        if make_preview:
            bot._diagnostic_preview_at = now
        checked = getattr(bot, "_diagnostic_yolo_checked", False)
        put("yolo", "pass" if checked else "pending",
            f"本帧检测完成；怪物 {len(bot.monsters)}（零怪物不是错误）" if checked
            else "模型已加载；本帧未推理，不沿用旧检测结果")
    for key, flag, good, bad in (
        ("roi", "current_minimap_roi_valid", "小地图区域有效", "小地图 ROI 无效或本帧未检查"),
        ("dot", "current_minimap_player_valid", "已找到小地图人物点", "未找到小地图人物点或上游未就绪"),
    ):
        if rows[key].status != "unused":
            valid = frame_valid and getattr(bot, flag, False)
            put(key, "pass" if valid else "warn", good if valid else bad)
    if rows["map"].status != "unused":
        pose = getattr(bot, "minimap_pose_snapshot", None)
        if frame_valid and getattr(bot, "_diagnostic_map_checked", False) and pose is not None:
            put("map", "pass" if pose.valid else "fail",
                f"位置 {pose.stable_position}；分数 {pose.score:.3f}；"
                f"{REASONS.get(pose.reason, pose.reason)}")
        else:
            put("map", "pending", "底图已加载；本帧尚未取得地图定位")
        put("route", "pass", "路线资源已加载；不等于实机走通")
    monitor = getattr(bot, "health_monitor", None)
    health_rows(rows, monitor.get_snapshot() if monitor is not None else None)
    pause_reason = getattr(bot, "_diagnostic_pause_reason", "")
    summary = REASONS.get(pause_reason, pause_reason)
    if not summary:
        summary = "运行中；请查看逐项识别结果"
        if getattr(getattr(bot, "kb", None), "is_need_force_heal", False):
            summary = "强制补血中；路线和攻击暂时让出控制"
    kb = getattr(bot, "kb", None)
    if kb is not None and getattr(kb, "_was_game_window_active", True) is False:
        summary = "等待游戏获得焦点；输入已暂停，请切回游戏窗口"
    return ReadinessReport(tuple(rows.values()), time.monotonic(), summary, preview,
                           int(getattr(bot, "run_generation", 0)))


class ReadinessChecker:
    """Short-lived perception only session; injected factories make safety testable."""

    def __init__(self, cfg, publish, cancel, *, capture_factory=None):
        self.cfg = copy.deepcopy(cfg)
        self.publish = publish
        self.cancel = cancel
        self.capture_factory = capture_factory
        self.rows = initial_rows(cfg)
        self.preview = None

    def put(self, key, status, detail):
        self.rows[key] = CheckResult(key, status, detail)

    def emit(self, summary):
        self.publish(ReadinessReport(tuple(self.rows.values()), time.monotonic(), summary, self.preview))

    def resource(self, key, fn):
        if self.cancel.is_set():
            return None
        self.put(key, "pending", "正在加载 / 检查…")
        self.emit("只读检查中，不会移动、攻击或喝药")
        try:
            result = fn()
            self.put(key, "pass", "资源加载成功；还需检查当前画面")
            return result
        except Exception as exc:
            self.put(key, "fail", f"检查失败：{exc}")
            return None

    def observe(self, key, fn):
        """A broken recognizer must not hide unrelated checks on the same frame."""
        try:
            return fn()
        except Exception as exc:
            self.put(key, "fail", f"当前画面检查失败：{exc}")
            return None

    def missing_frame(self):
        self.preview = None
        for key in ("name", "roi", "dot", "map", "yolo", "hp", "mp"):
            if self.rows[key].status not in {"unused", "fail"}:
                self.put(key, "pending", "没有新的有效画面；不沿用之前的识别成功结果")

    def run(self):
        cfg = self.cfg
        capture = None
        try:
            def load_name():
                repo = NameTagProfileRepository()
                name = cfg["nametag"]["name"]
                repo.load_profile(name, require_enabled=True)
                return NameTagLocalizer.from_profile(repo.resolve_profile(name))
            name = self.resource("profile", load_name)
            if name is None:
                self.put("name", "pending", "名字资料加载失败，无法检查当前名字")
            else:
                self.put("profile", "pass", f"已加载人物资料：{cfg['nametag']['name']}")
            roi_ok = True
            roi_source = "unused"
            if self.rows["roi"].status != "unused":
                roi_source = self.resource("roi", lambda: apply_map_minimap_roi_override(
                    cfg, str(cfg["bot"].get("map", "")), Path("minimaps")))
                roi_ok = roi_source is not None
            map_image = None
            if self.rows["map"].status != "unused":
                directory = Path("minimaps") / cfg["bot"]["map"]
                map_image = self.resource("map", lambda: load_image(str(directory / "map.png")))
                def load_routes():
                    if map_image is None:
                        raise ValueError("地图底图不可用，请先捕获并保存地图")
                    routes = [p for p in directory.glob("route*.png") if p.name != "route_rest.png"]
                    if not routes:
                        raise ValueError("没有路线预览文件，请在路线编辑器中保存路线")
                    for path in routes:
                        image = load_image(str(path))
                        if image.shape[:2] != map_image.shape[:2]:
                            raise ValueError(f"路线尺寸与底图不一致：{path.name}")
                    return load_semantic_route_set(directory, map_id=cfg["bot"]["map"],
                        canvas_size=(map_image.shape[1], map_image.shape[0]))
                self.resource("route", load_routes)
            dc = cfg["monster_detect"]
            detector = self.resource("yolo", lambda: OpenVinoMonsterDetector(
                dc["openvino_deployment_root"], confidence=dc.get("yolo_confidence", .25),
                min_monster_box_side=dc.get("yolo_min_monster_box_side", 10),
                player_exclusion_width=dc.get("yolo_player_exclusion_width", 80),
                player_exclusion_height=dc.get("yolo_player_exclusion_height", 100),
                max_det=50, variant=dc.get("yolo_variant", "int8_v2")))
            if self.cancel.is_set():
                return
            if self.capture_factory is None:
                from src.input.GameWindowCapturor import GameWindowCapturor
                self.capture_factory = GameWindowCapturor
            capture_cfg = copy.deepcopy(cfg)
            # Checking must not resize/activate the actual game window.
            capture_cfg["game_window"]["resize_on_start"] = False
            capture = self.resource("window", lambda: self.capture_factory(capture_cfg))
            if capture is None:
                return
            tracker = MinimapPoseTracker(find_pattern_sqdiff, max_score=cfg.get("route", {}).get("localization_max_score", .2))
            health = HealthBarRecognizer()
            deadline, last_sequence, frames = time.monotonic() + 3, object(), 0
            while not self.cancel.is_set() and time.monotonic() < deadline and frames < 20:
                packet = capture.get_frame_packet()
                if packet is None:
                    self.missing_frame()
                    self.put("window", "fail", "未收到新画面；检查游戏窗口是否最小化或失效")
                    self.emit("等待有效游戏画面（只读）")
                    self.cancel.wait(.1)
                    continue
                raw, captured_at, sequence = packet
                if sequence == last_sequence or time.monotonic() - captured_at > .3:
                    self.missing_frame()
                    self.put("window", "warn", "画面未更新；等待新的捕获帧")
                    self.cancel.wait(.1)
                    continue
                last_sequence = sequence
                frame = prepare_game_frame(raw, cfg)
                if frame is None:
                    self.missing_frame()
                    self.put("window", "fail", f"裁剪或尺寸不符；原图 {raw.shape[:2]}，配置 {cfg['game_window'].get('size')}")
                    break
                frames += 1
                self.put("window", "pass", f"有效新帧 {frames}/20；游戏画面 {frame.shape[1]}×{frame.shape[0]}")
                self.preview = None
                result = None
                if name is not None:
                    limit = cfg.get("nametag", {}).get("search_y_limit", 0) or cfg["ui_coords"]["ui_y_start"]
                    result = self.observe("name", lambda: name.locate(frame, y_limit=min(int(limit), frame.shape[0])))
                    if result is not None:
                        self.rows["name"], self.preview = name_result(result, frame)
                if detector is not None:
                    if result is not None and result.valid and self.rows["name"].status == "pass":
                        monsters = self.observe("yolo", lambda: detector.detect(frame, result.player, player_exclusion_anchor=result.player))
                        if monsters is not None:
                            self.put("yolo", "pass", f"推理正常；怪物 {len(monsters)}（零怪物不是错误）")
                    else:
                        self.put("yolo", "pending", "模型已加载，等待有效名字脚点后验证推理")
                if roi_ok and self.rows["roi"].status != "unused":
                    rect = self.observe("roi", lambda: get_minimap_loc_size(frame, cfg))
                    self.put("dot", "pending", "等待有效 ROI")
                    if map_image is not None:
                        self.put("map", "pending", "底图已加载，等待当前小地图人物点")
                    if rect is None:
                        self.put("roi", "fail", "没有识别到小地图区域，请展开或校准")
                    else:
                        x, y, w, h = map(int, rect)
                        if not cfg.get("minimap", {}).get("roi"):
                            x, y, w, h = x+1, y+1, w-2, h-2
                        minimap = frame[y:y+h, x:x+w]
                        source = {"global": "全局配置", "automatic": "自动识别"}.get(roi_source, roi_source)
                        self.put("roi", "pass", f"来源 {source}；实际像素区域 ({x}, {y}, {w}, {h})")
                        player = self.observe("dot", lambda: get_player_location_on_minimap(minimap, minimap_player_color=cfg["minimap"]["player_color"], player_hsv=cfg["minimap"].get("player_hsv")))
                        if self.rows["dot"].status != "fail":
                            self.put("dot", "pass" if player is not None else "fail", f"人物点 {player}" if player is not None else "未找到人物点，请检查颜色和遮挡")
                        if player is not None and map_image is not None:
                            mask = np.any(minimap != 0, axis=2).astype(np.uint8)*255
                            pose = self.observe("map", lambda: tracker.update(map_image, minimap, player, mask=mask, offset=cfg["minimap"].get("offset", [0, 0])))
                            if pose is not None:
                                self.put("map", "pass" if pose.valid else "fail", f"位置 {pose.stable_position}；分数 {pose.score:.3f}；{REASONS.get(pose.reason, pose.reason)}")
                if any(self.rows[k].status != "unused" for k in ("hp", "mp")):
                    y = int(cfg["ui_coords"]["ui_y_start"])
                    role = "hp" if self.rows["hp"].status != "unused" else "mp"
                    snapshot = self.observe(role, lambda: health.recognize(frame[y:, :], frames, captured_at, roi_origin=(0, y)))
                    if snapshot is not None:
                        health_rows(self.rows, snapshot)
                    else:
                        for key in ("hp", "mp"):
                            if self.rows[key].status != "unused":
                                self.put(key, "fail", self.rows[role].detail)
                self.emit("只读检查中；请保持名字、小地图和底部状态栏可见")
                self.cancel.wait(.1)
        finally:
            try:
                if capture is not None:
                    capture.stop()
            finally:
                self.emit("检查已取消；结果不用于启动" if self.cancel.is_set() else
                          "只读检查结束；结果为本次采样，F1 后仍持续检查；不会自动开始运行")
