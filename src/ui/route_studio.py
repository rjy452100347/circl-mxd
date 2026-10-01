"""Route Studio: ordered semantic route recording and editing."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QKeySequence, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.engine.RouteStudioSession import RouteStudioSession, RouteStudioSessionError
from src.engine.MapProjectConfig import (
    MAP_CONFIG_FILENAME,
    MapProjectConfigError,
    load_map_minimap_roi,
    normalized_roi_to_pixels,
    pixels_to_normalized_roi,
    save_map_minimap_roi_atomic,
)
from src.engine.SemanticRoute import (
    SCHEMA_VERSION,
    SEGMENT_TYPES,
    SemanticRouteError,
    calculate_ladder_mount_x,
    convert_legacy_png_to_draft,
    load_route_document,
    render_route_png,
    save_route_pair_atomic,
    validate_route_document,
)
from src.runtime_policy import RUNTIME_POLICY
from src.route_control_lock import RouteControlLock
from src.utils.common import (
    find_pattern_sqdiff,
    get_minimap_loc_size,
    get_player_location_on_minimap,
    load_image,
)


SEGMENT_CN = {
    "walk": "行走", "jump": "跳跃", "drop": "下跳", "ladder": "攀爬",
    "teleport": "传送", "stop": "停止", "goal": "目标",
}

ROUTE_STUDIO_QSS = """
QMainWindow, QWidget { background:#0b1220; color:#dbeafe; }
QGroupBox { background:#111c2e; border:1px solid #23324a; border-radius:7px;
  margin-top:10px; padding:10px; color:#67e8f9; }
QPushButton { background:#164e63; color:#ecfeff; border:1px solid #0e7490;
  border-radius:5px; padding:6px 10px; }
QPushButton:hover { background:#155e75; }
QPushButton:disabled { color:#64748b; background:#111827; border-color:#1f2937; }
QComboBox, QLineEdit, QListWidget { background:#0b1424; color:#e2e8f0;
  border:1px solid #334155; border-radius:4px; padding:4px; }
QComboBox QAbstractItemView { background:#111c2e; color:#e2e8f0;
  selection-background-color:#164e63; }
"""


def command_colors_from_cfg(cfg):
    result = {}
    route_cfg = cfg.get("route", {})
    for section in ("color_code", "color_code_up_down"):
        for rgb_text, command in route_cfg.get(section, {}).items():
            result[str(command)] = tuple(int(value) for value in rgb_text.split(","))
    return result


class SemanticRouteEditorModel:
    """Undoable in-memory editor; all disk writes go through paired atomic save."""

    def __init__(self, *, map_id, route_index, base_bgr, document=None):
        if base_bgr is None or base_bgr.ndim != 3:
            raise SemanticRouteError("编辑器需要三通道 map.png。")
        self.map_id = str(map_id)
        self.route_index = int(route_index)
        self.base_bgr = np.ascontiguousarray(base_bgr.copy())
        height, width = self.base_bgr.shape[:2]
        self.document = copy.deepcopy(document) if document is not None else {
            "schema_version": SCHEMA_VERSION,
            "map_id": self.map_id,
            "route_index": self.route_index,
            "canvas_size": [width, height],
            "loop": True,
            "segments": [{
                "id": "goal_001", "type": "goal",
                "position": [width // 2, height // 2], "radius": 6,
            }],
        }
        validate_route_document(self.document, allow_draft=True)
        self.selected_index = 0 if self.document["segments"] else -1
        self.dirty = False
        self._undo = []
        self._redo = []

    @classmethod
    def load(cls, json_path, base_bgr):
        path = Path(json_path)
        document = load_route_document(path, allow_draft=True)
        return cls(
            map_id=document["map_id"], route_index=document["route_index"],
            base_bgr=base_bgr, document=document,
        )

    def _snapshot(self):
        return copy.deepcopy(self.document), self.selected_index

    def _begin_change(self):
        self._undo.append(self._snapshot())
        self._redo.clear()

    def _changed(self):
        self.dirty = True
        if self.document["segments"]:
            self.selected_index = max(
                0, min(self.selected_index, len(self.document["segments"]) - 1)
            )
        else:
            self.selected_index = -1

    def undo(self):
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self.document, self.selected_index = self._undo.pop()
        self.dirty = True
        return True

    def redo(self):
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self.document, self.selected_index = self._redo.pop()
        self.dirty = True
        return True

    def replace_document(self, document, *, dirty=True):
        validate_route_document(document, allow_draft=True)
        self._begin_change()
        self.document = copy.deepcopy(document)
        self.map_id = self.document["map_id"]
        self.route_index = self.document["route_index"]
        self.selected_index = max(0, len(self.document["segments"]) - 1)
        self.dirty = bool(dirty)

    def selected_segment(self):
        if not 0 <= self.selected_index < len(self.document["segments"]):
            return None
        return self.document["segments"][self.selected_index]

    @staticmethod
    def _anchor(segment):
        if segment["type"] in {"walk", "ladder"}:
            return list(segment["points"][0])
        return list(
            segment.get("anchor") or segment.get("position")
            or segment.get("approach") or segment.get("exit")
        )

    def _unique_id(self, kind):
        used = {segment["id"] for segment in self.document["segments"]}
        serial = 1
        while f"{kind}_{serial:03d}" in used:
            serial += 1
        return f"{kind}_{serial:03d}"

    def delete_selected(self):
        segment = self.selected_segment()
        if segment is None:
            return False
        if segment["type"] == "goal":
            raise SemanticRouteError("路线必须保留唯一 Goal。")
        self._begin_change()
        self.document["segments"].pop(self.selected_index)
        self._changed()
        return True

    def split_selected(self, point_index=None):
        segment = self.selected_segment()
        if segment is None or segment["type"] != "walk":
            raise SemanticRouteError("只能分割行走线段。")
        points = segment["points"]
        point_index = len(points) // 2 if point_index is None else int(point_index)
        if not 0 < point_index < len(points) - 1:
            raise SemanticRouteError("分割点必须位于行走线段内部。")
        self._begin_change()
        first = copy.deepcopy(segment)
        second = copy.deepcopy(segment)
        first["points"] = copy.deepcopy(points[:point_index + 1])
        second["id"] = self._unique_id("walk")
        second["points"] = copy.deepcopy(points[point_index:])
        self.document["segments"][self.selected_index:self.selected_index + 1] = [
            first, second,
        ]
        self._changed()
        return True

    def merge_with_next(self):
        index = self.selected_index
        segments = self.document["segments"]
        if not 0 <= index < len(segments) - 1:
            raise SemanticRouteError("当前线段后没有可合并线段。")
        first, second = segments[index], segments[index + 1]
        if (
            first["type"] != "walk" or second["type"] != "walk"
            or first["direction"] != second["direction"]
        ):
            raise SemanticRouteError("只能合并相邻且同方向的行走线段。")
        self._begin_change()
        tail = second["points"]
        if first["points"][-1] == tail[0]:
            tail = tail[1:]
        first["points"].extend(copy.deepcopy(tail))
        segments.pop(index + 1)
        self._changed()
        return True

    def move_selected(self, delta):
        index = self.selected_index
        target = index + int(delta)
        segments = self.document["segments"]
        if not 0 <= index < len(segments) or not 0 <= target < len(segments):
            return False
        if segments[index]["type"] == "goal" or segments[target]["type"] == "goal":
            raise SemanticRouteError("Goal 必须保持在路线最后。")
        self._begin_change()
        segments[index], segments[target] = segments[target], segments[index]
        self.selected_index = target
        self._changed()
        return True

    def change_selected_type(self, kind):
        if kind not in SEGMENT_TYPES:
            raise SemanticRouteError(f"未知动作类型：{kind}。")
        old = self.selected_segment()
        if old is None or old["type"] == kind:
            return False
        if old["type"] == "goal" or kind == "goal":
            raise SemanticRouteError("Goal 不参与普通动作类型转换。")
        anchor = self._anchor(old)
        width, height = self.document["canvas_size"]
        next_x = min(width - 1, anchor[0] + 1)
        next_y = max(0, anchor[1] - 1)
        identifier = self._unique_id(kind)
        if kind == "walk":
            replacement = {"id": identifier, "type": kind, "direction": "right",
                           "points": [anchor, [next_x, anchor[1]]]}
        elif kind == "jump":
            replacement = {"id": identifier, "type": kind, "anchor": anchor,
                           "direction": "none"}
        elif kind == "drop":
            replacement = {"id": identifier, "type": kind, "anchor": anchor,
                           "landing": [anchor[0], min(height - 1, anchor[1] + 1)]}
        elif kind == "ladder":
            replacement = {
                "id": identifier, "type": kind, "approach": anchor,
                "mount": anchor, "points": [anchor, [anchor[0], next_y]],
                "mount_direction": "none",
                "exit": [anchor[0], next_y], "exit_direction": "none",
            }
        elif kind == "teleport":
            replacement = {"id": identifier, "type": kind, "anchor": anchor,
                           "direction": "right"}
        else:
            replacement = {"id": identifier, "type": "stop", "position": anchor}
        self._begin_change()
        self.document["segments"][self.selected_index] = replacement
        self._changed()
        return True

    def set_exit_direction(self, direction):
        segment = self.selected_segment()
        if segment is None or segment["type"] != "ladder":
            raise SemanticRouteError("只有攀爬线段可设置离梯方向。")
        if direction not in {"left", "right", "none"}:
            raise SemanticRouteError("离梯方向无效。")
        self._begin_change()
        segment["exit_direction"] = direction
        self._changed()

    def ladder_center_diagnostics(self):
        segment = self.selected_segment()
        if segment is None or segment.get("type") != "ladder":
            return None
        calculated = calculate_ladder_mount_x(segment)
        recorded = int(segment["mount"][0])
        return {
            "recorded": recorded,
            "calculated": calculated,
            "difference": None if calculated is None else calculated - recorded,
        }

    def recalculate_selected_ladder_mount(self):
        segment = self.selected_segment()
        if segment is None or segment.get("type") != "ladder":
            raise SemanticRouteError("只有攀爬线段可重算梯子中心。")
        calculated = calculate_ladder_mount_x(segment)
        if calculated is None:
            raise SemanticRouteError("当前攀爬线段没有足够的稳定攀爬样本。")
        if int(segment["mount"][0]) == calculated:
            return False
        self._begin_change()
        segment["mount"][0] = calculated
        self._changed()
        return True

    def set_direction(self, direction):
        segment = self.selected_segment()
        if segment is None:
            return False
        allowed = {
            "walk": {"left", "right"},
            "jump": {"left", "right", "none"},
            "teleport": {"left", "right", "up", "down"},
            "ladder": {"left", "right", "none"},
        }.get(segment["type"])
        if allowed is None:
            raise SemanticRouteError("当前动作没有可编辑的主方向。")
        if direction not in allowed:
            raise SemanticRouteError(
                f"{SEGMENT_CN[segment['type']]} 不支持方向 {direction}。"
            )
        field = "mount_direction" if segment["type"] == "ladder" else "direction"
        if segment.get(field, "none") == direction:
            return False
        self._begin_change()
        segment[field] = direction
        self._changed()
        return True

    def move_control(self, field, point_index, point):
        segment = self.selected_segment()
        if segment is None:
            return False
        x, y = int(round(point[0])), int(round(point[1]))
        width, height = self.document["canvas_size"]
        x, y = max(0, min(width - 1, x)), max(0, min(height - 1, y))
        if field == "points":
            target = segment.get("points")
            if target is None or not 0 <= point_index < len(target):
                return False
            new_value = [x, y]
            if target[point_index] == new_value:
                return False
            self._begin_change()
            target[point_index] = new_value
        else:
            if field not in segment:
                return False
            new_value = [x, y]
            if segment[field] == new_value:
                return False
            self._begin_change()
            segment[field] = new_value
        self._changed()
        return True

    def validate(self, *, finalize_draft=False):
        candidate = copy.deepcopy(self.document)
        candidate.pop("draft", None)
        validated = validate_route_document(
            candidate,
            expected_map_id=self.map_id,
            expected_canvas_size=(self.base_bgr.shape[1], self.base_bgr.shape[0]),
            expected_route_index=self.route_index,
        )
        if finalize_draft and self.document.get("draft"):
            self._begin_change()
            self.document = validated
            self._changed()
        return validated

    def save(self, directory, command_colors):
        if self.document.get("draft"):
            raise SemanticRouteError("草稿必须先在 Route Studio 中校正并点击验证。")
        document = self.validate()
        paths = save_route_pair_atomic(directory, document, self.base_bgr, command_colors)
        self.document.pop("draft", None)
        self.dirty = False
        return paths


class _ControlHandle(QGraphicsEllipseItem):
    def __init__(self, x, y, field, point_index, callback):
        super().__init__(-3, -3, 6, 6)
        self.field = field
        self.point_index = point_index
        self.callback = callback
        self.setPos(float(x), float(y))
        self.setBrush(QBrush(QColor("#f8fafc")))
        self.setPen(QPen(QColor("#22d3ee"), 1))
        self.setZValue(10)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        field = self.field
        point_index = self.point_index
        point = (self.pos().x(), self.pos().y())
        # The callback rebuilds the scene, so defer it until this graphics
        # item's mouse event has fully returned.
        QTimer.singleShot(0, lambda: self.callback(field, point_index, point))


class RouteCanvas(QGraphicsView):
    control_moved = Signal(str, int, object)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#050a12"))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(self.renderHints())
        self._pixmap_item = None
        self._handles = []

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def set_image(self, image):
        height, width = image.shape[:2]
        qimage = QImage(
            image.data, width, height, int(image.strides[0]), QImage.Format_BGR888
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        scene = self.scene()
        scene.clear()
        self._handles = []
        self._pixmap_item = scene.addPixmap(pixmap)
        scene.setSceneRect(0, 0, width, height)

    def set_controls(self, controls):
        for field, index, point in controls:
            handle = _ControlHandle(
                point[0], point[1], field, index,
                lambda f, i, p: self.control_moved.emit(f, i, p),
            )
            self.scene().addItem(handle)
            self._handles.append(handle)

    def set_player(self, point):
        if point is None:
            return
        item = self.scene().addEllipse(
            point[0] - 3, point[1] - 3, 6, 6,
            QPen(QColor("#fde047"), 1), QBrush(QColor("#fde047")),
        )
        item.setZValue(20)

    def set_ladder_diagnostics(self, segment, calculated_x):
        if segment is None or segment.get("type") != "ladder" or calculated_x is None:
            return
        y_values = [int(point[1]) for point in segment.get("points", [])]
        if not y_values:
            return
        item = self.scene().addLine(
            int(calculated_x), min(y_values), int(calculated_x), max(y_values),
            QPen(QColor("#22d3ee"), 2, Qt.DashLine),
        )
        item.setZValue(8)


class MinimapRoiCanvas(QGraphicsView):
    """A frozen-frame canvas whose scene coordinates equal source pixels."""

    selection_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#020617"))
        self.setDragMode(QGraphicsView.NoDrag)
        self._image_size = None
        self._drag_start = None
        self._current_item = None
        self._candidate_item = None
        self._player_item = None

    def set_image(self, image):
        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        qimage = QImage(
            image.data, width, height, int(image.strides[0]), QImage.Format_BGR888
        ).copy()
        scene = self.scene()
        scene.clear()
        scene.addPixmap(QPixmap.fromImage(qimage))
        scene.setSceneRect(0, 0, width, height)
        self._image_size = (width, height)
        self._current_item = None
        self._candidate_item = None
        self._player_item = None
        self._fit_image()

    def _fit_image(self):
        if self._image_size is not None:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_image()

    def _set_rect_item(self, attr, rect, color):
        item = getattr(self, attr)
        if item is not None:
            self.scene().removeItem(item)
        if rect is None:
            setattr(self, attr, None)
            return
        x, y, width, height = rect
        item = self.scene().addRect(
            float(x), float(y), float(width), float(height),
            QPen(QColor(color), 2),
        )
        item.setZValue(10)
        setattr(self, attr, item)

    def set_current_rect(self, rect):
        self._set_rect_item("_current_item", rect, "#22d3ee")

    def set_candidate_rect(self, rect):
        self._set_rect_item("_candidate_item", rect, "#fb923c")

    def set_player(self, point):
        if self._player_item is not None:
            self.scene().removeItem(self._player_item)
            self._player_item = None
        if point is None:
            return
        x, y = point
        self._player_item = self.scene().addEllipse(
            x - 4, y - 4, 8, 8,
            QPen(QColor("#22c55e"), 2), QBrush(Qt.NoBrush),
        )
        self._player_item.setZValue(20)

    def _scene_point(self, event):
        point = self.mapToScene(event.position().toPoint())
        rect = self.sceneRect()
        return QPointF(
            min(max(point.x(), rect.left()), rect.right()),
            min(max(point.y(), rect.top()), rect.bottom()),
        )

    @staticmethod
    def _integer_rect(rect):
        left = int(math.floor(rect.left()))
        top = int(math.floor(rect.top()))
        right = int(math.ceil(rect.right()))
        bottom = int(math.ceil(rect.bottom()))
        return left, top, max(0, right - left), max(0, bottom - top)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._image_size is not None:
            self._drag_start = self._scene_point(event)
            self.set_candidate_rect((self._drag_start.x(), self._drag_start.y(), 0, 0))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            rect = QRectF(self._drag_start, self._scene_point(event)).normalized()
            self.set_candidate_rect(self._integer_rect(rect))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_start is not None:
            rect = QRectF(self._drag_start, self._scene_point(event)).normalized()
            self._drag_start = None
            pixel_rect = self._integer_rect(rect)
            self.set_candidate_rect(pixel_rect)
            self.selection_changed.emit(pixel_rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MinimapRoiCalibrationDialog(QDialog):
    """Freeze one prepared game frame and return a normalized ROI."""

    def __init__(self, *, cfg, frame_provider, current_roi=None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.frame_provider = frame_provider
        self.current_roi = current_roi
        self.frame = None
        self.candidate_rect = None
        self.player = None
        self.restore_requested = False
        self.setWindowTitle("小地图 ROI 校准")
        self.resize(1100, 760)

        layout = QVBoxLayout(self)
        help_label = QLabel(
            "在冻结的游戏画面上拖框选择小地图。青色为当前区域，橙色为新区域，"
            "绿色圆圈为识别到的人物点。"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        self.canvas = MinimapRoiCanvas()
        self.canvas.selection_changed.connect(self._selection_changed)
        layout.addWidget(self.canvas, 1)

        options = QHBoxLayout()
        self.snap_checkbox = QCheckBox("自动吸附小地图白色边框")
        self.snap_checkbox.setChecked(True)
        options.addWidget(self.snap_checkbox)
        refresh = QPushButton("刷新截图")
        refresh.clicked.connect(self.refresh_frame)
        options.addWidget(refresh)
        restore = QPushButton("恢复使用全局 ROI")
        restore.clicked.connect(self._restore_global)
        options.addWidget(restore)
        options.addStretch()
        layout.addLayout(options)

        self.coordinates = QLabel("尚未框选")
        self.status = QLabel("正在获取游戏画面……")
        self.status.setWordWrap(True)
        layout.addWidget(self.coordinates)
        layout.addWidget(self.status)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("应用")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._accept_candidate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        QTimer.singleShot(0, self.refresh_frame)

    def refresh_frame(self):
        frame = self.frame_provider()
        if frame is None or frame.size == 0:
            self.status.setText("暂未取得游戏画面，请确认游戏窗口已打开后点击“刷新截图”。")
            return
        self.frame = np.ascontiguousarray(frame.copy())
        self.candidate_rect = None
        self.player = None
        self.canvas.set_image(self.frame)
        if self.current_roi:
            try:
                height, width = self.frame.shape[:2]
                self.canvas.set_current_rect(
                    normalized_roi_to_pixels(self.current_roi, width, height)
                )
            except MapProjectConfigError as exc:
                self.status.setText(f"当前 ROI 无效：{exc}")
                return
        self.coordinates.setText("尚未框选")
        self.status.setText("画面已冻结，请拖框选择小地图。")

    def _selection_changed(self, raw_rect):
        if self.frame is None:
            return
        x, y, width, height = (int(value) for value in raw_rect)
        warning = ""
        if width <= 0 or height <= 0:
            self.candidate_rect = raw_rect
            self.player = None
            self.status.setText("框选区域为空。")
            return
        if self.snap_checkbox.isChecked():
            crop = self.frame[y:y + height, x:x + width]
            snapped = get_minimap_loc_size(crop, None) if crop.size else None
            if snapped is not None:
                sx, sy, sw, sh = snapped
                x, y, width, height = x + sx, y + sy, sw, sh
            else:
                warning = "未识别到完整白色边框，已保留原始框选。"
        self.candidate_rect = (x, y, width, height)
        self.canvas.set_candidate_rect(self.candidate_rect)
        crop = self.frame[y:y + height, x:x + width]
        self.player = get_player_location_on_minimap(
            crop,
            minimap_player_color=self.cfg["minimap"]["player_color"],
            player_hsv=self.cfg["minimap"].get("player_hsv"),
        ) if crop.size else None
        absolute_player = (
            (x + self.player[0], y + self.player[1]) if self.player is not None else None
        )
        self.canvas.set_player(absolute_player)
        frame_height, frame_width = self.frame.shape[:2]
        try:
            normalized = pixels_to_normalized_roi(
                self.candidate_rect, frame_width, frame_height
            )
            self.coordinates.setText(
                f"像素：({x}, {y}, {width}, {height})    "
                f"归一化：[{', '.join(f'{value:.10f}' for value in normalized)}]"
            )
        except MapProjectConfigError as exc:
            self.coordinates.setText(str(exc))
        player_status = "已识别人物点。" if self.player is not None else "未识别到人物点。"
        self.status.setText(" ".join(item for item in (warning, player_status) if item))

    def _accept_candidate(self):
        if self.frame is None or self.candidate_rect is None:
            QMessageBox.warning(self, "小地图 ROI 校准", "请先在画面上框选小地图。")
            return
        if self.candidate_rect[2] < 20 or self.candidate_rect[3] < 20:
            QMessageBox.warning(self, "小地图 ROI 校准", "小地图 ROI 至少需要 20×20 px。")
            return
        if self.player is None:
            result = QMessageBox.question(
                self, "未识别人物点",
                "候选区域内未识别到人物黄点，可能还需要校正人物颜色。仍要采用吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return
        self.accept()

    def _restore_global(self):
        self.restore_requested = True
        self.accept()

    def selected_roi(self):
        if self.restore_requested:
            return None
        if self.frame is None or self.candidate_rect is None:
            return None
        height, width = self.frame.shape[:2]
        return pixels_to_normalized_roi(self.candidate_rect, width, height)


class RouteStudioWindow(QMainWindow):
    """One window shared by the main UI entry and standalone command."""

    studio_closed = Signal()
    project_saved = Signal(str)

    def __init__(self, *, cfg, initial_map="", project_root="minimaps", parent=None):
        super().__init__(parent)
        self.route_control_lock = RouteControlLock("route_studio")
        self.cfg = copy.deepcopy(cfg)
        self.global_minimap_roi = copy.deepcopy(
            self.cfg.get("minimap", {}).get("roi")
        )
        self.project_root = Path(project_root)
        self.model = None
        self.session = None
        self.local_replace_index = None
        self.active_map_id = ""
        self.map_roi_override = None
        self.map_config_dirty = False
        self.map_config_loaded_for = ""
        self.calibration_dialog_active = False
        self._updating = False
        self.setWindowTitle("路线录制与编辑")
        self.resize(1280, 820)
        self.setMinimumSize(980, 680)
        self.setStyleSheet(ROUTE_STUDIO_QSS)
        self._build_ui(initial_map)
        self.route_control_lock.acquire()
        self.timer = QTimer(self)
        self.timer.setInterval(round(1000 / RUNTIME_POLICY.route_recorder_fps))
        self.timer.timeout.connect(self._tick_session)
        self.timer.start()
        QShortcut(QKeySequence("F1"), self, activated=self.toggle_route_recording)
        QShortcut(QKeySequence("F3"), self, activated=self.finish_goal)
        QShortcut(QKeySequence("F4"), self, activated=self.save_project)

    @property
    def dirty(self):
        return bool(
            self.map_config_dirty
            or (self.model and self.model.dirty)
            or (
                self.session is not None
                and (
                    self.session.map_dirty
                    or (self.session.recorder is not None and self.session.recorder.dirty)
                )
            )
        )

    def _button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def _build_ui(self, initial_map):
        root = QWidget()
        layout = QVBoxLayout(root)
        project_bar = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("现有地图（锁定底图）", "existing")
        self.mode_combo.addItem("新地图（允许捕获底图）", "new")
        self.map_combo = QComboBox()
        self.map_combo.setEditable(True)
        if self.project_root.exists():
            for path in sorted(self.project_root.iterdir()):
                if path.is_dir() and not path.name.startswith("."):
                    self.map_combo.addItem(path.name, path.name)
        if initial_map:
            index = self.map_combo.findText(initial_map)
            if index >= 0:
                self.map_combo.setCurrentIndex(index)
            else:
                self.map_combo.setEditText(initial_map)
        self.route_combo = QComboBox()
        self.route_combo.setMinimumWidth(100)
        project_bar.addWidget(QLabel("项目："))
        project_bar.addWidget(self.mode_combo)
        project_bar.addWidget(QLabel("地图："))
        project_bar.addWidget(self.map_combo, 1)
        project_bar.addWidget(QLabel("路线："))
        project_bar.addWidget(self.route_combo)
        project_bar.addWidget(self._button("加载", self.load_project))
        project_bar.addWidget(self._button("新增路线", self.new_route))
        layout.addLayout(project_bar)

        capture_bar = QHBoxLayout()
        self.start_map_button = self._button("开始地图捕获", self.start_map_capture)
        capture_bar.addWidget(self.start_map_button)
        capture_bar.addWidget(self._button("停止捕获", self.stop_capture))
        capture_bar.addWidget(self._button(
            "放弃捕获/录制数据", self.discard_pending_capture_data
        ))
        capture_bar.addWidget(self._button("校准小地图 ROI", self.calibrate_minimap_roi))
        self.record_button = self._button("开始路线录制 (F1)", self.toggle_route_recording)
        capture_bar.addWidget(self.record_button)
        capture_bar.addWidget(self._button("完成并标记 Goal (F3)", self.finish_goal))
        capture_bar.addWidget(self._button("撤销", self.undo))
        capture_bar.addWidget(self._button("重做", self.redo))
        capture_bar.addStretch()
        layout.addLayout(capture_bar)

        splitter = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.segment_list = QListWidget()
        self.segment_list.currentRowChanged.connect(self._select_segment)
        left_layout.addWidget(QLabel("有序线段"))
        left_layout.addWidget(self.segment_list, 1)
        edit_box = QGroupBox("线段编辑")
        edit_form = QFormLayout(edit_box)
        self.type_combo = QComboBox()
        for kind in ("walk", "jump", "drop", "ladder", "teleport", "stop", "goal"):
            self.type_combo.addItem(SEGMENT_CN[kind], kind)
        self.type_combo.currentIndexChanged.connect(self.change_type)
        self.direction_combo = QComboBox()
        for label, value in (
            ("无", "none"), ("左", "left"), ("右", "right"),
            ("上", "up"), ("下", "down"),
        ):
            self.direction_combo.addItem(label, value)
        self.direction_combo.currentIndexChanged.connect(self.change_direction)
        self.exit_combo = QComboBox()
        for label, value in (("无", "none"), ("左", "left"), ("右", "right")):
            self.exit_combo.addItem(label, value)
        self.exit_combo.currentIndexChanged.connect(self.change_exit_direction)
        edit_form.addRow("动作类型", self.type_combo)
        self.direction_label = QLabel("动作方向")
        edit_form.addRow(self.direction_label, self.direction_combo)
        edit_form.addRow("离梯方向", self.exit_combo)
        self.ladder_center_label = QLabel("梯子中心：—")
        self.recalculate_ladder_button = self._button(
            "根据攀爬样本重新计算中心", self.recalculate_ladder_center
        )
        edit_form.addRow(self.ladder_center_label)
        edit_form.addRow(self.recalculate_ladder_button)
        row = QHBoxLayout()
        row.addWidget(self._button("删除", self.delete_segment))
        row.addWidget(self._button("分割", self.split_segment))
        row.addWidget(self._button("合并后续", self.merge_segment))
        edit_form.addRow(row)
        order_row = QHBoxLayout()
        order_row.addWidget(self._button("上移", lambda: self.move_segment(-1)))
        order_row.addWidget(self._button("下移", lambda: self.move_segment(1)))
        edit_form.addRow("顺序", order_row)
        edit_form.addRow(self._button("局部重录当前段", self.start_local_rerecord))
        left_layout.addWidget(edit_box)
        splitter.addWidget(left)

        self.canvas = RouteCanvas()
        self.canvas.control_moved.connect(self.move_control)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])
        layout.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        bottom.addWidget(self._button("转换旧 PNG 为草稿", self.convert_legacy))
        bottom.addWidget(self._button("验证", self.validate_project))
        bottom.addWidget(self._button("保存 (F4)", self.save_project))
        bottom.addWidget(self._button("另存路线", self.save_as_route))
        bottom.addStretch()
        self.status = QLabel("请加载地图项目")
        bottom.addWidget(self.status)
        layout.addLayout(bottom)
        self.setCentralWidget(root)
        self.map_combo.currentTextChanged.connect(self._refresh_routes)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self._refresh_routes()
        self._mode_changed()

    def _project_id(self):
        return self.map_combo.currentText().strip()

    def _project_dir(self):
        return self.project_root / self._project_id()

    def _active_project_dir(self):
        return self.project_root / (self.active_map_id or self._project_id())

    def _confirm_replace_project(self):
        if not self.dirty:
            return True
        result = QMessageBox.question(
            self, "未保存项目", "当前录制或编辑尚未保存，是否保存？",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if result == QMessageBox.Cancel:
            return False
        if result == QMessageBox.Save:
            return self.save_project()
        return True

    def _mode_changed(self):
        new_map = self.mode_combo.currentData() == "new"
        self.start_map_button.setEnabled(new_map)
        self.start_map_button.setToolTip(
            "只有新地图允许更新 map.png。" if not new_map else ""
        )

    def _refresh_routes(self):
        current = self.route_combo.currentData()
        self.route_combo.clear()
        directory = self._project_dir()
        indices = set()
        if directory.exists():
            for path in directory.glob("route*.png"):
                suffix = path.stem[5:]
                if suffix.isdigit():
                    indices.add(int(suffix))
            for path in directory.glob("route*.json"):
                suffix = path.stem[5:]
                if suffix.isdigit():
                    indices.add(int(suffix))
        for index in sorted(indices):
            self.route_combo.addItem(f"route{index}", index)
        next_index = max(indices, default=0) + 1
        self.route_combo.addItem(f"route{next_index}（新）", next_index)
        if current is not None:
            found = self.route_combo.findData(current)
            if found >= 0:
                self.route_combo.setCurrentIndex(found)

    def _load_base(self):
        path = self._project_dir() / "map.png"
        if not path.exists():
            raise SemanticRouteError(f"地图底图不存在：{path}")
        return load_image(str(path))

    def _apply_effective_map_roi(self):
        effective = (
            self.map_roi_override
            if self.map_roi_override is not None
            else self.global_minimap_roi
        )
        minimap = self.cfg.setdefault("minimap", {})
        if effective is None:
            minimap.pop("roi", None)
        else:
            minimap["roi"] = list(effective)
        if self.session is not None:
            self.session.apply_minimap_roi(effective)
        return effective

    def _load_map_config(self, map_id):
        map_id = str(map_id).strip()
        if not map_id:
            raise MapProjectConfigError("地图名不能为空。")
        self.map_roi_override = load_map_minimap_roi(self.project_root / map_id)
        self.map_config_loaded_for = map_id
        self.map_config_dirty = False
        self._apply_effective_map_roi()

    def _ensure_map_config(self, map_id):
        if self.map_config_loaded_for != map_id:
            self._load_map_config(map_id)

    def load_project(self):
        try:
            if not self._confirm_replace_project():
                return
            map_id = self._project_id()
            if not map_id:
                raise SemanticRouteError("地图名不能为空。")
            self._load_map_config(map_id)
            base = self._load_base()
            index = int(self.route_combo.currentData() or 1)
            json_path = self._project_dir() / f"route{index}.json"
            if json_path.exists():
                self.model = SemanticRouteEditorModel.load(json_path, base)
            else:
                self.model = SemanticRouteEditorModel(
                    map_id=map_id, route_index=index, base_bgr=base
                )
            self.active_map_id = map_id
            self._replace_session(base, new_map=False)
            self._refresh_editor()
            self.status.setText(f"已加载 {map_id} / route{index}")
        except Exception as exc:
            self._error(exc)

    def new_route(self):
        try:
            if not self._confirm_replace_project():
                return
            base = self._load_base()
            self._load_map_config(self._project_id())
            directory = self._project_dir()
            indices = [
                int(path.stem[5:]) for path in directory.glob("route*.png")
                if path.stem[5:].isdigit()
            ]
            index = max(indices, default=0) + 1
            self.model = SemanticRouteEditorModel(
                map_id=self._project_id(), route_index=index, base_bgr=base
            )
            self.active_map_id = self._project_id()
            self._replace_session(base, new_map=False)
            self._refresh_editor()
            self.status.setText(f"已新建 route{index}，底图已锁定")
        except Exception as exc:
            self._error(exc)

    def _replace_session(self, base, *, new_map):
        self.stop_capture()
        self.session = RouteStudioSession(
            self.cfg,
            map_id=self.active_map_id or self._project_id(),
            base_bgr=base,
            new_map=new_map,
        )

    def _confirm_new_map_capture_target(self, directory):
        """Allow an explicit bottom-map recapture only before routes exist."""
        directory = Path(directory)
        if not directory.exists():
            return True
        route_assets = sorted(
            path.name for path in directory.iterdir()
            if path.is_file()
            and path.name.startswith("route")
            and path.suffix.lower() in {".png", ".json"}
        )
        if route_assets:
            raise RouteStudioSessionError(
                "当前地图已经存在路线文件，重新捕获底图会使路线坐标失效。"
                "请使用新的地图名，或先人工备份并处理现有路线。"
            )
        allowed = {MAP_CONFIG_FILENAME, "map.png"}
        unexpected = sorted(
            path.name for path in directory.iterdir()
            if path.name not in allowed
        )
        if unexpected:
            raise RouteStudioSessionError(
                "新地图目录包含不能自动覆盖的文件："
                + "、".join(unexpected)
            )
        if (directory / "map.png").exists():
            result = QMessageBox.question(
                self, "重新捕获地图底图",
                "当前地图已有 map.png。继续后将从空白底图重新捕获，"
                "只有点击保存/F4 时才会替换原文件。是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            return result == QMessageBox.Yes
        return True

    def start_map_capture(self):
        try:
            map_id = self._project_id()
            if not map_id:
                raise RouteStudioSessionError("请输入新地图名。")
            continuing_calibrated_new_map = bool(
                self.active_map_id == map_id
                and self.model is None
                and self.session is not None
                and self.session.new_map
                and not self.session.map_dirty
            )
            if (
                not continuing_calibrated_new_map
                and not self._confirm_replace_project()
            ):
                return
            if self.mode_combo.currentData() != "new":
                raise RouteStudioSessionError("现有地图的 map.png 已锁定。")
            directory = self._project_dir()
            if not self._confirm_new_map_capture_target(directory):
                return
            self._ensure_map_config(map_id)
            self.active_map_id = map_id
            self.model = None
            self._replace_session(None, new_map=True)
            self.session.begin_map_capture()
            self.status.setText("新地图捕获中，请在游戏前台移动探索")
        except Exception as exc:
            self._error(exc)

    def stop_capture(self):
        if self.session is not None:
            self.session.stop()
        self.record_button.setText("开始路线录制 (F1)")

    def _confirm_discard_pending_data(self):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("放弃未保存数据")
        box.setText(
            "将放弃尚未保存的地图捕获和未完成的路线录制数据。"
            "磁盘上已经保存的 map.png、routeN.json 和 routeN.png 不会删除。"
        )
        box.setStandardButtons(QMessageBox.Discard | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Cancel)
        discard_button = box.button(QMessageBox.Discard)
        cancel_button = box.button(QMessageBox.Cancel)
        if discard_button is not None:
            discard_button.setText("放弃未保存数据")
        if cancel_button is not None:
            cancel_button.setText("取消")
        return box.exec() == QMessageBox.Discard

    def discard_pending_capture_data(self, *, confirm=True):
        """Discard in-memory capture/recording state, never saved assets."""
        if self.session is None:
            self.status.setText("当前没有可放弃的捕获或录制数据")
            return False
        recorder_dirty = bool(
            self.session.recorder is not None and self.session.recorder.dirty
        )
        has_pending = bool(
            self.session.map_dirty
            or recorder_dirty
            or self.session.map_capture_active
            or self.session.route_recording_active
        )
        if not has_pending:
            self.status.setText("当前没有可放弃的捕获或录制数据")
            return False
        if confirm and not self._confirm_discard_pending_data():
            return False

        had_map_capture = bool(self.session.map_dirty)
        self.session.stop()
        if recorder_dirty or self.session.recorder is not None:
            self.session.discard_route_recording()
            self.local_replace_index = None
        if had_map_capture:
            saved_path = self._active_project_dir() / "map.png"
            restored = load_image(str(saved_path)) if saved_path.is_file() else None
            self.session.discard_map_capture(restored)
            if restored is None:
                self.model = None
                self.segment_list.clear()
                self.canvas.scene().clear()
            else:
                self.model = SemanticRouteEditorModel(
                    map_id=self.active_map_id or self._project_id(),
                    route_index=int(self.route_combo.currentData() or 1),
                    base_bgr=restored,
                )
                self._refresh_editor()
        elif self.model is not None:
            self._refresh_editor()
        self.record_button.setText("开始路线录制 (F1)")
        self.status.setText("已放弃未保存的捕获/录制数据，可重新校准 ROI")
        return True

    def toggle_route_recording(self):
        try:
            if self.session is None:
                if self.model is None:
                    self.load_project()
                if self.model is None:
                    return
                self._replace_session(self.model.base_bgr, new_map=False)
            if self.session.route_recording_active:
                self.session.pause_route_recording()
                self.record_button.setText("继续路线录制 (F1)")
                self.status.setText("路线录制已暂停")
            else:
                if self.session.map_capture_active:
                    self.session.end_map_capture()
                    if self.session.base_bgr is None:
                        raise RouteStudioSessionError("尚未捕获到有效 map.png。")
                    route_index = (
                        self.model.route_index if self.model is not None
                        else int(self.route_combo.currentData() or 1)
                    )
                    self.model = SemanticRouteEditorModel(
                        map_id=self.active_map_id,
                        route_index=route_index,
                        base_bgr=self.session.base_bgr,
                    )
                    self._refresh_editor()
                if self.session.recorder is None:
                    index = self.model.route_index if self.model else int(self.route_combo.currentData())
                    self.session.begin_route_recording(index)
                else:
                    self.session.start()
                    self.session.route_recording_active = True
                self.record_button.setText("暂停路线录制 (F1)")
                self.status.setText("录制中；只接受游戏前台动作键")
        except Exception as exc:
            self._error(exc)

    def _calibration_is_blocked(self):
        if self.session is None:
            return False
        if self.session.map_capture_active or self.session.route_recording_active:
            self._error(RouteStudioSessionError("请先停止地图捕获或路线录制，再校准 ROI。"))
            return True
        if self.session.map_dirty:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("未保存的地图捕获数据")
            box.setText("存在未保存的地图捕获数据，如何处理后再校准 ROI？")
            box.setStandardButtons(
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            box.setDefaultButton(QMessageBox.Cancel)
            labels = (
                (QMessageBox.Save, "保存并继续"),
                (QMessageBox.Discard, "放弃捕获数据"),
                (QMessageBox.Cancel, "取消"),
            )
            for standard, label in labels:
                button = box.button(standard)
                if button is not None:
                    button.setText(label)
            result = box.exec()
            if result == QMessageBox.Save:
                return not self.save_project()
            if result == QMessageBox.Discard:
                return not self.discard_pending_capture_data(confirm=False)
            return True
        if self.session.recorder is not None and self.session.recorder.dirty:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("未完成的路线录制")
            box.setText("路线录制尚未完成。请按 F3 完成，或放弃本次录制后再校准 ROI。")
            box.setStandardButtons(QMessageBox.Discard | QMessageBox.Cancel)
            box.setDefaultButton(QMessageBox.Cancel)
            discard_button = box.button(QMessageBox.Discard)
            cancel_button = box.button(QMessageBox.Cancel)
            if discard_button is not None:
                discard_button.setText("放弃本次录制")
            if cancel_button is not None:
                cancel_button.setText("取消")
            if box.exec() == QMessageBox.Discard:
                return not self.discard_pending_capture_data(confirm=False)
            return True
        return False

    def _ensure_calibration_session(self, map_id):
        if self.session is not None and self.session.map_id == map_id:
            return
        base = None
        new_map = self.mode_combo.currentData() == "new"
        if not new_map:
            base = self._load_base()
        self.active_map_id = map_id
        self._replace_session(base, new_map=new_map)

    def _candidate_localization_score(self, frame, roi):
        if self.model is None and not (self._project_dir() / "map.png").exists():
            return None
        base = self.model.base_bgr if self.model is not None else self._load_base()
        frame_height, frame_width = frame.shape[:2]
        x, y, width, height = normalized_roi_to_pixels(
            roi, frame_width, frame_height
        )
        minimap = np.ascontiguousarray(frame[y:y + height, x:x + width].copy())
        player = get_player_location_on_minimap(
            minimap,
            minimap_player_color=self.cfg["minimap"]["player_color"],
            player_hsv=self.cfg["minimap"].get("player_hsv"),
        )
        if player is not None:
            minimap = RouteStudioSession._hide_player_dot(minimap, player)
        mask = RouteStudioSession._route_mask(minimap)
        try:
            _location, score, _cached = find_pattern_sqdiff(
                base, minimap, mask=mask
            )
        except (cv2.error, ValueError):
            return float("inf")
        return float(score)

    def calibrate_minimap_roi(self):
        try:
            map_id = self._project_id()
            if not map_id:
                raise RouteStudioSessionError("请先输入地图名。")
            if (
                self.active_map_id
                and self.active_map_id != map_id
                and not self._confirm_replace_project()
            ):
                return
            if self._calibration_is_blocked():
                return
            self._ensure_map_config(map_id)
            self._ensure_calibration_session(map_id)
            effective = self._apply_effective_map_roi()
            session_was_active = self.session.active
            dialog = MinimapRoiCalibrationDialog(
                cfg=self.cfg,
                frame_provider=self.session.capture_calibration_frame,
                current_roi=effective,
                parent=self,
            )
            self.calibration_dialog_active = True
            try:
                if dialog.exec() != QDialog.Accepted:
                    return
            finally:
                self.calibration_dialog_active = False
                if (
                    not session_was_active
                    and self.session is not None
                    and not self.session.map_capture_active
                    and not self.session.route_recording_active
                ):
                    self.session.stop()
                    if (
                        self.mode_combo.currentData() == "existing"
                        and self.model is None
                    ):
                        self.session = None
            candidate = dialog.selected_roi()
            if not dialog.restore_requested and dialog.frame is not None:
                score = self._candidate_localization_score(dialog.frame, candidate)
                threshold = float(
                    self.cfg.get("route", {}).get("localization_max_score", 0.20)
                )
                if score is not None and score > threshold:
                    shown = "无法匹配" if not math.isfinite(score) else f"{score:.4f}"
                    result = QMessageBox.question(
                        self, "地图定位警告",
                        f"新 ROI 与现有 map.png 的匹配分数为 {shown}，超过阈值 "
                        f"{threshold:.4f}，可能需要重新捕获地图。仍要采用吗？",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                    )
                    if result != QMessageBox.Yes:
                        return
            if candidate == self.map_roi_override:
                self.status.setText("小地图 ROI 未发生变化")
                return
            self.map_roi_override = candidate
            self.map_config_dirty = True
            self._apply_effective_map_roi()
            self.status.setText(
                "已恢复使用全局小地图 ROI，保存/F4 后写入地图配置"
                if candidate is None else
                "已应用地图独立小地图 ROI，保存/F4 后写入地图配置"
            )
        except Exception as exc:
            self._error(exc)

    def finish_goal(self):
        try:
            if self.session is None or self.session.recorder is None:
                raise SemanticRouteError("当前没有录制中的路线。")
            if self.session.last_position is None:
                raise SemanticRouteError("当前人物位置未确认，不能标记 Goal。")
            document = self.session.recorder.finish_goal(self.session.last_position)
            if self.local_replace_index is not None:
                replacement = [
                    segment for segment in document["segments"]
                    if segment["type"] != "goal"
                ]
                if not replacement:
                    raise SemanticRouteError("局部重录没有生成任何动作线段。")
                merged = copy.deepcopy(self.model.document)
                merged.pop("draft", None)
                index = self.local_replace_index
                merged["segments"][index:index + 1] = replacement
                # Reassign deterministic unique IDs after splicing.
                for serial, segment in enumerate(merged["segments"], 1):
                    segment["id"] = f"{segment['type']}_{serial:03d}"
                document = validate_route_document(merged)
                self.local_replace_index = None
            if self.model is None:
                self.model = SemanticRouteEditorModel(
                    map_id=self.session.map_id, route_index=document["route_index"],
                    base_bgr=self.session.base_bgr, document=document,
                )
                self.model.dirty = True
            else:
                self.model.replace_document(document)
                self.model.base_bgr = self.session.base_bgr.copy()
            self.session.pause_route_recording()
            self.record_button.setText("开始路线录制 (F1)")
            self._refresh_editor()
            self.status.setText("已完成录制并标记 Goal，请验证后保存")
        except Exception as exc:
            self._error(exc)

    def start_local_rerecord(self):
        try:
            if self.model is None or self.model.selected_segment() is None:
                raise SemanticRouteError("请先选择要局部重录的线段。")
            if self.model.selected_segment()["type"] == "goal":
                raise SemanticRouteError("Goal 不能通过局部重录替换。")
            if self.session is not None and self.session.map_capture_active:
                raise SemanticRouteError("请先停止新地图捕获。")
            if self.session is None:
                self._replace_session(self.model.base_bgr, new_map=False)
            self.session.recorder = None
            self.session.begin_route_recording(self.model.route_index)
            self.local_replace_index = self.model.selected_index
            self.record_button.setText("暂停局部重录 (F1)")
            self.status.setText(
                f"正在局部重录第 {self.local_replace_index + 1} 段；F3 完成替换"
            )
        except Exception as exc:
            self._error(exc)

    def _tick_session(self):
        if (
            self.calibration_dialog_active
            or self.session is None
            or not self.session.active
        ):
            return
        keyboard = self.session.keyboard
        if keyboard is not None:
            if keyboard.is_pressed_func_key[0]:
                keyboard.is_pressed_func_key[0] = False
                self.toggle_route_recording()
            if keyboard.is_pressed_func_key[2]:
                keyboard.is_pressed_func_key[2] = False
                self.finish_goal()
            if keyboard.is_pressed_func_key[3]:
                keyboard.is_pressed_func_key[3] = False
                self.save_project()
        result = self.session.tick()
        self.status.setText(self.session.last_status)
        if self.session.base_bgr is not None and self.model is None:
            height, width = self.session.base_bgr.shape[:2]
            self.model = SemanticRouteEditorModel(
                map_id=self.session.map_id,
                route_index=int(self.route_combo.currentData() or 1),
                base_bgr=self.session.base_bgr,
            )
        if result is not None and self.model is not None:
            self.model.base_bgr = self.session.base_bgr.copy()
            if (
                self.session.route_recording_active
                or self.session.map_capture_active
            ):
                self._render_canvas(player=result["position"])

    def _controls(self, segment):
        controls = []
        if segment is None:
            return controls
        for field in ("anchor", "landing", "approach", "mount", "exit", "position"):
            if field in segment:
                controls.append((field, -1, segment[field]))
        for index, point in enumerate(segment.get("points", [])):
            controls.append(("points", index, point))
        return controls

    def _render_canvas(self, player=None):
        if self.model is None:
            return
        document = self.model.document
        if (
            player is not None and self.session is not None
            and self.session.route_recording_active
            and self.session.recorder is not None
        ):
            preview = copy.deepcopy(self.session.recorder)
            document = preview.finish_goal(player)
        try:
            image = render_route_png(
                self.model.base_bgr, document,
                command_colors_from_cfg(self.cfg),
            )
        except Exception:
            image = self.model.base_bgr.copy()
        self.canvas.set_image(image)
        self.canvas.set_controls(self._controls(self.model.selected_segment()))
        ladder_diag = self.model.ladder_center_diagnostics()
        if ladder_diag is not None:
            self.canvas.set_ladder_diagnostics(
                self.model.selected_segment(), ladder_diag["calculated"]
            )
        self.canvas.set_player(player)

    def _refresh_editor(self):
        if self.model is None:
            return
        self._updating = True
        self.segment_list.clear()
        for index, segment in enumerate(self.model.document["segments"]):
            self.segment_list.addItem(
                f"{index + 1:02d}  {SEGMENT_CN[segment['type']]}  {segment['id']}"
            )
        if self.model.selected_index >= 0:
            self.segment_list.setCurrentRow(self.model.selected_index)
        self._updating = False
        self._select_segment(self.model.selected_index)

    def _select_segment(self, index):
        if self.model is None or index < 0:
            return
        self.model.selected_index = index
        segment = self.model.selected_segment()
        self._updating = True
        type_index = self.type_combo.findData(segment["type"])
        self.type_combo.setCurrentIndex(type_index)
        direction = segment.get("exit_direction", "none")
        self.exit_combo.setCurrentIndex(self.exit_combo.findData(direction))
        self.exit_combo.setEnabled(segment["type"] == "ladder")
        action_direction = (
            segment.get("mount_direction", "none")
            if segment["type"] == "ladder"
            else segment.get("direction", "none")
        )
        self.direction_combo.setCurrentIndex(
            self.direction_combo.findData(action_direction)
        )
        self.direction_combo.setEnabled(
            segment["type"] in {"walk", "jump", "ladder", "teleport"}
        )
        self.direction_label.setText(
            "挂梯方向" if segment["type"] == "ladder" else "动作方向"
        )
        ladder_diag = self.model.ladder_center_diagnostics()
        self.recalculate_ladder_button.setEnabled(ladder_diag is not None)
        if ladder_diag is None:
            self.ladder_center_label.setText("梯子中心：—")
        elif ladder_diag["calculated"] is None:
            self.ladder_center_label.setText(
                f"录制中心 {ladder_diag['recorded']}；稳定攀爬样本不足"
            )
        else:
            difference = int(ladder_diag["difference"])
            warning = " ⚠ 超过 2 px" if abs(difference) > 2 else ""
            self.ladder_center_label.setText(
                f"录制中心 {ladder_diag['recorded']}；攀爬中位线 "
                f"{ladder_diag['calculated']}；差值 {difference:+d} px{warning}"
            )
        self._updating = False
        self._render_canvas(
            player=self.session.last_position if self.session is not None else None
        )

    def _edit(self, callback):
        try:
            callback()
            self._refresh_editor()
        except Exception as exc:
            self._error(exc)

    def undo(self):
        if self.model and self.model.undo():
            self._refresh_editor()

    def redo(self):
        if self.model and self.model.redo():
            self._refresh_editor()

    def delete_segment(self):
        if self.model:
            self._edit(self.model.delete_selected)

    def split_segment(self):
        if self.model:
            self._edit(self.model.split_selected)

    def merge_segment(self):
        if self.model:
            self._edit(self.model.merge_with_next)

    def move_segment(self, delta):
        if self.model:
            self._edit(lambda: self.model.move_selected(delta))

    def change_type(self):
        if not self._updating and self.model:
            self._edit(lambda: self.model.change_selected_type(self.type_combo.currentData()))

    def change_exit_direction(self):
        if not self._updating and self.model and self.model.selected_segment():
            if self.model.selected_segment()["type"] == "ladder":
                self._edit(lambda: self.model.set_exit_direction(self.exit_combo.currentData()))

    def recalculate_ladder_center(self):
        if self.model is None:
            return
        diagnostics = self.model.ladder_center_diagnostics()
        if diagnostics is None or diagnostics["calculated"] is None:
            self._error("当前攀爬线段没有足够的稳定攀爬样本。")
            return
        answer = QMessageBox.question(
            self,
            "确认重算梯子中心",
            f"将梯子中心从 {diagnostics['recorded']} 修改为 "
            f"{diagnostics['calculated']}（差值 "
            f"{int(diagnostics['difference']):+d} px）。\n是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self._edit(self.model.recalculate_selected_ladder_mount)

    def change_direction(self):
        if not self._updating and self.model and self.model.selected_segment():
            if self.model.selected_segment()["type"] in {"walk", "jump", "ladder", "teleport"}:
                self._edit(lambda: self.model.set_direction(self.direction_combo.currentData()))

    def move_control(self, field, point_index, point):
        if self.model:
            self._edit(lambda: self.model.move_control(field, point_index, point))

    def validate_project(self):
        try:
            if self.model is None:
                raise SemanticRouteError("尚未加载路线。")
            self.model.validate(finalize_draft=True)
            self.status.setText("验证通过：顺序、Goal、坐标和动作语义均有效")
            return True
        except Exception as exc:
            self._error(exc)
            return False

    def save_project(self):
        try:
            if self.model is None and not self.map_config_dirty:
                raise SemanticRouteError("尚未加载路线。")
            if self.session is not None and self.session.route_recording_active:
                raise SemanticRouteError("请先暂停或完成路线录制。")
            directory = self._active_project_dir()
            directory.mkdir(parents=True, exist_ok=True)
            saved_map_only = False
            if self.session is not None and self.session.new_map:
                if self.session.map_capture_active:
                    self.session.end_map_capture()
                if self.session.base_bgr is not None:
                    self.session.save_map_atomic(directory / "map.png")
                if (
                    self.session.recorder is None
                    and self.model is not None and not self.model.dirty
                ):
                    self.model = SemanticRouteEditorModel(
                        map_id=self._project_id(),
                        route_index=int(self.route_combo.currentData() or 1),
                        base_bgr=self.session.base_bgr,
                    )
                    self._refresh_editor()
                    saved_map_only = True
                elif self.model is not None and self.session.base_bgr is not None:
                    self.model.base_bgr = self.session.base_bgr.copy()
            saved_pair = None
            saved_map_config = False
            if self.model is not None and not saved_map_only:
                saved_pair = self.model.save(
                    directory, command_colors_from_cfg(self.cfg)
                )
            if self.map_config_dirty:
                save_map_minimap_roi_atomic(directory, self.map_roi_override)
                self.map_config_dirty = False
                saved_map_config = True
            self._refresh_routes()
            if saved_map_only:
                self.status.setText(
                    "已原子保存 map.png"
                    + (" 和地图配置" if saved_map_config else "")
                    + "；尚未生成路线"
                )
            elif saved_pair is not None:
                json_path, png_path = saved_pair
                self.status.setText(
                    f"已原子保存 {json_path.name} + {png_path.name}"
                    + (" + 地图配置" if saved_map_config else "")
                )
            else:
                self.status.setText("已原子保存地图配置")
            if (
                (directory / "map.png").is_file()
                and any(directory.glob("route*.png"))
            ):
                self.project_saved.emit(directory.name)
            return True
        except Exception as exc:
            self._error(exc)
            return False

    def save_as_route(self):
        if self.model is None:
            self._error(SemanticRouteError("尚未加载路线。"))
            return
        directory = self._active_project_dir()
        indices = [
            int(path.stem[5:]) for path in directory.glob("route*.png")
            if path.stem[5:].isdigit()
        ]
        index = max(indices, default=0) + 1
        self.model._begin_change()
        self.model.route_index = index
        self.model.document["route_index"] = index
        self.model._changed()
        self.save_project()

    def convert_legacy(self):
        try:
            if not self._confirm_replace_project():
                return
            index = int(self.route_combo.currentData() or 1)
            png_path = self._project_dir() / f"route{index}.png"
            draft_path = convert_legacy_png_to_draft(
                png_path,
                map_id=self._project_id(), route_index=index,
                command_colors=command_colors_from_cfg(self.cfg),
            )
            self._load_map_config(self._project_id())
            base = self._load_base()
            self.model = SemanticRouteEditorModel.load(draft_path, base)
            self.model.dirty = True
            self.active_map_id = self._project_id()
            self._replace_session(base, new_map=False)
            self._refresh_editor()
            self.status.setText(f"已生成 {draft_path.name}；校正并验证后才可保存正式 JSON")
        except Exception as exc:
            self._error(exc)

    def _error(self, exc):
        self.status.setText(f"错误：{exc}")
        QMessageBox.warning(self, "路线录制与编辑", str(exc))

    def closeEvent(self, event):
        if self.dirty:
            result = QMessageBox.question(
                self, "未保存路线", "当前路线已修改，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if result == QMessageBox.Cancel:
                event.ignore()
                return
            if result == QMessageBox.Save and not self.save_project():
                event.ignore()
                return
        self.stop_capture()
        self.timer.stop()
        self.route_control_lock.release()
        self.studio_closed.emit()
        event.accept()
