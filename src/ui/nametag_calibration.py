"""In-app Qt workflow for creating and validating name-tag profiles."""

from __future__ import annotations

import copy
import math
import time

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QBrush, QIcon, QImage, QKeySequence, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.engine.NameTagLocalizer import NameTagLocalizer, name_search_y_limit
from src.engine.NameTagProfileRepository import (
    NameTagProfileError,
    NameTagProfileRepository,
    StagedNameTagProfile,
    load_bgr_image,
    validate_profile_name,
)
from src.input.GameWindowCapturor import GameWindowCapturor
from src.route_control_lock import RouteControlBusyError, RouteControlLock
from src.utils.common import prepare_game_frame


class NameTagCalibrationCanvas(QGraphicsView):
    rectangle_selected = Signal(tuple)
    point_selected = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#020617"))
        self.setDragMode(QGraphicsView.NoDrag)
        self._image_size = None
        self._drag_start = None
        self._mode = "none"
        self._rect_item = None
        self._point_item = None
        self._line_item = None

    def set_mode(self, mode):
        self._mode = mode
        self._drag_start = None

    def set_image(self, image):
        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        qimage = QImage(
            image.data, width, height, int(image.strides[0]), QImage.Format_BGR888
        ).copy()
        self.scene().clear()
        self.scene().addPixmap(QPixmap.fromImage(qimage))
        self.scene().setSceneRect(0, 0, width, height)
        self._image_size = (width, height)
        self._rect_item = self._point_item = self._line_item = None
        self._fit_image()

    def _fit_image(self):
        if self._image_size:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_image()

    def set_overlay(self, rect=None, point=None, *, valid=True):
        for item in (self._rect_item, self._point_item, self._line_item):
            if item is not None:
                self.scene().removeItem(item)
        self._rect_item = self._point_item = self._line_item = None
        color = QColor("#22c55e" if valid else "#f97316")
        if rect is not None:
            x, y, width, height = rect
            self._rect_item = self.scene().addRect(
                float(x), float(y), float(width), float(height), QPen(color, 2)
            )
            self._rect_item.setZValue(10)
        if point is not None:
            x, y = point
            self._point_item = self.scene().addEllipse(
                x - 5, y - 5, 10, 10, QPen(color, 2), QBrush(Qt.NoBrush)
            )
            self._point_item.setZValue(20)
            if rect is not None:
                rx, ry, _rw, _rh = rect
                self._line_item = self.scene().addLine(
                    rx, ry, x, y, QPen(QColor("#38bdf8"), 1)
                )
                self._line_item.setZValue(15)

    def _scene_point(self, event):
        point = self.mapToScene(event.position().toPoint())
        bounds = self.sceneRect()
        return QPointF(
            min(max(point.x(), bounds.left()), bounds.right()),
            min(max(point.y(), bounds.top()), bounds.bottom()),
        )

    @staticmethod
    def _integer_rect(rect):
        left = int(math.floor(rect.left()))
        top = int(math.floor(rect.top()))
        right = int(math.ceil(rect.right()))
        bottom = int(math.ceil(rect.bottom()))
        return left, top, max(0, right - left), max(0, bottom - top)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._image_size:
            if self._mode == "point":
                point = self._scene_point(event)
                self.point_selected.emit((round(point.x()), round(point.y())))
                event.accept()
                return
            if self._mode == "rect":
                self._drag_start = self._scene_point(event)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            rect = QRectF(self._drag_start, self._scene_point(event)).normalized()
            self.set_overlay(self._integer_rect(rect), valid=False)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_start is not None:
            rect = QRectF(self._drag_start, self._scene_point(event)).normalized()
            self._drag_start = None
            result = self._integer_rect(rect)
            self.set_overlay(result, valid=False)
            self.rectangle_selected.emit(result)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class NameTagCalibrationDialog(QDialog):
    VALIDATION_FRAMES = 20
    MIN_CAPTURED_FRAMES = 10
    REQUIRED_VALID_FRAMES = 16
    MAX_CONSECUTIVE_MISSES = 3
    VALIDATION_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        *,
        cfg,
        current_profile="",
        repository=None,
        capture_factory=None,
        lock_factory=None,
        parent=None,
    ):
        super().__init__(parent)
        self.cfg = copy.deepcopy(cfg)
        self.repository = repository or NameTagProfileRepository()
        self.capture_factory = capture_factory or GameWindowCapturor
        self.control_lock = (lock_factory or RouteControlLock)("nametag_calibration")
        try:
            self.control_lock.acquire()
        except RouteControlBusyError as exc:
            raise RuntimeError(str(exc)) from exc

        self.capture = None
        self.latest_frame = None
        self.frozen_frame = None
        self.stage: StagedNameTagProfile | None = None
        self.stage_dirty = False
        self.applied_name = None
        self._updating_samples = False
        self._pending_rect = None
        self._pending_image = None
        self._pending_offset = None
        self._pending_filename = None
        self._pending_kind = None
        self._pending_sample_kind = 'full'
        self._low_quality_samples = set()
        self._validation_localizer = None
        self._pet_test = None
        self._capture_sequence = None
        self._validation_mode = None
        self._validation_attempts = 0
        self._validation_captured = 0
        self._validation_valid = 0
        self._validation_consecutive = 0
        self._validation_max_consecutive = 0
        self._validation_started_at = 0.0
        self._validation_all_scores_ok = True
        self._validation_points_inside = True

        self.setWindowTitle("人物名字标定与管理")
        self.resize(1180, 780)
        self.setMinimumSize(960, 650)
        self._build_ui()
        self._populate_profiles(current_profile)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick_capture)
        self.timer.start()
        self.freeze_shortcut = QShortcut(
            QKeySequence("Space"), self.canvas,
            activated=self.freeze_for_new_sample,
        )
        self.freeze_shortcut.setContext(Qt.WidgetWithChildrenShortcut)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        intro = QLabel(
            "先选择或新建角色配置，再冻结画面并紧密框选角色名字，最后点击人物脚底。"
            "完整游戏截图不会保存。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        profile_bar = QHBoxLayout()
        profile_bar.addWidget(QLabel("人物配置："))
        self.profile_combo = QComboBox()
        self.profile_combo.setEditable(True)
        self.profile_combo.setMinimumWidth(260)
        self.profile_combo.activated.connect(self._profile_activated)
        profile_bar.addWidget(self.profile_combo, 1)
        new_profile = QPushButton("新建角色")
        new_profile.clicked.connect(self.create_profile)
        profile_bar.addWidget(new_profile)
        layout.addLayout(profile_bar)

        splitter = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("名字样本"))
        self.sample_list = QListWidget()
        self.sample_list.itemChanged.connect(self._sample_enabled_changed)
        left_layout.addWidget(self.sample_list, 1)
        self.sample_kind_combo = QComboBox()
        self.sample_kind_combo.addItem('完整名字（首次确认身份）', 'full')
        self.sample_kind_combo.addItem('局部名字（仅附近跟踪）', 'partial')
        self.sample_kind_combo.setToolTip('添加样本时选择类型；不要框入宠物文字。局部样本不能单独启动定位。')
        left_layout.addWidget(self.sample_kind_combo)
        self.change_kind_button = QPushButton('将选中样本设为此类型')
        self.change_kind_button.clicked.connect(self.change_selected_kind)
        left_layout.addWidget(self.change_kind_button)
        add_sample = QPushButton("冻结并添加样本 (Space)")
        add_sample.clicked.connect(self.freeze_for_new_sample)
        left_layout.addWidget(add_sample)
        refoot = QPushButton("重新标定选中样本脚点")
        refoot.clicked.connect(self.recalibrate_selected_foot)
        left_layout.addWidget(refoot)
        test_sample = QPushButton("测试选中样本")
        test_sample.clicked.connect(self.test_selected_sample)
        left_layout.addWidget(test_sample)
        test_profile = QPushButton("测试整个配置")
        test_profile.clicked.connect(self.test_profile)
        left_layout.addWidget(test_profile)
        pet_test = QPushButton('名字遮挡测试（20 秒／停止）')
        pet_test.clicked.connect(self.test_pet_occlusion)
        left_layout.addWidget(pet_test)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.canvas = NameTagCalibrationCanvas()
        self.canvas.rectangle_selected.connect(self._name_rect_selected)
        self.canvas.point_selected.connect(self._foot_selected)
        right_layout.addWidget(self.canvas, 1)
        self.coordinates = QLabel("等待游戏画面……")
        self.status = QLabel("正在连接游戏窗口。")
        self.status.setWordWrap(True)
        right_layout.addWidget(self.coordinates)
        right_layout.addWidget(self.status)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        buttons = QDialogButtonBox()
        self.save_button = buttons.addButton("保存资料", QDialogButtonBox.ActionRole)
        self.apply_button = buttons.addButton("应用并使用", QDialogButtonBox.AcceptRole)
        cancel_button = buttons.addButton("取消", QDialogButtonBox.RejectRole)
        self.save_button.clicked.connect(self.save_profile)
        self.apply_button.clicked.connect(self.apply_profile)
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_profiles(self, preferred=""):
        names = self.repository.list_profiles()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(names)
        self.profile_combo.setEditText(preferred or (names[0] if names else ""))
        self.profile_combo.blockSignals(False)
        if preferred in names:
            self._load_existing_profile(preferred)
        elif names:
            self._load_existing_profile(names[0])

    def _confirm_discard_stage(self):
        if not self.stage_dirty:
            return True
        result = QMessageBox.question(
            self,
            "未保存的名字资料",
            "当前人物名字资料尚未保存，切换后将放弃这些修改。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return result == QMessageBox.Yes

    def _profile_activated(self, _index):
        name = self.profile_combo.currentText().strip()
        if self.stage is not None and name == self.stage.name:
            return
        if not self._confirm_discard_stage():
            self.profile_combo.setEditText(self.stage.name if self.stage else "")
            return
        try:
            self._load_existing_profile(name)
        except NameTagProfileError as exc:
            QMessageBox.warning(self, "人物名字配置", str(exc))

    def _load_existing_profile(self, name):
        self._pet_test = None
        self._validation_localizer = self._validation_mode = None
        self._clear_pending()
        self.stage = self.repository.create_stage(name, existing=True)
        self.stage_dirty = False
        self._low_quality_samples.clear()
        self.profile_combo.setEditText(self.stage.name)
        self._refresh_samples()
        self.status.setText(f"已加载人物名字配置：{self.stage.name}")

    def create_profile(self):
        self._pet_test = None
        if not self._confirm_discard_stage():
            return
        name, ok = QInputDialog.getText(self, "新建角色", "人物配置名称：")
        if not ok:
            return
        try:
            name = validate_profile_name(name)
            self.stage = self.repository.create_stage(name, existing=False)
        except NameTagProfileError as exc:
            QMessageBox.warning(self, "新建角色", str(exc))
            return
        self.stage_dirty = True
        self.profile_combo.setEditText(name)
        self._validation_localizer = self._validation_mode = None
        self._clear_pending()
        self._low_quality_samples.clear()
        self._refresh_samples()
        self.status.setText("新配置已暂存，请冻结画面并添加第一个名字样本。")

    def _stage_image(self, filename):
        if self.stage is None:
            raise NameTagProfileError("请先选择或新建人物名字配置。")
        if filename in self.stage.new_images:
            return self.stage.new_images[filename]
        if self.stage.source_dir is None:
            raise NameTagProfileError(f"找不到名字样本：{filename}")
        return load_bgr_image(self.stage.source_dir / filename)

    def _refresh_samples(self):
        self._updating_samples = True
        try:
            self.sample_list.clear()
            if self.stage is None:
                return
            for entry in self.stage.data.get("samples", []):
                filename = entry["file"]
                offset = entry["player_offset"]
                quality = "  [验证未通过]" if filename in self._low_quality_samples else ""
                label = '完整名字' if entry.get('kind', 'full') == 'full' else '局部名字'
                item = QListWidgetItem(
                    f"{filename} [{label}]    脚点偏移 {offset[0]}, {offset[1]}{quality}"
                )
                item.setData(Qt.UserRole, filename)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if entry.get("enabled", True) else Qt.Unchecked)
                try:
                    image = self._stage_image(filename)
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    height, width = rgb.shape[:2]
                    qimage = QImage(
                        rgb.data, width, height, int(rgb.strides[0]), QImage.Format_RGB888
                    ).copy()
                    item.setIcon(QIcon(QPixmap.fromImage(qimage).scaled(
                        120, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )))
                except NameTagProfileError:
                    pass
                self.sample_list.addItem(item)
        finally:
            self._updating_samples = False

    def _sample_enabled_changed(self, item):
        if self._updating_samples or self.stage is None:
            return
        filename = item.data(Qt.UserRole)
        enabled = item.checkState() == Qt.Checked
        if enabled and filename in self._low_quality_samples:
            result = QMessageBox.question(
                self,
                "启用低质量样本",
                "该样本的连续验证未通过，启用后可能造成错误人物定位。仍要启用吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                self._updating_samples = True
                item.setCheckState(Qt.Unchecked)
                self._updating_samples = False
                return
        self.repository.stage_sample_enabled(self.stage, filename, enabled)
        self.stage_dirty = True

    def change_selected_kind(self):
        filename = self._selected_filename()
        if not filename or self.stage is None:
            self.status.setText('请先选择一个名字样本。')
            return
        if self._validation_localizer is not None or self._pending_kind or self._pet_test is not None:
            self.status.setText('请先完成当前标定或停止测试，再修改样本类型。')
            return
        try:
            self.repository.stage_sample_kind(self.stage, filename, self.sample_kind_combo.currentData())
        except NameTagProfileError as exc:
            self.status.setText(str(exc))
            return
        self.stage_dirty = True
        self._refresh_samples()
        self.status.setText('样本类型已暂存；请保留至少一个启用的完整名字样本，再保存资料。')

    def _ensure_capture(self):
        if self.capture is None:
            self.capture = self.capture_factory(self.cfg)

    def _capture_frame(self):
        try:
            self._ensure_capture()
            if hasattr(self.capture, 'get_frame_packet'):
                packet = self.capture.get_frame_packet()
                if packet is None:
                    return None
                raw, captured_at, sequence = packet
                if time.monotonic()-captured_at > .3:
                    return None
                if sequence == self._capture_sequence:
                    return None
                self._capture_sequence = sequence
            else:
                raw = self.capture.get_frame()
            frame = prepare_game_frame(raw, self.cfg) if raw is not None else None
        except Exception:
            return None
        if frame is None or frame.size == 0:
            return None
        return np.ascontiguousarray(frame.copy())

    def _tick_capture(self):
        frame = self._capture_frame()
        if frame is not None:
            self.latest_frame = frame
        if self._pet_test is not None:
            self._pet_test_tick(frame)
        elif self._validation_localizer is not None:
            self._validation_tick(frame)
        elif self.frozen_frame is None and frame is not None:
            self.canvas.set_image(frame)
            self.coordinates.setText(f"实时画面：{frame.shape[1]}×{frame.shape[0]}")
        elif frame is None and self.latest_frame is None:
            self.status.setText("暂未取得游戏画面，请确认游戏窗口已打开。")

    def freeze_for_new_sample(self):
        self._pet_test = None
        self._validation_localizer = None
        if self.stage is None:
            QMessageBox.warning(self, "人物名字标定", "请先选择或新建人物名字配置。")
            return
        frame = self.latest_frame
        if frame is None:
            frame = self._capture_frame()
        if frame is None:
            QMessageBox.warning(self, "人物名字标定", "暂未取得有效游戏画面。")
            return
        self.frozen_frame = np.ascontiguousarray(frame.copy())
        self.canvas.set_image(self.frozen_frame)
        self.canvas.set_mode("rect")
        self._pending_kind = "new"
        self._pending_sample_kind = self.sample_kind_combo.currentData()
        self._pending_filename = None
        self.status.setText('画面已冻结，请只框选角色露出的文字，不包含宠物名字。'
                            + ('局部模板至少 24×5 px。' if self._pending_sample_kind == 'partial' else '请选择完整名字。'))

    def _name_rect_selected(self, rect):
        if self.frozen_frame is None or self._pending_kind != "new":
            return
        x, y, width, height = (int(value) for value in rect)
        frame_height, frame_width = self.frozen_frame.shape[:2]
        if (width < 5 or height < 5 or x < 0 or y < 0
                or x + width > frame_width or y + height > frame_height):
            self.status.setText("名字框必须位于游戏画面内且至少为 5×5 px。")
            return
        image = self.frozen_frame[y:y + height, x:x + width].copy()
        try:
            self.repository._validate_image_content(image, "候选样本")
            if self._pending_sample_kind == 'partial':
                self.repository.validate_partial_image(image)
        except NameTagProfileError as exc:
            self.status.setText(str(exc))
            return
        self._pending_rect = (x, y, width, height)
        self._pending_image = image
        self.canvas.set_mode("point")
        self.canvas.set_overlay(self._pending_rect, valid=False)
        self.status.setText("名字框已选择，请点击人物脚底中心。")

    def _foot_selected(self, point):
        if self.frozen_frame is None or self._pending_rect is None:
            return
        x, y = (int(value) for value in point)
        height, width = self.frozen_frame.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            self.status.setText("人物脚点必须位于游戏画面内。")
            return
        rx, ry, _rw, _rh = self._pending_rect
        self._pending_offset = (x - rx, y - ry)
        self.canvas.set_overlay(self._pending_rect, point, valid=False)
        self.coordinates.setText(
            f"名字框：{self._pending_rect}    人物脚点：({x}, {y})    "
            f"偏移：{self._pending_offset}"
        )
        self._start_validation(
            [(self._pending_image, self._pending_offset, self._pending_sample_kind)],
            mode=self._pending_kind,
        )

    def _localizer_settings(self):
        settings = copy.deepcopy(
            (self.stage.data if self.stage else {}).get("settings", {})
        )
        return settings

    def _start_validation(self, samples, *, mode):
        self._pet_test = None
        try:
            settings = self._localizer_settings()
            partial_only = all(len(s) > 2 and s[2] == 'partial' for s in samples)
            settings['allow_partial'] = partial_only or (mode == 'test' and len(samples) > 1)
            self._validation_localizer = NameTagLocalizer(
                samples, **settings
            )
            if partial_only:
                # Manual click authorizes only this read-only, nearby sample
                # validation. Runtime must still establish a complete name.
                if mode in ('new', 'refoot') and self._pending_rect and self._pending_offset:
                    anchor = tuple(a+b for a,b in zip(self._pending_rect[:2], self._pending_offset))
                else:
                    anchor = self._complete_name_anchor(self.latest_frame)
                if anchor is None:
                    self._validation_localizer = None
                    self.status.setText('测试局部样本前请先露出完整名字，以确认人物身份；或重新冻结框选并点击脚点。')
                    return
                loc = self._validation_localizer
                loc.last_player = anchor
                loc.last_tag = tuple(a-b for a,b in zip(anchor, samples[0][1]))
                loc.last_kind, loc.partial_confirmed = 'partial', True
        except (TypeError, ValueError) as exc:
            self.status.setText(f"无法开始名字样本验证：{exc}")
            return
        self.frozen_frame = None
        self.canvas.set_mode("none")
        self._validation_mode = mode
        self._validation_attempts = self._validation_captured = 0
        self._validation_valid = self._validation_consecutive = 0
        self._validation_max_consecutive = 0
        self._validation_started_at = time.monotonic()
        self._validation_all_scores_ok = True
        self._validation_points_inside = True
        self.status.setText("正在连续验证 20 帧，请保持名字可见并轻微左右移动人物。")

    def _validation_tick(self, frame):
        if time.monotonic() - self._validation_started_at > self.VALIDATION_TIMEOUT_SECONDS:
            self._finish_validation(environment_error=self._validation_captured < self.MIN_CAPTURED_FRAMES)
            return
        if frame is None:
            return
        self._validation_captured += 1
        self._validation_attempts += 1
        y_limit = name_search_y_limit(self.cfg, frame.shape[0])
        result = self._validation_localizer.locate(frame, y_limit=y_limit)
        valid = bool(result.valid and result.player is not None)
        rect = point = None
        if valid:
            px, py = result.player
            inside = 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]
            self._validation_points_inside &= inside
            self._validation_all_scores_ok &= (
                result.score <= self._validation_localizer.max_score
            )
            valid = valid and inside
            if result.tag_top_left and result.tag_size:
                rect = (*result.tag_top_left, *result.tag_size)
            point = result.player
        if valid:
            self._validation_valid += 1
            self._validation_consecutive = 0
        else:
            self._validation_consecutive += 1
            self._validation_max_consecutive = max(
                self._validation_max_consecutive, self._validation_consecutive
            )
        self.canvas.set_image(frame)
        self.canvas.set_overlay(rect, point, valid=valid)
        self.coordinates.setText(
            f"连续验证：{self._validation_attempts}/{self.VALIDATION_FRAMES}    "
            f"有效：{self._validation_valid}    分数：{result.score:.3f}    "
            f"状态：{'通过' if valid else result.reason}"
        )
        if self._validation_attempts >= self.VALIDATION_FRAMES:
            self._finish_validation(environment_error=False)

    def _finish_validation(self, *, environment_error):
        mode = self._validation_mode
        self._validation_localizer = None
        self._validation_mode = None
        if environment_error:
            self.status.setText("有效游戏画面不足 10 帧，未保存样本，请检查游戏窗口后重试。")
            self._clear_pending()
            return
        passed = (
            self._validation_attempts >= self.VALIDATION_FRAMES
            and self._validation_valid >= self.REQUIRED_VALID_FRAMES
            and self._validation_max_consecutive <= self.MAX_CONSECUTIVE_MISSES
            and self._validation_all_scores_ok
            and self._validation_points_inside
        )
        if mode == "new":
            filename = self.repository.stage_sample(
                self.stage,
                self._pending_image,
                self._pending_offset,
                enabled=passed,
                kind=self._pending_sample_kind,
            )
            if not passed:
                self._low_quality_samples.add(filename)
            self.stage_dirty = True
            self._refresh_samples()
            self.sample_list.setCurrentRow(self.sample_list.count() - 1)
        elif mode == "refoot" and self._pending_filename:
            self.repository.stage_player_offset(
                self.stage, self._pending_filename, self._pending_offset
            )
            if not passed:
                self.repository.stage_sample_enabled(
                    self.stage, self._pending_filename, False
                )
                self._low_quality_samples.add(self._pending_filename)
            self.stage_dirty = True
            self._refresh_samples()
        if passed:
            result_text = (
                f"验证通过：{self._validation_valid}/{self._validation_attempts} 帧有效。"
            )
        elif mode in {"new", "refoot"}:
            result_text = (
                f"验证未通过：{self._validation_valid}/{self._validation_attempts} 帧有效，"
                "样本已暂存为禁用状态。"
            )
        else:
            result_text = (
                f"测试未通过：{self._validation_valid}/{self._validation_attempts} 帧有效；"
                "现有样本状态未修改。"
            )
        self.status.setText(result_text)
        self._clear_pending()

    def _clear_pending(self):
        self.frozen_frame = None
        self._pending_rect = self._pending_image = self._pending_offset = None
        self._pending_filename = self._pending_kind = None
        self.canvas.set_mode("none")

    def _selected_filename(self):
        item = self.sample_list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def recalibrate_selected_foot(self):
        filename = self._selected_filename()
        frame = self.latest_frame
        if frame is None:
            frame = self._capture_frame()
        if not filename or frame is None:
            QMessageBox.warning(self, "重新标定脚点", "请选择样本并确认游戏画面可用。")
            return
        image = self._stage_image(filename)
        entry = self.repository._sample_entry(self.stage, filename)
        try:
            localizer = NameTagLocalizer(
                [(image, tuple(entry["player_offset"]), entry.get('kind', 'full'))], **self._localizer_settings()
            )
            if entry.get('kind', 'full') == 'partial':
                anchor = self._complete_name_anchor(frame)
                if anchor is None:
                    QMessageBox.warning(self, '重新标定脚点', '请先露出完整名字以确认人物身份，再定位局部样本。')
                    return
                localizer.last_player = anchor
                localizer.last_tag = tuple(a-b for a,b in zip(anchor, entry['player_offset']))
                localizer.partial_confirmed = True
            result = localizer.locate(frame)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "重新标定脚点", str(exc))
            return
        if not result.valid or result.tag_top_left is None or result.tag_size is None:
            QMessageBox.warning(self, "重新标定脚点", "当前画面未找到该名字样本。")
            return
        self.frozen_frame = np.ascontiguousarray(frame.copy())
        self._pending_filename = filename
        self._pending_kind = "refoot"
        self._pending_sample_kind = entry.get('kind', 'full')
        self._pending_image = image
        self._pending_rect = (*result.tag_top_left, *result.tag_size)
        self.canvas.set_image(self.frozen_frame)
        self.canvas.set_overlay(self._pending_rect, result.player, valid=True)
        self.canvas.set_mode("point")
        self.status.setText("已定位选中样本，请重新点击人物脚底中心。")

    def _samples_for_validation(self, filenames):
        samples = []
        for entry in self.stage.data.get("samples", []):
            if entry["file"] in filenames:
                samples.append((
                    self._stage_image(entry["file"]),
                    tuple(entry["player_offset"]),
                    entry.get('kind', 'full'),
                ))
        return samples

    def _complete_name_anchor(self, frame):
        if frame is None or self.stage is None:
            return None
        files = {e['file'] for e in self.stage.data.get('samples', [])
                 if e.get('enabled', True) and e.get('kind', 'full') == 'full'}
        if not files:
            return None
        loc = NameTagLocalizer(self._samples_for_validation(files), allow_partial=False, **self._localizer_settings())
        result = loc.locate(frame, y_limit=name_search_y_limit(self.cfg, frame.shape[0]))
        return result.player if result.valid else None

    def test_pet_occlusion(self):
        if self._pet_test is not None:
            self._finish_pet_test()
            return
        if self.stage is None:
            self.status.setText('请先选择人物资料，并准备无遮挡的完整名字样本。')
            return
        enabled = {e['file'] for e in self.stage.data.get('samples', []) if e.get('enabled', True)}
        if not enabled or not any(e['file'] in enabled and e.get('kind', 'full') == 'full'
                                  for e in self.stage.data.get('samples', [])):
            self.status.setText('没有启用的完整名字样本，无法测试。')
            return
        try:
            settings = self._localizer_settings()
            settings['allow_partial'] = True
            loc = NameTagLocalizer(self._samples_for_validation(enabled), **settings)
        except (ValueError, NameTagProfileError) as exc:
            self.status.setText(f'无法进行宠物遮挡测试：{exc}')
            return
        self._validation_localizer = None
        self.frozen_frame = None
        self.canvas.set_mode('none')
        self._pet_test = dict(localizer=loc, started=time.monotonic(),
                              counts=dict(full=0, left=0, right=0, partial=0, missing=0),
                              missing_at=None, recoveries=[], longest=0.0)
        self.status.setText('20 秒宠物遮挡测试：先露出完整名字，再手动左右换向。软件不会按键或保存截图。'
                            + (' 此样本文字不足，不支持局部恢复。' if not loc.fragments else ''))

    def _pet_test_tick(self, frame):
        test = self._pet_test
        now = time.monotonic()
        if now-test['started'] >= 20:
            self._finish_pet_test()
            return
        if frame is None:
            return
        result = test['localizer'].locate(frame, y_limit=name_search_y_limit(self.cfg, frame.shape[0]))
        kind = result.match_kind if result.valid else 'missing'
        test['counts'][kind] += 1
        if not result.valid:
            if test['missing_at'] is None:
                test['missing_at'] = now
        elif test['missing_at'] is not None:
            duration = now-test['missing_at']
            test['recoveries'].append(duration)
            test['longest'] = max(test['longest'], duration)
            test['missing_at'] = None
        self.canvas.set_image(frame)
        if result.valid:
            rect = result.fragment_rect or (*result.tag_top_left, *result.tag_size)
            self.canvas.set_overlay(rect, result.player)
        labels = dict(full='完整名字', left='左侧文字', right='右侧文字', partial='局部样本', missing='失配／等待确认')
        self.coordinates.setText(f'剩余 {20-(now-test["started"]):.1f} 秒；{labels[kind]}；'
                                 f'分数 {result.score:.3f}；有效帧统计 {test["counts"]}')

    def _finish_pet_test(self):
        test = self._pet_test
        if test is None:
            return
        now = time.monotonic()
        if test['missing_at'] is not None:
            test['longest'] = max(test['longest'], now-test['missing_at'])
        counts = test['counts']
        recovered = test['recoveries']
        self.status.setText(
            f'宠物测试结束：完整 {counts["full"]}，左侧 {counts["left"]}，右侧 {counts["right"]}，'
            f'局部样本 {counts["partial"]}，'
            f'失配 {counts["missing"]} 帧；最长失配 {test["longest"]:.2f} 秒；'
            f'恢复 {len(recovered)} 次，平均失配至恢复耗时 {np.mean(recovered) if recovered else 0:.2f} 秒。'
            '此统计包含遮挡持续时间，不等于文字露出后的算法延迟。资料及启用状态未改变。'
            + (' 有效截图不足 10 帧，请检查游戏窗口后重试。' if sum(counts.values()) < 10 else '')
        )
        self._pet_test = None

    def test_selected_sample(self):
        filename = self._selected_filename()
        if not filename:
            QMessageBox.warning(self, "测试名字样本", "请先选择一个名字样本。")
            return
        self._pending_filename = filename
        self._start_validation(
            self._samples_for_validation({filename}), mode="test"
        )

    def test_profile(self):
        if self.stage is None:
            return
        enabled = {
            entry["file"] for entry in self.stage.data.get("samples", [])
            if entry.get("enabled", True)
        }
        if not enabled:
            QMessageBox.warning(self, "测试人物配置", "当前配置没有已启用样本。")
            return
        self._start_validation(
            self._samples_for_validation(enabled), mode="test"
        )

    def save_profile(self):
        if self.stage is None:
            QMessageBox.warning(self, "保存人物资料", "请先选择或新建人物配置。")
            return False
        try:
            self.repository.commit_profile(self.stage, require_enabled=False)
        except NameTagProfileError as exc:
            QMessageBox.warning(self, "保存人物资料", str(exc))
            return False
        self.stage_dirty = False
        self.status.setText(f"人物名字资料已保存：{self.stage.name}")
        return True

    def apply_profile(self):
        if self.stage is None:
            QMessageBox.warning(self, "应用人物配置", "请先选择或新建人物配置。")
            return
        try:
            self.repository.commit_profile(self.stage, require_enabled=True)
        except NameTagProfileError as exc:
            QMessageBox.warning(self, "应用人物配置", str(exc))
            return
        self.stage_dirty = False
        self.applied_name = self.stage.name
        self.accept()

    def done(self, result):
        if result == QDialog.Rejected and self.stage_dirty:
            answer = QMessageBox.question(
                self,
                "放弃未保存修改",
                "人物名字资料存在未保存修改，确定放弃吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        if hasattr(self, "timer"):
            self.timer.stop()
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        self.control_lock.release()
        super().done(result)
