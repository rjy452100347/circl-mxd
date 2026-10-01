import copy
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication, QDialog, QPushButton

import src.engine.RouteStudioSession as session_module
import src.ui.route_studio as route_studio_module
from src.engine.RouteStudioSession import RouteStudioSession, RouteStudioSessionError
from src.engine.SemanticRoute import SemanticRouteError, load_route_document
from src.ui.route_studio import (
    MinimapRoiCalibrationDialog,
    MinimapRoiCanvas,
    RouteStudioWindow,
    SemanticRouteEditorModel,
    command_colors_from_cfg,
)
from src.engine.MapProjectConfig import load_map_minimap_roi
from src.ui.ui import MainWindow
from src.utils.common import load_yaml


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


def _cfg():
    color_code = {}
    color_code_up_down = {}
    for command, color in COLORS.items():
        target = color_code_up_down if command in {"none up none", "none down none"} else color_code
        target[",".join(str(value) for value in color)] = command
    return {
        "key": {"jump": "space", "teleport": ""},
        "route": {
            "localization_max_score": 0.2,
            "color_code": color_code,
            "color_code_up_down": color_code_up_down,
        },
        "minimap": {
            "roi": [0.0, 0.0, 1.0, 1.0],
            "player_color": [0, 255, 255],
            "offset": [0, 0],
        },
        "game_window": {"title": "game", "size": [10, 10], "title_bar_height": 0},
    }


def _document():
    return {
        "schema_version": 1,
        "map_id": "training",
        "route_index": 1,
        "canvas_size": [80, 50],
        "loop": True,
        "segments": [
            {"id": "walk_001", "type": "walk", "direction": "right",
             "points": [[4, 30], [12, 30], [20, 30], [28, 30]]},
            {"id": "walk_002", "type": "walk", "direction": "right",
             "points": [[28, 30], [40, 30]]},
            {"id": "ladder_001", "type": "ladder", "approach": [42, 30],
             "mount": [42, 30], "points": [[42, 30], [42, 15]],
             "exit": [45, 15], "exit_direction": "right"},
            {"id": "goal_001", "type": "goal", "position": [70, 15], "radius": 6},
        ],
    }


def _model(document=None):
    return SemanticRouteEditorModel(
        map_id="training", route_index=1,
        base_bgr=np.zeros((50, 80, 3), dtype=np.uint8),
        document=document or _document(),
    )


def test_editor_split_merge_move_type_and_undo():
    model = _model()
    model.selected_index = 0
    assert model.split_selected(2)
    assert len(model.document["segments"]) == 5
    assert model.merge_with_next()
    assert model.document["segments"][0]["points"] == _document()["segments"][0]["points"]
    assert model.move_control("points", 1, (13, 29))
    assert model.document["segments"][0]["points"][1] == [13, 29]
    assert model.change_selected_type("jump")
    assert model.selected_segment()["type"] == "jump"
    assert model.undo()
    assert model.selected_segment()["type"] == "walk"
    assert model.redo()
    assert model.selected_segment()["type"] == "jump"


def test_editor_delete_goal_is_blocked_and_ladder_exit_is_editable():
    model = _model()
    model.selected_index = 3
    with pytest.raises(SemanticRouteError, match="Goal"):
        model.delete_selected()
    model.selected_index = 2
    model.set_direction("right")
    model.set_exit_direction("left")
    assert model.selected_segment()["mount_direction"] == "right"
    assert model.selected_segment()["exit_direction"] == "left"


def test_editor_reports_and_explicitly_recalculates_ladder_center():
    document = _document()
    ladder = document["segments"][2]
    ladder["mount"] = [42, 30]
    ladder["points"] = [[38, 30], [40, 24], [45, 18], [45, 16], [46, 15]]
    model = _model(document)
    model.selected_index = 2

    diagnostics = model.ladder_center_diagnostics()
    assert diagnostics == {"recorded": 42, "calculated": 45, "difference": 3}
    assert model.recalculate_selected_ladder_mount()
    assert model.selected_segment()["mount"] == [45, 30]
    assert model.dirty


def test_editor_can_reorder_and_change_action_direction_without_moving_goal():
    model = _model()
    model.selected_index = 1
    assert model.move_selected(-1)
    assert model.selected_index == 0
    model.set_direction("left")
    assert model.selected_segment()["direction"] == "left"
    model.selected_index = len(model.document["segments"]) - 2
    with pytest.raises(SemanticRouteError, match="Goal"):
        model.move_selected(1)


def test_draft_must_be_explicitly_validated_before_atomic_save(tmp_path):
    document = _document()
    document["draft"] = True
    model = _model(document)
    with pytest.raises(SemanticRouteError, match="校正"):
        model.save(tmp_path, COLORS)
    model.validate(finalize_draft=True)
    assert "draft" not in model.document
    json_path, png_path = model.save(tmp_path, COLORS)
    assert json_path.exists() and png_path.exists()
    assert load_route_document(json_path)["route_index"] == 1


def test_save_pair_stays_one_pixel_for_walk(tmp_path):
    model = _model()
    model.save(tmp_path, COLORS)
    png = cv2.imread(str(tmp_path / "route1.png"))
    bgr = tuple(reversed(COLORS["right none none"]))
    mask = np.all(png == bgr, axis=2)
    # Away from 2px event markers, horizontal visualization remains exactly 1px.
    assert not mask[29].any()
    assert mask[30].any()
    assert not mask[31].any()


class _Capture:
    def __init__(self, frame):
        self.frame = frame

    def get_frame(self):
        return self.frame.copy()

    def stop(self):
        pass


class _Keyboard:
    def __init__(self, keys=()):
        self.key_pressing = list(keys)
        self.is_pressed_func_key = [False] * 12

    def stop(self):
        pass


def test_existing_map_session_localizes_and_never_mutates_base(monkeypatch):
    cfg = _cfg()
    base = np.full((30, 40, 3), 17, dtype=np.uint8)
    original = base.copy()
    session = RouteStudioSession(cfg, map_id="training", base_bgr=base, new_map=False)
    session.capture = _Capture(np.zeros((10, 10, 3), dtype=np.uint8))
    session.keyboard = _Keyboard(["right"])
    monkeypatch.setattr(session_module, "prepare_game_frame", lambda frame, cfg: frame)
    monkeypatch.setattr(session_module, "get_minimap_loc_size", lambda frame, cfg: (0, 0, 10, 10))
    monkeypatch.setattr(session_module, "get_player_location_on_minimap", lambda *a, **k: (3, 4))
    monkeypatch.setattr(session_module, "find_pattern_sqdiff", lambda *a, **k: ((5, 6), 0.04, False))
    result = session.tick(now=1.0)
    assert result["position"] == (8, 10)
    assert result["valid"]
    assert np.array_equal(session.base_bgr, original)


def test_recording_session_reads_configured_space_and_blank_teleport(monkeypatch):
    cfg = _cfg()
    base = np.zeros((30, 40, 3), dtype=np.uint8)
    session = RouteStudioSession(cfg, map_id="training", base_bgr=base, new_map=False)
    session.capture = _Capture(np.zeros((10, 10, 3), dtype=np.uint8))
    session.keyboard = _Keyboard(["space"])
    monkeypatch.setattr(session_module, "prepare_game_frame", lambda frame, cfg: frame)
    monkeypatch.setattr(session_module, "get_minimap_loc_size", lambda frame, cfg: (0, 0, 10, 10))
    monkeypatch.setattr(session_module, "get_player_location_on_minimap", lambda *a, **k: (3, 4))
    monkeypatch.setattr(session_module, "find_pattern_sqdiff", lambda *a, **k: ((5, 6), 0.04, False))
    session.begin_route_recording(1)
    # begin_route_recording starts real capture; replace it with deterministic fakes again.
    session.capture = _Capture(np.zeros((10, 10, 3), dtype=np.uint8))
    session.keyboard = _Keyboard(["space"])
    session.tick(now=1.0)
    session.keyboard.key_pressing = []
    session.tick(now=1.4)
    document = session.recorder.finish_goal((9, 10))
    assert [segment["type"] for segment in document["segments"]] == ["jump", "goal"]
    assert all(segment["type"] != "teleport" for segment in document["segments"])


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_route_studio_has_required_controls_and_locked_existing_map(qt_app, tmp_path):
    project = tmp_path / "training"
    project.mkdir()
    cv2.imwrite(str(project / "map.png"), np.zeros((50, 80, 3), dtype=np.uint8))
    window = RouteStudioWindow(cfg=_cfg(), initial_map="training", project_root=tmp_path)
    try:
        assert window.windowTitle() == "路线录制与编辑"
        assert not window.start_map_button.isEnabled()
        assert "F1" in window.record_button.text()
        assert "校准小地图 ROI" in {
            button.text() for button in window.findChildren(QPushButton)
        }
        window.load_project()
        assert window.model.map_id == "training"
        assert window.model.route_index == 1
    finally:
        window.model.dirty = False
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def _calibration_frame():
    frame = np.zeros((260, 360, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 40), (270, 210), (255, 255, 255), 1)
    cv2.rectangle(frame, (145, 110), (150, 115), (0, 255, 255), -1)
    return frame


def test_roi_calibrator_snaps_to_white_border_and_detects_player(qt_app):
    frame = _calibration_frame()
    dialog = MinimapRoiCalibrationDialog(
        cfg=_cfg(), frame_provider=lambda: frame, current_roi=None,
    )
    try:
        dialog.refresh_frame()
        dialog._selection_changed((40, 30, 250, 200))

        assert dialog.candidate_rect == (51, 41, 219, 169)
        assert dialog.player is not None
        roi = dialog.selected_roi()
        assert roi is not None
        assert len(roi) == 4
    finally:
        dialog.close()
        dialog.deleteLater()
        qt_app.processEvents()


def test_roi_calibrator_can_disable_snap_and_keep_exact_rectangle(qt_app):
    frame = _calibration_frame()
    dialog = MinimapRoiCalibrationDialog(
        cfg=_cfg(), frame_provider=lambda: frame, current_roi=None,
    )
    try:
        dialog.refresh_frame()
        dialog.snap_checkbox.setChecked(False)
        dialog._selection_changed((40, 30, 250, 200))

        assert dialog.candidate_rect == (40, 30, 250, 200)
    finally:
        dialog.close()
        dialog.deleteLater()
        qt_app.processEvents()


def test_roi_calibrator_warns_but_can_accept_without_player(qt_app, monkeypatch):
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    dialog = MinimapRoiCalibrationDialog(
        cfg=_cfg(), frame_provider=lambda: frame, current_roi=None,
    )
    try:
        dialog.refresh_frame()
        dialog.snap_checkbox.setChecked(False)
        dialog._selection_changed((20, 20, 80, 60))
        monkeypatch.setattr(
            route_studio_module.QMessageBox, "question",
            lambda *args, **kwargs: route_studio_module.QMessageBox.Yes,
        )

        dialog._accept_candidate()

        assert dialog.result() == QDialog.Accepted
    finally:
        dialog.close()
        dialog.deleteLater()
        qt_app.processEvents()


def test_roi_canvas_rounds_scene_selection_outward():
    assert MinimapRoiCanvas._integer_rect(
        QRectF(10.8, 20.2, 30.1, 40.1)
    ) == (10, 20, 31, 41)


def test_route_studio_can_save_only_a_map_roi_override(qt_app, tmp_path):
    window = RouteStudioWindow(
        cfg=_cfg(), initial_map="training", project_root=tmp_path,
    )
    try:
        window.active_map_id = "training"
        window.map_config_loaded_for = "training"
        window.map_roi_override = (0.1, 0.2, 0.3, 0.4)
        window.map_config_dirty = True

        assert window.save_project()
        assert load_map_minimap_roi(tmp_path / "training") == (0.1, 0.2, 0.3, 0.4)
        assert not window.map_config_dirty
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_route_studio_emits_project_saved_for_runnable_route(qt_app, tmp_path):
    directory = tmp_path / "测试地图"
    directory.mkdir()
    base = np.zeros((50, 80, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", base)
    assert success
    encoded.tofile(directory / "map.png")
    window = RouteStudioWindow(
        cfg=_cfg(), initial_map="测试地图", project_root=tmp_path,
    )
    saved = []
    try:
        window.active_map_id = "测试地图"
        window.model = _model()
        window.model.map_id = "测试地图"
        window.model.document["map_id"] = "测试地图"
        window.model.base_bgr = base
        window.project_saved.connect(saved.append)

        assert window.save_project()
        assert saved == ["测试地图"]
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_route_studio_blocks_calibration_during_capture(qt_app, tmp_path):
    window = RouteStudioWindow(
        cfg=_cfg(), initial_map="training", project_root=tmp_path,
    )
    errors = []
    try:
        window.session = RouteStudioSession(
            _cfg(), map_id="training", base_bgr=None, new_map=True,
        )
        window.session.map_capture_active = True
        window._error = errors.append

        assert window._calibration_is_blocked()
        assert "先停止" in str(errors[-1])
    finally:
        window.session.map_capture_active = False
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_session_applies_roi_and_clears_localization_cache():
    session = RouteStudioSession(
        _cfg(), map_id="training",
        base_bgr=np.zeros((50, 80, 3), dtype=np.uint8), new_map=False,
    )
    session.minimap_roi = (1, 2, 3, 4)
    session.last_minimap_global = (5, 6)
    session.last_position = (7, 8)
    session.last_score = 0.1

    session.apply_minimap_roi((0.1, 0.2, 0.3, 0.4))

    assert session.cfg["minimap"]["roi"] == [0.1, 0.2, 0.3, 0.4]
    assert session.minimap_roi is None
    assert session.last_minimap_global is None
    assert session.last_position is None
    assert session.last_score is None


def test_session_discard_helpers_clear_only_in_memory_data():
    captured = np.full((20, 30, 3), 99, dtype=np.uint8)
    restored = np.full((20, 30, 3), 7, dtype=np.uint8)
    session = RouteStudioSession(
        _cfg(), map_id="测试地图", base_bgr=captured, new_map=True,
    )
    session.map_dirty = True
    session.last_minimap_global = (4, 5)
    session.last_position = (6, 7)
    session.discard_map_capture(restored)

    assert not session.map_dirty
    assert np.array_equal(session.base_bgr, restored)
    assert session.last_minimap_global is None
    assert session.last_position is None

    session.recorder = object()
    session.route_recording_active = True
    session.discard_route_recording()
    assert session.recorder is None
    assert not session.route_recording_active


def test_discard_button_restores_saved_map_without_deleting_disk(
    qt_app, tmp_path
):
    directory = tmp_path / "测试地图"
    directory.mkdir()
    saved = np.full((20, 30, 3), (3, 7, 11), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", saved)
    assert success
    encoded.tofile(directory / "map.png")
    window = RouteStudioWindow(
        cfg=_cfg(), initial_map="测试地图", project_root=tmp_path,
    )
    try:
        window.active_map_id = "测试地图"
        window.session = RouteStudioSession(
            _cfg(), map_id="测试地图",
            base_bgr=np.full_like(saved, 200), new_map=True,
        )
        window.session.map_dirty = True

        assert window.discard_pending_capture_data(confirm=False)
        assert directory.joinpath("map.png").is_file()
        assert np.array_equal(window.session.base_bgr, saved)
        assert not window.session.map_dirty
        assert window.model is not None
        assert "可重新校准" in window.status.text()
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_new_map_can_transition_from_first_capture_to_route_without_model(qt_app, tmp_path):
    class Session:
        active = False
        map_capture_active = True
        route_recording_active = False
        base_bgr = np.zeros((50, 80, 3), dtype=np.uint8)
        recorder = None
        last_position = None

        def end_map_capture(self):
            self.map_capture_active = False

        def begin_route_recording(self, index):
            self.started_index = index
            self.recorder = object()

    window = RouteStudioWindow(
        cfg=_cfg(), initial_map="brand_new", project_root=tmp_path,
    )
    try:
        window.active_map_id = "brand_new"
        window.model = None
        window.session = Session()

        window.toggle_route_recording()

        assert window.model is not None
        assert window.model.route_index == 1
        assert window.session.started_index == 1
    finally:
        window.session = None
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_new_map_can_explicitly_recapture_map_before_routes_exist(
    qt_app, tmp_path, monkeypatch
):
    directory = tmp_path / "training"
    directory.mkdir()
    cv2.imwrite(str(directory / "map.png"), np.zeros((50, 80, 3), dtype=np.uint8))
    window = RouteStudioWindow(
        cfg=_cfg(), initial_map="training", project_root=tmp_path,
    )
    try:
        monkeypatch.setattr(
            route_studio_module.QMessageBox, "question",
            lambda *args, **kwargs: route_studio_module.QMessageBox.Yes,
        )

        assert window._confirm_new_map_capture_target(directory)

        (directory / "route1.png").write_bytes(b"route")
        with pytest.raises(RouteStudioSessionError, match="路线坐标失效"):
            window._confirm_new_map_capture_target(directory)
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_classic_profile_uses_space_and_disables_teleport():
    cfg = load_yaml("config/config_classic_cn.yaml")
    assert cfg["key"]["jump"] == "space"
    assert cfg["key"]["teleport"] == ""


def test_main_window_entry_disables_start_until_studio_closes(qt_app, monkeypatch):
    class Controller:
        def enable_bot_viz(self, mode="game"):
            pass

        def disable_bot_viz(self):
            pass

        def terminate_bot(self):
            pass

    monkeypatch.setattr(MainWindow, "load_ui_state", lambda self: None)
    monkeypatch.setattr(MainWindow, "save_ui_state", lambda self: None)
    main = MainWindow(Controller())
    try:
        main.open_route_studio()
        assert main.route_studio_window is not None
        assert not main.button_start_pause.isEnabled()
        main.route_studio_window.close()
        qt_app.processEvents()
        assert main.route_studio_window is None
        assert main.button_start_pause.isEnabled()
    finally:
        main.close()
