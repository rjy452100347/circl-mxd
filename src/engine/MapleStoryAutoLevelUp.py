'''
Execute this script:
python mapleStoryAutoLevelUp.py --map cloud_balcony --monster brown_windup_bear,pink_windup_bear
'''
# Standard import
import time
import random
import argparse
import copy
import glob
import sys
import logging
import threading
from collections import deque
from pathlib import Path

# Library import
import numpy as np
import cv2
import yaml

# Local import
from src.utils.global_var import WINDOW_WORKING_SIZE
from src.app_paths import writable_path
from src.utils.logger import logger
from src.utils.common import (find_pattern_sqdiff, draw_rectangle,
    load_image, get_minimap_loc_size, get_player_location_on_minimap,
    is_mac, override_cfg, load_yaml, get_all_other_player_locations_on_minimap,
    click_in_game_window, mask_route_colors, debug_minimap_colors,
    activate_game_window, is_img_16_to_9, normalize_pixel_coordinate, resize_window,
    prepare_game_frame,
)
from src.input.KeyBoardController import KeyBoardController
from src.input.KeyBoardListener import KeyBoardListener
if is_mac():
    from src.input.GameWindowCapturorForMac import GameWindowCapturor
else:
    from src.input.GameWindowCapturor import GameWindowCapturor
from src.engine.HealthMonitor import HealthMonitor
from src.engine.MinimapPoseTracker import MinimapPoseTracker
from src.engine.Profiler import Profiler
from src.engine.FiniteStateMachine import FiniteStateMachine
from src.engine.RouteNavigator import RouteNavigator
from src.engine.SemanticRoute import (
    SemanticRouteError,
    load_semantic_route_set,
)
from src.engine.SemanticRouteNavigator import SemanticRouteNavigator
from src.engine.RouteCombatArbiter import (
    CombatIntent,
    RouteCombatArbiter,
    RouteIntent,
)
from src.engine.NameTagLocalizer import NameTagLocalizer
from src.engine.NameTagProfileRepository import (
    NameTagProfileError,
    NameTagProfileRepository,
)
from src.engine.OpenVinoMonsterDetector import (
    OpenVinoDeploymentError,
    OpenVinoMonsterDetector,
)
from src.engine.MapProjectConfig import (
    apply_map_minimap_roi_override,
    MapProjectConfigError,
)
from src.states.hunting import HuntingState
from src.states.auxiliary import AuxiliaryState
from src.states.patrol import PatrolState
from src.states.fixed_platform import FixedPlatformState
from src.states.continuous_attack import ContinuousAttackState
from src.runtime_policy import RUNTIME_POLICY
from src.config_compat import migrate_health_monitor_config, validate_custom_config



class MapleStoryAutoBot:
    '''
    MapleStoryAutoBot
    '''
    def __init__(self, args):
        '''
        Init MapleStoryAutoBot
        '''
        self.args = args # User args
        self.cfg = None # Configuration
        self.idx_routes = 0 # Index of route map
        self.monsters = [] # monster detected in current frame
        self.monster_detector = None # required monster/player OpenVINO detector
        control_window = RUNTIME_POLICY.main_fps * 5 * 60
        preview_window = RUNTIME_POLICY.preview_fps * 5 * 60
        self.monster_detection_times = deque(maxlen=control_window)
        self.monster_inference_times = deque(maxlen=control_window)
        self.monster_frame_pipeline_times = deque(maxlen=control_window)
        self.visualization_compose_times = deque(maxlen=preview_window)
        self.visualization_publish_times = deque(maxlen=preview_window)
        self.debug_attack_direction = "none" # passive combat decision display
        self.fps = 0 # Frame per second
        self.control_frame_sequence = 0
        self.red_dot_center_prev = None # previous other player location in minimap
        self.color_code = {} # For color code instruction
        self.color_code_up_down = {} # Color code only contain 'up' and 'down'
        self.thread_auto_bot = None # thread for running autobot
        self.run_generation = 0
        self.cmd_move_x = "none" # "left" "right"
        self.cmd_move_y = "none" # "up" "down"
        self.cmd_action = "none" # "jump" "attack" ....
        # Signals (for UI)
        self.visualization_sink = None
        self.termination_sink = None
        self._visualization_request = threading.Event()
        self._frame_visualization_mode = None
        self.viz_mode = "game"
        self.t_last_viz_frame = 0.0
        # Flags
        self.is_first_frame = True # first frame flag
        self.is_terminated = False # Close all object and thread if True
        self.is_on_ladder = False # Character is on ladder or not
        self.is_show_debug_window = not args.disable_viz #
        self.is_need_show_debug_window = not args.disable_viz #
        self.is_disable_control = args.disable_control
        self.is_ui = args.is_ui # Whether is using UI framework to invoke engine
        self.is_frame_done = False #
        # Coordinate (top-left coordinate)
        self.loc_nametag = (0, 0) # nametag location on game screen
        self.loc_minimap = (0, 0) # minimap location on game screen
        self.loc_player = (0, 0) # player location on game screen
        self.loc_player_minimap = (0, 0) # player location on minimap
        self.loc_minimap_global = (0, 0) # minimap location on global map
        self.loc_player_global = (0, 0) # player location on global map
        self.loc_watch_dog = (0, 0) # watch dog location on global map
        # Images
        self.frame = None # raw image
        self.frame_captured_at = 0.0
        self.capture_frame_sequence = None
        self.img_frame = None # game window frame
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = np.zeros((10, 10, 3), dtype=np.uint8) # minimap on game screen
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_watch_dog = time.time() # Last movement timer
        self.t_last_teleport = time.time() # Last teleport timer
        self.t_last_attack = time.time() # Last attack timer for cooldown
        self.t_last_minimap_update = time.time()
        self.t_to_change_channel = time.time()
        # Images
        self.img_map = None
        self.img_routes = []
        self.img_login_button = None

        # Database
        self.data = load_yaml("config/config_data.yaml")
        # Threads & Objects
        self.kb = None # Keyboard controller
        self.capture = None # Game window capturor
        self.health_monitor = None # Health monitor
        self.profiler = None # Profiler, for performance issue debugging
        self.route_navigator = None
        self.route_source = "legacy"
        self.route_combat_arbiter = RouteCombatArbiter()
        self.last_route_intent = RouteIntent("none", "none", "none")
        self.last_combat_intent = CombatIntent()
        self.combat_target = None
        self.combat_target_last_seen_at = None
        self.combat_target_last_direction = None
        self.combat_target_last_state = None
        self.route_commit_until = 0.0
        self.route_commit_kind = None
        self.debug_combat_state = "observe"
        self.nametag_localizer = None
        self.nametag_last_result = None
        self.localization_score = 1.0
        self.minimap_pose_tracker = MinimapPoseTracker(
            lambda *args, **kwargs: find_pattern_sqdiff(*args, **kwargs)
        )
        self.minimap_pose_snapshot = self.minimap_pose_tracker.last_snapshot
        self.last_visual_pause_reason = ""
        self.current_minimap_roi_valid = False
        self.current_minimap_player_valid = False
        self.current_nametag_valid = False
        self.minimap_roi_source = "automatic"
        self._minimap_roi_pixels_logged = False

        # Finite State Machine
        self.fsm = FiniteStateMachine()
        self.fsm.add_state(HuntingState    ("hunting"     , self))
        self.fsm.add_state(AuxiliaryState  ("aux"         , self))
        self.fsm.add_state(PatrolState     ("patrol"      , self))
        self.fixed_platform_state = FixedPlatformState("fixed_platform", self)
        self.fsm.add_state(self.fixed_platform_state)
        self.continuous_attack_state = ContinuousAttackState(
            "continuous_attack", self
        )
        self.fsm.add_state(self.continuous_attack_state)
        self.fsm.set_init_state("hunting")

    def set_visualization_sink(self, sink):
        """Install the controller-owned capacity-one preview publisher."""
        self.visualization_sink = sink

    def set_termination_sink(self, sink):
        """Install a lightweight worker-stop notifier owned by the UI layer."""
        self.termination_sink = sink

    def request_visualization(self):
        """Request one preview; repeated requests coalesce into one event."""
        if self.is_need_show_debug_window:
            if not hasattr(self, "_visualization_request"):
                self._visualization_request = threading.Event()
            self._visualization_request.set()

    def _claim_visualization_mode(self):
        if not self.is_need_show_debug_window:
            return None
        if not getattr(self, "is_ui", False):
            # The standalone debug runner still owns both OpenCV windows.
            return "both"
        if not self._visualization_request.is_set():
            return None
        self._visualization_request.clear()
        return self.viz_mode

    @staticmethod
    def prepare_runtime_config(cfg):
        """Validate mode-specific invariants on an isolated runtime copy."""
        runtime_cfg = copy.deepcopy(cfg)
        migrate_health_monitor_config(runtime_cfg)
        mode = runtime_cfg.get("bot", {}).get("mode")
        if mode not in {
            "normal", "aux", "patrol", "fixed_platform", "continuous_attack"
        }:
            raise ValueError(f"不支持的运行模式：{mode}")
        if mode == "fixed_platform":
            width_px = runtime_cfg.get("fixed_platform", {}).get("width_px")
            if (isinstance(width_px, bool) or not isinstance(width_px, int) or
                    not 10 <= width_px <= 1000):
                raise ValueError(
                    "小地图巡逻总宽度必须是 10–1000 px 之间的整数。"
                )
            runtime_cfg["bot"]["route_only"] = False
            runtime_cfg["bot"]["route_attack"] = False
        elif mode == "continuous_attack":
            if runtime_cfg.get("bot", {}).get("attack") != "directional":
                raise ValueError("持续定向攻击模式仅支持方向攻击。")
            if not str(runtime_cfg.get("key", {}).get(
                    "directional_attack", "")).strip():
                raise ValueError("持续定向攻击模式的攻击键不能为空。")
            if not str(runtime_cfg.get("key", {}).get("jump", "")).strip():
                raise ValueError("持续定向攻击模式的跳跃键不能为空。")
            runtime_cfg["bot"]["route_only"] = False
            runtime_cfg["bot"]["route_attack"] = False
        return runtime_cfg

    def load_config(self, cfg):
        """Load route assets, the required name-tag profile and YOLO model."""
        # Runtime normalization must not mutate the UI/profile snapshot.  The
        # classic profile enables route_only, while fixed-platform mode is an
        # independent local patrol mode that must never enter route control.
        self._report_load_stage("window", "正在校验运行配置")
        try:
            cfg = self.prepare_runtime_config(cfg)
        except ValueError as exc:
            logger.error(f"[配置] {exc}")
            return -1

        self.minimap_pose_tracker = MinimapPoseTracker(
            lambda *args, **kwargs: find_pattern_sqdiff(*args, **kwargs),
            max_score=float(
                cfg.get("route", {}).get("localization_max_score", 0.20)
            ),
        )
        self.minimap_pose_snapshot = self.minimap_pose_tracker.last_snapshot

        if cfg["bot"]["mode"] != "continuous_attack":
            self._report_load_stage("roi", "正在加载当前地图 ROI 配置")
            map_name = str(cfg.get("bot", {}).get("map", "")).strip()
            try:
                self.minimap_roi_source = apply_map_minimap_roi_override(
                    cfg, map_name, Path("minimaps")
                )
            except MapProjectConfigError as exc:
                logger.error(f"[小地图ROI] 加载失败，已阻止启动：{exc}")
                return -1
            self._minimap_roi_pixels_logged = False
            logger.info(
                f"[小地图ROI] source={self.minimap_roi_source} "
                f"normalized={cfg.get('minimap', {}).get('roi')}"
            )
        else:
            self.minimap_roi_source = "unused_continuous_attack"
            self._minimap_roi_pixels_logged = False
            logger.info("[持续定向攻击] 不加载小地图 ROI、地图或路线资源。")

        # Parse color code in config
        self.color_code = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code"].items()
        }
        self.color_code_up_down = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code_up_down"].items()
        }

        if cfg["bot"]["mode"] == "normal":
            self._report_load_stage("map", "正在加载地图底图")
            map_name = cfg['bot']['map']
            self.img_map = load_image(f"minimaps/{map_name}/map.png",
                                      cv2.IMREAD_COLOR)
            self._report_load_stage("route", "正在加载并校验路线资源")
            # Load route*.png from minimaps/
            route_files = sorted(glob.glob(f"minimaps/{map_name}/route*.png"))
            route_files = [p for p in route_files if not p.endswith("route_rest.png")]
            self.img_routes = []
            for route_file in route_files:
                img = cv2.cvtColor(load_image(route_file), cv2.COLOR_BGR2RGB)
                # Remove pixel in map that is color code
                img = mask_route_colors(self.img_map, img, cfg["route"]["color_code"])
                img = mask_route_colors(self.img_map, img, cfg["route"]["color_code_up_down"])
                self.img_routes.append(img)
            if not self.img_routes:
                logger.error(f"No route images found for map: {map_name}")
                return -1
            try:
                semantic_routes = load_semantic_route_set(
                    f"minimaps/{map_name}",
                    map_id=map_name,
                    canvas_size=(self.img_map.shape[1], self.img_map.shape[0]),
                )
            except SemanticRouteError as exc:
                logger.error(f"[语义路线] 加载失败，已阻止启动：{exc}")
                return -1
            self.route_source = semantic_routes.source
            if semantic_routes.source == "semantic":
                if cfg["key"].get("teleport", "") == "" and any(
                    segment["type"] == "teleport"
                    for document in semantic_routes.documents
                    for segment in document["segments"]
                ):
                    logger.error("[语义路线] 路线包含传送动作，但传送键为空。")
                    return -1
                self.route_navigator = SemanticRouteNavigator(
                    semantic_routes.documents,
                    ladder_align_tolerance=int(cfg["route"].get(
                        "mount_horizontal_tolerance", 6
                    )),
                    ladder_retry_frames=int(cfg["route"].get(
                        "mount_retry_frames", 5
                    )),
                    ladder_success_distance=int(cfg["route"].get(
                        "mount_success_distance", 12
                    )),
                    ladder_max_attempts=int(cfg["route"].get(
                        "mount_max_attempts", 5
                    )),
                    ladder_precision_tolerance=int(cfg["route"].get(
                        "mount_precision_tolerance", 1
                    )),
                    ladder_settle_frames=int(cfg["route"].get(
                        "mount_settle_frames", 2
                    )),
                    ladder_forward_hold_frames=int(cfg["route"].get(
                        "mount_forward_hold_frames", 4
                    )),
                    ladder_retry_runup_distance=int(cfg["route"].get(
                        "mount_retry_runup_distance", 8
                    )),
                    ladder_attempt_timeout=float(cfg["route"].get(
                        "mount_attempt_timeout", 1.5
                    )),
                    ladder_confirm_frames=int(cfg["route"].get(
                        "mount_confirm_frames", 3
                    )),
                    event_commit_seconds=float(cfg["route"].get(
                        "traversal_commit_seconds", 0.45
                    )),
                )
                logger.info(
                    f"[语义路线] 已加载 {len(semantic_routes.documents)} 条有序路线。"
                )
            else:
                self.route_navigator = RouteNavigator(
                    self.img_routes,
                    self.color_code,
                    self.color_code_up_down,
                    search_range=cfg["route"]["search_range"],
                    event_rearm_distance=cfg["route"].get("event_rearm_distance", 5),
                    mount_search_range=cfg["route"].get("mount_search_range", 10),
                    mount_retry_frames=cfg["route"].get("mount_retry_frames", 5),
                    mount_success_distance=cfg["route"].get("mount_success_distance", 12),
                    mount_horizontal_tolerance=cfg["route"].get("mount_horizontal_tolerance", 6),
                )
                logger.info("[路线] 未发现 JSON，继续使用旧 PNG 执行器。")


        self.monster_detector = None
        self.monster_detection_times.clear()
        self.monster_inference_times.clear()
        self.monster_frame_pipeline_times.clear()
        self.visualization_compose_times.clear()
        self.visualization_publish_times.clear()
        self._report_load_stage("yolo", "正在加载 YOLO 部署和模型")
        detect_cfg = cfg["monster_detect"]
        try:
            self.monster_detector = OpenVinoMonsterDetector(
                detect_cfg["openvino_deployment_root"],
                confidence=float(detect_cfg.get("yolo_confidence", 0.25)),
                min_monster_box_side=detect_cfg.get(
                    "yolo_min_monster_box_side", 10
                ),
                player_exclusion_width=detect_cfg.get(
                    "yolo_player_exclusion_width", 80
                ),
                player_exclusion_height=detect_cfg.get(
                    "yolo_player_exclusion_height", 100
                ),
                max_det=RUNTIME_POLICY.yolo_max_det,
                variant=detect_cfg.get("yolo_variant", "int8_v2"),
            )
        except (KeyError, ValueError, TypeError, OpenVinoDeploymentError) as exc:
            logger.error(f"[YOLO怪物检测] 初始化失败，已阻止启动：{exc}")
            return -1
        logger.info(
            "[YOLO怪物检测] 已加载双类 monster/player 模型："
            f"variant={self.monster_detector.variant} root={self.monster_detector.root}"
        )

        self._report_load_stage("profile", "正在加载人物名字样本")
        try:
            profile_repository = NameTagProfileRepository()
            profile_name = cfg["nametag"]["name"]
            profile_dir = profile_repository.resolve_profile(profile_name)
            profile_repository.load_profile(profile_name, require_enabled=True)
            self.nametag_localizer = NameTagLocalizer.from_profile(profile_dir)
        except (KeyError, TypeError, ValueError, NameTagProfileError) as exc:
            logger.error(f"[名字定位] 加载失败，已阻止启动：{exc}")
            return -1
        logger.info(f"Loaded name-tag profile: {profile_dir}")

        if cfg["bot"]["mode"] != "continuous_attack":
            self._report_load_stage("window", "正在加载游戏界面模板")
            self.img_login_button = load_image(
                f"misc/login_button_{RUNTIME_POLICY.client_language}.png"
            )
            cfg['ui_coords']['login_button_top_left'] = normalize_pixel_coordinate(
                cfg['ui_coords']['login_button_top_left'], cfg['game_window']['size'])
            cfg['ui_coords']['login_button_bottom_right'] = normalize_pixel_coordinate(
                cfg['ui_coords']['login_button_bottom_right'], cfg['game_window']['size'])
        else:
            self.img_login_button = None

        # Print mode on log
        logger.info(f"[load_config] Config AutoBot as {cfg['bot']['mode']} mode")

        # Update cfg
        self.cfg = cfg
        self._report_load_stage(None, "资源检查完成，等待当前画面识别")

        return 0 # load successfully

    def _report_load_stage(self, key, detail):
        sink = getattr(self, "load_status_sink", None)
        if sink is not None:
            sink(key, detail)

    def start(self):
        '''
        Start all threads
        '''
        previous_thread = getattr(self, "thread_auto_bot", None)
        if previous_thread is not None and previous_thread.is_alive():
            raise RuntimeError("上一次运行尚未安全退出，请稍后再启动。")
        self.is_terminated = False
        self.minimap_pose_tracker.reset()
        self.minimap_pose_snapshot = self.minimap_pose_tracker.last_snapshot
        self.run_generation += 1
        run_generation = self.run_generation
        # Resolve the exact target window before any input thread exists.
        if self.args.test_image == '':
            self.capture = GameWindowCapturor(self.cfg)
        else:
            self.capture = GameWindowCapturor(self.cfg, self.args.test_image)

        # Start keyboard controller thread.  Live capture supplies the exact
        # title used to resolve a stable HWND instead of a title substring.
        target_window_title = self.capture.window_title or None
        self.kb = KeyBoardController(
            self.cfg,
            target_window_title=target_window_title,
            target_hwnd=getattr(self.capture, "window_hwnd", None),
        )
        if self.is_disable_control:
            self.kb.disable() # Disable keyboard controller for debugging

        # Start health monitoring thread
        self.health_monitor = HealthMonitor(self.cfg, self.kb)
        if self.health_monitor.active:
            self.health_monitor.start()

        # Init profiler
        self.profiler = Profiler(self.cfg)

        # Reset all timers
        self.t_last_frame = time.time()
        self.t_watch_dog = time.time()
        self.t_last_teleport = time.time()
        self.t_last_attack = time.time()
        self.t_last_minimap_update = time.time()
        self.t_to_change_channel = time.time()

        # Set init state
        if self.args.init_state != "":
            self.fsm.set_init_state(self.args.init_state) # For debugging
        else:
            self.set_state_for_configured_mode()

        # Start Auto Bot main thread
        self.thread_auto_bot = threading.Thread(
            target=self.loop,
            args=(run_generation,),
        )
        self.thread_auto_bot.start()
        self.is_first_frame = True

        logger.info("[MapleStoryAutoBot] Started")

    def set_state_for_configured_mode(self):
        """Enter the FSM state owned by the active main-program mode."""
        state_name = {
            "aux": "aux",
            "patrol": "patrol",
            "fixed_platform": "fixed_platform",
            "continuous_attack": "continuous_attack",
        }.get(self.cfg["bot"]["mode"], "hunting")
        self.fsm.set_init_state(state_name)

    def pause(self):
        '''
        Terminate thread except main thread
        '''
        self.terminate_threads()

    def enable_viz(self, mode="game"):
        '''
        Enable AutoBot to generate debug image
        '''
        if mode not in {"game", "route"}:
            raise ValueError(f"Unsupported visualization mode: {mode}")
        self.viz_mode = mode
        self.t_last_viz_frame = 0.0
        self.is_need_show_debug_window = True
        if not hasattr(self, "_visualization_request"):
            self._visualization_request = threading.Event()
        self._visualization_request.clear()
        self.request_visualization()
        logger.debug(f"[enable_viz] mode={mode}")

    def disable_viz(self):
        '''
        Disable AutoBot to generate debug image
        '''
        self.is_need_show_debug_window = False
        self.is_show_debug_window = False
        self._frame_visualization_mode = None
        if hasattr(self, "_visualization_request"):
            self._visualization_request.clear()
        logger.debug("[disable_viz] is_show_debug_window = False")

    def is_viz_frame_due(self, now=None):
        """Return whether the current frontend has requested a fresh preview."""
        if not self.is_need_show_debug_window:
            return False
        return (not getattr(self, "is_ui", False) or
                getattr(self, "_visualization_request", threading.Event()).is_set())

    def get_player_location_by_nametag(self):
        """Return the player foot from the required calibrated name-tag profile."""
        if self.nametag_localizer is None:
            raise RuntimeError("名字定位器尚未初始化。")
        # Name-tag search and health-bar extraction are different ROIs.  The
        # classic client can place the player name below ui_y_start, so tying
        # both features to that boundary silently makes the tag unreachable.
        configured_limit = self.cfg.get("nametag", {}).get("search_y_limit", 0)
        y_limit = int(configured_limit) if int(configured_limit) > 0 else \
            int(self.cfg["ui_coords"]["ui_y_start"])
        result = self.nametag_localizer.locate(
            self.img_frame,
            y_limit=min(y_limit, self.img_frame.shape[0]),
        )
        self.nametag_last_result = result
        if not result.valid:
            return None
        self.loc_nametag = result.tag_top_left
        if self.img_frame_debug is not None:
            width, height = result.tag_size
            draw_rectangle(
                self.img_frame_debug, result.tag_top_left,
                (height, width), (0, 255, 0),
                f"NameTag {result.score:.3f}", thickness=1, text_height=0.5,
            )
            cv2.circle(
                self.img_frame_debug, result.player, radius=4,
                color=(255, 0, 255), thickness=-1,
            )
        return result.player

    def get_player_location_on_global_map(self):
        '''
        get_player_location_on_global_map
        '''
        self._diagnostic_map_checked = True
        mask = np.any(self.img_minimap != [0, 0, 0], axis=2).astype(
            np.uint8
        ) * 255
        snapshot = self.minimap_pose_tracker.update(
            self.img_map,
            self.img_minimap,
            self.loc_player_minimap,
            mask=mask,
            offset=self.cfg["minimap"].get("offset", [0, 0]),
            allow_large_jump=(
                getattr(self, "last_route_intent", None) is not None
                and self.last_route_intent.action == "teleport"
            ),
        )
        self.minimap_pose_snapshot = snapshot
        self.localization_score = float(
            snapshot.score if snapshot.score is not None else 1.0
        )
        if not snapshot.valid or snapshot.stable_position is None:
            return None
        self.loc_minimap_global = snapshot.camera_position
        loc_player_global = snapshot.stable_position

        # Draw local minimap rectangle
        camera_bottom_right = (
            self.loc_minimap_global[0] + self.img_minimap.shape[1],
            self.loc_minimap_global[1] + self.img_minimap.shape[0]
        )
        if self.img_route_debug is not None:
            cv2.rectangle(self.img_route_debug, self.loc_minimap_global,
                          camera_bottom_right, (0, 255, 255), 1)
            cv2.putText(
                self.img_route_debug,
                f"Minimap,{snapshot.source},score({round(self.localization_score, 2)})",
                (self.loc_minimap_global[0], self.loc_minimap_global[1]+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 255), 1
            )
            cv2.circle(self.img_route_debug, loc_player_global, radius=2,
                       color=(0, 255, 255), thickness=-1)

        return loc_player_global

    def get_nearest_color_code(self):
        '''
        Searches for the nearest color-coded action marker
        around the player on the route map.

        This function:
        - Scans each pixel in the search box to find nearest color code
        - Tracks the closest matching pixel using Manhattan distance (|dx| + |dy|).
        - Returns a dictionary containing the nearest matching
          pixel's position, color, action label, and distance.

        Returns:
            dict or None: Dictionary containing:
                - "pixel": (x, y) coordinate of the matched pixel
                - "color": matched RGB color tuple
                - "action": corresponding action string from config
                - "distance": Manhattan distance from player
            Returns None if no matching color is found within the region.
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        x_min = max(0, x0 - self.cfg["route"]["search_range"])
        x_max = min(w, x0 + self.cfg["route"]["search_range"])
        y_min = max(0, y0 - self.cfg["route"]["search_range"])
        y_max = min(h, y0 + self.cfg["route"]["search_range"])

        nearest = None
        nearest_up_down = None
        min_dist = float('inf')
        min_dist_up_down = float('inf')
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                pixel = tuple(self.img_route[y, x])  # (R, G, B)
                dist = abs(x - x0) + abs(y - y0)
                # Get nearest color
                if pixel in self.color_code and dist < min_dist:
                    nearest = {
                        "pixel": (x, y),
                        "color": pixel,
                        "command": self.color_code[pixel],
                        "distance": dist
                    }
                    min_dist = dist
                # Get nearest color (up, dowm)
                if pixel in self.color_code_up_down and dist < min_dist_up_down:
                    nearest_up_down = {
                        "pixel": (x, y),
                        "color": pixel,
                        "command": self.color_code_up_down[pixel],
                        "distance": dist
                    }
                    min_dist_up_down = dist

        # Debug is strictly optional and must not allocate or affect routing.
        if self.img_route_debug is not None:
            draw_rectangle(
                self.img_route_debug,
                (x_min, y_min),
                (self.cfg["route"]["search_range"]*2,
                 self.cfg["route"]["search_range"]*2),
                (0, 0, 255), "", text_height=0.4, thickness=1,
            )
        # Draw a straigt line from map_loc_player to color_code["pixel"]
        if nearest is not None:
            if self.img_route_debug is not None:
                cv2.line(
                    self.img_route_debug,
                    self.loc_player_global,
                    nearest["pixel"],
                    (0, 255, 0),
                    1,
                )
            # Print color code on debug image
            if self.img_frame_debug is not None:
                cv2.putText(
                    self.img_frame_debug, f"Route Action: {nearest['command']}",
                    (650, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                    2, cv2.LINE_AA
                )
                cv2.putText(
                    self.img_frame_debug, f"Route Index: {self.idx_routes}",
                    (650, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                    2, cv2.LINE_AA
                )

        if nearest_up_down is not None:
            if self.img_frame_debug is not None:
                cv2.putText(
                    self.img_frame_debug,
                    f"Route Action: {nearest_up_down['command']}",
                    (650, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                    2, cv2.LINE_AA
                )
            if self.img_route_debug is not None:
                cv2.line(
                    self.img_route_debug,
                    self.loc_player_global,
                    nearest_up_down["pixel"],
                    (0, 0, 255),
                    1,
                )

        return nearest, nearest_up_down  # if not found return none

    def get_attack_range(self, is_left=True):
        '''
        get_attack_range
        '''
        if self.cfg["bot"]["attack"] == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2
            dy = self.cfg["aoe_skill"]["range_y"] // 2
            x0 = max(0, self.loc_player[0] - dx)
            x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
            y0 = max(0, self.loc_player[1] - dy)
            y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        elif self.cfg["bot"]["attack"] == "directional":
            if is_left:
                x0 = self.loc_player[0] - self.cfg["directional_attack"]["range_x"]
                x1 = self.loc_player[0]
            else:
                x0 = self.loc_player[0]
                x1 = x0 + self.cfg["directional_attack"]["range_x"]
            y0 = self.loc_player[1] - self.cfg["directional_attack"]["range_y"] // 2
            y1 = y0 + self.cfg["directional_attack"]["range_y"]
        else:
            raise RuntimeError(f"Unsupported attack mode: {self.cfg['bot']['attack']}")

        return (x0, y0, x1, y1)

    def get_monster_search_range(self):
        """Return the visible part of the fixed 1280x224 YOLO input strip."""
        if self.img_frame is None or self.loc_player is None:
            return (0, 0, 0, 0)
        return OpenVinoMonsterDetector.visible_input_rect(
            self.img_frame.shape, self.loc_player
        )

    def get_combined_attack_range(self):
        """Return the attack-area rectangle shown in Window Viz."""
        if self.cfg["bot"]["attack"] == "aoe_skill":
            attack_range = self.get_attack_range()
        else:
            left = self.get_attack_range(is_left=True)
            right = self.get_attack_range(is_left=False)
            attack_range = (left[0], left[1], right[2], right[3])

        x0, y0, x1, y1 = attack_range
        frame_h, frame_w = self.img_frame.shape[:2]
        return (max(0, x0), max(0, y0), min(frame_w, x1), min(frame_h, y1))

    def draw_combat_ranges_debug(self, canvas=None):
        """Always render the theoretical detection and attack geometry."""
        if canvas is None:
            canvas = self.img_frame_debug
        if canvas is None or self.loc_player is None:
            return

        detect_x0, detect_y0, detect_x1, detect_y1 = \
            self.get_monster_search_range()
        cv2.rectangle(
            canvas,
            (detect_x0, detect_y0), (detect_x1, detect_y1),
            (255, 0, 0), 2,
        )

        attack_x0, attack_y0, attack_x1, attack_y1 = \
            self.get_combined_attack_range()
        cv2.rectangle(
            canvas,
            (attack_x0, attack_y0), (attack_x1, attack_y1),
            (0, 0, 255), 2,
        )
        if self.cfg["bot"]["attack"] == "directional":
            center_x = min(max(0, self.loc_player[0]), self.img_frame.shape[1])
            cv2.line(
                canvas,
                (center_x, attack_y0), (center_x, attack_y1),
                (0, 0, 255), 1,
            )

    def draw_player_exclusion_debug(self, canvas=None):
        """Draw the exact player-area filter used by the current YOLO frame."""
        if canvas is None:
            canvas = self.img_frame_debug
        if (canvas is None or self.monster_detector is None or
                not getattr(self, "current_nametag_valid", False)):
            return
        rect = self.monster_detector.player_exclusion_rect(
            canvas.shape, self.loc_player
        )
        if rect is None:
            return
        x0, y0, x1, y1 = rect
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 165, 255), 2)

    def draw_fixed_platform_debug(self, canvas=None):
        """Draw fixed-platform bounds inside the existing minimap preview."""
        if canvas is None:
            canvas = self.img_frame_debug
        if (canvas is None or self.cfg["bot"].get("mode") != "fixed_platform" or
                self.img_minimap is None or self.img_minimap.size == 0):
            return

        diagnostics = self.fixed_platform_state.get_diagnostics()
        minimap_x, minimap_y = self.loc_minimap
        minimap_h, _ = self.img_minimap.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]
        top = min(max(0, int(minimap_y)), max(0, canvas_h - 1))
        bottom = min(max(top, int(minimap_y + minimap_h - 1)), canvas_h - 1)

        def draw_marker(local_x, color, thickness):
            if local_x is None:
                return
            screen_x = min(
                max(0, int(round(minimap_x + local_x))), canvas_w - 1
            )
            cv2.line(canvas, (screen_x, top), (screen_x, bottom), color, thickness)

        draw_marker(diagnostics["left_x"], (0, 165, 255), 2)
        draw_marker(diagnostics["right_x"], (0, 165, 255), 2)
        draw_marker(diagnostics["anchor_x"], (255, 255, 0), 1)

        if diagnostics["anchor_x"] is None:
            status = "Fixed platform: waiting localization"
            color = (0, 0, 255)
        else:
            state = "active" if diagnostics["localized"] else "waiting"
            status = (
                f"Fixed platform: {state} dir={diagnostics['direction']} "
                f"range={diagnostics['left_x']}-{diagnostics['right_x']} "
                f"width={diagnostics['actual_width']}/"
                f"{diagnostics['requested_width']}"
            )
            color = (0, 255, 255) if diagnostics["localized"] else (0, 0, 255)
        text_y = min(canvas_h - 5, max(15, bottom + 18))
        cv2.putText(
            canvas, status, (max(5, int(minimap_x)), text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    def draw_continuous_attack_debug(self, canvas=None):
        """Draw locked direction and timing without repeating inference."""
        if canvas is None:
            canvas = self.img_frame_debug
        if (canvas is None or
                self.cfg["bot"].get("mode") != "continuous_attack"):
            return
        diagnostics = self.continuous_attack_state.get_diagnostics()
        direction = diagnostics["locked_direction"] or "waiting"
        remaining = diagnostics["remaining_seconds"]
        remaining_text = "--" if remaining is None else f"{remaining:.1f}s"
        color = (0, 165, 255) if direction == "waiting" else (0, 255, 0)
        lines = (
            f"Continuous: phase={diagnostics['phase']}",
            f"Locked direction: {direction} remaining={remaining_text}",
            f"Healing priority: {diagnostics['healing']}",
        )
        for index, text in enumerate(lines):
            cv2.putText(
                canvas, text, (10, 400 + index * 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

    def record_control_frame_pipeline(self, started_at):
        """Record capture-through-command time, excluding optional preview work."""
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        self.monster_frame_pipeline_times.append(elapsed_ms)

    @staticmethod
    def _p95(samples):
        values = np.asarray(samples, dtype=np.float64)
        return float(np.percentile(values, 95)) if values.size else 0.0

    def get_visualization_performance(self):
        return {
            "control_p95_ms": self._p95(
                getattr(self, "monster_frame_pipeline_times", ())
            ),
            "infer_p95_ms": self._p95(
                getattr(self, "monster_inference_times", ())
            ),
            "viz_p95_ms": self._p95(
                getattr(self, "visualization_compose_times", ())
            ),
            "publish_p95_ms": self._p95(
                getattr(self, "visualization_publish_times", ())
            ),
        }

    @staticmethod
    def _resize_preview(image, max_width, interpolation):
        if image is None or image.size == 0:
            return None
        height, width = image.shape[:2]
        if width <= max_width:
            return np.ascontiguousarray(image)
        target_height = max(1, round(height * max_width / width))
        return cv2.resize(
            image, (max_width, target_height), interpolation=interpolation
        )

    def _prepare_visualization_buffers(self):
        """Allocate only the debug buffer requested for this control frame."""
        mode = self._frame_visualization_mode
        self.img_frame_debug = (
            self.img_frame.copy() if mode in {"game", "both"} else None
        )
        self.img_route_debug = (
            cv2.cvtColor(self.img_route, cv2.COLOR_RGB2BGR)
            if mode in {"route", "both"} and self.img_route is not None
            else None
        )

    def get_frame_debug_for_viz(self):
        """Compose the requested game preview once on its dedicated copy."""
        if self.img_frame_debug is None:
            return None
        source_canvas = self.img_frame_debug
        is_ui_preview = getattr(self, "is_ui", False)
        # UI previews are composed exactly once.  The standalone debug runner
        # may ask for the same frame repeatedly, so keep its canvas immutable.
        canvas = (
            source_canvas if is_ui_preview else source_canvas.copy()
        )
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        elif canvas.ndim != 3 or canvas.shape[2] != 3:
            logger.error(f"[Viz] Unsupported debug frame shape: {canvas.shape}")
            return None
        if is_ui_preview:
            self.img_frame_debug = canvas
            if hasattr(self, "t_last_frame"):
                self.update_info_on_img_frame_debug()
        elif hasattr(self, "t_last_frame"):
            self.img_frame_debug = canvas
            try:
                self.update_info_on_img_frame_debug()
            finally:
                self.img_frame_debug = source_canvas
        can_draw_detection = self.loc_player is not None
        if (self.cfg["bot"].get("mode") == "fixed_platform" and
                not self.current_nametag_valid):
            can_draw_detection = False
        if self.cfg["bot"].get("mode") == "continuous_attack":
            # The lock frame may show the boxes that established direction.
            # The next frame clears current_nametag_valid and all detections,
            # so this never paints a stale startup observation.
            can_draw_detection = self.current_nametag_valid
        if can_draw_detection:
            x0, y0, x1, y1 = self.get_monster_search_range()
            self.draw_monster_detection_debug(
                (x0, y0), (x1, y1), self.monsters, canvas=canvas
            )
        if self.cfg["bot"].get("mode") != "continuous_attack" or \
                can_draw_detection:
            self.draw_combat_ranges_debug(canvas)
            self.draw_player_exclusion_debug(canvas)
        self.draw_fixed_platform_debug(canvas)
        self.draw_continuous_attack_debug(canvas)
        self.draw_ladder_execution_debug(canvas)
        return self._resize_preview(
            canvas,
            RUNTIME_POLICY.preview_max_width,
            cv2.INTER_AREA,
        )

    def get_route_debug_for_viz(self):
        """Scale the requested route preview without touching the game frame."""
        if self.img_route_debug is None:
            return None
        scale = float(self.cfg["minimap"]["debug_window_upscale"])
        image = self.img_route_debug
        if scale != 1.0:
            image = cv2.resize(
                image, (0, 0), fx=scale, fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
        return self._resize_preview(
            image, RUNTIME_POLICY.preview_max_width, cv2.INTER_NEAREST
        )

    def _ensure_route_combat_state(self):
        """Initialize combat state for normal construction and lightweight tests."""
        defaults = {
            "route_combat_arbiter": RouteCombatArbiter(),
            "last_route_intent": RouteIntent("none", "none", "none"),
            "last_combat_intent": CombatIntent(),
            "combat_target": None,
            "combat_target_last_seen_at": None,
            "combat_target_last_direction": None,
            "combat_target_last_state": None,
            "route_commit_until": 0.0,
            "route_commit_kind": None,
            "debug_combat_state": "observe",
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    @staticmethod
    def _monster_center(monster):
        x, y = monster["position"]
        width, height = monster["size"]
        return x + width // 2, y + height // 2

    @staticmethod
    def _monster_foot(monster):
        x, y = monster["position"]
        width, height = monster["size"]
        return x + width // 2, y + height

    def _monster_direction(self, monster):
        center_x, _ = self._monster_center(monster)
        if center_x <= self.loc_player[0]:
            return "left"
        return "right"

    def is_monster_in_attack_range(self, monster):
        """Return whether one candidate overlaps its directional attack box."""
        direction = self._monster_direction(monster)
        if direction not in {"left", "right"}:
            return False
        x0, y0, x1, y1 = self.get_attack_range(is_left=direction == "left")
        mx, my = monster["position"]
        width, height = monster["size"]
        overlap_width = max(0, min(x1, mx + width) - max(x0, mx))
        overlap_height = max(0, min(y1, my + height) - max(y0, my))
        return overlap_width > 0 and overlap_height > 0

    def _monster_is_same_level(self, monster):
        """Conservative no-A* pursuit gate based on player and monster feet."""
        _, monster_foot_y = self._monster_foot(monster)
        tolerance = int(self.cfg["combat_tracking"].get(
            "pursuit_vertical_tolerance", 70
        ))
        return abs(monster_foot_y - self.loc_player[1]) <= tolerance

    def _monster_horizontal_distance(self, monster):
        """Shortest horizontal gap from the player X to one monster box."""
        x, _ = monster["position"]
        width, _ = monster["size"]
        player_x = self.loc_player[0]
        if player_x < x:
            return x - player_x
        if player_x > x + width:
            return player_x - (x + width)
        return 0

    def _combat_candidate_rank(self, monster):
        """Order same-level candidates by horizontal proximity and confidence."""
        center_x, _ = self._monster_center(monster)
        _, foot_y = self._monster_foot(monster)
        return (
            self._monster_horizontal_distance(monster),
            float(monster.get("score", 1.0)),
            abs(center_x - self.loc_player[0]),
            center_x,
            foot_y,
        )

    def _target_lost_grace_seconds(self):
        return float(self.cfg["combat_tracking"].get(
            "target_lost_grace_seconds", 0.3
        ))

    def reset_route_combat(self, now=None):
        """Release target ownership; Route will be sampled again next frame."""
        self._ensure_route_combat_state()
        self.combat_target = None
        self.combat_target_last_seen_at = None
        self.combat_target_last_direction = None
        self.combat_target_last_state = None
        self.last_combat_intent = CombatIntent()
        self.debug_combat_state = "observe"

    def _select_nearest_combat_target(self, now):
        """Select the nearest same-level detection on every fresh frame."""
        self._ensure_route_combat_state()
        eligible = [
            monster for monster in self.monsters
            if self._monster_is_same_level(monster)
        ]
        if eligible:
            self.combat_target = min(eligible, key=self._combat_candidate_rank)
            self.combat_target_last_seen_at = now
            self.combat_target_last_direction = \
                self._monster_direction(self.combat_target)
            return self.combat_target, True

        if (self.combat_target is not None and
                self.combat_target_last_seen_at is not None and
                now - self.combat_target_last_seen_at <=
                self._target_lost_grace_seconds()):
            return self.combat_target, False

        self.reset_route_combat(now=now)
        return None, False

    def build_route_combat_intent(
        self, route_intent, now=None, control_allowed=None
    ):
        """Build high-priority local pursuit/attack intent without pathfinding."""
        self._ensure_route_combat_state()
        now = time.time() if now is None else now
        target, visible = self._select_nearest_combat_target(now)
        if target is None:
            return CombatIntent()

        direction = (self._monster_direction(target) if visible else
                     self.combat_target_last_direction)
        if direction not in {"left", "right"}:
            return CombatIntent(target=target, target_visible=visible)

        if control_allowed is None:
            control_allowed = not route_intent.is_critical
        if not control_allowed:
            return CombatIntent(
                "traversal_wait", direction, target=target,
                target_visible=visible,
            )

        cooldown = float(self.cfg["directional_attack"]["cooldown"])
        if not visible:
            if self.combat_target_last_state == "visible_attack":
                action = "none"
                if now - self.t_last_attack > cooldown:
                    action = f"attack_{direction}"
                    self.t_last_attack = now
                return CombatIntent(
                    "lost_grace_attack", direction, action, target, False
                )
            return CombatIntent(
                "lost_grace_pursuit", direction, "none", target, False
            )

        if self.is_monster_in_attack_range(target):
            self.combat_target_last_state = "visible_attack"
            action = "none"
            if now - self.t_last_attack > cooldown:
                action = f"attack_{direction}"
                self.t_last_attack = now
            return CombatIntent(
                "visible_attack", direction, action, target, True
            )

        self.combat_target_last_state = "visible_pursuit"
        return CombatIntent(
            "visible_pursuit", direction, "none", target, True
        )

    def get_nearest_monster(self, is_left=True):
        """Return the nearest YOLO monster overlapping one attack half."""
        x0, y0, x1, y1 = self.get_attack_range(is_left=is_left)
        candidates = []
        for monster in self.monsters:
            mx, my = monster["position"]
            width, height = monster["size"]
            if min(x1, mx + width) <= max(x0, mx):
                continue
            if min(y1, my + height) <= max(y0, my):
                continue
            center_x, center_y = self._monster_center(monster)
            distance = abs(center_x - self.loc_player[0]) + abs(
                center_y - self.loc_player[1]
            )
            candidates.append((distance, float(monster.get("score", 1.0)), monster))
        return min(candidates, default=(None, None, None))[-1]

    def draw_monster_detection_debug(
        self, top_left, bottom_right, monsters, canvas=None
    ):
        """Draw YOLO diagnostics without invoking detection a second time."""
        if canvas is None:
            canvas = self.img_frame_debug
        if canvas is None:
            return

        for monster in monsters:
            width, height = monster["size"]
            draw_rectangle(
                canvas, monster["position"], (height, width),
                (0, 255, 0),
                f"monster {float(monster.get('confidence', 0.0)):.2f}",
            )

        detector = self.monster_detector
        if detector is not None:
            for player in getattr(detector, "last_players", ()):
                width, height = player["size"]
                draw_rectangle(
                    canvas, player["position"], (height, width),
                    (255, 0, 255),
                    f"player {float(player.get('confidence', 0.0)):.2f}",
                    thickness=1, text_height=0.45,
                )
            detected_foot = getattr(detector, "last_player_foot", None)
            if detected_foot is not None:
                cv2.circle(
                    canvas, detected_foot, radius=3,
                    color=(255, 0, 255), thickness=-1,
                )

        timing = detector.last_timing if detector is not None else None
        performance = self.get_visualization_performance()
        timing_text = (
            f"YOLO: monsters={timing.monsters} players={timing.players} "
            f"crop={timing.crop_ms:.1f} infer={timing.infer_ms:.1f} "
            f"convert={timing.convert_ms:.1f} total={timing.total_ms:.1f}ms"
            if timing is not None else "YOLO: not initialized"
        )
        performance_text = (
            f"control_p95={performance['control_p95_ms']:.1f}ms "
            f"infer_p95={performance['infer_p95_ms']:.1f}ms "
            f"viz_p95={performance['viz_p95_ms']:.1f}ms"
        )
        cv2.putText(
            canvas, timing_text,
            (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 255), 2,
        )
        cv2.putText(
            canvas, performance_text,
            (10, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 255), 2,
        )
        cv2.putText(
            canvas,
            "Attack direction (observe only): "
            f"{getattr(self, 'debug_attack_direction', 'none')}",
            (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 255), 2,
        )

    def get_img_frame(self):
        '''
        get_img_frame
        '''
        # Get window game raw frame
        if hasattr(self.capture, "get_frame_packet"):
            packet = self.capture.get_frame_packet()
            if packet is None:
                self.frame = None
                self.frame_captured_at = 0.0
                self.capture_frame_sequence = None
            else:
                (self.frame, self.frame_captured_at,
                 self.capture_frame_sequence) = packet
        else:
            self.frame = self.capture.get_frame()
            self.frame_captured_at = (
                time.monotonic() if self.frame is not None else 0.0
            )
            self.capture_frame_sequence = None
        if self.frame is None:
            logger.warning("Failed to capture game frame.")
            return

        if not self.cfg["game_window"].get("resize_on_start", True):
            return prepare_game_frame(self.frame, self.cfg)

        # Cut the title bar and resize raw frame to (1296, 759)
        frame_no_title = self.frame[self.cfg["game_window"]["title_bar_height"]:, :]

        # Make sure the window ratio is as expected
        if self.args.test_image != "":
            pass # Disable size check if using test image for debugging
        elif self.cfg["bot"]["mode"] == "aux":
            if not is_img_16_to_9(frame_no_title, self.cfg): # Aux mode allow 16:9 resolution
                text = f"Unexpeted window size: {frame_no_title.shape[:2]} (expect window ratio 16:9)\n"
                text += "Please use windowed mode & smallest resolution."
                logger.error(text)
                return
        else:
            # Other mode only allow specific resolution
            if self.cfg["game_window"]["size"] != frame_no_title.shape[:2]:
                text = f"Unexpeted window size: {frame_no_title.shape[:2]} "\
                       f"(expect {self.cfg['game_window']['size']})\n"
                text += "Please use windowed mode & smallest resolution."
                logger.error(text)
                return

        return cv2.resize(frame_no_title, WINDOW_WORKING_SIZE,
                   interpolation=cv2.INTER_NEAREST)

    def is_player_stuck(self):
        """
        Checks whether the player is stuck (not moving)
        based on their global position on map.

        This function:
        - Compares the player's current position with their last known position
          tracked by the watchdog.
        - If the player has moved beyond a threshold (`watch_dog_range`),
          it resets the watchdog timer.
        - If the player hasn't moved and the elapsed time exceeds (`watch_dog_timeout`),
          it flags the player as stuck and resets the watchdog.

        Returns:
            bool: True if the player is stuck, False otherwise.
        """
        dx = abs(self.loc_player_global[0] - self.loc_watch_dog[0])
        dy = abs(self.loc_player_global[1] - self.loc_watch_dog[1])

        current_time = time.time()
        if dx + dy > self.cfg["watchdog"]["range"]:
            # Player moved, reset watchdog timer
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            return False

        dt = current_time - self.t_watch_dog
        if dt > self.cfg["watchdog"]["timeout"]:
            # watch dog idle for too long, player stuck
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            logger.warning(f"[is_player_stuck] Player stuck for {round(dt, 2)} seconds.")
            return True
        return False

    def is_near_edge(self):
        '''
        Detects whether the player is near a teleport edge region

        This function:
        - Defines a rectangular search region around the player's current global location.
        - Scans for pixels matching a specific edge teleport color code within the region.
        - If matching pixels are found, it computes the average X position of those pixels.
        - Compares that average to the player's X position to determine whether the edge is on the left or right.

        Returns:
            str: One of:
                - "edge on left"
                - "edge on right"
                - "" (empty string if no edge is detected nearby)
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        h_trigger_box = self.cfg["edge_teleport"]["trigger_box_height"]
        w_trigger_box = self.cfg["edge_teleport"]["trigger_box_width"]
        x_min = max(0, x0 - w_trigger_box//2)
        x_max = min(w, x0 + w_trigger_box//2)
        y_min = max(0, y0 - h_trigger_box//2)
        y_max = min(h, y0 + h_trigger_box//2)

        # Debug: draw search box
        # draw_rectangle(
        #     self.img_route_debug,
        #     (x_min, y_min),
        #     (y_max - y_min, x_max - x_min),
        #     (0, 0, 255), "Edge Check", thickness=1, text_height=0.4
        # )

        # Find mask of matching pixels
        roi = self.img_route[y_min:y_max, x_min:x_max]
        mask = np.all(roi == self.cfg["edge_teleport"]["color_code"], axis=2)
        coords = np.column_stack(np.where(mask))

        # No edge pixel
        if coords.size == 0:
            return ""

        # Calculate mean position of matching pixels
        mean_x = np.mean(coords[:, 1])

        # Compare to roi center
        if mean_x < x0:
            return "edge on left"
        else:
            return "edge on right"

    def update_info_on_img_frame_debug(self):
        '''
        update_info_on_img_frame_debug
        '''
        if self.img_frame_debug is None:
            return
        # Print text at bottom left corner
        elapsed = max(time.time() - self.t_last_frame, 1e-6)
        self.fps = round(1.0 / elapsed)
        text_y_interval = 23
        text_y_start = 460
        h, w = self.frame.shape[:2]
        text_list = [
            f"FPS: {self.fps}",
            f"State: {self.fsm.state.name}",
            f"Resolution: {h}x{w}, Ratio: {round(w/h, 2)}",
            f"Press 'F1' to {'pause' if self.kb.is_enable else 'start'} Bot",
             "Press 'F12' to quit"]
        for idx, text in enumerate(text_list):
            cv2.putText(
                self.img_frame_debug, text,
                (10, text_y_start + text_y_interval*idx),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        self.draw_health_monitor_debug()
        if self.cfg["bot"]["mode"] == "continuous_attack":
            return

        # Draw minimap rectangle on img debug
        draw_rectangle(
            self.img_frame_debug,
            self.loc_minimap,
            self.img_minimap.shape[:2],
            (0, 0, 255), "minimap",thickness=2
        )

        # Modes without a route image stop after drawing the live minimap.
        if self.cfg["bot"]["mode"] in ["patrol", "aux", "fixed_platform"]:
            return

        # Compute crop region with boundary check
        crop_w, crop_h = 80, 80
        x0 = max(0, self.loc_player_global[0] - crop_w // 2)
        y0 = max(0, self.loc_player_global[1] - crop_h // 2)
        route_h, route_w = self.img_route.shape[:2]
        x1 = min(route_w, x0 + crop_w)
        y1 = min(route_h, y0 + crop_h)

        # Check if valid crop region
        if x1 <= x0 or y1 <= y0:
            return

        # Crop region
        mini_map_crop = cv2.cvtColor(
            self.img_route[y0:y1, x0:x1], cv2.COLOR_RGB2BGR
        )
        mini_map_crop = cv2.resize(mini_map_crop,
                                (int(mini_map_crop.shape[1] * 3),
                                 int(mini_map_crop.shape[0] * 3)),
                                interpolation=cv2.INTER_NEAREST)
        # Paste into top-right corner of self.img_frame_debug
        h_crop, w_crop = mini_map_crop.shape[:2]
        h_frame, w_frame = self.img_frame_debug.shape[:2]
        x_paste = w_frame - w_crop - 10  # 10px margin from right
        y_paste = 10
        self.img_frame_debug[y_paste:y_paste + h_crop, x_paste:x_paste + w_crop] = mini_map_crop

        # Draw border around minimap
        cv2.rectangle(
            self.img_frame_debug,
            (x_paste, y_paste),
            (x_paste + w_crop, y_paste + h_crop),
            color=(255, 255, 255),   # White border
            thickness=2
        )

        # Print command on screen
        cv2.putText(self.img_frame_debug, f"Cmd: {self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}",
                    (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def draw_health_monitor_debug(self):
        """Render one internally consistent health snapshot on game preview."""
        health_monitor = getattr(self, "health_monitor", None)
        if (self.img_frame_debug is None or health_monitor is None or
                not health_monitor.active):
            return
        snapshot = health_monitor.get_snapshot()
        values = (
            snapshot.hp_percent, snapshot.mp_percent, snapshot.exp_percent
        )
        rects = (snapshot.hp_rect, snapshot.mp_rect, snapshot.exp_rect)
        colors = ((0, 0, 255), (255, 0, 0), (0, 255, 255))
        frame_h, frame_w = self.img_frame_debug.shape[:2]
        origin_x, origin_y = snapshot.roi_origin

        if snapshot.state == "CALIBRATING":
            search_y = min(max(0, origin_y), frame_h - 1)
            cv2.rectangle(
                self.img_frame_debug, (0, search_y),
                (frame_w - 1, frame_h - 1), (0, 165, 255), 1,
            )

        for index, (bar_name, value, rect, color) in enumerate(zip(
                ("HP", "MP", "EXP"), values, rects, colors)):
            value_text = "--" if value is None else f"{value:.1f}%"
            cv2.putText(
                self.img_frame_debug, f"{bar_name}: {value_text}",
                (250, 30 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2,
            )
            if rect is None:
                continue
            x, y, width, height = rect
            x0 = max(0, origin_x + x)
            y0 = max(0, origin_y + y)
            x1 = min(frame_w, origin_x + x + width)
            y1 = min(frame_h, origin_y + y + height)
            if x1 <= x0 or y1 <= y0:
                continue
            cv2.rectangle(
                self.img_frame_debug, (x0, y0), (x1 - 1, y1 - 1), color, 2
            )
            bar = self.img_frame[y0:y1, x0:x1]
            paste_x, paste_y = 410, 13 + 30 * index
            paste_x1 = paste_x + bar.shape[1]
            paste_y1 = paste_y + bar.shape[0]
            if paste_x1 <= frame_w and paste_y1 <= frame_h:
                self.img_frame_debug[paste_y:paste_y1, paste_x:paste_x1] = bar

        cv2.putText(
            self.img_frame_debug,
            f"Health: {snapshot.state} conf={snapshot.confidence:.2f} "
            f"reason={snapshot.reason}",
            (250, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 165, 255), 1, cv2.LINE_AA,
        )

    def update_img_frame_debug(self):
        '''
        update_img_frame_debug
        '''
        cv2.imshow("Game Window Debug", self.img_frame_debug)
        # Update FPS timer
        self.t_last_frame = time.time()

    def channel_change(self):
        '''
        channel_change
        '''
        logger.info("[channel_change] Start")

        window_title = self.capture.window_title
        ui_coords = self.cfg["ui_coords"]
        click_in_game_window(window_title, ui_coords["menu"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["channel"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["random_channel"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["random_channel_confirm"])
        time.sleep(1)

        loc_login_button = None
        while loc_login_button is None and not self.is_terminated:
            try:
                self.img_frame = self.get_img_frame()
                loc_login_button = self.get_login_button_location()
                if loc_login_button is None:
                    logger.info("Waiting for login button to show up...")
            except Exception as e:
                logger.warning(f"Exception occurred while waiting for login button: {e}")
                if not is_mac():
                    resize_window(window_title, width=1296, height=759)
                logger.info("Retrying login button detection...")

            time.sleep(3)
        logger.info(f"login_button button found: {loc_login_button}")

        time.sleep(3)  # wait the screen to be brighter

        # Click login button
        click_in_game_window(window_title, loc_login_button)
        time.sleep(2)

        # Click "Select Character"
        click_in_game_window(window_title, ui_coords["select_character"])
        time.sleep(5)

        self.kb.enable()
        self.kb.set_command("none none none")
        self.kb.release_all_key()

        self.set_state_for_configured_mode()
        self.t_last_attack = time.time() # Update timer

    def terminate_threads(self):
        '''
        terminate all threads
        '''
        # Publish the caller-owned stop intent before terminating KBC.  This
        # prevents the worker loop from misreporting a normal F1 pause/close as
        # an unexpected runtime stop while both threads are winding down.
        self.is_terminated = True
        # Close the input gate first. Health recognition may be blocked in an
        # in-flight frame, so waiting for it before releasing movement keys
        # would leave the character controllable during an F1 pause.
        if self.kb is not None:
            # Safety stop is synchronous: do not wait for the controller
            # thread's next iteration to release a held direction key.
            self.kb.set_command("none none none")
            if hasattr(self.kb, "terminate"):
                self.kb.terminate()
            else:
                if hasattr(self.kb, "clear_aux_actions"):
                    self.kb.clear_aux_actions()
                self.kb.release_all_key()
                self.kb.is_terminated = True
        # With input synchronously closed, wait for recovery production to end.
        if self.health_monitor is not None:
            self.health_monitor.stop()
        # Terminate game window capturor
        if self.capture is not None:
            self.capture.stop()
        worker = getattr(self, "thread_auto_bot", None)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)
            if worker.is_alive():
                logger.warning(
                    "[terminate_threads] 主控制线程仍在退出；退出前将阻止重新启动。"
                )
        logger.info(f"[terminate_threads] Terminated all threads")

    def request_return_home(self, source="watchdog"):
        """Queue return-home through the same guarded input path as recovery."""
        self.kb.set_command("none none none")
        accepted = self.kb.request_aux_action(
            "return_home",
            source_sequence=int(getattr(self, "control_frame_sequence", 0)),
            requested_at=time.monotonic(),
            priority="emergency",
        )
        if accepted:
            logger.info(f"[ReturnHome] Queued safely from {source}.")
        return accepted

    def get_attack_direction(self, monster_left, monster_right):
        '''
        get_attack_direction
        '''
        # Compute distance for left
        distance_left = float('inf')
        if monster_left is not None:
            mx, my = monster_left["position"]
            mw, mh = monster_left["size"]
            center_left = (mx + mw // 2, my + mh // 2)
            distance_left = abs(center_left[0] - self.loc_player[0]) + \
                            abs(center_left[1] - self.loc_player[1])
        # Compute distance for right
        distance_right = float('inf')
        if monster_right is not None:
            mx, my = monster_right["position"]
            mw, mh = monster_right["size"]
            center_right = (mx + mw // 2, my + mh // 2)
            distance_right = abs(center_right[0] - self.loc_player[0]) + \
                            abs(center_right[1] - self.loc_player[1])
        # Choose attack direction and nearest monster
        attack_direction = None
        # nearest_monster = None

        # Additional validation: check if monster is actually on the correct side
        def is_monster_on_correct_side(monster, direction):
            if monster is None:
                return False
            mx, my = monster["position"]
            mw, mh = monster["size"]
            monster_center_x = mx + mw // 2
            player_x = self.loc_player[0]

            if direction == "left":
                return monster_center_x < player_x  # Monster should be left of player
            else:  # direction == "right"
                return monster_center_x > player_x  # Monster should be right of player

        # Only choose direction if there's a clear winner and monster is on correct side
        if monster_left is not None and monster_right is None and \
            is_monster_on_correct_side(monster_left, "left"):
            attack_direction = "left"
            # nearest_monster = monster_left
        elif monster_right is not None and monster_left is None and \
            is_monster_on_correct_side(monster_right, "right"):
            attack_direction = "right"
            # nearest_monster = monster_right
        elif monster_left is not None and monster_right is not None:
            # Both sides have monsters, check distance and side validation
            left_valid = is_monster_on_correct_side(monster_left, "left")
            right_valid = is_monster_on_correct_side(monster_right, "right")

            if left_valid and not right_valid:
                attack_direction = "left"
                # nearest_monster = monster_left
            elif right_valid and not left_valid:
                attack_direction = "right"
                # nearest_monster = monster_right
            elif left_valid and right_valid and distance_left < distance_right - 50:
                attack_direction = "left"
                # nearest_monster = monster_left
            elif left_valid and right_valid and distance_right < distance_left - 50:
                attack_direction = "right"
                # nearest_monster = monster_right
            elif left_valid and right_valid:
                # While following a route, an ambiguous pair should not
                # suppress combat indefinitely. Prefer the route's current
                # horizontal direction, which also avoids rapid turnarounds.
                route_attack = (
                    self.cfg["bot"].get("route_only") and
                    self.cfg["bot"].get("route_attack", False)
                )
                if route_attack and self.cmd_move_x in {"left", "right"}:
                    attack_direction = self.cmd_move_x

        # Debug attack direction selection
        if ((monster_left is not None or monster_right is not None) and
                self.img_frame_debug is not None):
            left_side_ok = is_monster_on_correct_side(monster_left, "left") if monster_left else False
            right_side_ok = is_monster_on_correct_side(monster_right, "right") if monster_right else False
            debug_text = f"L:{distance_left:.0f}({left_side_ok}) R:{distance_right:.0f}({right_side_ok}) Dir:{attack_direction}"
            cv2.putText(self.img_frame_debug, debug_text,
                        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        return attack_direction

    def is_need_change_channel(self, loc_other_players):
        '''
        is_need_change_channel
        '''
        # Calculate center value
        xs = [x for (x, _) in loc_other_players]
        ys = [y for (_, y) in loc_other_players]
        if len(xs) == 0 or len(ys) == 0:
            return False
        center_x, center_y = (np.mean(xs), np.mean(ys))
        if np.isnan(center_x) or np.isnan(center_y):
            return False
        center = (int(np.mean(xs)), int(np.mean(ys)))
        #logger.info(f"[is_need_change_channel] Center of mass = {center}")

        # Change channel
        mode = self.cfg["channel_change"]["mode"]
        if mode == "true":
            logger.warning("[is_need_change_channel] Player detected, immediately change channel.")
            return True
        elif mode == "pixel":
            if self.red_dot_center_prev is None:
                self.red_dot_center_prev = center
            else:
                dx = abs(center[0] - self.red_dot_center_prev[0])
                dy = abs(center[1] - self.red_dot_center_prev[1])
                total = dx + dy
                logger.debug(f"[is_need_change_channel] Movement dx={dx}, dy={dy}, total={total}")
                thres = self.cfg["channel_change"]["other_player_move_thres"]
                if total > thres:
                    logger.warning(f"Other player movement > {thres} pixel detected. "
                                "Trigger channel change.")
                    return True
        else:
            logger.error(f"[is_need_change_channel] Unsupported mode: {mode}")

        return False

    def is_time_to_change_channel(self):
        '''
        is_time_to_change_channel
        '''
        if not self.cfg["scheduled_channel_switching"]["enable"]:
            return False
        dt = time.time() - self.t_to_change_channel
        if dt > self.cfg["scheduled_channel_switching"]["interval_seconds"]:
            self.t_to_change_channel = time.time()
            return True
        return False

    def get_login_button_location(self):
        '''
        get_login_button_location
        '''
        # Extract the region where the login button should appear
        x0, y0 = self.cfg["ui_coords"]["login_button_top_left"]
        x1, y1 = self.cfg["ui_coords"]["login_button_bottom_right"]
        img_roi = self.img_frame[y0:y1, x0:x1]

        # Draw rectange on debug image
        if self.img_frame_debug is not None:
            draw_rectangle(self.img_frame_debug, (x0, y0),
                           (y1-y0, x1-x0), (0, 255, 0), "login_button box")

        # Find the 'login' button
        loc, score, _ = find_pattern_sqdiff(
                        img_roi, self.img_login_button)
        if score < self.cfg["ui_coords"]["login_button_thres"]:
            h, w = self.img_login_button.shape[:2]
            logger.info(f"[get_login_button_location] Found login button with score({score})")
            return (x0 + loc[0] + w // 2,
                    y0 + loc[1] + h // 2 + self.cfg['game_window']['title_bar_height'])
        else:
            return None

    def is_route_traversal_committed(self, now=None):
        """Protect route actions that were already emitted before combat."""
        self._ensure_route_combat_state()
        now = time.time() if now is None else now
        navigator = getattr(self, "route_navigator", None)
        pending_mount = bool(
            navigator is not None and
            (
                getattr(navigator, "has_pending_traversal", False)
                or getattr(navigator, "requires_route_exclusive_control", False)
            )
        )
        if (pending_mount or getattr(self, "is_on_ladder", False) or
                now < self.route_commit_until):
            return True
        self.route_commit_until = 0.0
        self.route_commit_kind = None
        return False

    def mark_route_traversal_commit(self, route_intent, now=None):
        """Start/refresh the short visual lease for a route traversal action."""
        self._ensure_route_combat_state()
        now = time.time() if now is None else now
        action_started = route_intent.action in {"jump", "mount", "teleport"}
        vertical_started = route_intent.move_y in {"up", "down"}
        if not action_started and not vertical_started:
            return
        duration = float(self.cfg["route"].get(
            "traversal_commit_seconds", 0.45
        ))
        self.route_commit_until = max(self.route_commit_until, now + duration)
        self.route_commit_kind = route_intent.action if action_started else \
            route_intent.move_y

    def update_cmd_by_route(self):
        if (
            isinstance(self.route_navigator, SemanticRouteNavigator)
            or self.cfg["bot"].get("route_only")
        ):
            if isinstance(self.route_navigator, SemanticRouteNavigator):
                decision = self.route_navigator.decide(
                    self.loc_player_global,
                    is_on_ladder=self.is_on_ladder,
                    suspend_traversal=bool(
                        getattr(self, "kb", None) is not None
                        and self.kb.is_need_force_heal
                    ),
                )
            else:
                decision = self.route_navigator.decide(self.loc_player_global)
            self.idx_routes = self.route_navigator.route_index
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = decision.command.split()
            self.last_route_intent = RouteIntent.from_command(
                decision.command, reason=decision.reason
            )
            if decision.reason.startswith("semantic_ladder_retry_"):
                diagnostics = self.route_navigator.diagnostics
                segment = self.route_navigator._segment
                logger.warning(
                    "[语义路线] 挂梯尝试失败，准备重新对位："
                    f"route={diagnostics.route_index + 1} "
                    f"segment={segment.get('id')} "
                    f"attempt={diagnostics.ladder_attempts}/"
                    f"{self.route_navigator.ladder_max_attempts} "
                    f"reason={decision.reason.removeprefix('semantic_ladder_retry_')} "
                    f"direction={diagnostics.mount_direction} "
                    f"player={self.loc_player_global} "
                    f"approach={segment.get('approach')} "
                    f"mount={segment.get('mount')}"
                    f" candidate={self.route_navigator.ladder_debug.get('candidate_mount')}"
                    f" offset={self.route_navigator.ladder_debug.get('candidate_offset')}"
                )
            if (decision.reason == "semantic_ladder_failed" and
                    getattr(self, "kb", None) is not None):
                diagnostics = self.route_navigator.diagnostics
                segment = self.route_navigator._segment
                logger.error(
                    "[语义路线] 挂梯连续失败，已安全停止："
                    f"route={diagnostics.route_index + 1} "
                    f"segment={segment.get('id')} "
                    f"direction={diagnostics.mount_direction} "
                    f"player={self.loc_player_global} "
                    f"approach={segment.get('approach')} "
                    f"mount={segment.get('mount')} "
                    f"candidate={self.route_navigator.ladder_debug.get('candidate_mount')} "
                    f"offset={self.route_navigator.ladder_debug.get('candidate_offset')} "
                    f"attempts={diagnostics.ladder_attempts}"
                )
                self.kb.set_command("stop stop none")
                self.kb.release_all_key()
                self.kb.termination_reason = "route_ladder_failed"
                self.kb.terminate()
            return self.last_route_intent
        # get color code from img_route
        color_code, color_code_up_down = self.get_nearest_color_code()
        # Use color_code and color_code_up_down to complement each other
        # To prevent character stuck at the end of ladder, we use two color color pixels
        # and let them complement with each other, to ensure smoothy ladder climbing
        if color_code and color_code_up_down:
            if color_code["distance"] < color_code_up_down["distance"]:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code["command"].split()
                _, cmd, _ = color_code_up_down["command"].split()
                if self.cmd_move_y == "none" and self.is_on_ladder:
                    self.cmd_move_y = cmd # only complement cmd_move_y when player is on ladder
            else:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code_up_down["command"].split()
                cmd, _, _ = color_code["command"].split()
                if self.cmd_move_x == "none" and self.is_on_ladder:
                    self.cmd_move_x = cmd # only complement cmd_move_x when player is on ladder
        elif color_code:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code["command"].split()
        elif color_code_up_down:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code_up_down["command"].split()

        # teleport away from edge to avoid falling off cliff
        if self.is_near_edge() and \
            time.time() - self.t_last_teleport > self.cfg["teleport"]["cooldown"]:
            self.cmd_action = "teleport"
            self.t_last_teleport = time.time() # update timer

        # Use teleport while walking
        if self.cfg['teleport']['is_use_teleport_to_walk'] and \
            time.time() - self.t_last_teleport > self.cfg['teleport']['cooldown']:
            self.cmd_action = "teleport"
            self.t_last_teleport = time.time() # update timer

        # replace teleport to jump if user doesn't set teleport key
        if self.cfg["key"]["teleport"] == "" and self.cmd_action == "teleport":
            self.cmd_action = "jump"

        self.last_route_intent = RouteIntent(
            self.cmd_move_x, self.cmd_move_y, self.cmd_action
        )
        return self.last_route_intent

    def draw_route_execution_debug(self):
        """Render cached route progress without adding another update loop."""
        if self.img_route_debug is None or self.route_navigator is None:
            return
        diagnostics = getattr(self.route_navigator, "diagnostics", None)
        if diagnostics is None:
            text = f"Route: {self.idx_routes + 1} source=legacy"
        else:
            text = (
                f"Route: {diagnostics.route_index + 1} "
                f"segment={diagnostics.segment_index + 1} "
                f"point={diagnostics.point_index + 1} "
                f"traversal={diagnostics.traversal_state} "
                f"reason={diagnostics.reason}"
            )
        cv2.putText(
            self.img_route_debug, text, (5, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
        )
        pose = getattr(self, "minimap_pose_snapshot", None)
        if pose is not None:
            pose_text = (
                f"Pose raw={pose.raw_position} stable={pose.stable_position} "
                f"score={pose.score} src={pose.source} reason={pose.reason}"
            )
            cv2.putText(
                self.img_route_debug, pose_text, (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1,
            )
        ladder = getattr(self.route_navigator, "ladder_debug", None)
        if ladder is not None:
            cv2.circle(
                self.img_route_debug, ladder["approach"], 3,
                (0, 255, 255), 1, cv2.LINE_8,
            )
            cv2.circle(
                self.img_route_debug, ladder["mount"], 4,
                (0, 165, 255), 1, cv2.LINE_8,
            )
            cv2.circle(
                self.img_route_debug, ladder.get("candidate_mount", ladder["mount"]), 5,
                (255, 255, 0), 1, cv2.LINE_8,
            )
            detail = (
                f"Ladder: {ladder['stage']} dir={ladder['mount_direction']} "
                f"dx={ladder['align_error']} settle={ladder['settle_frames']} "
                f"try={ladder['attempts']}/{ladder['max_attempts']} "
                f"off={ladder.get('candidate_offset', 0)} "
                f"pulse={ladder.get('pulse_count', 0)} "
                f"rise={ladder.get('max_rise', 0)} "
                f"cache={ladder.get('cached_offset')}"
            )
            cv2.putText(
                self.img_route_debug, detail, (5, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1,
            )

    def draw_ladder_execution_debug(self, canvas=None):
        """Draw the active semantic ladder stage on the game minimap."""
        if canvas is None:
            canvas = self.img_frame_debug
        navigator = getattr(self, "route_navigator", None)
        ladder = getattr(navigator, "ladder_debug", None)
        if (canvas is None or ladder is None or self.img_minimap is None or
                self.img_minimap.size == 0):
            return
        offset_x, offset_y = self.cfg["minimap"].get("offset", [0, 0])
        minimap_h, minimap_w = self.img_minimap.shape[:2]

        def to_screen(point):
            local_x = int(point[0]) - int(self.loc_minimap_global[0]) - int(offset_x)
            local_y = int(point[1]) - int(self.loc_minimap_global[1]) - int(offset_y)
            if not (0 <= local_x < minimap_w and 0 <= local_y < minimap_h):
                return None
            return int(self.loc_minimap[0]) + local_x, int(self.loc_minimap[1]) + local_y

        approach = to_screen(ladder["approach"])
        mount = to_screen(ladder["mount"])
        candidate_mount = to_screen(ladder.get("candidate_mount", ladder["mount"]))
        if approach is not None:
            cv2.circle(canvas, approach, 4, (0, 255, 255), 1, cv2.LINE_8)
        if mount is not None:
            cv2.circle(canvas, mount, 5, (0, 165, 255), 1, cv2.LINE_8)
        if candidate_mount is not None:
            cv2.circle(canvas, candidate_mount, 6, (255, 255, 0), 1, cv2.LINE_8)
        detail = (
            f"Ladder {ladder['stage']} dir={ladder['mount_direction']} "
            f"dx={ladder['align_error']} settle={ladder['settle_frames']} "
            f"try={ladder['attempts']}/{ladder['max_attempts']} "
            f"off={ladder.get('candidate_offset', 0)} "
            f"pulse={ladder.get('pulse_count', 0)} "
            f"rise={ladder.get('max_rise', 0)} "
            f"cache={ladder.get('cached_offset')}"
        )
        cv2.putText(
            canvas, detail, (10, 470), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 165, 255), 1, cv2.LINE_AA,
        )

    def update_monster_observations(self):
        """Refresh detector output without granting either subsystem control."""
        detection_only = self.is_disable_control
        x0, y0, x1, y1 = self.get_monster_search_range()

        if self.monster_detector is None:
            raise RuntimeError("YOLO 怪物检测器尚未初始化。")
        if getattr(self, "current_nametag_valid", False):
            self.monsters = self.monster_detector.detect(
                self.img_frame,
                self.loc_player,
                player_exclusion_anchor=self.loc_player,
            )
        else:
            # Keep the historical detector call compatible while explicitly
            # withholding a stale name-tag anchor from the exclusion filter.
            self.monsters = self.monster_detector.detect(
                self.img_frame, self.loc_player
            )
        self._diagnostic_yolo_checked = True
        timing = self.monster_detector.last_timing
        self.monster_detection_times.append(timing.total_ms)
        if not hasattr(self, "monster_inference_times"):
            self.monster_inference_times = deque(maxlen=300)
        self.monster_inference_times.append(timing.infer_ms)

        if detection_only:
            now = time.time()
            last_log = getattr(self, "t_last_mob_score_log", 0.0)
            if now - last_log >= 1.0:
                samples = np.asarray(self.monster_detection_times, dtype=np.float64)
                mean_ms = float(np.mean(samples)) if samples.size else 0.0
                p95_ms = float(np.percentile(samples, 95)) if samples.size else 0.0
                frame_samples = np.asarray(
                    self.monster_frame_pipeline_times, dtype=np.float64
                )
                control_p95_ms = (
                    float(np.percentile(frame_samples, 95))
                    if frame_samples.size else 0.0
                )
                logger.info(
                    "[MobDetect:YOLO] "
                    f"monsters={timing.monsters} players={timing.players} "
                    f"crop={timing.crop_ms:.2f}ms infer={timing.infer_ms:.2f}ms "
                    f"convert={timing.convert_ms:.2f}ms total={timing.total_ms:.2f}ms "
                    f"mean={mean_ms:.2f}ms p95={p95_ms:.2f}ms "
                    f"control_p95={control_p95_ms:.2f}ms"
                )
                self.t_last_mob_score_log = now

        # --disable_control is also the visual calibration mode.  It must run
        # the detector regardless of the route pixel under the player, while
        # remaining incapable of producing an attack command.
        if detection_only:
            if self.cfg["bot"]["attack"] == "directional":
                monster_left = self.get_nearest_monster(is_left=True)
                monster_right = self.get_nearest_monster(is_left=False)
                self.debug_attack_direction = (
                    self.get_attack_direction(monster_left, monster_right) or "none"
                )
            else:
                self.debug_attack_direction = "aoe"
        return True

    def apply_route_combat_intent(
        self, route_intent, now=None, control_allowed=None,
        combat_intent=None,
    ):
        """Resolve a fresh combat observation against one route intent."""
        now = time.time() if now is None else now
        if combat_intent is None:
            combat_intent = self.build_route_combat_intent(
                route_intent, now=now, control_allowed=control_allowed
            )
        resolved = self.route_combat_arbiter.resolve(
            route_intent, combat_intent
        )
        self.cmd_move_x = resolved.move_x
        self.cmd_move_y = resolved.move_y
        self.cmd_action = resolved.action
        self.last_combat_intent = combat_intent

        previous_state = self.debug_combat_state
        self.debug_combat_state = combat_intent.state
        self.debug_attack_direction = combat_intent.direction or "none"
        if previous_state != combat_intent.state:
            logger.info(
                f"[RouteCombat] {previous_state}->{combat_intent.state} "
                f"owner={resolved.owner} direction={combat_intent.direction}"
            )
        if combat_intent.action != "none":
            logger.info(
                f"[RouteAttack] intent={combat_intent.action} "
                f"route={route_intent.move_x} final={resolved.move_x} "
                f"key={self.cfg['key']['directional_attack']}"
            )
        return combat_intent

    def update_cmd_by_mob_detection(self, route_intent=None):
        """Compatibility wrapper for non-route states and focused tests."""
        route_attack = (self.cfg["bot"].get("route_only") and
                        self.cfg["bot"].get("route_attack", False))
        detection_only = self.is_disable_control
        if route_intent is None:
            route_intent = getattr(
                self,
                "last_route_intent",
                RouteIntent(self.cmd_move_x, self.cmd_move_y, self.cmd_action),
            )

        self.update_monster_observations()
        if detection_only:
            return CombatIntent(target_visible=bool(self.monsters))

        if route_attack and self.cfg["bot"]["attack"] == "directional":
            return self.apply_route_combat_intent(route_intent)

        # Check if no mob to attack in legacy/AOE modes.
        if len(self.monsters) == 0:
            self.debug_attack_direction = "none"
            return CombatIntent()

        if route_attack and route_intent.is_critical:
            # AOE retains its legacy detector but cannot interrupt a committed
            # route event.
            return CombatIntent(target_visible=True)

        # Update attack command
        if self.cfg["bot"]["attack"] == "aoe_skill":
            cooldown = self.cfg["aoe_skill"]["cooldown"]
            if time.time() - self.t_last_attack > cooldown:
                self.cmd_action = "attack"
                self.t_last_attack = time.time()

        elif self.cfg["bot"]["attack"] == "directional":
            cooldown = self.cfg["directional_attack"]["cooldown"]
            # Get nearest monster to player
            monster_left  = self.get_nearest_monster(is_left = True)
            monster_right = self.get_nearest_monster(is_left = False)
            # Determine attack direction
            attack_direction = self.get_attack_direction(monster_left, monster_right)
            # Attack Command
            if time.time() - self.t_last_attack > cooldown and attack_direction is not None:
                if route_attack:
                    # Keep the route movement command intact. The keyboard
                    # controller performs a short facing pulse, attacks, then
                    # restores the held route direction.
                    self.cmd_action = f"attack_{attack_direction}"
                    self.debug_attack_direction = attack_direction
                    logger.info(
                        f"[RouteAttack] intent={self.cmd_action} "
                        f"route={self.cmd_move_x} key={self.cfg['key']['directional_attack']}"
                    )
                else:
                    self.cmd_action = "attack"
                    # Legacy mode faces by replacing its movement command.
                    self.cmd_move_x = attack_direction
                self.t_last_attack = time.time()
        return CombatIntent(target_visible=bool(self.monsters))

    def update_route_only_commands(self, combat_available=True, now=None):
        """Run perception-first route/combat arbitration for one frame."""
        now = time.time() if now is None else now
        route_attack = (
            self.cfg["bot"].get("route_attack", False) and
            self.cfg["bot"].get("attack") == "directional"
        )
        if not route_attack or not combat_available:
            if route_attack:
                self.monsters = []
                self.reset_route_combat(now=now)
            route_intent = self.update_cmd_by_route()
            self.mark_route_traversal_commit(route_intent, now=now)
            return route_intent, CombatIntent()

        self.update_monster_observations()
        if self.is_disable_control:
            route_intent = self.update_cmd_by_route()
            self.mark_route_traversal_commit(route_intent, now=now)
            return route_intent, CombatIntent(target_visible=bool(self.monsters))

        if self.is_route_traversal_committed(now=now):
            route_intent = self.update_cmd_by_route()
            self.mark_route_traversal_commit(route_intent, now=now)
            combat_intent = self.apply_route_combat_intent(
                route_intent, now=now, control_allowed=False
            )
            return route_intent, combat_intent

        route_reference = self.last_route_intent
        previous_combat_owned = self.last_combat_intent.owns_control
        combat_intent = self.build_route_combat_intent(
            route_reference, now=now, control_allowed=True
        )
        if combat_intent.owns_control:
            # Do not sample RouteNavigator here: event pixels must remain
            # untouched until the target lock has ended.
            combat_base = RouteIntent(
                "none", "none", "none", reason="combat_hold"
            )
            self.apply_route_combat_intent(
                combat_base, now=now, combat_intent=combat_intent
            )
            return None, combat_intent

        if previous_combat_owned and isinstance(
            self.route_navigator, SemanticRouteNavigator
        ):
            self.route_navigator.request_reacquire(
                "semantic_combat_resume_reacquire"
            )
        route_intent = self.update_cmd_by_route()
        self.mark_route_traversal_commit(route_intent, now=now)
        self.apply_route_combat_intent(
            route_intent, now=now, combat_intent=combat_intent
        )
        return route_intent, combat_intent

    def update_cmd_by_random(self):
        '''
        update_cmd_by_random - pick a random action except 'up' and teleport command
        '''
        self.cmd_move_x = random.choice(["left", "right", "none"])
        self.cmd_move_y = random.choice(["down", "none"])
        self.cmd_action = random.choice(["jump", "none"])
        logger.warning("[update_cmd_by_random]"\
                    f"{self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}")

    def check_reach_goal(self):
        if self.cmd_action == "goal":
            if isinstance(self.route_navigator, SemanticRouteNavigator):
                # The ordered navigator advanced atomically when it emitted
                # the Goal intent; advancing again here would skip a route.
                return
            # Switch to next route map
            self.idx_routes = (self.idx_routes+1)%len(self.img_routes)
            logger.debug(f"Change to new route:{self.idx_routes}")

    def pause_for_visual(self, reason):
        """Stop physical movement while retaining the route for safe recovery."""
        self.last_visual_pause_reason = reason
        self._diagnostic_pause_reason = reason
        self.reset_route_combat()
        self.route_commit_until = 0.0
        if isinstance(self.route_navigator, SemanticRouteNavigator):
            self.route_navigator.suspend_traversal()
            if not self.route_navigator.requires_route_exclusive_control:
                self.route_navigator.request_reacquire(
                    f"semantic_visual_resume_{reason}"
                )
        self.route_commit_kind = None
        self.cmd_move_x, self.cmd_move_y, self.cmd_action = "stop", "stop", "none"
        if self.kb is not None:
            self.kb.set_command("stop stop none")
            self.kb.release_all_key()

    def _update_health_monitor_frame(self):
        """Publish one owned bottom-UI ROI to the health worker."""
        if self.health_monitor is None or not self.health_monitor.active:
            return
        ui_y_start = self.cfg["ui_coords"]["ui_y_start"]
        health_roi = self.img_frame[ui_y_start:, :]
        self.health_monitor.update_frame(
            health_roi,
            frame_sequence=(
                self.capture_frame_sequence
                if self.capture_frame_sequence is not None
                else self.control_frame_sequence
            ),
            roi_origin=(0, ui_y_start),
            captured_at=self.frame_captured_at,
        )

    def _run_continuous_attack_frame(self):
        """Run only the one-time perception needed to lock attack direction."""
        state = self.continuous_attack_state
        was_locked = state.direction_locked
        if was_locked:
            # Never display or reuse the startup detections after direction is
            # locked. The state keeps only the selected direction/diagnostics.
            self.monsters = []
            self.current_nametag_valid = False
        else:
            loc_player = self.get_player_location_by_nametag()
            self.current_nametag_valid = loc_player is not None
            if loc_player is not None:
                self.loc_player = loc_player
                if self.img_frame_debug is not None:
                    cv2.circle(
                        self.img_frame_debug,
                        self.loc_player,
                        radius=3,
                        color=(0, 0, 255),
                        thickness=-1,
                    )
            else:
                self.monsters = []

        self.last_visual_pause_reason = ""
        self.fsm.do_state_stuff()
        self.is_first_frame = False
        self.profiler.mark("Continuous directional attack")
        return 0

    def run_once(self):
        # Diagnostic delivery is independent of debug rendering and never changes
        # navigation decisions. Publish one immutable latest snapshot, not a Qt
        # event containing a full-sized image for every control frame.
        enabled = getattr(self, "diagnostics_enabled", False)
        if enabled:
            self._diagnostic_frame_valid = False
            self._diagnostic_map_checked = False
            self._diagnostic_yolo_checked = False
            self._diagnostic_pause_reason = ""
            self.nametag_last_result = None
        try:
            return self._run_once()
        finally:
            now = time.monotonic()
            if enabled and now - getattr(self, "_diagnostic_published_at", 0) >= .2:
                try:
                    from src.engine.ReadinessDiagnostics import runtime_report
                    self.readiness_report = runtime_report(self, self._diagnostic_frame_valid)
                    self._diagnostic_published_at = now
                except Exception as exc:
                    # Diagnostics must not mask a control-loop failure.
                    logger.warning(f"[识别状态] 无法生成诊断：{exc}")

    def _run_once(self):
        '''
        Process one game window frame
        '''
        # Start profiler for performance debugging
        self.profiler.start()
        # These flags describe this exact control frame.  Fixed-platform mode
        # fails closed instead of reusing stale coordinates from a prior frame.
        self.current_minimap_roi_valid = False
        self.current_minimap_player_valid = False
        self.current_nametag_valid = False

        # A UI request is a one-shot token.  Only the selected preview is
        # allocated on this control frame.
        self._frame_visualization_mode = self._claim_visualization_mode()
        self.is_show_debug_window = self._frame_visualization_mode is not None
        if not self.is_show_debug_window:
            self.img_frame_debug = None
            self.img_route_debug = None

        ###########################
        ### Image Preprocessing ###
        ###########################
        # Get game window frame
        img_frame = self.get_img_frame()
        if img_frame is None:
            self.pause_for_visual("capture_invalid")
            if not is_mac() and not self.cfg["bot"].get("route_only"):
                activate_game_window(self.capture.window_title)
            return -1 # Wait for game window to be ready
        else:
            self.img_frame = img_frame
            self._diagnostic_frame_valid = True
            self.control_frame_sequence += 1

        # Get current route image
        if self.cfg["bot"]["mode"] == "normal":
            self.img_route = self.img_routes[self.idx_routes]
        else:
            self.img_route = None
        self._prepare_visualization_buffers()

        self.profiler.mark("Image Preprocessing")

        # Health monitoring remains active in the lightweight continuous-
        # attack path and receives this frame before any optional perception.
        self._update_health_monitor_frame()

        if self.cfg["bot"]["mode"] == "continuous_attack":
            return self._run_continuous_attack_frame()

        ###################
        ### Get Minimap ###
        ###################
        # Get minimap coordinate and size on game window
        minimap_result = get_minimap_loc_size(self.img_frame, self.cfg)
        if minimap_result is None:
            if self.cfg["bot"].get("route_only"):
                self.pause_for_visual("minimap_roi_invalid")
                return -1
            if time.time() - self.t_last_minimap_update > 30:
                # Unable to get minimap for 30 seconds -> assume it's login screen
                loc_login_button = self.get_login_button_location()
                if loc_login_button:
                    logger.info("Found login button on screen. Proceed to login.")
                    click_in_game_window(self.capture.window_title,
                                         loc_login_button)
                    time.sleep(3)
                    click_in_game_window(self.capture.window_title,
                                         self.cfg["ui_coords"]["select_character"])
                    time.sleep(2)
        else:
            x, y, w, h = minimap_result
            if not self.cfg.get("minimap", {}).get("roi"):
                # Legacy dynamic detector includes a one-pixel border.
                x += 1
                y += 1
                w -= 2
                h -= 2
            # update minimap image
            self.loc_minimap = (x, y)
            self.img_minimap = self.img_frame[y:y+h, x:x+w]
            self.current_minimap_roi_valid = True
            self.t_last_minimap_update = time.time()
            if not self._minimap_roi_pixels_logged:
                logger.info(
                    f"[小地图ROI] source={self.minimap_roi_source} "
                    f"pixels=({x}, {y}, {w}, {h}) "
                    f"frame={self.img_frame.shape[1]}x{self.img_frame.shape[0]}"
                )
                self._minimap_roi_pixels_logged = True

        self.profiler.mark("Get Minimap Location and Size")

        #################################
        ### Player Location Detection ###
        #################################
        # Get player location in game window
        route_attack = (self.cfg["bot"].get("route_only") and
                        self.cfg["bot"].get("route_attack", False))
        # Name tag is the only viewport player-localization source. A missed
        # tag disables combat for this frame; minimap route navigation remains
        # available independently.
        loc_player = self.get_player_location_by_nametag()
        self.current_nametag_valid = loc_player is not None

        # Update player location
        if loc_player is not None:
            # Check if character is on ladder
            dx = abs(loc_player[0] - self.loc_player[0])
            dy = abs(loc_player[1] - self.loc_player[1])
            if self.is_on_ladder:
                if dx > 3: # Leave ladder if there is horizontal move
                    self.is_on_ladder = False
            else:
                if dx < 3 and dy != 0:
                    self.is_on_ladder = True
            # logger.info((self.is_on_ladder, dx, dy))
            # Update player location
            self.loc_player = loc_player

        # Draw player center for debugging
        if self.img_frame_debug is not None and loc_player is not None:
            cv2.circle(self.img_frame_debug,
                    self.loc_player, radius=3,
                    color=(0, 0, 255), thickness=-1)

        # Get player location on minimap
        loc_player_minimap = get_player_location_on_minimap(
                                self.img_minimap,
                                minimap_player_color=self.cfg["minimap"]["player_color"],
                                player_hsv=self.cfg["minimap"].get("player_hsv"))
        if loc_player_minimap:
            self.loc_player_minimap = loc_player_minimap
            self.current_minimap_player_valid = self.current_minimap_roi_valid
        elif self.cfg["bot"].get("route_only"):
            self.pause_for_visual("player_not_visible")
            return -1

        # Get other player location on minimap
        loc_other_players = [] if self.cfg["bot"].get("route_only") else \
            get_all_other_player_locations_on_minimap(
                self.img_minimap, self.cfg['minimap']['other_player_color'])
        # Debug
        # if self.is_first_frame:
        #     logger.info("Running minimap color analysis...")
        #     debug_minimap_colors(self.img_minimap, other_player_color)

        # Get player location on global map
        if self.cfg["bot"]["mode"] in ["patrol", "aux", "fixed_platform"]:
            self.loc_player_global = self.loc_player_minimap
        else:
            loc_player_global = self.get_player_location_on_global_map()
            if loc_player_global is None:
                self.pause_for_visual("map_localization_invalid")
                return -1
            self.loc_player_global = loc_player_global

        self.profiler.mark("Player Location Detection")

        if self.cfg["bot"].get("route_only"):
            self.last_visual_pause_reason = ""
            self.update_route_only_commands(
                combat_available=(not route_attack or loc_player is not None)
            )
            self.draw_route_execution_debug()
            self.kb.set_command(
                f"{self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}"
            )
            self.is_first_frame = False
            return 0

        ######################
        ### Change Channel ###
        ######################
        if self.cfg['channel_change']['enable'] and \
            self.is_need_change_channel(loc_other_players):
            self.kb.set_command("none none none")
            self.kb.release_all_key()
            self.kb.disable()
            time.sleep(1)
            self.channel_change()
            self.red_dot_center_prev = None
            return 0

        if self.is_time_to_change_channel():
            self.kb.set_command("none none none")
            self.kb.release_all_key()
            self.kb.disable()
            time.sleep(1)
            self.channel_change()
            return 0

        self.profiler.mark("Change Channel")

        #######################
        ### Attack WatchDog ###
        ####################### Check if last attack is timeout
        dt = time.time() - self.t_last_attack
        if self.cfg['bot']['mode'] == 'normal' and \
            dt > self.cfg["watchdog"]["last_attack_timeout"]:
            logger.info(f"[Attack Timeout] Last attack timeout for {round(dt, 2)} seconds")
            cfg_action = self.cfg["watchdog"]["last_attack_timeout_action"]
            if cfg_action == "change_channel":
                logger.info("[Attack Timeout] Change channel!")
                self.channel_change()
            elif cfg_action == "go_home":
                logger.info("[Attack Timeout] Return home!")
                self.request_return_home("attack_watchdog")
                # Do not emit a movement or attack command while the guarded
                # input worker is preparing the return-home action.
                return 0
            else:
                logger.info(f"Unsupported timeout mode: {cfg_action}")

        self.profiler.mark("Attack WatchDog")

        ######################
        ### State Behavior ###
        ######################
        self.fsm.do_state_stuff()

        if self.img_route_debug is not None:
            self.draw_route_execution_debug()

        self.is_first_frame = False

        self.profiler.mark("State per-frame behavior")

        # Print profiler result
        if self.cfg["profiler"]["enable"] and \
            self.profiler.total_frames % self.cfg["profiler"]["print_frequency"] == 0:
            logger.info('\n' + self.profiler.report())

        return 0 # frame done

    def loop(self, run_generation=None):
        '''
        Auto Bot main loop
        Only run when call autobot from UI framework and AutoBotController
        '''
        if run_generation is None:
            run_generation = int(getattr(self, "run_generation", 0))
        runtime_stop_reason = None
        try:
            while not self.kb.is_terminated:

                frame_started = time.perf_counter()

                # Process one game window frame
                self.is_frame_done = False
                ret = self.run_once()

                # Only proceed if the frame is valid
                if ret == 0:
                    # Control latency excludes optional diagnostic rendering.
                    self.record_control_frame_pipeline(frame_started)

                    requested_mode = self._frame_visualization_mode
                    if (self.is_ui and requested_mode in {"game", "route"} and
                            self.visualization_sink is not None):
                        compose_started = time.perf_counter()
                        if requested_mode == "game":
                            preview = self.get_frame_debug_for_viz()
                        else:
                            preview = self.get_route_debug_for_viz()
                        compose_ms = (
                            time.perf_counter() - compose_started
                        ) * 1000.0
                        self.visualization_compose_times.append(compose_ms)

                        if preview is not None:
                            publish_started = time.perf_counter()
                            self.visualization_sink(
                                requested_mode,
                                preview,
                                self.get_visualization_performance(),
                            )
                            self.visualization_publish_times.append(
                                (time.perf_counter() - publish_started) * 1000.0
                            )
                            self.t_last_viz_frame = time.time()

                        # Release the only full-size debug copy immediately.
                        self.img_frame_debug = None
                        self.img_route_debug = None

                    self.t_last_frame = time.time()

                self.is_frame_done = True

                # Cap FPS to save system resource
                frame_duration = time.perf_counter() - frame_started
                target_duration = 1.0 / RUNTIME_POLICY.main_fps
                if frame_duration < target_duration:
                    time.sleep(target_duration - frame_duration)
        except Exception as exc:
            # Fail closed before stopping the remaining workers. This is
            # deliberately synchronous: command TTL is a second line of
            # defense, not a substitute for immediately releasing held keys.
            logger.error(f"[MapleStoryAutoBot] Main loop failed closed: {exc}")
            if not self.is_terminated:
                runtime_stop_reason = "engine_error"
            try:
                self.kb.set_command("stop stop none")
            except Exception as command_exc:
                logger.error(f"[MapleStoryAutoBot] Failed to set stop command: {command_exc}")
            if hasattr(self.kb, "terminate"):
                self.kb.terminate()
            else:
                self.kb.is_terminated = True
                self.kb.release_all_key()
            try:
                self.capture.stop()
            except Exception as capture_exc:
                logger.error(f"[MapleStoryAutoBot] Failed to stop capture: {capture_exc}")
            try:
                self.health_monitor.stop()
            except Exception as health_exc:
                logger.error(f"[MapleStoryAutoBot] Failed to stop health monitor: {health_exc}")
            self.is_terminated = True
        finally:
            self.is_frame_done = True
            self.kb.release_all_key()
            if self.kb.is_terminated and not getattr(self, "is_terminated", False):
                runtime_stop_reason = (
                    getattr(self.kb, "termination_reason", "") or
                    "input_stopped"
                )
                try:
                    if self.health_monitor is not None:
                        self.health_monitor.stop()
                except Exception as health_exc:
                    logger.error(
                        "[MapleStoryAutoBot] Failed to stop health monitor: "
                        f"{health_exc}"
                    )
                try:
                    if self.capture is not None:
                        self.capture.stop()
                except Exception as capture_exc:
                    logger.error(
                        "[MapleStoryAutoBot] Failed to stop capture: "
                        f"{capture_exc}"
                    )
                self.is_terminated = True
            termination_sink = getattr(self, "termination_sink", None)
            if runtime_stop_reason is not None and termination_sink is not None:
                try:
                    termination_sink(run_generation, runtime_stop_reason)
                except Exception as notify_exc:
                    logger.error(
                        "[MapleStoryAutoBot] Failed to notify runtime stop: "
                        f"{notify_exc}"
                    )

def main(args):
    '''
    This main function works as a fake autoBotController
    This function will only be called when the using terminal to
    run this script
    '''
    #####################
    ### Init Auto Bot ###
    #####################
    try:
        mapleStoryAutoBot = MapleStoryAutoBot(args)
    except Exception as e:
        logger.error(f"MapleStoryAutoBot Init failed: {e}")
        sys.exit(1)
    else:
        logger.info("MapleStoryAutoBot Init Successfully")

    ####################
    ### Apply Config ###
    ####################
    # Load defautl yaml config
    cfg = load_yaml("config/config_default.yaml")
    # Override with platform config
    if is_mac():
        cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
    # Override with user customized config
    custom_cfg = migrate_health_monitor_config(validate_custom_config(
        load_yaml(f"config/config_{args.cfg}.yaml")
    ))
    cfg = override_cfg(cfg, custom_cfg)
    # Dump config to log for debugging
    logger.debug(yaml.dump(cfg, sort_keys=False,
                 indent=2, default_flow_style=False))
    # autoBot load config
    mapleStoryAutoBot.load_config(cfg)

    #####################
    ### Start AutoBot ###
    #####################
    try:
        mapleStoryAutoBot.start() # Start all threads in autoBot
    except Exception as e:
        logger.error(f"MapleStoryAutoBot start failed: {e}")
        mapleStoryAutoBot.terminate_threads() # Terminate all threads
        sys.exit(1)
    else:
        logger.info("MapleStoryAutoBot Start Successfully")

    kb_listener = KeyBoardListener(is_autobot=True)
    kb_listener.register_func_key_handler('f1', mapleStoryAutoBot.kb.toggle_enable)
    kb_listener.register_func_key_handler('emergency', mapleStoryAutoBot.terminate_threads)

    # Create CLI debug windows immediately. Previously their creation was
    # gated by the short-lived is_frame_done flag, so the display loop could
    # miss every update even while detection and logging were healthy.
    if not args.disable_viz:
        cv2.namedWindow("Game Window Debug", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Route Map Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Game Window Debug", 1100, 650)
        cv2.resizeWindow("Route Map Debug", 600, 600)

    # While loop
    while not mapleStoryAutoBot.is_terminated:
        # Show the latest completed-or-in-progress debug buffers. Repainting a
        # stale frame is preferable to losing the window because of a polling
        # race; this path is diagnostic-only and never controls the bot.
        if not args.disable_viz:
            if mapleStoryAutoBot.img_frame_debug is not None:
                img_frame_debug_viz = mapleStoryAutoBot.get_frame_debug_for_viz()
                if img_frame_debug_viz is not None:
                    cv2.imshow("Game Window Debug", img_frame_debug_viz)

            if mapleStoryAutoBot.img_route_debug is not None:
                route_debug_viz = mapleStoryAutoBot.get_route_debug_for_viz()
                if route_debug_viz is not None:
                    cv2.imshow("Route Map Debug", route_debug_viz)

            cv2.waitKey(1)

        time.sleep(0.01)

    #########################
    ### Terminate AutoBot ###
    #########################
    mapleStoryAutoBot.terminate_threads() # Terminate all threads

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--disable_control',
        action='store_true',
        help='Disable simulated keyboard input'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='custom',
        help='Choose customized config yaml file in config/'
    )

    parser.add_argument(
        '--debug',
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        '--disable_viz',
        action="store_true",
        help="Disable viz debug window"
    )

    parser.add_argument(
        '--test_image',
        default="",
        help="Pass in image in test/XXX.png"
    )

    parser.add_argument(
        '--init_state',
        default="",
        help="choose the init_state"
    )

    args = parser.parse_args()
    args.is_ui = False # Always set False for command line

    # Set logger level
    if args.debug:
        logger.set_level(logging.DEBUG)

    main(args)
