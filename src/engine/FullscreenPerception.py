"""Control-loop integration for one shared three-class scene per capture."""
import time
from dataclasses import dataclass

import cv2

from src.engine.FullscreenSelfDetector import SelfLocalizationGate, uses_fullscreen_self
from src.engine.OpenVinoMonsterDetector import OpenVinoMonsterDetector
from src.utils.logger import logger


def current_player_valid(bot):
    if uses_fullscreen_self(getattr(bot, "cfg", {}) or {}):
        return bool(getattr(bot, "current_player_valid", False))
    return bool(getattr(bot, "current_nametag_valid", False))


def fixed_fullscreen_directional(cfg):
    return (uses_fullscreen_self(cfg) and cfg.get('bot', {}).get('mode') == 'fixed_platform'
            and cfg['bot'].get('attack') == 'directional')


@dataclass(frozen=True)
class FixedAttackGeometry:
    frame_token: object
    self_box: tuple
    center: tuple
    foot: tuple
    left: tuple
    right: tuple
    combined: tuple
    pursuit_rect: tuple

    def direction(self, monster):
        x, _ = monster['position']
        w, _ = monster['size']
        return 'left' if x+w//2 <= self.center[0] else 'right'

    def contains(self, monster):
        x, y = monster['position']
        w, h = monster['size']
        x0, y0, x1, y1 = self.left if self.direction(monster) == 'left' else self.right
        return min(x1, x+w) > max(x0, x) and min(y1, y+h) > max(y0, y)

    def same_level(self, monster, tolerance):
        y = monster['position'][1] + monster['size'][1]
        return abs(y-self.foot[1]) <= tolerance


def fixed_attack_geometry(scene, cfg):
    if scene is None or len(scene.selves) != 1:
        return None
    body = scene.selves[0]
    x, y = body['position']
    w, h = body['size']
    cx, cy = x+w//2, y+h//2
    x0, y0, x1, y1 = scene.visible_rect
    radius = int(cfg['directional_attack']['range_x'])
    height = int(cfg['directional_attack']['range_y'])
    top = max(y0, cy-height//2)
    bottom = min(y1, cy-height//2+height)
    left = (max(x0, cx-radius), top, cx, bottom)
    right = (cx, top, min(x1, cx+radius), bottom)
    foot = (cx, min(scene.frame_shape[0]-1, y+h))
    pursuit_rect = OpenVinoMonsterDetector.visible_input_rect(
        scene.frame_shape, foot)
    return FixedAttackGeometry(scene.frame_token, (x, y, w, h), (cx, cy),
                               foot, left, right,
                               (left[0], top, right[2], bottom), pursuit_rect)


class FullscreenPerceptionMixin:
    def _uses_fullscreen_self(self):
        return uses_fullscreen_self(getattr(self, "cfg", {}) or {})

    def _fullscreen_frame_problem(self):
        captured_at = getattr(self, "frame_captured_at", None)
        if captured_at is None or not 0 <= time.monotonic()-captured_at <= .3:
            return "frame_stale"
        active_check = getattr(self.kb, "is_game_window_active", None)
        active = active_check() if callable(active_check) else getattr(self.kb, "_was_game_window_active", True)
        return "" if active else "window_inactive"

    def fullscreen_control_ready(self):
        if not self._uses_fullscreen_self():
            return True
        problem = self._fullscreen_frame_problem()
        if problem:
            self.pause_for_visual(problem)
            return False
        return bool(self.current_player_valid and not self.is_disable_control
                    and self.cfg['bot']['mode'] != 'aux')

    def _invalidate_fullscreen(self, reason):
        self.current_player_valid = False
        self.monsters = []
        self.debug_attack_direction = "none"
        self.fixed_attack_geometry = None
        if hasattr(self, "self_localization_gate"):
            self.self_localization_gate.invalidate(reason)
        self.fullscreen_scene = None

    def _update_fullscreen_perception(self):
        token = self._detection_frame_token()
        self.current_player_valid = False
        self.monsters = []
        self.fixed_attack_geometry = None
        if getattr(self, 'capture_frame_sequence', None) is None:
            self.pause_for_visual('self_frame_unknown')
            return False
        # Never retry a failed inference or advance recovery on the same capture.
        if token is None or token == getattr(self, "_self_attempt_token", None):
            if fixed_fullscreen_directional(self.cfg):
                self.pause_for_visual('self_frame_duplicate')
            return False
        self._self_attempt_token = token
        self.fullscreen_scene = None
        if not hasattr(self, "self_localization_gate"):
            self.self_localization_gate = SelfLocalizationGate()
        try:
            scene = self.monster_detector.detect_scene(self.img_frame, frame_token=token)
        except Exception as exc:
            self.pause_for_visual("self_inference_failed")
            now = time.monotonic()
            if now-getattr(self, "_self_error_logged_at", float("-inf")) >= 5:
                logger.error(f"[三类全屏检测] 推理失败，停止控制：{exc}")
                self._self_error_logged_at = now
            return False
        problem = self._fullscreen_frame_problem()
        if problem and not (problem == 'window_inactive'
                            and fixed_fullscreen_directional(self.cfg)):
            self.pause_for_visual(problem)
            return False
        self.fullscreen_scene = scene
        if fixed_fullscreen_directional(self.cfg):
            self.fixed_attack_geometry = fixed_attack_geometry(scene, self.cfg)
        self._diagnostic_yolo_checked = True
        self.monster_detection_times.append(scene.timing.total_ms)
        self.monster_inference_times.append(scene.timing.infer_ms)
        location = self.self_localization_gate.update(scene, token)
        self.current_player_valid = location.valid
        if not location.valid:
            self.pause_for_visual(location.reason)
            return False
        previous = getattr(self, 'loc_player', None)
        if previous is not None:
            dx = abs(location.foot[0]-previous[0])
            dy = abs(location.foot[1]-previous[1])
            if getattr(self, 'is_on_ladder', False):
                if dx > 3:
                    self.is_on_ladder = False
            elif dx < 3 and dy != 0:
                self.is_on_ladder = True
        self.loc_player = location.foot
        self.monster_detector.last_player_foot = location.foot
        self.monsters = (list(scene.monsters) if fixed_fullscreen_directional(self.cfg)
                         else self.monster_detector.finalize_scene(scene, location.foot))
        self._monster_observation_token = token
        self.last_visual_pause_reason = ""
        self._diagnostic_pause_reason = ""
        return True

    def _draw_fullscreen_scene(self, canvas):
        scene = getattr(self, "fullscreen_scene", None)
        if scene is None or scene.frame_token != self._detection_frame_token():
            return
        x0, y0, x1, y1 = scene.visible_rect
        cv2.rectangle(canvas, (x0, y0), (x1-1, y1-1), (255, 0, 0), 2)
        for group, color in ((scene.monsters, (0, 165, 255)),
                             (scene.players, (255, 255, 0)), (scene.selves, (0, 255, 0))):
            for item in group:
                x, y = item["position"]
                w, h = item["size"]
                cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 2)
                cv2.putText(canvas, f'{item["name"]} {item["confidence"]:.2f}',
                            (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        geometry = getattr(self, 'fixed_attack_geometry', None)
        if (fixed_fullscreen_directional(self.cfg) and geometry is not None
                and geometry.frame_token == scene.frame_token):
            confirmed = bool(getattr(getattr(self, 'self_localization_gate', None),
                                     'snapshot', None) and self.self_localization_gate.snapshot.valid)
            self._draw_fixed_pursuit_geometry(canvas, geometry, confirmed)
            self._draw_fixed_attack_geometry(canvas, geometry, scene, confirmed)
        elif self.current_player_valid:
            self.draw_combat_ranges_debug(canvas)
            self.draw_player_exclusion_debug(canvas)
            cv2.circle(canvas, self.loc_player, 4, (0, 0, 255), -1)
        self.draw_fixed_platform_debug(canvas)
        self.draw_continuous_attack_debug(canvas)
        self.draw_ladder_execution_debug(canvas)

    def _draw_fixed_pursuit_geometry(self, canvas, geometry, confirmed):
        x0, y0, x1, y1 = geometry.pursuit_rect
        if x1 <= x0 or y1 <= y0:
            return
        color = (255, 255, 0)
        cv2.rectangle(canvas, (x0, y0), (x1-1, y1-1), color,
                      2 if confirmed else 1)
        label = f'Pursuit {x1-x0}x{y1-y0} | same level + attack height'
        if not confirmed:
            label += ' | self pending'
        cv2.putText(canvas, label, (x0+8, min(y1-8, y0+22)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)

    def _draw_fixed_attack_geometry(self, canvas, geometry, scene, confirmed):
        x0, y0, x1, y1 = geometry.combined
        color = (0, 0, 255)
        if confirmed:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        else:
            for start, end in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                               ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
                length = max(abs(end[0]-start[0]), abs(end[1]-start[1]))
                for offset in range(0, length, 12):
                    finish = min(length, offset+6)
                    point = lambda n: (start[0]+(end[0]-start[0])*n//max(1, length),
                                       start[1]+(end[1]-start[1])*n//max(1, length))
                    cv2.line(canvas, point(offset), point(finish), color, 2)
        cv2.line(canvas, (geometry.center[0], y0), (geometry.center[0], y1), color, 1)
        cv2.circle(canvas, geometry.center, 4, color, -1)
        tolerance = int(self.cfg.get('combat_tracking', {}).get('pursuit_vertical_tolerance', 70))
        in_box = sum(geometry.contains(m) for m in scene.monsters)
        eligible = sum(geometry.contains(m) and geometry.same_level(m, tolerance)
                       for m in scene.monsters)
        state = getattr(self, 'fixed_platform_state', None)
        target = (getattr(state, 'current_target', None)
                  if self.current_player_valid and state is not None and
                  getattr(self, 'current_minimap_player_valid', False) and
                  not self.is_disable_control else None)
        if target is not None:
            tx, ty = target['position']
            tw, th = target['size']
            cv2.rectangle(canvas, (tx, ty), (tx+tw, ty+th), (255, 0, 255), 2)
        reason = (getattr(self, 'last_visual_pause_reason', '') or
                  ('disabled' if self.is_disable_control else
                   'minimap_invalid' if not getattr(self, 'current_minimap_player_valid', False)
                   else 'self_unconfirmed' if not confirmed else 'active'))
        cv2.putText(canvas, f'Attack box {in_box} / same level {eligible} | {reason}',
                    (max(8, x0), max(20, y0-8)), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, color, 2)
