"""Fixed-width minimap patrol with YOLO-gated attacks."""

import time

from src.states.base_state import State
from src.utils.logger import logger
from src.engine.FullscreenPerception import current_player_valid
from src.engine.FullscreenSelfDetector import uses_fullscreen_self
from src.engine.OpenVinoMonsterDetector import OpenVinoMonsterDetector


class FixedPlatformState(State):
    """Patrol a captured minimap interval without yielding movement to combat."""

    TURN_MARGIN_PX = 2
    STALL_SECONDS = 2.0
    PROGRESS_PX = 2
    MAX_SAMPLE_GAP = 0.5

    def __init__(self, name, bot):
        super().__init__(name, bot)
        self.anchor_x = None
        self.left_x = None
        self.right_x = None
        self.direction = "left"
        self.localized = False
        self._neutral_attack_required = False
        self._clear_progress()
        self._clear_pursuit()
        self.combat_state = "patrol"
        self.current_target = None
        self.last_status = "等待人物定位"

    def on_enter(self):
        """A fresh state entry always captures a new platform anchor."""
        self.anchor_x = None
        self.left_x = None
        self.right_x = None
        self.direction = "left"
        self.localized = False
        self._neutral_attack_required = False
        self._clear_progress()
        self._clear_pursuit()
        self.combat_state = "patrol"
        self.current_target = None
        self.last_status = "等待人物定位"

    def on_exit(self):
        pass

    def check_transitions(self):
        return None

    def get_diagnostics(self):
        """Return stable, rendering-friendly patrol diagnostics."""
        requested_width = int(
            self.bot.cfg.get("fixed_platform", {}).get("width_px", 80)
        )
        actual_width = (
            None if self.left_x is None or self.right_x is None
            else self.right_x - self.left_x
        )
        return {
            "anchor_x": self.anchor_x,
            "left_x": self.left_x,
            "right_x": self.right_x,
            "direction": self.direction,
            "localized": self.localized,
            "requested_width": requested_width,
            "actual_width": actual_width,
            "combat_state": self.combat_state,
        }

    def _has_current_localization(self):
        return bool(
            current_player_valid(self.bot)
            and getattr(self.bot, "current_minimap_player_valid", False)
        )

    def _capture_bounds(self):
        minimap_width = int(self.bot.img_minimap.shape[1])
        if minimap_width <= 0:
            return False

        observed_x = int(self.bot.loc_player_minimap[0])
        max_x = minimap_width - 1
        self.anchor_x = min(max(observed_x, 0), max_x)

        requested_width = int(self.bot.cfg["fixed_platform"]["width_px"])
        left_extent = requested_width // 2
        right_extent = requested_width - left_extent
        self.left_x = max(0, self.anchor_x - left_extent)
        self.right_x = min(max_x, self.anchor_x + right_extent)
        logger.info(
            "[固定平台] 已锚定小地图区域："
            f"center={self.anchor_x} left={self.left_x} right={self.right_x} "
            f"requested_width={requested_width}"
        )
        return True

    def _clear_progress(self):
        self._progress_paused_at = None
        self._progress_x = None
        self._progress_at = None
        self._sample_at = None

    def _clear_pursuit(self):
        self._pursuit_direction = None
        self._pursuit_x = None
        self._pursuit_at = None
        self._pursuit_sample_at = None
        self._pursuit_blocked_direction = None
        self._pursuit_blocked_until = 0.0

    def suspend_tracking(self, preserve_progress=False):
        """Also used when capture fails before the FSM receives a frame."""
        self._clear_pursuit()
        self._neutral_attack_required = False
        self.combat_state = "patrol"
        self.current_target = None
        if preserve_progress:
            if self._progress_paused_at is None:
                self._progress_paused_at = time.monotonic()
            return
        self._clear_progress()
        self.localized = False

    def _reset_progress(self, x, now):
        self._progress_x = x
        self._progress_at = self._sample_at = now

    def _update_patrol_direction(self):
        """Track horizontal progress in this mode, independent of the FSM watchdog.

        Return True for a single neutral frame when inward input needs to be
        reasserted at a boundary. Never turn outward to escape a blocked edge.
        """
        now = time.monotonic()
        if self._progress_paused_at is not None:
            elapsed = max(0., now-self._progress_paused_at)
            if self._progress_at is not None:
                self._progress_at += elapsed
            if self._sample_at is not None:
                self._sample_at += elapsed
            self._progress_paused_at = None
        player_x = float(self.bot.loc_player_minimap[0])
        inward = None
        if player_x <= self.left_x + self.TURN_MARGIN_PX:
            inward = "right"
        elif player_x >= self.right_x - self.TURN_MARGIN_PX:
            inward = "left"
        if inward is not None and self.direction != inward:
            self.direction = inward
            self._reset_progress(player_x, now)
            self.last_status = f"到达边界，向{'右' if inward == 'right' else '左'}返回"
            logger.info(f"[固定平台] 边界反向 x={player_x} direction={inward}")
            return False

        kb = getattr(self.bot, "kb", None)
        paused = (self.bot.is_disable_control or
                  getattr(kb, "is_need_force_heal", False) or
                  not getattr(kb, "_was_game_window_active", True) or
                  not getattr(kb, "is_enable", True))
        if (paused or self._progress_at is None or self._sample_at is None or
                now - self._sample_at > self.MAX_SAMPLE_GAP):
            self._reset_progress(player_x, now)
            return False
        self._sample_at = now
        progress = (player_x - self._progress_x) * (1 if self.direction == "right" else -1)
        if progress >= self.PROGRESS_PX:
            self._reset_progress(player_x, now)
            self.last_status = f"正在向{'右' if self.direction == 'right' else '左'}巡逻"
        elif now - self._progress_at >= self.STALL_SECONDS:
            self._reset_progress(player_x, now)
            if inward is not None:
                self.last_status = "边界内侧移动无进展，松键一帧后重新向内移动"
                logger.warning(f"[固定平台] 边界内侧无进展，重发方向 x={player_x} direction={inward}")
                return True
            self.direction = "right" if self.direction == "left" else "left"
            self.last_status = "连续 2 秒横向移动无进展，已反向"
            logger.warning(f"[固定平台] 无进展反向 x={player_x} direction={self.direction}")
        return False

    @staticmethod
    def _target_distance(bot, monster):
        x, y = monster["position"]
        width, height = monster["size"]
        geometry = getattr(bot, 'fixed_attack_geometry', None)
        anchor = geometry.center if geometry is not None else bot.loc_player
        return (
            abs(x + width // 2 - anchor[0])
            + abs(y + height // 2 - anchor[1])
        )

    def _directional_attack_direction(self):
        left_target = self.bot.get_nearest_monster(is_left=True)
        right_target = self.bot.get_nearest_monster(is_left=False)

        def is_on_side(target, direction):
            if target is None:
                return False
            x, _ = target["position"]
            width, _ = target["size"]
            center_x = x + width // 2
            if direction == "left":
                return center_x < self.bot.loc_player[0]
            return center_x > self.bot.loc_player[0]

        if not is_on_side(left_target, "left"):
            left_target = None
        if not is_on_side(right_target, "right"):
            right_target = None
        if left_target is None:
            return "right" if right_target is not None else None
        if right_target is None:
            return "left"

        left_distance = self._target_distance(self.bot, left_target)
        right_distance = self._target_distance(self.bot, right_target)
        if left_distance == right_distance:
            return self.direction
        return "left" if left_distance < right_distance else "right"

    def _attack_action(self, now):
        attack_mode = self.bot.cfg["bot"]["attack"]
        if attack_mode == "directional":
            direction = self._directional_attack_direction()
            cooldown = float(self.bot.cfg["directional_attack"]["cooldown"])
            if direction is not None and now - self.bot.t_last_attack > cooldown:
                self.bot.debug_attack_direction = direction
                return f"attack_{direction}"
        elif attack_mode == "aoe_skill":
            target = self.bot.get_nearest_monster(is_left=True)
            cooldown = float(self.bot.cfg["aoe_skill"]["cooldown"])
            if target is not None and now - self.bot.t_last_attack > cooldown:
                self.bot.debug_attack_direction = "aoe"
                return "attack"
        return "none"

    def _uses_fullscreen_directional_combat(self):
        return (uses_fullscreen_self(self.bot.cfg)
                and self.bot.cfg["bot"]["attack"] == "directional")

    @staticmethod
    def _box_overlap(monster, rect):
        x, y = monster["position"]
        w, h = monster["size"]
        x0, y0, x1, y1 = rect
        return min(x1, x+w) > max(x0, x) and min(y1, y+h) > max(y0, y)

    def _rank_monster(self, monster):
        x, y = monster["position"]
        w, h = monster["size"]
        confidence = float(monster.get("confidence", 1-monster.get("score", 1)))
        return (self._target_distance(self.bot, monster), -confidence,
                x+w//2, y+h, x, y)

    def _pursuit_rank(self, monster):
        x, y = monster["position"]
        w, h = monster["size"]
        confidence = float(monster.get("confidence", 1-monster.get("score", 1)))
        return (self.bot._monster_horizontal_distance(monster), -confidence,
                x+w//2, y+h, x, y)

    def _in_range_target(self):
        geometry = getattr(self.bot, 'fixed_attack_geometry', None)
        if geometry is not None:
            tolerance = int(self.bot.cfg.get('combat_tracking', {}).get(
                'pursuit_vertical_tolerance', 70))
            candidates = [monster for monster in self.bot.monsters
                if geometry.same_level(monster, tolerance) and geometry.contains(monster)]
        else:
            candidates = [monster for monster in self.bot.monsters
                if self.bot._monster_is_same_level(monster)
                and self.bot.is_monster_in_attack_range(monster)]
        return min(candidates, key=self._rank_monster, default=None)

    def _pursuit_candidates(self):
        geometry = getattr(self.bot, 'fixed_attack_geometry', None)
        rect = (geometry.pursuit_rect if geometry is not None else
                OpenVinoMonsterDetector.visible_input_rect(
                    self.bot.img_frame.shape, self.bot.loc_player))
        _, attack_top, _, attack_bottom = self.bot.get_attack_range(is_left=True)
        result = []
        for monster in self.bot.monsters:
            x, y = monster["position"]
            _, h = monster["size"]
            if (self._box_overlap(monster, rect)
                    and self.bot._monster_is_same_level(monster)
                    and min(attack_bottom, y+h) > max(attack_top, y)):
                result.append(monster)
        return sorted(result, key=self._pursuit_rank)

    def _pursuit_has_progress(self, direction, now):
        x = float(self.bot.loc_player_minimap[0])
        if (direction != self._pursuit_direction
                or self._pursuit_sample_at is None
                or now-self._pursuit_sample_at > self.MAX_SAMPLE_GAP):
            self._pursuit_direction = direction
            self._pursuit_x = x
            self._pursuit_at = self._pursuit_sample_at = now
            return True
        self._pursuit_sample_at = now
        progress = (x-self._pursuit_x) * (1 if direction == "right" else -1)
        if progress >= self.PROGRESS_PX:
            self._pursuit_x = x
            self._pursuit_at = now
            return True
        return now-self._pursuit_at < self.STALL_SECONDS

    def _patrol_after_combat(self, *, unreachable=False):
        if self.combat_state in {"attack", "pursuit"}:
            self._clear_progress()
        self.combat_state = "unreachable" if unreachable else "patrol"
        self.current_target = None
        self._neutral_attack_required = False
        refresh_input = self._update_patrol_direction()
        self.bot.cmd_move_x = "stop" if refresh_input else self.direction
        self.bot.cmd_move_y = "none"
        self.bot.cmd_action = "none"
        self.bot.debug_attack_direction = "none"
        if unreachable:
            self.last_status = "目标不可达，沿原边界巡逻"
        elif self.combat_state == "patrol":
            self.last_status = "无同层可攻击目标，正在巡逻"
        self.bot.kb.set_command(
            f"{self.bot.cmd_move_x} {self.bot.cmd_move_y} none"
        )

    def _on_fullscreen_directional_frame(self):
        # The full-scene inference has already run for this capture frame.
        geometry = getattr(self.bot, 'fixed_attack_geometry', None)
        if (geometry is None or
                geometry.frame_token != self.bot._detection_frame_token()):
            self.bot.pause_for_visual('self_geometry_missing')
            return
        self.bot.update_monster_observations()
        if not self.bot.fullscreen_control_ready():
            return
        if self.bot.is_disable_control:
            return

        target = self._in_range_target()
        if target is not None:
            self._clear_progress()
            self._pursuit_direction = None
            self.combat_state = "attack"
            self.current_target = target
            direction = self.bot._monster_direction(target)
            self.bot.debug_attack_direction = direction
            self.bot.cmd_move_x = "stop"
            self.bot.cmd_move_y = "none"
            self.bot.cmd_action = "none"
            self.last_status = f"停步朝{('左' if direction == 'left' else '右')}攻击"
            if self._neutral_attack_required:
                self._neutral_attack_required = False
            else:
                now = time.time()
                cooldown = float(self.bot.cfg["directional_attack"]["cooldown"])
                if now-self.bot.t_last_attack > cooldown:
                    self.bot.cmd_action = f"attack_{direction}"
                    self.bot.t_last_attack = now
                    self._neutral_attack_required = True
            self.bot.kb.set_command(f"stop none {self.bot.cmd_action}")
            return

        self._neutral_attack_required = False
        candidates = self._pursuit_candidates()
        now = time.monotonic()
        player_x = float(self.bot.loc_player_minimap[0])
        unreachable = False
        for target in candidates:
            direction = self.bot._monster_direction(target)
            if (direction == self._pursuit_blocked_direction
                    and now < self._pursuit_blocked_until):
                unreachable = True
                continue
            if ((direction == "left" and player_x <= self.left_x+self.TURN_MARGIN_PX)
                    or (direction == "right" and player_x >= self.right_x-self.TURN_MARGIN_PX)):
                unreachable = True
                continue
            if not self._pursuit_has_progress(direction, now):
                self._pursuit_blocked_direction = direction
                self._pursuit_blocked_until = now+self.STALL_SECONDS
                self._pursuit_direction = None
                self.direction = "right" if direction == "left" else "left"
                self._clear_progress()
                self._patrol_after_combat(unreachable=True)
                return
            self._clear_progress()
            self.combat_state = "pursuit"
            self.current_target = target
            self.bot.cmd_move_x = direction
            self.bot.cmd_move_y = "none"
            self.bot.cmd_action = "none"
            self.bot.debug_attack_direction = "none"
            self.last_status = f"正在向{'左' if direction == 'left' else '右'}靠近同层怪物"
            self.bot.kb.set_command(f"{direction} none none")
            return

        self._pursuit_direction = None
        self._patrol_after_combat(unreachable=unreachable)

    def on_frame(self):
        if not self._has_current_localization():
            if self.localized:
                logger.warning("[固定平台] 定位丢失，暂停巡逻："
                               f"player={current_player_valid(self.bot)} "
                               f"minimap={self.bot.current_minimap_player_valid}")
            self.suspend_tracking()
            self.last_status = "人物定位或小地图人物点无效，等待定位恢复"
            self.localized = False
            self.bot.monsters = []
            self.bot.debug_attack_direction = "none"
            self.bot.pause_for_visual("fixed_platform_localization_invalid")
            return

        if self.anchor_x is None and not self._capture_bounds():
            self.localized = False
            self.bot.pause_for_visual("fixed_platform_localization_invalid")
            return

        if not self.localized:
            self._clear_progress()
            self.last_status = "定位已恢复，沿原边界继续巡逻"
        self.localized = True
        if hasattr(self.bot, "last_visual_pause_reason"):
            self.bot.last_visual_pause_reason = ""

        if self._uses_fullscreen_directional_combat():
            self._on_fullscreen_directional_frame()
            return

        refresh_input = self._update_patrol_direction()
        self.bot.cmd_move_x = self.direction
        self.bot.cmd_move_y = "none"
        self.bot.cmd_action = "none"
        self.bot.debug_attack_direction = "none"

        self.bot.update_monster_observations()
        if uses_fullscreen_self(self.bot.cfg) and not self.bot.fullscreen_control_ready():
            return

        if refresh_input:
            self.bot.cmd_move_x = "stop"
            self.bot.kb.set_command("stop stop none")
            self._neutral_attack_required = False
            return

        force_neutral = self._neutral_attack_required
        self._neutral_attack_required = False
        if not force_neutral and not self.bot.is_disable_control:
            now = time.time()
            action = self._attack_action(now)
            if action != "none":
                self.bot.cmd_action = action
                self.bot.t_last_attack = now
                self._neutral_attack_required = True

        self.bot.kb.set_command(
            f"{self.bot.cmd_move_x} {self.bot.cmd_move_y} "
            f"{self.bot.cmd_action}"
        )
