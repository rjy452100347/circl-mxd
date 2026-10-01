"""Fixed-width minimap patrol with YOLO-gated attacks."""

import time

from src.states.base_state import State
from src.utils.logger import logger


class FixedPlatformState(State):
    """Patrol a captured minimap interval without yielding movement to combat."""

    TURN_MARGIN_PX = 2

    def __init__(self, name, bot):
        super().__init__(name, bot)
        self.anchor_x = None
        self.left_x = None
        self.right_x = None
        self.direction = "left"
        self.localized = False
        self._neutral_attack_required = False

    def on_enter(self):
        """A fresh state entry always captures a new platform anchor."""
        self.anchor_x = None
        self.left_x = None
        self.right_x = None
        self.direction = "left"
        self.localized = False
        self._neutral_attack_required = False

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
        }

    def _has_current_localization(self):
        return bool(
            getattr(self.bot, "current_nametag_valid", False)
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

    def _update_patrol_direction(self):
        player_x = float(self.bot.loc_player_minimap[0])
        if player_x <= self.left_x + self.TURN_MARGIN_PX:
            self.direction = "right"
        elif player_x >= self.right_x - self.TURN_MARGIN_PX:
            self.direction = "left"
        elif self.bot.is_player_stuck():
            self.direction = "right" if self.direction == "left" else "left"

    @staticmethod
    def _target_distance(bot, monster):
        x, y = monster["position"]
        width, height = monster["size"]
        return (
            abs(x + width // 2 - bot.loc_player[0])
            + abs(y + height // 2 - bot.loc_player[1])
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

    def on_frame(self):
        if not self._has_current_localization():
            self.localized = False
            self.bot.monsters = []
            self.bot.debug_attack_direction = "none"
            self.bot.pause_for_visual("fixed_platform_localization_invalid")
            return

        if self.anchor_x is None and not self._capture_bounds():
            self.localized = False
            self.bot.pause_for_visual("fixed_platform_localization_invalid")
            return

        self.localized = True
        if hasattr(self.bot, "last_visual_pause_reason"):
            self.bot.last_visual_pause_reason = ""

        self._update_patrol_direction()
        self.bot.cmd_move_x = self.direction
        self.bot.cmd_move_y = "none"
        self.bot.cmd_action = "none"
        self.bot.debug_attack_direction = "none"

        self.bot.update_monster_observations()

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
