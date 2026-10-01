"""Continuous directional attacks with a periodic anti-idle sequence."""

import random
import time

from src.states.base_state import State
from src.utils.logger import logger


class ContinuousAttackState(State):
    """Lock one YOLO-derived direction, then attack without further inference."""

    ATTACK_MIN_SECONDS = 20.0
    ATTACK_MAX_SECONDS = 25.0
    ACTION_GAP_SECONDS = 1.0

    def __init__(self, name, bot, clock=None, random_uniform=None):
        super().__init__(name, bot)
        self._clock = clock or time.monotonic
        self._random_uniform = random_uniform or random.uniform
        self.locked_direction = None
        self.selected_target = None
        self.phase = "waiting_direction"
        self.cycle_duration = None
        self.cycle_deadline = None
        self.phase_deadline = None
        self._neutral_attack_required = False
        self._healing = False

    def on_enter(self):
        self.locked_direction = None
        self.selected_target = None
        self.phase = "waiting_direction"
        self.cycle_duration = None
        self.cycle_deadline = None
        self.phase_deadline = None
        self._neutral_attack_required = False
        self._healing = False
        self.bot.monsters = []
        self.bot.debug_attack_direction = "none"

    def on_exit(self):
        self._send_command("stop", "none", "none")

    def check_transitions(self):
        return None

    @property
    def direction_locked(self):
        return self.locked_direction in {"left", "right"}

    @staticmethod
    def _confidence(monster):
        return float(monster.get("confidence", monster.get("score", 0.0)))

    def _select_direction(self):
        player_x, player_y = self.bot.loc_player
        candidates = []
        for monster in self.bot.monsters:
            monster_x, monster_y = monster["position"]
            width, height = monster["size"]
            center_x = monster_x + width // 2
            center_y = monster_y + height // 2
            if center_x == player_x:
                continue
            direction = "left" if center_x < player_x else "right"
            distance = abs(center_x - player_x) + abs(center_y - player_y)
            # Left wins only a complete distance/confidence tie.
            side_rank = 0 if direction == "left" else 1
            candidates.append((
                distance,
                -self._confidence(monster),
                side_rank,
                direction,
                monster,
            ))
        if not candidates:
            return None, None
        _, _, _, direction, monster = min(candidates, key=lambda item: item[:3])
        return direction, monster

    def _send_command(self, move_x, move_y, action):
        self.bot.cmd_move_x = move_x
        self.bot.cmd_move_y = move_y
        self.bot.cmd_action = action
        if getattr(self.bot, "is_disable_control", False):
            move_x, move_y, action = "none", "none", "none"
        self.bot.kb.set_command(f"{move_x} {move_y} {action}")

    def _start_attack_cycle(self, now, directed_attack=True):
        self.cycle_duration = float(self._random_uniform(
            self.ATTACK_MIN_SECONDS,
            self.ATTACK_MAX_SECONDS,
        ))
        self.cycle_deadline = now + self.cycle_duration
        self.phase_deadline = None
        self.phase = "attacking"
        action = (
            f"attack_{self.locked_direction}"
            if directed_attack else "attack"
        )
        self._neutral_attack_required = True
        self.bot.debug_attack_direction = self.locked_direction
        self._send_command("none", "none", action)

    def _enter_healing(self):
        self.phase = "healing"
        self.cycle_duration = None
        self.cycle_deadline = None
        self.phase_deadline = None
        self._neutral_attack_required = False
        self._healing = True
        self._send_command("none", "none", "none")

    def _handle_waiting_direction(self, now):
        if not getattr(self.bot, "current_nametag_valid", False):
            self.bot.monsters = []
            self.bot.debug_attack_direction = "none"
            self._send_command("none", "none", "none")
            return

        self.bot.update_monster_observations()
        direction, target = self._select_direction()
        if direction is None:
            self.bot.debug_attack_direction = "none"
            self._send_command("none", "none", "none")
            return

        self.locked_direction = direction
        self.selected_target = target
        logger.info(
            "[持续定向攻击] 已锁定攻击方向："
            f"direction={direction} target={target}"
        )
        self._start_attack_cycle(now, directed_attack=True)

    def _handle_attacking(self, now):
        if now >= self.cycle_deadline:
            self.phase = "gap_after_jump"
            self.phase_deadline = now + self.ACTION_GAP_SECONDS
            self._neutral_attack_required = False
            self._send_command("none", "none", "jump")
            return

        if self._neutral_attack_required:
            self._neutral_attack_required = False
            self._send_command("none", "none", "none")
            return

        self._neutral_attack_required = True
        self._send_command("none", "none", "attack")

    def _handle_sequence(self, now):
        if self.phase == "gap_after_jump":
            if now < self.phase_deadline:
                self._send_command("none", "none", "none")
                return
            self.phase = "gap_after_left"
            self.phase_deadline = now + self.ACTION_GAP_SECONDS
            self._send_command("left", "none", "none")
            return

        if self.phase == "gap_after_left":
            if now < self.phase_deadline:
                self._send_command("none", "none", "none")
                return
            self.phase = "restore_direction"
            self.phase_deadline = None
            self._send_command("right", "none", "none")
            return

        if self.phase == "restore_direction":
            self._start_attack_cycle(now, directed_attack=True)

    def on_frame(self):
        now = self._clock()
        force_heal = bool(getattr(self.bot.kb, "is_need_force_heal", False))
        if force_heal:
            self._enter_healing()
            return

        if self._healing:
            self._healing = False
            if self.direction_locked:
                self._start_attack_cycle(now, directed_attack=True)
            else:
                self.phase = "waiting_direction"
                self._handle_waiting_direction(now)
            return

        if not self.direction_locked:
            self.phase = "waiting_direction"
            self._handle_waiting_direction(now)
            return

        if self.phase == "attacking":
            self._handle_attacking(now)
            return

        self._handle_sequence(now)

    def get_diagnostics(self, now=None):
        now = self._clock() if now is None else now
        remaining = None
        if self.phase == "attacking" and self.cycle_deadline is not None:
            remaining = max(0.0, self.cycle_deadline - now)
        elif self.phase in {"gap_after_jump", "gap_after_left"} and \
                self.phase_deadline is not None:
            remaining = max(0.0, self.phase_deadline - now)
        return {
            "phase": self.phase,
            "locked_direction": self.locked_direction,
            "selected_target": self.selected_target,
            "cycle_duration": self.cycle_duration,
            "remaining_seconds": remaining,
            "healing": self._healing,
        }
