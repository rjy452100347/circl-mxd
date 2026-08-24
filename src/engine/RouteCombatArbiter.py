from __future__ import annotations

from dataclasses import dataclass


CRITICAL_ROUTE_ACTIONS = {"jump", "mount", "teleport", "goal"}


@dataclass(frozen=True)
class RouteIntent:
    """One route-map decision before combat is allowed to modify control."""

    move_x: str
    move_y: str
    action: str
    reason: str = "route_hit"

    @classmethod
    def from_command(cls, command: str, reason: str = "route_hit"):
        move_x, move_y, action = command.split()
        return cls(move_x, move_y, action, reason)

    @property
    def is_critical(self):
        return (
            self.reason == "route_lost" or
            self.move_x == "stop" or
            self.move_y != "none" or
            self.action in CRITICAL_ROUTE_ACTIONS
        )


@dataclass(frozen=True)
class CombatIntent:
    """Combat request on independent locomotion and action channels."""

    state: str = "observe"
    direction: str | None = None
    action: str = "none"
    target: dict | None = None
    target_visible: bool = False

    @property
    def owns_control(self):
        return self.state in {
            "visible_pursuit",
            "visible_attack",
            "attack_wait",
            "lost_grace_pursuit",
            "lost_grace_attack",
        }


@dataclass(frozen=True)
class ResolvedCommand:
    move_x: str
    move_y: str
    action: str
    owner: str

    @property
    def command(self):
        return f"{self.move_x} {self.move_y} {self.action}"


class RouteCombatArbiter:
    """Resolve route and combat without allowing combat to break traversal."""

    def resolve(self, route: RouteIntent, combat: CombatIntent):
        if route.is_critical:
            return ResolvedCommand(
                route.move_x, route.move_y, route.action, "route_critical"
            )

        if (combat.state in {"visible_pursuit", "lost_grace_pursuit"} and
                combat.direction in {"left", "right"}):
            return ResolvedCommand(
                combat.direction, "none", combat.action, "combat_pursuit"
            )

        if combat.state in {
            "visible_attack", "attack_wait", "lost_grace_attack"
        }:
            return ResolvedCommand(
                "stop", "none", combat.action, "combat_attack"
            )

        return ResolvedCommand(
            route.move_x, route.move_y, route.action, "route_cruise"
        )
