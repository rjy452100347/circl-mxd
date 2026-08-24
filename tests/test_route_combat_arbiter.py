from src.engine.RouteCombatArbiter import (
    CombatIntent,
    RouteCombatArbiter,
    RouteIntent,
)


def test_critical_route_action_has_exclusive_control():
    arbiter = RouteCombatArbiter()
    route = RouteIntent("none", "up", "mount", "mount_retry")
    combat = CombatIntent("visible_pursuit", "left")

    result = arbiter.resolve(route, combat)

    assert result.command == "none up mount"
    assert result.owner == "route_critical"


def test_pursuit_overrides_only_horizontal_route_cruise():
    arbiter = RouteCombatArbiter()
    route = RouteIntent("right", "none", "none")

    result = arbiter.resolve(route, CombatIntent("visible_pursuit", "left"))

    assert result.command == "left none none"
    assert result.owner == "combat_pursuit"


def test_attack_stops_even_when_route_direction_is_aligned():
    arbiter = RouteCombatArbiter()
    route = RouteIntent("right", "none", "none")

    result = arbiter.resolve(
        route, CombatIntent("visible_attack", "right", "attack_right")
    )

    assert result.command == "stop none attack_right"


def test_attack_stops_route_that_would_walk_away_from_target():
    arbiter = RouteCombatArbiter()
    route = RouteIntent("right", "none", "none")

    result = arbiter.resolve(
        route, CombatIntent("visible_attack", "left", "attack_left")
    )

    assert result.command == "stop none attack_left"


def test_lost_pursuit_keeps_last_direction_without_attack():
    arbiter = RouteCombatArbiter()
    route = RouteIntent("right", "none", "jump")

    result = arbiter.resolve(
        RouteIntent("none", "none", "none", "combat_hold"),
        CombatIntent("lost_grace_pursuit", "left", "none"),
    )

    assert result.command == "left none none"
    assert result.owner == "combat_pursuit"


def test_lost_attack_stays_stopped_and_keeps_attack_action():
    arbiter = RouteCombatArbiter()

    result = arbiter.resolve(
        RouteIntent("none", "none", "none", "combat_hold"),
        CombatIntent("lost_grace_attack", "right", "attack_right"),
    )

    assert result.command == "stop none attack_right"
    assert result.owner == "combat_attack"
