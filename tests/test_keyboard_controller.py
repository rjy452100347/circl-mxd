import threading

import src.input.KeyBoardController as module
from src.input.KeyBoardController import (
    KeyBoardController,
    press_key,
    _scan_code_for_key,
)


def _controller(command, iterations=2):
    controller = object.__new__(KeyBoardController)
    controller.cfg = {
        "key": {
            "jump": "space", "teleport": "", "add_hp": "1", "add_mp": "2",
            "return_home": "home",
        },
        "health_monitor": {"add_hp_cooldown": 0.5, "add_mp_cooldown": 0.5},
        "route": {"mount_press_duration": 0.10},
        "directional_attack": {"character_turn_delay": 0.02},
    }
    controller.cmd_left_right, controller.cmd_up_down, controller.cmd_action = command.split()
    controller.cmd_left_right_last = ""
    controller.cmd_up_down_last = ""
    controller.cmd_action_last = "none"
    controller._command_lock = threading.Lock()
    controller._aux_action_lock = threading.Lock()
    controller._input_state_lock = threading.RLock()
    controller._aux_actions = {}
    controller._aux_generations = {
        "add_hp": 0, "add_mp": 0, "return_home": 0,
    }
    controller._aux_last_dispatched = {
        "add_hp": float("-inf"), "add_mp": float("-inf"),
        "return_home": float("-inf"),
    }
    controller.command_timeout_seconds = 10.0
    controller.aux_action_timeout_seconds = 0.3
    controller.command_updated_at = module.time.monotonic()
    controller._foreground_acquired_at = controller.command_updated_at - 1.0
    controller._was_game_window_active = True
    controller.is_enable = True
    controller.is_terminated = False
    controller.is_need_force_heal = False
    controller.attack_key = "w"
    controller.t_last_skill = 0.0
    controller.is_game_window_active = lambda: True
    calls = {"count": 0}

    def finish_after_iterations():
        calls["count"] += 1
        if calls["count"] >= iterations:
            controller.is_terminated = True

    controller.limit_fps = finish_after_iterations
    return controller


def test_continuous_direction_is_pressed_once_not_repeated(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "key_down", lambda key: operations.append(("down", key)))
    monkeypatch.setattr(module, "key_up", lambda key: operations.append(("up", key)))
    controller = _controller("right none none", iterations=3)

    controller.run()

    assert operations.count(("down", "right")) == 1


def test_mount_presses_space_then_up_once_and_keeps_up_held(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "key_down", lambda key: operations.append(("down", key)))
    monkeypatch.setattr(module, "key_up", lambda key: operations.append(("up", key)))
    monkeypatch.setattr(module.time, "sleep", lambda _duration: None)
    controller = _controller("none up mount", iterations=3)

    controller.run()

    assert operations.count(("down", "space")) == 1
    assert operations.count(("down", "up")) == 1
    mount_start = operations.index(("down", "space"))
    assert operations[mount_start:mount_start + 3] == [
        ("down", "space"),
        ("down", "up"),
        ("up", "space"),
    ]


def test_foreground_loss_releases_held_keys(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "key_down", lambda key: operations.append(("down", key)))
    monkeypatch.setattr(module, "key_up", lambda key: operations.append(("up", key)))
    controller = _controller("right none none", iterations=1)
    controller._was_game_window_active = True
    controller.is_game_window_active = lambda: False

    controller.run()

    assert ("up", "left") in operations
    assert ("up", "right") in operations
    assert ("up", "space") in operations


def test_stale_command_releases_held_direction_without_replay(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "key_down", lambda key: operations.append(("down", key)))
    monkeypatch.setattr(module, "key_up", lambda key: operations.append(("up", key)))
    monkeypatch.setattr(module.time, "monotonic", lambda: 10.0)
    controller = _controller("right none none", iterations=1)
    controller.command_timeout_seconds = 0.5
    controller.command_updated_at = 1.0
    controller._foreground_acquired_at = 0.0
    controller.cmd_left_right_last = "right"

    controller.run()

    assert ("down", "right") not in operations
    assert ("up", "right") in operations


def test_foreground_reacquisition_does_not_replay_old_command(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "key_down", lambda key: operations.append(("down", key)))
    monkeypatch.setattr(module, "key_up", lambda key: operations.append(("up", key)))
    monkeypatch.setattr(module.time, "monotonic", lambda: 10.0)
    controller = _controller("right none none", iterations=2)
    controller._was_game_window_active = False
    controller.command_updated_at = 1.0

    controller.run()

    assert ("down", "right") not in operations


def test_keyboard_loop_backend_failure_terminates_and_releases(monkeypatch):
    operations = []

    def failing_key_down(key):
        operations.append(("down", key))
        if key == "right":
            raise RuntimeError("backend failure")

    monkeypatch.setattr(module, "key_down", failing_key_down)
    monkeypatch.setattr(module, "key_up", lambda key: operations.append(("up", key)))
    controller = _controller("right none none", iterations=3)

    controller.run()

    assert controller.is_terminated
    assert ("up", "left") in operations
    assert ("up", "right") in operations
    assert ("up", "space") in operations


def test_press_key_releases_key_when_sleep_fails(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "key_down", lambda key: operations.append(("down", key)))
    monkeypatch.setattr(module, "key_up", lambda key: operations.append(("up", key)))

    def fail_sleep(_duration):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(module.time, "sleep", fail_sleep)

    try:
        press_key("q")
    except RuntimeError:
        pass
    else:
        raise AssertionError("press_key should propagate the backend interruption")

    assert operations == [("down", "q"), ("up", "q")]


def test_press_key_attempts_release_when_key_down_fails(monkeypatch):
    operations = []

    def fail_key_down(key):
        operations.append(("down", key))
        raise RuntimeError("uncertain injection result")

    monkeypatch.setattr(module, "key_down", fail_key_down)
    monkeypatch.setattr(module, "key_up", lambda key: operations.append(("up", key)))

    try:
        press_key("q")
    except RuntimeError:
        pass
    else:
        raise AssertionError("press_key should propagate the backend failure")

    assert operations == [("down", "q"), ("up", "q")]


def test_directional_attack_turns_then_restores_route_direction_once(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "key_down", lambda key: operations.append(("down", key)))
    monkeypatch.setattr(module, "key_up", lambda key: operations.append(("up", key)))
    monkeypatch.setattr(module, "press_key", lambda key: operations.append(("press", key)))
    monkeypatch.setattr(module.time, "sleep", lambda _duration: None)
    controller = _controller("right none attack_left", iterations=3)

    controller.run()

    attack_index = operations.index(("press", "w"))
    assert operations[attack_index - 2:attack_index + 3] == [
        ("up", "right"),
        ("down", "left"),
        ("press", "w"),
        ("up", "left"),
        ("down", "right"),
    ]
    assert operations.count(("press", "w")) == 1


def test_ui_letter_attack_key_has_windows_scan_code():
    assert _scan_code_for_key("w") == 0x11


def test_windows_navigation_keys_have_fixed_extended_scan_codes():
    expected = {
        "home": 0x47, "end": 0x4F,
        "pageup": 0x49, "pgup": 0x49,
        "pagedown": 0x51, "pgdown": 0x51,
        "insert": 0x52, "ins": 0x52,
        "delete": 0x53, "del": 0x53,
    }
    for key, scan_code in expected.items():
        assert _scan_code_for_key(key) == scan_code
        assert key in module._EXTENDED_KEYS


def test_windows_keyboard_is_always_dispatched_by_sendinput(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "is_mac", lambda: False)
    monkeypatch.setattr(
        module, "_send_input_key",
        lambda key, is_key_up: operations.append(("sendinput", key, is_key_up))
    )
    monkeypatch.setattr(
        module.pyautogui, "keyDown",
        lambda key: operations.append(("pyautogui_down", key))
    )
    module.key_down("q")

    assert operations == [("sendinput", "q", False)]


def test_auxiliary_requests_are_latest_only_and_hp_has_priority():
    controller = _controller("none none none", iterations=1)
    now = controller.command_updated_at

    assert controller.request_aux_action("add_mp", 1, now)
    assert controller.request_aux_action("add_hp", 1, now)
    assert not controller.request_aux_action("add_hp", 1, now)

    request = controller._take_aux_action(now + 0.01)

    assert request.action == "add_hp"
    assert set(controller._aux_actions) == {"add_mp"}


def test_stale_or_pre_focus_auxiliary_request_is_discarded():
    controller = _controller("none none none", iterations=1)
    controller._foreground_acquired_at = 10.0
    controller.request_aux_action("add_hp", 1, requested_at=9.9)

    assert controller._take_aux_action(10.1) is None
    assert controller._aux_actions == {}


def test_auxiliary_cooldown_uses_successful_dispatch_time():
    controller = _controller("none none none", iterations=1)
    controller._foreground_acquired_at = 0.0
    controller._aux_last_dispatched["add_hp"] = 10.0
    controller.request_aux_action("add_hp", 1, requested_at=10.1)

    assert controller._take_aux_action(10.2) is None
    # A new health frame replaces the request while cooldown is active.  The
    # old request is never replayed after becoming stale.
    controller.request_aux_action("add_hp", 2, requested_at=10.49)
    assert controller._take_aux_action(10.5).action == "add_hp"


def test_recovery_owns_action_cycle_without_concurrent_attack(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "key_down", lambda key: operations.append(("down", key)))
    monkeypatch.setattr(module, "key_up", lambda key: operations.append(("up", key)))
    monkeypatch.setattr(module, "press_key", lambda key: operations.append(("press", key)))
    controller = _controller("right none attack", iterations=1)
    controller.request_aux_action(
        "add_hp", 1, requested_at=controller.command_updated_at
    )

    controller.run()

    assert ("press", "1") in operations
    assert ("press", "w") not in operations


def test_force_heal_suppresses_attack_and_mp_but_keeps_movement(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "key_down", lambda key: operations.append(("down", key)))
    monkeypatch.setattr(module, "key_up", lambda key: operations.append(("up", key)))
    monkeypatch.setattr(module, "press_key", lambda key: operations.append(("press", key)))
    controller = _controller("right none attack", iterations=1)
    controller.set_force_heal(True)
    controller.request_aux_action(
        "add_mp", 1, requested_at=controller.command_updated_at
    )

    controller.run()

    assert ("down", "right") in operations
    assert ("press", "w") not in operations
    assert ("press", "2") not in operations


def test_force_heal_releases_vertical_mount_input(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "key_down", lambda key: operations.append(("down", key)))
    monkeypatch.setattr(module, "key_up", lambda key: operations.append(("up", key)))
    monkeypatch.setattr(module, "press_key", lambda key: operations.append(("press", key)))
    controller = _controller("right up mount", iterations=1)
    controller.cmd_up_down_last = "up"
    controller.set_force_heal(True)

    controller.run()

    assert ("down", "right") in operations
    assert ("up", "up") in operations
    assert ("down", "up") not in operations
    assert ("down", "space") not in operations


def test_auxiliary_dispatch_rechecks_foreground_immediately_before_press(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "press_key", lambda key: operations.append(key))
    controller = _controller("none none none", iterations=1)
    controller.is_game_window_active = lambda: False
    request = module.AuxActionRequest("add_hp", 1, controller.command_updated_at)

    assert controller._dispatch_aux_action(request, controller.command_updated_at) is False
    assert operations == []
    assert controller._aux_last_dispatched["add_hp"] == float("-inf")


def test_dequeued_request_is_cancelled_by_newer_recovery_state(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "press_key", lambda key: operations.append(key))
    controller = _controller("none none none", iterations=1)
    now = controller.command_updated_at
    controller.request_aux_action("add_hp", 1, now)
    request = controller._take_aux_action(now + 0.01)

    controller.clear_aux_actions({"add_hp"})

    assert controller._dispatch_aux_action(request, now + 0.02) is False
    assert operations == []


def test_legacy_add_hp_command_is_queued_instead_of_bypassing_cooldown(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "press_key", lambda key: operations.append(key))
    controller = _controller("none none add_hp", iterations=1)

    controller.run()

    assert "1" not in operations
    assert "add_hp" in controller._aux_actions


def test_return_home_dispatch_terminates_keyboard_after_success(monkeypatch):
    operations = []
    monkeypatch.setattr(module, "press_key", lambda key: operations.append(key))
    monkeypatch.setattr(module, "key_up", lambda _key: None)
    monkeypatch.setattr(module.time, "monotonic", lambda: 12.5)
    controller = _controller("none none none", iterations=1)
    request = module.AuxActionRequest("return_home", 4, 12.4, "emergency")

    assert controller._dispatch_aux_action(request, 12.4) is True

    assert operations == ["home"]
    assert controller._aux_last_dispatched["return_home"] == 12.5
    assert controller.termination_reason == "return_home"
    assert not controller.is_enable
    assert controller.is_terminated


def test_required_target_hwnd_never_falls_back_to_matching_title(monkeypatch):
    if module.is_mac():
        return
    controller = object.__new__(KeyBoardController)
    controller.target_hwnd = 1234
    controller._require_target_hwnd = True
    controller._window_title_is_exact = True
    controller.window_title = "冒险岛怀旧服"
    monkeypatch.setattr(module.win32gui, "IsWindow", lambda _hwnd: False)
    monkeypatch.setattr(
        module.gw, "getActiveWindow",
        lambda: type("Window", (), {"title": "冒险岛怀旧服"})(),
    )

    assert not KeyBoardController.is_game_window_active(controller)
