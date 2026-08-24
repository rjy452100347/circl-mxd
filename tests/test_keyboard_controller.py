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
        "key": {"jump": "space", "teleport": "", "add_hp": "1", "add_mp": "2"},
        "route": {"mount_press_duration": 0.10},
        "directional_attack": {"character_turn_delay": 0.02},
    }
    controller.cmd_left_right, controller.cmd_up_down, controller.cmd_action = command.split()
    controller.cmd_left_right_last = ""
    controller.cmd_up_down_last = ""
    controller.cmd_action_last = "none"
    controller._command_lock = threading.Lock()
    controller.command_timeout_seconds = 10.0
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
