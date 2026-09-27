import pytest

from src.states.continuous_attack import ContinuousAttackState


class FakeClock:
    def __init__(self, value=0.0):
        self.value = float(value)

    def __call__(self):
        return self.value


class FakeKeyboard:
    def __init__(self):
        self.commands = []
        self.is_need_force_heal = False

    def set_command(self, command):
        self.commands.append(command)


class FakeBot:
    def __init__(self):
        self.kb = FakeKeyboard()
        self.loc_player = (100, 100)
        self.monsters = []
        self.current_nametag_valid = True
        self.debug_attack_direction = "none"
        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        self.is_disable_control = False
        self.detect_calls = 0

    def update_monster_observations(self):
        self.detect_calls += 1


def monster(center_x, center_y=100, confidence=0.5, size=(10, 10)):
    width, height = size
    return {
        "position": (center_x - width // 2, center_y - height // 2),
        "size": (width, height),
        "confidence": confidence,
    }


def make_state(direction="left", duration=20.0):
    clock = FakeClock()
    bot = FakeBot()
    bot.monsters = [monster(80 if direction == "left" else 120)]
    state = ContinuousAttackState(
        "continuous_attack",
        bot,
        clock=clock,
        random_uniform=lambda _minimum, _maximum: duration,
    )
    state.on_enter()
    bot.monsters = [monster(80 if direction == "left" else 120)]
    return state, bot, clock


def test_waits_without_input_until_name_and_monster_are_available():
    state, bot, _clock = make_state()
    bot.current_nametag_valid = False

    state.on_frame()

    assert state.locked_direction is None
    assert bot.detect_calls == 0
    assert bot.kb.commands[-1] == "none none none"

    bot.current_nametag_valid = True
    bot.monsters = []
    state.on_frame()

    assert state.locked_direction is None
    assert bot.detect_calls == 1
    assert bot.kb.commands[-1] == "none none none"


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("left", "none none attack_left"), ("right", "none none attack_right")],
)
def test_first_valid_detection_locks_and_faces_once(direction, expected):
    state, bot, _clock = make_state(direction=direction, duration=23.0)

    state.on_frame()

    assert state.locked_direction == direction
    assert state.cycle_duration == 23.0
    assert bot.detect_calls == 1
    assert bot.kb.commands[-1] == expected

    state.on_frame()
    state.on_frame()
    assert bot.detect_calls == 1
    assert bot.kb.commands[-2:] == ["none none none", "none none attack"]


def test_nearest_target_then_confidence_then_left_breaks_ties():
    state, bot, _clock = make_state()
    bot.monsters = [monster(80, confidence=0.99), monster(110, confidence=0.1)]
    state.on_frame()
    assert state.locked_direction == "right"

    state.on_enter()
    bot.monsters = [monster(90, confidence=0.2), monster(110, confidence=0.9)]
    state.on_frame()
    assert state.locked_direction == "right"

    state.on_enter()
    bot.monsters = [monster(90, confidence=0.9), monster(110, confidence=0.9)]
    state.on_frame()
    assert state.locked_direction == "left"


def test_monster_centered_on_player_does_not_lock_direction():
    state, bot, _clock = make_state()
    bot.monsters = [monster(100)]

    state.on_frame()

    assert state.locked_direction is None
    assert bot.kb.commands[-1] == "none none none"


def test_attack_sequence_is_non_blocking_and_restores_locked_left_direction():
    state, bot, clock = make_state(direction="left", duration=20.0)
    state.on_frame()

    clock.value = 20.0
    state.on_frame()
    assert bot.kb.commands[-1] == "none none jump"

    clock.value = 20.9
    state.on_frame()
    assert bot.kb.commands[-1] == "none none none"

    clock.value = 21.0
    state.on_frame()
    assert bot.kb.commands[-1] == "left none none"

    clock.value = 21.1
    state.on_frame()
    assert bot.kb.commands[-1] == "none none none"

    clock.value = 22.0
    state.on_frame()
    assert bot.kb.commands[-1] == "right none none"

    clock.value = 22.1
    state.on_frame()
    assert bot.kb.commands[-1] == "none none attack_left"
    assert state.locked_direction == "left"
    assert state.cycle_deadline == pytest.approx(42.1)


def test_right_direction_is_explicitly_restored_after_movement():
    state, bot, clock = make_state(direction="right", duration=20.0)
    state.on_frame()
    clock.value = 20.0
    state.on_frame()
    clock.value = 21.0
    state.on_frame()
    clock.value = 22.0
    state.on_frame()
    clock.value = 22.1
    state.on_frame()

    assert bot.kb.commands[-1] == "none none attack_right"


def test_force_heal_stops_sequence_and_restarts_cycle_with_same_direction():
    state, bot, clock = make_state(direction="left", duration=24.0)
    state.on_frame()
    original_deadline = state.cycle_deadline

    clock.value = 5.0
    bot.kb.is_need_force_heal = True
    state.on_frame()
    assert state.phase == "healing"
    assert bot.kb.commands[-1] == "none none none"

    clock.value = 8.0
    bot.kb.is_need_force_heal = False
    state.on_frame()

    assert state.locked_direction == "left"
    assert state.cycle_deadline == pytest.approx(32.0)
    assert state.cycle_deadline != original_deadline
    assert bot.kb.commands[-1] == "none none attack_left"


def test_observe_mode_never_emits_control_commands():
    state, bot, _clock = make_state(direction="left")
    bot.is_disable_control = True

    state.on_frame()

    assert state.locked_direction == "left"
    assert bot.kb.commands[-1] == "none none none"
