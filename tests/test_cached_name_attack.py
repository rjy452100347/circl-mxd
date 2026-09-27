import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.engine.NameTagTracking import NameTagTracking
from src.states.fixed_platform import FixedPlatformState

_CLOCK = [1000.0]


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    _CLOCK[0] = 1000.0
    monkeypatch.setattr(time, 'time', lambda: _CLOCK[0])

def bot_fixture(mode='fixed_platform', attack='directional'):
    bot = object.__new__(MapleStoryAutoBot)
    bot.cfg = dict(bot=dict(mode=mode, attack=attack),
                   directional_attack=dict(range_x=120, range_y=60, cooldown=0),
                   aoe_skill=dict(range_x=120, range_y=60, cooldown=0),
                   fixed_platform=dict(width_px=80), combat_tracking={})
    bot.name_tracking = NameTagTracking()
    bot.is_disable_control = False
    bot.is_terminated = False
    bot.img_frame = np.zeros((220, 500, 3), np.uint8)
    bot.img_frame_debug = None
    bot.img_minimap = np.zeros((80, 180, 3), np.uint8)
    bot.loc_player = (150, 100)
    bot.loc_player_minimap = bot.loc_player_global = (50, 40)
    bot.current_minimap_player_valid = bot.current_nametag_valid = True
    bot.monsters = []
    bot.t_last_attack = time.time()-100
    bot.monster_detection_times = deque(maxlen=300)
    bot.monster_frame_pipeline_times = deque(maxlen=300)
    bot.route_navigator = None
    bot.reset_route_combat = lambda: None
    bot.control_frame_sequence = 1
    bot.capture_frame_sequence = 1
    bot.kb = SimpleNamespace(commands=[], releases=0, is_need_force_heal=False, is_enable=True)
    bot.kb.set_command = bot.kb.commands.append
    def release():
        bot.kb.releases += 1
    bot.kb.release_all_key = release
    bot.fixed_platform_state = FixedPlatformState('fixed_platform', bot)
    bot.fixed_platform_state._capture_bounds()
    bot._targets = [dict(position=(90, 90), size=(20, 20), score=.1, name='monster')]
    bot._calls = []
    class Detector:
        last_timing = SimpleNamespace(total_ms=1, infer_ms=.5, crop_ms=.1, convert_ms=.1, monsters=1, players=0)
        last_players = []
        last_player_foot = None
        def detect(self, frame, anchor, **kwargs):
            bot._calls.append((bot.capture_frame_sequence, anchor, kwargs))
            return list(bot._targets)
    bot.monster_detector = Detector()
    assert not bot._process_name_tracking(bot.loc_player)
    return bot


def tick(bot, *, live=False, next_frame=True):
    _CLOCK[0] += .1
    if next_frame:
        bot.capture_frame_sequence += 1
        bot.control_frame_sequence += 1
    bot.current_nametag_valid = live
    return bot._process_name_tracking(bot.loc_player if live else None)


@pytest.mark.parametrize('flicker', [False, True])
def test_fixed_name_templates_control_without_body_binding(monkeypatch, flicker):
    bot = bot_fixture()
    bot.monster_detector.detect_scene = lambda *a: pytest.fail('No body localization inference')
    start = time.monotonic()
    for i in range(8):
        monkeypatch.setattr(time, 'monotonic', lambda: start+i*.1)
        bot.frame_captured_at = start+i*.1
        bot.capture_frame_sequence = i
        visible = i < 5 or (flicker and i == 6)
        name_point = (156, 100) if i >= 5 else (150, 100)
        bot.current_nametag_valid = visible
        bot.nametag_last_result = SimpleNamespace(valid=visible, player=name_point, match_kind='partial' if i >= 5 else 'full')
        foot = bot._process_fixed_player_localization(name_point if visible else None)
        assert foot == (name_point if visible else None)
        paused = bot._process_name_tracking(foot)
        assert paused == (not visible)
        if visible:
            bot.loc_player = foot
            bot.fixed_platform_state.on_frame()
            assert bot.kb.commands[-1].startswith('left ')
        else:
            assert bot.kb.commands[-1].startswith('stop stop ')
        bot.update_monster_observations()
    assert len(bot._calls) == 8
    assert bot.player_localization is None
    assert not hasattr(bot, 'player_localization_tracker')
    assert bot.fixed_platform_state.anchor_x == 50
    assert bot.fixed_platform_state.direction == 'left'
    bot.kb.is_need_force_heal = True
    bot.capture_frame_sequence += 1
    assert bot._process_name_tracking(None)
    assert bot.kb.commands[-1] == 'stop stop none'


def test_body_occlusion_freezes_existing_patrol_progress_clock(monkeypatch):
    bot = bot_fixture()
    state = bot.fixed_platform_state
    state.localized = True
    state._reset_progress(50, 10.)
    monkeypatch.setattr(time, 'monotonic', lambda: 10.5)
    state.suspend_tracking(preserve_progress=True)
    monkeypatch.setattr(time, 'monotonic', lambda: 12.5)
    state._update_patrol_direction()
    assert state._progress_at == 12.
    assert state._progress_x == 50 and state.localized
    assert state.anchor_x == 50 and state.direction == 'left'


@pytest.mark.parametrize('mode', ['fixed_platform', 'normal', 'patrol'])
def test_cache_stops_movement_but_attacks_fresh_targets_with_neutral_gap(mode):
    bot = bot_fixture(mode)
    assert tick(bot)
    assert bot.kb.commands[-1] == 'stop stop none' and bot.kb.releases == 1
    assert tick(bot)
    assert bot.kb.commands[-1] == 'stop stop attack_left'
    assert not bot.current_nametag_valid
    assert bot._calls[-1][2]['player_exclusion_anchor'] == (150, 100)
    tick(bot)
    assert bot.kb.commands[-1] == 'stop stop none'
    tick(bot)
    assert bot.kb.commands[-1] == 'stop stop attack_left'
    assert all(c.startswith('stop stop ') for c in bot.kb.commands)


def test_yolo_and_attack_do_not_replay_a_capture_packet():
    bot = bot_fixture()
    tick(bot)
    tick(bot)
    count = len(bot._calls)
    tick(bot, next_frame=False)
    bot.update_monster_observations()
    assert len(bot._calls) == count
    assert bot.kb.commands[-1] == 'stop stop none'
    bot._targets = []
    tick(bot)
    assert not bot.monsters and bot.kb.commands[-1] == 'stop stop none'


@pytest.mark.parametrize('attack', ['directional', 'aoe_skill'])
def test_out_of_range_and_cooldown_prevent_cache_attack(attack):
    bot = bot_fixture(attack=attack)
    tick(bot)
    bot._targets[0]['position'] = (450, 190)
    tick(bot)
    assert bot.kb.commands[-1] == 'stop stop none'
    bot._targets[0]['position'] = (130, 90)
    bot.cfg['directional_attack' if attack == 'directional' else 'aoe_skill']['cooldown'] = 100
    bot.t_last_attack = time.time()
    tick(bot)
    assert bot.kb.commands[-1] == 'stop stop none'


def test_displacement_latches_and_recovery_preserves_bounds_direction():
    bot = bot_fixture()
    state = bot.fixed_platform_state
    state.direction = 'right'
    bounds = state.left_x, state.right_x, state.anchor_x
    tick(bot)
    bot.loc_player_minimap = (51, 40)
    tick(bot)
    assert bot.name_tracking.snapshot.attack_allowed
    bot.loc_player_minimap = (53, 40)
    tick(bot)
    assert not bot.name_tracking.snapshot.attack_allowed
    bot.loc_player_minimap = (50, 40)
    tick(bot)
    assert not bot.name_tracking.snapshot.attack_allowed
    assert 'player_exclusion_anchor' not in bot._calls[-1][2]
    assert bot.kb.commands[-1] == 'stop stop none'
    assert not tick(bot, live=True)
    assert (state.left_x, state.right_x, state.anchor_x) == bounds
    assert state.direction == 'right' and state._progress_at is None


def test_healing_is_temporary_but_minimap_failure_latches():
    bot = bot_fixture()
    tick(bot)
    bot.kb.is_need_force_heal = True
    tick(bot)
    assert bot.kb.commands[-1] == 'stop stop none'
    bot.kb.is_need_force_heal = False
    tick(bot)
    assert bot.kb.commands[-1] == 'stop stop attack_left'
    bot.current_minimap_player_valid = False
    tick(bot)
    bot.current_minimap_player_valid = True
    tick(bot)
    assert bot.kb.commands[-1] == 'stop stop none'


def test_name_recovery_during_healing_does_not_resume_travel_until_healed():
    bot = bot_fixture()
    tick(bot)
    tick(bot)
    bot.kb.is_need_force_heal = True
    assert tick(bot, live=True)
    assert bot.name_tracking.snapshot.live
    assert bot.kb.commands[-1] == 'stop stop none'
    assert bot._name_cache_active
    bot.kb.is_need_force_heal = False
    assert not tick(bot, live=True)
    assert not bot._name_cache_active


@pytest.mark.parametrize('mode,debug', [('aux', False), ('normal', True)])
def test_observation_modes_never_attack(mode, debug):
    bot = bot_fixture(mode)
    bot.is_disable_control = debug
    for _ in range(4):
        tick(bot)
    assert bot._calls
    assert all(c == 'stop stop none' for c in bot.kb.commands)


@pytest.mark.parametrize('reason', ['capture_invalid', 'frame_stale', 'window_inactive', 'name_context_changed'])
def test_capture_focus_and_context_failure_clear_cache_and_release(reason):
    bot = bot_fixture()
    tick(bot)
    bot.pause_for_visual(reason)
    assert bot.name_tracking.snapshot.anchor is None
    assert not bot.monsters and bot.kb.releases >= 2
    previous = len(bot._calls)
    tick(bot)
    assert len(bot._calls) == previous and bot.kb.commands[-1] == 'stop stop none'


def test_continuous_attack_initial_live_name_still_runs_detector():
    bot = bot_fixture('continuous_attack')
    bot.name_tracking.reset()
    bot.current_nametag_valid = True
    bot.update_monster_observations()
    assert len(bot._calls) == 1


@pytest.mark.parametrize('failure', ['stale', 'focus'])
def test_inference_cannot_emit_attack_after_frame_or_focus_became_invalid(failure):
    bot = bot_fixture()
    tick(bot)
    if failure == 'stale':
        bot.frame_captured_at = time.monotonic()-1
    else:
        bot.kb.is_game_window_active = lambda: False
    tick(bot)
    assert bot.kb.commands[-1] == 'stop stop none'
    assert bot.name_tracking.snapshot.anchor is None


@pytest.mark.parametrize('mode', ['normal', 'patrol', 'fixed_platform'])
def test_main_pipeline_cache_gate_precedes_routes_channel_watchdog_and_fsm(monkeypatch, mode):
    import importlib
    engine = importlib.import_module('src.engine.MapleStoryAutoLevelUp')
    bot = bot_fixture(mode)
    bot.cfg['bot']['route_only'] = mode == 'normal'
    bot.cfg['bot']['route_attack'] = True
    bot.cfg['minimap'] = dict(roi=[0, 0, 1, 1], player_color=[136, 255, 255])
    bot.profiler = SimpleNamespace(start=lambda: None, mark=lambda _: None)
    bot._claim_visualization_mode = lambda: None
    bot._prepare_visualization_buffers = lambda: None
    bot.health_monitor = None
    bot.idx_routes = 0
    bot.img_routes = [np.zeros((10, 10, 3), np.uint8)]
    bot.is_on_ladder = False
    bot.get_img_frame = lambda: bot.img_frame
    bot.get_player_location_by_nametag = lambda: None
    from src.engine.MinimapObservation import MinimapObservationSnapshot, RoiEvidence
    bot.frame_captured_at = time.monotonic()
    bot.minimap_observer = SimpleNamespace(update=lambda *a, **k:
        MinimapObservationSnapshot(a[2], a[3], RoiEvidence((0, 0, 80, 80), None, True, 'ok', 'global'), (50, 40), 1, 'ok'))
    bot._minimap_roi_pixels_logged = True
    monkeypatch.setattr(engine, 'get_minimap_loc_size', lambda *_: (0, 0, 80, 80))
    monkeypatch.setattr(engine, 'get_player_location_on_minimap', lambda *a, **k: (50, 40))
    monkeypatch.setattr(engine, 'get_all_other_player_locations_on_minimap',
                        lambda *a, **k: pytest.fail('must return before channel detection'))
    bot.fsm = SimpleNamespace(do_state_stuff=lambda: pytest.fail('must not enter moving FSM'))
    bot.update_route_only_commands = lambda **k: pytest.fail('must not execute route')
    assert bot.run_once() == 0
    assert bot.kb.commands[-1] == 'stop stop none'
    bot.capture_frame_sequence += 1
    assert bot.run_once() == 0
    assert bot.kb.commands[-1] == 'stop stop attack_left'


def test_ladder_pause_retains_segment_without_reacquiring_or_consuming_timer():
    from src.engine.SemanticRouteNavigator import SemanticRouteNavigator
    bot = bot_fixture('normal')
    nav = object.__new__(SemanticRouteNavigator)
    calls = []
    nav.suspend_traversal = lambda: calls.append('suspend')
    nav.request_reacquire = lambda *a: pytest.fail('must not change route segment')
    bot.route_navigator = nav
    tick(bot)
    for _ in range(3):
        tick(bot)
    assert calls == ['suspend']


def test_detection_and_exclusion_draw_use_cache_without_marking_name_live():
    bot = bot_fixture()
    bot.is_ui = True
    bot.img_frame_debug = None
    bot.monster_detector.player_exclusion_rect = lambda shape, anchor: (110, 0, 190, 100)
    bot.draw_fixed_platform_debug = lambda *a, **k: None
    bot.draw_continuous_attack_debug = lambda *a, **k: None
    bot.draw_ladder_execution_debug = lambda *a, **k: None
    tick(bot)
    bot.img_frame_debug = bot.img_frame.copy()
    canvas = bot.get_frame_debug_for_viz()
    assert not bot.current_nametag_valid
    assert tuple(canvas[50, 110]) == (0, 165, 255)
    assert tuple(canvas[212, 250]) == (255, 0, 0)
    bot.current_minimap_player_valid = False
    tick(bot)
    bot.img_frame_debug = bot.img_frame.copy()
    canvas = bot.get_frame_debug_for_viz()
    assert tuple(canvas[50, 110]) != (0, 165, 255)
    assert tuple(canvas[212, 250]) == (255, 0, 0)
