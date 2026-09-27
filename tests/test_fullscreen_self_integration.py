import copy
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.engine.FullscreenSelfDetector import FullscreenScene, SceneTiming, SelfLocation
from src.engine.FullscreenPerception import fixed_attack_geometry
from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.engine.RouteCombatArbiter import RouteIntent
from src.engine.ReadinessDiagnostics import runtime_report
from src.utils.common import load_yaml


def box(cid, x, y=160, w=40, h=40):
    return dict(class_id=cid, name=('monster', 'player', 'self')[cid],
                position=(x, y), size=(w, h), confidence=.9, score=.1)


class Detector:
    def __init__(self):
        self.calls = 0
        self.selves = (box(2, 780),)
        self.monsters = (box(0, 850),)
        self.players = (box(1, 200),)
        self.after = lambda: None
        self.error = None
    def detect_scene(self, frame, *, frame_token):
        self.calls += 1
        if self.error:
            raise self.error
        self.after()
        self.last_timing = SceneTiming(monsters=len(self.monsters), players=len(self.players), selves=len(self.selves))
        return FullscreenScene(self.monsters, self.players, self.selves,
                               (0, 0, 1600, 400), frame.shape, self.last_timing, frame_token)
    def finalize_scene(self, scene, anchor):
        return list(scene.monsters)


@pytest.fixture
def bot(monkeypatch):
    b = MapleStoryAutoBot(SimpleNamespace(disable_viz=True, disable_control=False, is_ui=False))
    b.cfg = copy.deepcopy(load_yaml('config/config_default.yaml'))
    b.cfg['bot'].update(mode='continuous_attack', attack='directional', route_only=False)
    b.cfg['monster_detect'].update(yolo_variant='int8_head_fp', yolo_max_det=300)
    b.cfg['directional_attack'].update(range_x=100, range_y=100, cooldown=0)
    b.cfg['aoe_skill'].update(range_x=200, range_y=100, cooldown=0)
    b.kb = SimpleNamespace(commands=[], releases=0, active=True, is_need_force_heal=False, is_enable=True)
    b.kb.set_command = b.kb.commands.append
    def release():
        b.kb.releases += 1
    b.kb.release_all_key = release
    b.kb.is_game_window_active = lambda: b.kb.active
    b.profiler = SimpleNamespace(start=lambda: None, mark=lambda _: None)
    b.monster_detector = Detector()
    b.get_img_frame = lambda: np.zeros((400, 1600, 3), np.uint8)
    b.get_player_location_by_nametag = lambda: pytest.fail('Three-class flow must not read name templates')
    b.set_state_for_configured_mode()
    b.t_last_attack = time.time()-100
    return b


def tick(bot, sequence, age=0):
    bot.capture_frame_sequence = sequence
    bot.frame_captured_at = time.monotonic()-age
    return bot.run_once()


def test_main_loop_infers_once_per_capture_and_recovers_without_names(bot):
    tick(bot, 1)
    assert not bot.current_player_valid and bot.kb.commands[-1] == 'stop stop none'
    tick(bot, 1)
    assert bot.monster_detector.calls == 1
    tick(bot, 2)
    assert bot.current_player_valid and bot.loc_player == (800, 200)
    assert not bot.current_nametag_valid
    assert bot.kb.commands[-1] == 'none none attack_right'
    bot.update_monster_observations()
    bot.update_monster_observations()
    assert bot.monster_detector.calls == 2
    bot.monster_detector.selves = ()
    tick(bot, 3)
    assert bot.kb.commands[-1] == 'stop stop none'
    assert not bot.monsters and bot.combat_target is None
    assert bot.kb.releases >= 2
    bot.monster_detector.selves = (box(2, 780),)
    tick(bot, 4)
    assert not bot.current_player_valid
    tick(bot, 5)
    assert bot.current_player_valid


def test_missing_capture_identity_stops_without_reinferring_old_frame(bot):
    tick(bot, 1); tick(bot, 2)
    count = bot.monster_detector.calls
    tick(bot, None)
    assert bot.monster_detector.calls == count
    assert not bot.current_player_valid
    assert bot.kb.commands[-1] == 'stop stop none'
    tick(bot, 3)
    assert not bot.current_player_valid
    tick(bot, 4)
    assert bot.current_player_valid


def test_equal_distance_equal_confidence_monsters_have_stable_selection(bot):
    tick(bot, 1); tick(bot, 2)
    first, second = box(0, 730, y=160), box(0, 730, y=200)
    bot.monsters = [first, second]
    assert bot.get_nearest_monster(is_left=True) is first


@pytest.mark.parametrize('fault', ['multiple', 'stale_before', 'stale_after', 'focus_before', 'focus_after', 'exception', 'capture'])
def test_each_failure_releases_keys_and_requires_fresh_recovery(bot, fault):
    tick(bot, 1); tick(bot, 2)
    assert bot.current_player_valid
    if fault == 'multiple':
        bot.monster_detector.selves *= 2
    elif fault == 'focus_before':
        bot.kb.active = False
    elif fault == 'focus_after':
        bot.monster_detector.after = lambda: setattr(bot.kb, 'active', False)
    elif fault == 'stale_after':
        bot.monster_detector.after = lambda: setattr(bot, 'frame_captured_at', time.monotonic()-1)
    elif fault == 'exception':
        bot.monster_detector.error = RuntimeError('inference failed')
    elif fault == 'capture':
        bot.get_img_frame = lambda: None
    tick(bot, 3, age=1 if fault == 'stale_before' else 0)
    assert not bot.current_player_valid
    assert bot.kb.commands[-1] == 'stop stop none'
    assert not bot.monsters and bot.combat_target is None
    bot.monster_detector.selves = (box(2, 780),)
    bot.monster_detector.after = lambda: None
    bot.monster_detector.error = None
    bot.kb.active = True
    bot.get_img_frame = lambda: np.zeros((400, 1600, 3), np.uint8)
    tick(bot, 4)
    assert not bot.current_player_valid
    tick(bot, 5)
    assert bot.current_player_valid


def test_continuous_attack_tracks_current_in_range_direction_and_pauses_timers(bot):
    state = bot.continuous_attack_state
    clock = [10.]
    state._clock = lambda: clock[0]
    state._random_uniform = lambda *args: 20.
    tick(bot, 1); tick(bot, 2)
    assert state.cycle_deadline == 30.
    clock[0] = 12.
    bot.monster_detector.monsters = (box(0, 1450),)
    tick(bot, 3)
    assert bot.kb.commands[-1] == 'stop stop none'
    assert state.get_diagnostics()['remaining_seconds'] == 18.
    clock[0] = 112.
    bot.monster_detector.monsters = (box(0, 720),)
    tick(bot, 4); tick(bot, 5)
    assert state.cycle_deadline == 130.
    assert state.locked_direction == 'left'
    assert bot.kb.commands[-1] == 'none none attack_left'
    assert not any('jump' in command for command in bot.kb.commands)


def test_periodic_sequence_freezes_during_self_loss(bot):
    state = bot.continuous_attack_state
    clock = [10.]
    state._clock = lambda: clock[0]
    state._random_uniform = lambda *args: 20.
    tick(bot, 1); tick(bot, 2)
    clock[0] = 30.
    tick(bot, 3)
    assert state.phase == 'gap_after_jump'
    assert bot.kb.commands[-1] == 'none none jump'
    clock[0] = 30.5
    bot.monster_detector.selves = ()
    tick(bot, 4)
    clock[0] = 100.
    bot.monster_detector.selves = (box(2, 780),)
    tick(bot, 5); tick(bot, 6)
    assert state.phase == 'gap_after_jump'
    assert state.phase_deadline == 100.5
    assert bot.kb.commands[-1] == 'none none none'


def test_readonly_mode_never_dispatches_attack_or_movement(bot):
    bot.is_disable_control = True
    for i in range(1, 7):
        tick(bot, i)
    assert bot.monster_detector.calls == 6
    assert all(c == 'stop stop none' for c in bot.kb.commands)


def test_route_pursuit_stays_local_and_self_loss_cancels_grace(bot):
    tick(bot, 1); tick(bot, 2)
    bot.monsters = [box(0, 1480)]  # Outside the old 1280-wide local strip.
    assert bot.build_route_combat_intent(RouteIntent('left', 'none', 'none'), now=100).state == 'observe'
    bot.monsters = [box(0, 1100)]
    assert bot.build_route_combat_intent(RouteIntent('left', 'none', 'none'), now=101).state == 'visible_pursuit'
    bot.monsters = [box(0, 850)]
    assert bot.build_route_combat_intent(RouteIntent('left', 'none', 'none'), now=102).state == 'visible_attack'
    bot.monsters = []
    assert bot.build_route_combat_intent(RouteIntent('left', 'none', 'none'), now=102.1).state == 'lost_grace_attack'
    bot.pause_for_visual('self_missing')
    assert bot.build_route_combat_intent(RouteIntent('left', 'none', 'none'), now=102.2).state == 'observe'
    assert bot.combat_target is None


def test_aoe_out_of_range_monster_does_not_trigger_attack(bot):
    tick(bot, 1); tick(bot, 2)
    bot.cfg['bot']['attack'] = 'aoe_skill'
    bot.cmd_action = 'none'
    bot.fullscreen_scene = replace(bot.fullscreen_scene, monsters=(box(0, 1450),))
    bot.update_cmd_by_mob_detection()
    assert bot.cmd_action == 'none'


def test_previous_attack_is_not_replayed_when_monsters_disappear(bot):
    tick(bot, 1); tick(bot, 2)
    bot.fullscreen_scene = replace(bot.fullscreen_scene, monsters=())
    bot.cmd_action = 'attack_right'
    bot.update_cmd_by_mob_detection()
    assert bot.cmd_action == 'none'


def test_fullscreen_updates_existing_ladder_motion_evidence(bot):
    tick(bot, 1); tick(bot, 2)
    bot.monster_detector.selves = (box(2, 780, y=140),)
    tick(bot, 3)
    assert bot.is_on_ladder
    bot.monster_detector.selves = (box(2, 790, y=140),)
    tick(bot, 4)
    assert not bot.is_on_ladder


def test_runtime_diagnostics_show_self_and_all_three_counts(bot):
    tick(bot, 1); tick(bot, 2)
    bot.cfg.pop('nametag')
    report = runtime_report(bot, True)
    rows = {r.key: r for r in report.rows}
    assert rows['profile'].status == 'unused'
    assert rows['name'].status == rows['yolo'].status == 'pass'
    assert 'self 1' in rows['yolo'].detail and 'player 1' in rows['yolo'].detail
    assert report.name_preview is None


def test_load_configuration_requires_no_name_assets(bot, monkeypatch):
    import importlib
    engine = importlib.import_module('src.engine.MapleStoryAutoLevelUp')
    monkeypatch.setattr(engine, 'NameTagProfileRepository', lambda: pytest.fail('Must not open name assets'))
    monkeypatch.setattr(engine, 'detector_from_config', lambda *a, **k: SimpleNamespace(variant='int8_head_fp', root='example'))
    bot.cfg.pop('nametag')
    bot.cfg['key'].update(directional_attack='q', jump='space')
    assert bot.load_config(bot.cfg) == 0
    assert bot.nametag_localizer is None


def test_fixed_platform_keeps_bounds_after_self_loss(bot):
    tick(bot, 1); tick(bot, 2)
    bot.cfg['bot']['mode'] = 'fixed_platform'
    bot.cfg['fixed_platform']['width_px'] = 80
    bot.loc_player_minimap = (80, 40)
    bot.img_minimap = np.zeros((80, 180, 3), np.uint8)
    bot.current_minimap_player_valid = True
    state = bot.fixed_platform_state
    state.on_enter()
    state.on_frame()
    bounds = (state.anchor_x, state.left_x, state.right_x)
    assert bounds == (80, 40, 120)
    bot.pause_for_visual('self_missing')
    assert not bot.current_player_valid
    bot.current_player_valid = True
    bot.loc_player_minimap = (83, 40)
    state.on_frame()
    assert (state.anchor_x, state.left_x, state.right_x) == bounds


def test_fixed_attack_geometry_uses_self_box_center_and_configured_size(bot):
    bot.cfg['bot'].update(mode='fixed_platform', attack='directional')
    scene = FullscreenScene((), (), (box(2, 780, y=160, w=41, h=61),),
                            (0, 0, 1600, 400), (400, 1600, 3), SceneTiming(), ('capture', 2))
    geometry = fixed_attack_geometry(scene, bot.cfg)
    assert geometry.center == (800, 190)
    assert geometry.foot == (800, 221)
    assert geometry.left == (700, 140, 800, 240)
    assert geometry.right == (800, 140, 900, 240)
    assert geometry.combined == (700, 140, 900, 240)
    assert geometry.contains(box(0, 850, y=190))
    assert not geometry.contains(box(0, 900, y=190))


def test_fixed_attack_geometry_clips_to_visible_game_region(bot):
    bot.cfg['bot'].update(mode='fixed_platform', attack='directional')
    scene = FullscreenScene((), (), (box(2, 10, y=20, w=40, h=40),),
                            (4, 10, 1500, 330), (400, 1600, 3), SceneTiming(), ('capture', 2))
    geometry = fixed_attack_geometry(scene, bot.cfg)
    assert geometry.combined == (4, 10, 130, 90)
    assert fixed_attack_geometry(replace(scene, selves=()), bot.cfg) is None
    assert fixed_attack_geometry(replace(scene, selves=scene.selves*2), bot.cfg) is None


def test_fixed_preview_draws_attack_range_from_unique_self_without_control(bot):
    bot.cfg['bot'].update(mode='fixed_platform', attack='directional')
    bot.capture_frame_sequence = 2
    token = bot._detection_frame_token()
    scene = FullscreenScene((), (), (box(2, 780),), (0, 0, 1600, 400),
                            (400, 1600, 3), SceneTiming(), token)
    bot.fullscreen_scene = scene
    bot.fixed_attack_geometry = fixed_attack_geometry(scene, bot.cfg)
    assert bot.fixed_attack_geometry.pursuit_rect == (160, 88, 1440, 312)
    bot.current_player_valid = False
    bot.current_minimap_player_valid = False
    bot.is_disable_control = True
    bot.last_visual_pause_reason = 'window_inactive'
    canvas = np.zeros((400, 1600, 3), np.uint8)
    bot._draw_fullscreen_scene(canvas)
    assert tuple(canvas[0, 400]) == (255, 0, 0)  # Full-screen detection border.
    assert tuple(canvas[250, 160]) == (255, 255, 0)  # Local pursuit border.
    assert tuple(canvas[150, 800]) == (0, 0, 255)  # Attack border from self center.
    bot.fixed_attack_geometry = None
    canvas.fill(0)
    bot._draw_fullscreen_scene(canvas)
    assert not np.any(canvas[150, 680:920, 2])


def test_fixed_focus_loss_keeps_fresh_self_preview_but_releases_input(bot):
    bot.cfg['bot'].update(mode='fixed_platform', attack='directional')
    bot.kb.active = False
    tick(bot, 1)
    tick(bot, 2)
    assert bot.monster_detector.calls == 2
    assert bot.fixed_attack_geometry is not None
    assert bot.fullscreen_scene is not None
    assert not bot.current_player_valid
    assert bot.kb.commands[-1] == 'stop stop none'
    assert bot.kb.releases >= 1


def test_fixed_minimap_loss_keeps_current_attack_preview_and_stops(bot):
    bot.cfg['bot'].update(mode='fixed_platform', attack='directional')
    bot.capture_frame_sequence = 2
    scene = FullscreenScene((), (), (box(2, 780),), (0, 0, 1600, 400),
                            (400, 1600, 3), SceneTiming(), bot._detection_frame_token())
    bot.fullscreen_scene = scene
    bot.fixed_attack_geometry = fixed_attack_geometry(scene, bot.cfg)
    bot.current_player_valid = True
    bot.pause_for_visual('fixed_platform_localization_invalid')
    assert bot.fixed_attack_geometry is not None
    assert bot.fullscreen_scene is scene
    assert bot.kb.commands[-1] == 'stop stop none'
    bot.pause_for_visual('frame_stale')
    assert bot.fixed_attack_geometry is None
    assert bot.fullscreen_scene is None


def test_fixed_duplicate_frame_releases_input_without_new_inference(bot):
    bot.cfg['bot'].update(mode='fixed_platform', attack='directional')
    for sequence in (1, 2):
        bot.capture_frame_sequence = sequence
        bot.frame_captured_at = time.monotonic()
        bot.img_frame = np.zeros((400, 1600, 3), np.uint8)
        bot._update_fullscreen_perception()
    assert bot.current_player_valid
    assert bot.fixed_attack_geometry is not None
    calls = bot.monster_detector.calls
    assert not bot._update_fullscreen_perception()
    assert bot.monster_detector.calls == calls
    assert bot.fixed_attack_geometry is None
    assert bot.kb.commands[-1] == 'stop stop none'


def test_fixed_fullscreen_keeps_nearby_monster_classified_by_three_class_model(bot):
    bot.cfg['bot'].update(mode='fixed_platform', attack='directional')
    bot.monster_detector.monsters = (box(0, 810),)
    bot.monster_detector.finalize_scene = lambda *_: pytest.fail('Fixed three-class mode must not apply the legacy exclusion area')
    for sequence in (1, 2):
        bot.capture_frame_sequence = sequence
        bot.frame_captured_at = time.monotonic()
        bot.img_frame = np.zeros((400, 1600, 3), np.uint8)
        bot._update_fullscreen_perception()
    assert bot.current_player_valid
    assert len(bot.monsters) == 1
    assert bot.fixed_attack_geometry.contains(bot.monsters[0])


def test_fixed_attack_uses_the_same_centered_rectangle_as_preview(bot):
    state = fixed_combat(bot)
    bot.capture_frame_sequence = 5
    token = bot._detection_frame_token()
    scene = FullscreenScene((), (), (box(2, 780, y=160, w=40, h=60),),
                            (0, 0, 1600, 400), (400, 1600, 3), SceneTiming(), token)
    bot.fixed_attack_geometry = fixed_attack_geometry(scene, bot.cfg)
    bot.loc_player = bot.fixed_attack_geometry.foot
    assert bot.get_combined_attack_range() == bot.fixed_attack_geometry.combined
    bot.monsters = [box(0, 850, y=110)]
    state.on_frame()
    assert bot.kb.commands[-1] == 'stop none attack_right'


def fixed_combat(bot):
    bot.cfg['bot'].update(mode='fixed_platform', attack='directional')
    bot.cfg['fixed_platform']['width_px'] = 80
    bot.capture_frame_sequence = 2
    bot.fullscreen_scene = FullscreenScene((), (), (box(2, 780),),
        (0, 0, 1600, 400), (400, 1600, 3), SceneTiming(), bot._detection_frame_token())
    bot.fixed_attack_geometry = fixed_attack_geometry(bot.fullscreen_scene, bot.cfg)
    bot.loc_player = (800, 200)
    bot.loc_player_minimap = (80, 40)
    bot.img_frame = np.zeros((400, 1600, 3), np.uint8)
    bot.img_minimap = np.zeros((80, 180, 3), np.uint8)
    bot.current_player_valid = True
    bot.current_minimap_player_valid = True
    bot.monsters = []
    bot.update_monster_observations = lambda: True
    bot.fullscreen_control_ready = lambda: True
    bot.t_last_attack = time.time()-100
    state = bot.fixed_platform_state
    state.on_enter()
    return state


def test_fixed_fullscreen_pursues_then_stops_to_attack_and_resumes_patrol(bot):
    state = fixed_combat(bot)
    bot.monsters = [box(0, 1050)]
    state.on_frame()
    assert bot.kb.commands[-1] == 'right none none'
    assert state.combat_state == 'pursuit' and state.direction == 'left'
    assert state.get_diagnostics()['combat_state'] == 'pursuit'

    bot.monsters = [box(0, 850)]
    state.on_frame()
    assert bot.kb.commands[-1] == 'stop none attack_right'
    assert state.combat_state == 'attack'
    state.on_frame()
    assert bot.kb.commands[-1] == 'stop none none'  # Rising-edge neutral gap.
    bot.monsters = []
    state.on_frame()
    assert bot.kb.commands[-1] == 'left none none'
    assert state.combat_state == 'patrol'


def test_fixed_fullscreen_selects_nearest_in_range_before_pursuit(bot):
    state = fixed_combat(bot)
    bot.monsters = [box(0, 700), box(0, 850), box(0, 1050)]
    state.on_frame()
    assert bot.kb.commands[-1] == 'stop none attack_right'
    assert state.combat_state == 'attack'


def test_fixed_fullscreen_tie_breaks_in_range_by_confidence(bot):
    state = fixed_combat(bot)
    left, right = box(0, 710), box(0, 850)
    left['confidence'] = .8
    right['confidence'] = .9
    bot.monsters = [left, right]
    state.on_frame()
    assert bot.kb.commands[-1] == 'stop none attack_right'


def test_fixed_fullscreen_pursuit_prefers_horizontal_distance(bot):
    state = fixed_combat(bot)
    # The nearer horizontal gap has a larger vertical center offset.
    bot.monsters = [box(0, 550, y=190), box(0, 990, y=120)]
    state.on_frame()
    assert bot.kb.commands[-1] == 'right none none'
    assert state.combat_state == 'pursuit'


def test_fixed_fullscreen_attacking_stays_stopped_during_cooldown(bot):
    state = fixed_combat(bot)
    bot.cfg['directional_attack']['cooldown'] = 10
    bot.monsters = [box(0, 850)]
    state.on_frame()
    assert bot.kb.commands[-1] == 'stop none attack_right'
    state.on_frame()
    assert bot.kb.commands[-1] == 'stop none none'
    state.on_frame()
    assert bot.kb.commands[-1] == 'stop none none'
    assert state.combat_state == 'attack'


def test_fixed_fullscreen_switches_attack_direction_after_neutral_gap(bot):
    state = fixed_combat(bot)
    bot.monsters = [box(0, 850)]
    state.on_frame()
    assert bot.kb.commands[-1] == 'stop none attack_right'
    bot.monsters = [box(0, 710)]
    state.on_frame()
    assert bot.kb.commands[-1] == 'stop none none'
    bot.t_last_attack = time.time()-100
    state.on_frame()
    assert bot.kb.commands[-1] == 'stop none attack_left'


def test_fixed_fullscreen_pursuit_requires_positive_attack_height_overlap(bot):
    state = fixed_combat(bot)
    # Foot offset is within 70 px, but the box only touches the attack band.
    bot.monsters = [box(0, 1050, y=90)]
    state.on_frame()
    assert state.combat_state == 'patrol'
    assert bot.kb.commands[-1] == 'left none none'


def test_fixed_fullscreen_ignores_other_levels_and_blocks_outward_boundary(bot):
    state = fixed_combat(bot)
    bot.monsters = [box(0, 1050, y=300)]  # Different level.
    state.on_frame()
    assert bot.kb.commands[-1] == 'left none none'
    assert state.combat_state == 'patrol'

    bot.loc_player_minimap = (120, 40)
    bot.monsters = [box(0, 1050)]
    state.on_frame()
    assert bot.kb.commands[-1] == 'left none none'
    assert state.combat_state == 'unreachable'


def test_fixed_fullscreen_blocks_left_boundary_and_ignores_remote_monsters(bot):
    state = fixed_combat(bot)
    bot.monsters = [box(0, 1450)]  # Outside the local 1280-pixel strip.
    state.on_frame()
    assert state.combat_state == 'patrol'
    bot.loc_player_minimap = (40, 40)
    bot.monsters = [box(0, 550)]
    state.on_frame()
    assert bot.kb.commands[-1] in {'stop none none', 'right none none'}
    assert state.combat_state == 'unreachable'


def test_fixed_fullscreen_stalled_pursuit_reverses_and_cools_down(bot, monkeypatch):
    import src.states.fixed_platform as platform_module
    clock = [0.0]
    monkeypatch.setattr(platform_module.time, 'monotonic', lambda: clock[0])
    state = fixed_combat(bot)
    bot.monsters = [box(0, 1050)]
    state.on_frame()
    assert bot.kb.commands[-1] == 'right none none'
    for tick in range(1, 22):
        clock[0] = tick*.1
        state.on_frame()
    assert state.combat_state == 'unreachable'
    assert state.direction == 'left'
    assert bot.kb.commands[-1] == 'left none none'
    clock[0] += .1
    state.on_frame()
    assert bot.kb.commands[-1] == 'left none none'


@pytest.mark.parametrize('invalid', ['self', 'minimap', 'focus', 'stale', 'disabled'])
def test_fixed_fullscreen_invalid_control_never_pursues_or_attacks(bot, invalid):
    state = fixed_combat(bot)
    bot.monsters = [box(0, 850)]
    if invalid == 'self':
        bot.current_player_valid = False
    elif invalid == 'minimap':
        bot.current_minimap_player_valid = False
    elif invalid == 'focus':
        bot.kb.active = False
        bot.fullscreen_control_ready = MapleStoryAutoBot.fullscreen_control_ready.__get__(bot)
        bot.frame_captured_at = time.monotonic()
    elif invalid == 'stale':
        bot.fullscreen_control_ready = MapleStoryAutoBot.fullscreen_control_ready.__get__(bot)
        bot.frame_captured_at = time.monotonic()-1
    else:
        bot.is_disable_control = True
    state.on_frame()
    assert not any('attack' in command or command.startswith('right')
                   for command in bot.kb.commands)
    if invalid in {'self', 'minimap', 'focus', 'stale'}:
        assert bot.kb.commands[-1] == 'stop stop none'
        assert bot.kb.releases >= 1


def test_readiness_checker_uses_three_class_model_without_name_or_keyboard(bot, monkeypatch):
    from src.engine import ReadinessDiagnostics as diag
    from src.input import KeyBoardController as keyboard
    monkeypatch.setattr(diag, 'NameTagProfileRepository', lambda: pytest.fail('Must not load name assets'))
    monkeypatch.setattr(keyboard, 'KeyBoardController', lambda *a, **k: pytest.fail('Must not create input controller'))
    monkeypatch.setattr(diag, 'detector_from_config', lambda *a, **k: bot.monster_detector)
    bot.cfg.pop('nametag')
    bot.cfg['game_window'].update(size=(400, 1600), client_crop=None, title_bar_height=0)
    bot.cfg['health_monitor'].update(auto_hp_enabled=False, auto_mp_enabled=False)
    class Capture:
        sequence = 0
        stopped = False
        def get_frame_packet(self):
            self.sequence += 1
            return np.zeros((400, 1600, 3), np.uint8), time.monotonic(), self.sequence
        def stop(self):
            self.stopped = True
    capture = Capture()
    cancel = SimpleNamespace(is_set=lambda: False, wait=lambda _: None)
    reports = []
    diag.ReadinessChecker(bot.cfg, reports.append, cancel, capture_factory=lambda _: capture).run()
    rows = {r.key: r for r in reports[-1].rows}
    assert rows['profile'].status == 'unused'
    assert rows['name'].status == rows['yolo'].status == 'pass'
    assert capture.stopped and bot.monster_detector.calls == 20
    assert not any(report.name_preview for report in reports)
