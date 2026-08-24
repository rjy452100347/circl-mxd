'''
KeyBoardController
Simulate user keyboard input to control character in the game 
'''
# Standard Import
import threading
import time
import ctypes
from ctypes import wintypes

# Library import
import pyautogui
from pynput import keyboard

# Local import
from src.utils.logger import logger
from src.utils.common import is_mac
from src.runtime_policy import RUNTIME_POLICY

if is_mac():
    import Quartz
else:
    import pygetwindow as gw

pyautogui.PAUSE = 0  # remove delay


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput), ("ki", _KeyboardInput), ("hi", _HardwareInput)]


class _Input(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [("type", wintypes.DWORD), ("union", _InputUnion)]


_SCAN_CODES = {"left": 0x4B, "right": 0x4D, "up": 0x48, "down": 0x50, "space": 0x39}
_EXTENDED_KEYS = {"left", "right", "up", "down"}
def _scan_code_for_key(key):
    if key in _SCAN_CODES:
        return _SCAN_CODES[key]
    if not isinstance(key, str) or not key:
        return None

    special_vks = {
        "ctrl": 0x11, "shift": 0x10, "alt": 0x12,
        "enter": 0x0D, "tab": 0x09, "esc": 0x1B,
        "backspace": 0x08, "delete": 0x2E,
    }
    if len(key) == 1:
        translated = int(ctypes.windll.user32.VkKeyScanW(key))
        if translated == -1:
            return None
        vk = translated & 0xFF
    elif key in special_vks:
        vk = special_vks[key]
    elif key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 12:
        vk = 0x70 + int(key[1:]) - 1
    else:
        return None

    scan_code = int(ctypes.windll.user32.MapVirtualKeyW(vk, 0))
    return scan_code or None


def _send_input_key(key, is_key_up):
    scan_code = _scan_code_for_key(key)
    if scan_code is None:
        raise ValueError(f"No scan code available for key: {key}")
    flags = 0x0008
    if key in _EXTENDED_KEYS:
        flags |= 0x0001
    if is_key_up:
        flags |= 0x0002
    value = _Input(
        type=1,
        ki=_KeyboardInput(0, scan_code, flags, 0, 0),
    )
    user32 = ctypes.windll.user32
    sent = int(user32.SendInput(1, ctypes.byref(value), ctypes.sizeof(_Input)))
    if sent != 1:
        raise OSError("SendInput did not send exactly one keyboard event")

def key_down(key):
    '''
    Press key down
    '''
    if not key:
        return
    try:
        if is_mac():
            pyautogui.keyDown(key)
        else:
            _send_input_key(key, False)
    except pyautogui.FailSafeException:
        logger.warning("[key_down] pyautogui failsafe triggered during key_down.")
        recover_mouse()

def key_up(key):
    '''
    Release key
    '''
    if not key:
        return
    try:
        if is_mac():
            pyautogui.keyUp(key)
        else:
            _send_input_key(key, True)
    except pyautogui.FailSafeException:
        logger.warning("[key_up] pyautogui failsafe triggered during key_up.")
        recover_mouse()

def recover_mouse():
    '''
    Move mouse back to center to avoid pyautogui failsafe
    '''
    pyautogui.FAILSAFE = False # Temp disasble failsafe to avoid nested exception

    screen_w, screen_h = pyautogui.size()
    pyautogui.moveTo(screen_w // 2, screen_h // 2)
    time.sleep(0.2) # Give it a moment to "cool down"

    pyautogui.FAILSAFE = True # Recover failsafe

def press_key(key, duration=0.05):
    '''
    Simulates a key press for a specified duration
    '''
    if key:
        try:
            key_down(key)
            time.sleep(duration)
        finally:
            # A backend can fail after partially injecting key-down, so an
            # unconditional best-effort key-up is safer than assuming failure
            # means nothing reached the target window.
            key_up(key)

class KeyBoardController():
    '''
    KeyBoardController
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        if not is_mac():
            logger.info("[KeyBoardController] Windows input backend: SendInput (fixed)")
        self.cmd_action = "none"
        self.cmd_up_down = "none"
        self.cmd_left_right = "none"
        self.cmd_up_down_last = ""
        self.cmd_left_right_last = ""
        self.cmd_action_last = "none"
        self._was_game_window_active = False
        self._foreground_acquired_at = 0.0
        self._command_lock = threading.Lock()
        self.command_updated_at = time.monotonic()
        self.command_timeout_seconds = max(
            0.1,
            RUNTIME_POLICY.command_timeout_seconds,
        )
        self.window_title = cfg["game_window"]["title"]
        self.fps = 0 # Frame per seconds
        # Timer
        self.t_last_up = 0.0
        self.t_last_down = 0.0
        self.t_last_toggle = 0.0
        self.t_last_jump_down = 0.0
        self.t_last_run = time.time()
        self.t_last_skill = 0.0 # Last time character perform action(attack, cast spell, ...)
        # Flags
        self.is_enable = True
        self.is_need_force_heal = False
        self.is_terminated = False
        # Parameters
        self.debounce_interval = RUNTIME_POLICY.hotkey_debounce_seconds
        self.fps_limit = RUNTIME_POLICY.keyboard_fps

        # use 'ctrl', 'alt' for mac, because it's hard to get around
        # macOS's security settings
        if is_mac():
            self.toggle_key = keyboard.Key.ctrl
            self.terminate_key = keyboard.Key.esc
        else:
            self.toggle_key = keyboard.Key.f1
            self.terminate_key = keyboard.Key.f12

        # set up attack key
        self.attack_key = ""
        if (cfg["bot"].get("route_only") and
                not cfg["bot"].get("route_attack", False)):
            self.attack_key = ""
        elif cfg["bot"]["attack"] == "aoe_skill":
            self.attack_key = cfg["key"]["aoe_skill"]
        elif cfg["bot"]["attack"] == "directional":
            self.attack_key = cfg["key"]["directional_attack"]
        else:
            raise ValueError(f"Unexpected attack type: {cfg['bot']['attack']}")

        # Start keyboard control thread
        threading.Thread(target=self.run, daemon=True).start()

        logger.info("[KeyBoardController] Init done")

    def toggle_enable(self):
        '''
        toggle_enable
        '''
        self.is_enable = not self.is_enable
        logger.info(f"Player pressed F1, is_enable:{self.is_enable}")

        # Make sure all key are released
        self.release_all_key()

    def disable(self):
        '''
        disable keyboard controlller
        '''
        self.is_enable = False

    def enable(self):
        '''
        enable keyboard controlller
        '''
        self.is_enable = True

    def set_command(self, new_command):
        '''
        Set keyboard command
        '''
        parts = new_command.split()
        if len(parts) != 3:
            raise ValueError(f"Keyboard command must have three fields: {new_command!r}")
        with self._command_lock:
            self.cmd_left_right, self.cmd_up_down, self.cmd_action = parts
            self.command_updated_at = time.monotonic()

    def _command_snapshot(self):
        with self._command_lock:
            return (
                self.cmd_left_right,
                self.cmd_up_down,
                self.cmd_action,
                self.command_updated_at,
            )

    def press_directional_attack(self, direction, route_direction=None):
        """Briefly face a mob, attack, then restore route movement."""
        if route_direction is None:
            route_direction = self.cmd_left_right
        needs_turn = direction in {"left", "right"} and route_direction != direction
        attack_completed = False
        try:
            if needs_turn:
                if route_direction in {"left", "right"}:
                    key_up(route_direction)
                key_down(direction)
                time.sleep(float(self.cfg["directional_attack"].get(
                    "character_turn_delay", 0.02)))

            logger.info(
                f"[AttackInput] key={self.attack_key} face={direction} "
                f"route={route_direction}"
            )
            press_key(self.attack_key)
            attack_completed = True
        finally:
            if needs_turn:
                try:
                    key_up(direction)
                finally:
                    if attack_completed and route_direction in {"left", "right"}:
                        key_down(route_direction)

    def is_game_window_active(self):
        '''
        Check if the game window is currently the active (foreground) window.

        Returns:
        - True
        - False
        '''
        if is_mac():
            active_window = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID
            )
            for window in active_window:
                window_name = window.get(Quartz.kCGWindowName, '')
                if window_name and self.window_title in window_name:
                    return True
            return False
        else:
            try:
                active_window = gw.getActiveWindow()
                if not active_window:
                    return False
                return self.window_title in active_window.title
            except Exception as e:
                return False

    def release_all_key(self):
        '''
        Release all key
        '''
        keys = [
            "left", "right", "up", "down",
            self.cfg["key"].get("jump", ""),
            self.cfg["key"].get("teleport", ""),
            self.attack_key,
        ]
        released = set()
        for key in keys:
            if not key or key in released:
                continue
            released.add(key)
            try:
                key_up(key)
            except Exception as exc:
                logger.error(f"[release_all_key] Failed to release {key!r}: {exc}")

    def limit_fps(self):
        '''
        Limit FPS
        '''
        # If the loop finished early, sleep to maintain target FPS
        target_duration = 1.0 / self.fps_limit  # seconds per frame
        frame_duration = time.time() - self.t_last_run
        if frame_duration < target_duration:
            time.sleep(target_duration - frame_duration)

        # Update FPS
        self.fps = round(1.0 / (time.time() - self.t_last_run))
        self.t_last_run = time.time()
        # logger.info(f"FPS = {self.fps}")

    def run(self):
        '''
        run
        '''
        try:
            while not self.is_terminated:
                # Check if game window is active
                is_active = self.is_game_window_active()
                if not self.is_enable or not is_active:
                    if self._was_game_window_active:
                        self.release_all_key()
                    self._was_game_window_active = False
                    self._foreground_acquired_at = 0.0
                    self.cmd_left_right_last = ""
                    self.cmd_up_down_last = ""
                    self.cmd_action_last = "none"
                    self.limit_fps()
                    continue

                now_monotonic = time.monotonic()
                if not self._was_game_window_active:
                    # Never replay a command that was produced while another
                    # window owned the keyboard. Wait for one new perception
                    # result after the game becomes foreground again.
                    self._was_game_window_active = True
                    self._foreground_acquired_at = now_monotonic
                    self.release_all_key()
                    self.cmd_left_right_last = ""
                    self.cmd_up_down_last = ""
                    self.cmd_action_last = "none"
                    self.limit_fps()
                    continue

                cmd_left_right, cmd_up_down, cmd_action, command_updated_at = \
                    self._command_snapshot()
                command_is_fresh = (
                    command_updated_at > self._foreground_acquired_at and
                    now_monotonic - command_updated_at <= self.command_timeout_seconds
                )
                if not command_is_fresh:
                    if (self.cmd_left_right_last or self.cmd_up_down_last or
                            self.cmd_action_last != "none"):
                        self.release_all_key()
                    self.cmd_left_right_last = ""
                    self.cmd_up_down_last = ""
                    self.cmd_action_last = "none"
                    self.limit_fps()
                    continue

                # Force Heal
                if self.is_need_force_heal:
                    cmd_action = "add_hp"

                ##########################
                ### Left-Right Command ###
                ##########################
                if cmd_left_right == "left":
                    if self.cmd_left_right_last != "left":
                        key_up("right")
                        key_down("left")
                elif cmd_left_right == "right":
                    if self.cmd_left_right_last != "right":
                        key_up("left")
                        key_down("right")
                elif cmd_left_right == "stop":
                    if self.cmd_left_right_last != "stop":
                        key_up("left")
                        key_up("right")
                elif cmd_left_right == "none":
                    if self.cmd_left_right_last != "none":
                        key_up("left")
                        key_up("right")
                else:
                    logger.error("[KeyBoardController] Unsupported left-right command: "
                                 f"{cmd_left_right}")
                self.cmd_left_right_last = cmd_left_right

                #######################
                ### Up-Down Command ###
                #######################
                if cmd_up_down == "up" and cmd_action == "mount":
                    if self.cmd_up_down_last != "up":
                        key_up("down")
                elif cmd_up_down == "up":
                    if self.cmd_up_down_last != "up":
                        key_up("down")
                        key_down("up")
                elif cmd_up_down == "down":
                    if self.cmd_up_down_last != "down":
                        key_up("up")
                        key_down("down")
                elif cmd_up_down == "stop":
                    if self.cmd_up_down_last != "stop":
                        key_up("up")
                        key_up("down")
                elif cmd_up_down == "none":
                    if self.cmd_up_down_last != "none":
                        key_up("up")
                        key_up("down")
                else:
                    logger.error("[KeyBoardController] Unsupported up-down command: "
                                 f"{cmd_up_down}")
                self.cmd_up_down_last = cmd_up_down

                ######################
                ### Action Command ###
                ######################
                action_rising_edge = cmd_action != self.cmd_action_last
                if cmd_action == "mount":
                    if action_rising_edge:
                        key_down(self.cfg["key"]["jump"])
                        key_down("up")
                        time.sleep(float(self.cfg["route"].get("mount_press_duration", 0.10)))
                        key_up(self.cfg["key"]["jump"])
                elif cmd_action == "jump":
                    if action_rising_edge:
                        press_key(self.cfg["key"]["jump"])
                elif cmd_action == "teleport":
                    if action_rising_edge:
                        press_key(self.cfg["key"]["teleport"])
                elif cmd_action in {"attack", "attack_left", "attack_right"}:
                    if action_rising_edge:
                        if cmd_action == "attack":
                            press_key(self.attack_key)
                        else:
                            self.press_directional_attack(
                                cmd_action.split("_", 1)[1],
                                route_direction=cmd_left_right,
                            )
                        self.t_last_skill = time.time()
                elif cmd_action == "add_hp":
                    press_key(self.cfg["key"]["add_hp"])
                    cmd_action = "none"
                elif cmd_action == "add_mp":
                    press_key(self.cfg["key"]["add_mp"])
                    cmd_action = "none"
                elif cmd_action == "goal":
                    pass
                elif cmd_action == "none":
                    pass
                else:
                    logger.error("[KeyBoardController] Unsupported action command: "
                                 f"{cmd_action}")
                self.cmd_action_last = cmd_action

                self.limit_fps()
        except Exception as exc:
            logger.error(f"[KeyBoardController] Control loop failed closed: {exc}")
            self.is_terminated = True
        finally:
            self.release_all_key()
            logger.info("[KeyBoardController] terminated")
