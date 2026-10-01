'''
Auto generate route map 
'''
# Standard import
import time
import argparse
import sys
import os

# CV import
import numpy as np
import cv2

# local import
from src.utils.global_var import WINDOW_WORKING_SIZE
from src.utils.logger import logger
from src.runtime_policy import RUNTIME_POLICY
from src.utils.common import (
    find_pattern_sqdiff, draw_rectangle, screenshot,
    get_minimap_loc_size, get_player_location_on_minimap,
    to_opencv_hsv, load_yaml, override_cfg, is_mac, load_image,
    prepare_game_frame,
)
from src.input.KeyBoardListener import KeyBoardListener
from src.input.GameWindowCapturor import GameWindowCapturor

class RouteRecorder():
    '''
    Route recorder
    '''
    def update_info_on_img_frame_debug(self):
        '''
        update_info_on_img_frame_debug
        '''
        # Print text at bottom left corner
        self.fps = round(1.0 / (time.time() - self.t_last_frame))
        text_y_interval = 23
        text_y_start = 550
        dt_screenshot = time.time() - self.kb.t_func_key[1]
        dt_save_route = time.time() - self.kb.t_func_key[2]
        dt_save_map = time.time() - self.kb.t_func_key[3]
        text_list = [
            f"FPS: {self.fps}",
            f"Press 'F1' to {'pause' if self.is_enable else 'start'} route record",
            f"Press 'F2' to save screenshot{' : Saved' if dt_screenshot < 0.7 else ''}",
            f"Press 'F3' to save route{' : Saved' if dt_save_route < 0.7 else ''}",
            f"Press 'F4' to save map{' : Saved' if dt_save_map < 0.7 else ''}",
        ]
        for idx, text in enumerate(text_list):
            cv2.putText(
                self.img_frame_debug, text,
                (10, text_y_start + text_y_interval*idx),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        # Draw minimap rectangle on img debug
        draw_rectangle(
            self.img_frame_debug,
            self.loc_minimap,
            self.img_minimap.shape[:2],
            (0, 0, 255), "minimap",thickness=1
        )

        # Compute crop region with boundary check
        crop_w, crop_h = 80, 80
        x0 = max(0, self.loc_player_global[0] - crop_w // 2)
        y0 = max(0, self.loc_player_global[1] - crop_h // 2)
        x1 = min(self.img_route_debug.shape[1], x0 + crop_w)
        y1 = min(self.img_route_debug.shape[0], y0 + crop_h)

        # Crop region
        mini_map_crop = self.img_route_debug[y0:y1, x0:x1]
        mini_map_crop = cv2.resize(mini_map_crop,
                                (int(mini_map_crop.shape[1] * 3),
                                 int(mini_map_crop.shape[0] * 3)),
                                interpolation=cv2.INTER_NEAREST)
        # Paste into top-right corner of self.img_frame_debug
        h_crop, w_crop = mini_map_crop.shape[:2]
        h_frame, w_frame = self.img_frame_debug.shape[:2]
        x_paste = w_frame - w_crop - 10  # 10px margin from right
        y_paste = 70
        self.img_frame_debug[y_paste:y_paste + h_crop, x_paste:x_paste + w_crop] = mini_map_crop

        # Draw border around minimap
        cv2.rectangle(
            self.img_frame_debug,
            (x_paste, y_paste),
            (x_paste + w_crop, y_paste + h_crop),
            color=(255, 255, 255),   # White border
            thickness=2
        )

    def update_img_frame_debug(self):
        '''
        update_img_frame_debug
        '''
        debug_size = self.cfg["game_window"].get("debug_size")
        if debug_size:
            debug_h, debug_w = (int(value) for value in debug_size)
            debug_frame = cv2.resize(
                self.img_frame_debug,
                (debug_w, debug_h),
                interpolation=cv2.INTER_AREA,
            )
        else:
            debug_frame = self.img_frame_debug[:self.cfg["ui_coords"]["ui_y_start"], :]
        cv2.imshow("Game Window Debug", debug_frame)
        # Update FPS timer
        self.t_last_frame = time.time()

    def get_player_location_on_global_map(self):
        '''
        get_player_location_on_global_map
        '''
        self.loc_minimap_global, score, _ = find_pattern_sqdiff(
                                        self.img_map,
                                        self.img_minimap)
        loc_player_global = (
            self.loc_minimap_global[0] + self.loc_player_minimap[0],
            self.loc_minimap_global[1] + self.loc_player_minimap[1]
        )

        # Draw local minimap rectangle
        camera_bottom_right = (
            self.loc_minimap_global[0] + self.img_minimap.shape[1],
            self.loc_minimap_global[1] + self.img_minimap.shape[0]
        )
        cv2.rectangle(self.img_route_debug, self.loc_minimap_global,
                      camera_bottom_right, (0, 255, 255), 1)
        cv2.putText(
            self.img_route_debug,
            f"Minimap,score({round(score, 2)})",
            (self.loc_minimap_global[0], self.loc_minimap_global[1]+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 255, 255), 1
        )

        # Draw player center
        cv2.circle(self.img_route_debug,
                   loc_player_global, radius=2,
                   color=(0, 255, 255), thickness=-1)

        return loc_player_global

    def replace_color_on_map(self, lower_hsv, upper_hsv, replace_color=(0, 0, 0)):
        '''
        Replace pixels in self.img_map that fall within the given HSV range
        and are part of a connected component with area > 15.
        '''
        hsv_map = cv2.cvtColor(self.img_map, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_map, to_opencv_hsv(lower_hsv), to_opencv_hsv(upper_hsv))

        # Connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 10:
                component_mask = (labels == i)
                self.img_map[component_mask] = replace_color

    def get_img_frame(self):
        '''
        get_img_frame
        '''
        # Get window game raw frame
        self.frame = self.capture.get_frame()
        if self.frame is None:
            logger.warning("Failed to capture game frame.")
            return

        return prepare_game_frame(self.frame, self.cfg)

    def __init__(self, args):
        '''
        Init MapleStoryBot
        '''
        self.args = args # User arguments
        self.idx_routes = 0 # Index of route map
        self.fps = 0 # Frame per second
        self.is_first_frame = True # first frame flag
        self.is_enable = True
        # Coordinate (top-left coordinate)
        self.loc_minimap = (0, 0) # minimap location on game screen
        self.loc_player = (0, 0) # player location on game screen
        self.loc_player_minimap = (0, 0) # player location on minimap
        self.loc_minimap_global = (0, 0) # minimap location on global map
        self.loc_player_global = (0, 0) # player location on global map
        self.loc_player_global_last = None # playeer location on global map last frame
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = None # minimap on game screen
        self.img_map = None # map
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_last_draw_blob = time.time() # Last draw blob timer
        self.previous_key_press = frozenset()
        self.localization_gap_started_at = None

        # Load defautl yaml config
        cfg = load_yaml("config/config_default.yaml")
        # Override with platform config
        if is_mac():
            cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
        # Override with user customized config
        self.cfg = override_cfg(cfg, load_yaml(f"config/config_{args.cfg}.yaml"))

        # Parse color_code
        self.color_code = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code"].items()
        }
        color_code_up_down = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code_up_down"].items()
        }
        self.color_code.update(color_code_up_down) # Combine both dictionaries

        self.fps_limit = RUNTIME_POLICY.route_recorder_fps

        # Check create new map directory
        map_dir = os.path.join("minimaps", args.new_map)
        if os.path.exists(map_dir):
            if os.listdir(map_dir):
                raise FileExistsError(
                    f"Map directory already exists and will not be overwritten: {map_dir}"
                )
            logger.info(f"Reusing empty map directory: {map_dir}")
        else:
            os.makedirs(map_dir)
            logger.info(f"Created new directory: {map_dir}")

        # Load exist map
        if self.args.map != '':
            self.img_map = load_image(f"{self.args.map}")

        # Start keyboard listener thread
        self.kb = KeyBoardListener(self.cfg, is_autobot=False)

        # Start game window capturing thread
        logger.info("Waiting for game window to activate, please click on game window")
        self.capture = GameWindowCapturor(self.cfg)

    def ensure_img_map_capacity(self, x, y, h, w):
        '''
        Ensure that self.img_map is large enough to contain the region defined by (x, y, h, w).
        Always add at least "map_padding" when expanding in any direction.
        '''
        map_h, map_w = self.img_map.shape[:2]
        pad = RUNTIME_POLICY.route_recorder_map_padding

        # Compute required expansion margins
        expand_top = pad - y if y < pad else 0
        expand_left = pad - x if x < pad else 0
        expand_bottom = y + h + pad - map_h if y + h + pad > map_h else 0
        expand_right = x + w + pad - map_w if x + w + pad > map_w else 0
        expand_top = max(0, expand_top)
        expand_left = max(0, expand_left)
        expand_bottom = max(0, expand_bottom)
        expand_right = max(0, expand_right)
        # If no expansion needed, return
        if expand_top == 0 and expand_bottom == 0 and expand_left == 0 and expand_right == 0:
            return

        # Create new canvas and paste old image
        new_h = map_h + expand_top + expand_bottom
        new_w = map_w + expand_left + expand_right
        new_map = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        new_map[expand_top:expand_top + map_h, expand_left:expand_left + map_w] = self.img_map
        self.img_map = new_map

        # Also update all global coordinates that depend on the map (optional)
        self.loc_minimap_global = (
            self.loc_minimap_global[0] + expand_left,
            self.loc_minimap_global[1] + expand_top
        )

    def remove_color_code_pixels(self, img):
        """
        Set all pixels in self.img_map to black if they match any color in color_code (assumed RGB).
        """
        for rgb in self.color_code.keys():
            bgr = (rgb[2], rgb[1], rgb[0])  # Convert RGB → BGR
            mask = np.all(img == bgr, axis=2)
            img[mask] = (0, 0, 0)
        return img

    def update_minimap(self):
        '''
        update_minimap
        '''

    def run_once(self):
        '''
        Process with one game window frame
        '''
        # Get lastest game screen frame buffer
        img_frame = self.get_img_frame()
        if img_frame is None:
            return -1 # Wait for game window to be ready
        else:
            self.img_frame = img_frame

        # Image for debug use
        self.img_frame_debug = self.img_frame.copy()

        # Get minimap from game window
        if self.is_first_frame:
            minimap_result = get_minimap_loc_size(self.img_frame, self.cfg)
            if minimap_result is None:
                logger.warning("Unable to locate configured minimap ROI")
                return -1
            x, y, w, h = minimap_result
            if not self.cfg.get("minimap", {}).get("roi"):
                # Legacy dynamic detection includes its one-pixel border.
                x += 1
                y += 1
                w -= 2
                h -= 2
            self.loc_minimap = (x, y)
            self.img_minimap = self.img_frame[y:y+h, x:x+w]
        else:
            x, y = self.loc_minimap
            h, w = self.img_minimap.shape[:2]
            self.img_minimap = self.img_frame[y:y+h, x:x+w]

        # Replace black pixels (0, 0, 0) with (1, 1, 1)
        black_mask = np.all(self.img_minimap == [0, 0, 0], axis=-1)
        self.img_minimap[black_mask] = [1, 1, 1]

        # Get player location on minimap
        loc_player_minimap = get_player_location_on_minimap(
            self.img_minimap,
            minimap_player_color=self.cfg["minimap"]["player_color"],
            player_hsv=self.cfg["minimap"].get("player_hsv"),
        )
        if loc_player_minimap is None:
            if self.localization_gap_started_at is None:
                self.localization_gap_started_at = time.monotonic()
            self.previous_key_press = frozenset(self.kb.key_pressing)
            return -1
        self.loc_player_minimap = loc_player_minimap

        # Update map
        if self.is_first_frame:
            # copy minimap to map
            if self.img_map is None:
                self.img_map = self.img_minimap.copy()
                pad = RUNTIME_POLICY.route_recorder_map_padding
                self.img_map = cv2.copyMakeBorder(
                    self.img_map,
                    top=pad, bottom=pad, left=pad, right=pad,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)  # Black padding
                )

            # Replace player "yellow" dot to black on map
            self.replace_color_on_map(
                (55, 40, 80),
                (60, 100, 100)
            )
            # Replace other player "red" dot to black on map
            self.replace_color_on_map((0, 80, 80),
                                      (5, 100, 100))

            # Update route
            self.img_route = self.remove_color_code_pixels(self.img_map.copy())
            self.img_route_debug = self.img_route.copy()

        else:
            # Create mask where pixels are not black
            mask = np.any(self.img_minimap != [0, 0, 0], axis=2).astype(np.uint8)
            mask = mask * 255

            # Perform template matching to find where the current minimap fits in the global map
            self.loc_minimap_global, score, _ = find_pattern_sqdiff(
                self.img_map,
                self.img_minimap,
                mask=mask
            )
            x, y = self.loc_minimap_global
            h, w = self.img_minimap.shape[:2]
            # Ensure img_map is big enough to fit the newly explored region
            self.ensure_img_map_capacity(x, y, h, w)

            # Don't copy pixel near player
            player_yellow_dot_radius = 5
            px, py = self.loc_player_minimap
            h, w = self.img_minimap.shape[:2]
            x_min = max(0, px - player_yellow_dot_radius)
            x_max = min(w, px + player_yellow_dot_radius)
            y_min = max(0, py - player_yellow_dot_radius)
            y_max = min(h, py + player_yellow_dot_radius)
            # Apply the black color mask to mask player yellow dot
            self.img_minimap[y_min:y_max, x_min:x_max] = (0, 0, 0)

            # Update map
            if self.args.map == '':
                map_slice = self.img_map[y:y+h, x:x+w]
                black_mask = np.all(map_slice == [0, 0, 0], axis=2)
                map_slice[black_mask] = self.img_minimap[black_mask]

            # Replace other player "red" dot to black on map
            self.replace_color_on_map((0, 78, 78),
                                      (5, 100, 100))

        cv2.imshow("Map", self.img_map)
        self.img_route_debug = self.img_route.copy()

        # Get player location on global map
        self.loc_player_global = self.get_player_location_on_global_map()
        if self.localization_gap_started_at is not None:
            gap_ms = (time.monotonic() - self.localization_gap_started_at) * 1000.0
            distance = 0
            if self.loc_player_global_last is not None:
                distance = (
                    abs(self.loc_player_global[0] - self.loc_player_global_last[0])
                    + abs(self.loc_player_global[1] - self.loc_player_global_last[1])
                )
            if gap_ms > 500 or distance > 18:
                self.loc_player_global_last = None
            self.localization_gap_started_at = None

        # Determine which color code to use based on user input
        action = ""
        is_draw_blob = False
        key_press = self.kb.key_pressing
        jump_key = str(self.cfg["key"].get("jump", "")).lower()
        teleport_key = str(self.cfg["key"].get("teleport", "")).lower()
        if jump_key and jump_key in key_press:
            if "left" in key_press:
                action = "left none jump"
            elif "right" in key_press:
                action = "right none jump"
            elif "down" in key_press:
                action = "none down jump"
            else:
                action = "none none jump"
            is_draw_blob = True
        elif teleport_key and teleport_key in key_press: # Teleport skill
            if "left" in key_press:
                action = "left none teleport"
            elif "right" in key_press:
                action = "right none teleport"
            elif "down" in key_press:
                action = "none down teleport"
            elif "up" in key_press:
                action = "none up teleport"
            else:
                action = ""
            is_draw_blob = True
        elif "up" in key_press:
            action = "none up none"
        elif "down" in key_press:
            action = "none down none"
        elif "left" in key_press:
            action = "left none none"
        elif "right" in key_press:
            action = "right none none"
        else:
            action = ""

        # Check if need to change route
        if self.kb.is_pressed_func_key[2]: # 'F3' is pressed
            action = "none none goal"
            is_draw_blob = True
            self.kb.is_pressed_func_key[2] = False
        elif self.kb.is_pressed_func_key[0]: # 'F1' is pressed
            self.is_enable = not self.is_enable
            logger.info(f"User press F1, is_enable = {self.is_enable}")
            self.kb.is_pressed_func_key[0] = False

        # Update route image
        if self.is_enable and action != "":
            # Get color from action
            dict_action_to_color = {v: k for k, v in self.color_code.items()}
            color_rgb = dict_action_to_color.get(action, None)
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            # Draw a line from the last position to the current one (if available)
            px, py = self.loc_player_global
            if is_draw_blob:
                event_key = jump_key if action.endswith("jump") else teleport_key
                if action == "none none goal" or event_key not in self.previous_key_press:
                    # Draw a small filled circle at current position
                    cv2.circle(self.img_route,
                            (px, py),
                            radius=2,
                            color=color_bgr,
                            thickness=-1)  # filled circle
                    self.loc_player_global_last = None
            else:
                if self.loc_player_global_last is None:
                    px_last, py_last = self.loc_player_global
                else:
                    px_last, py_last = self.loc_player_global_last
                cv2.line(self.img_route,
                        (px_last, py_last),
                        (px     , py),
                        color=color_bgr,
                        thickness=1)
                self.loc_player_global_last = self.loc_player_global
        elif action == "":
            self.loc_player_global_last = None

        self.previous_key_press = frozenset(key_press)

        # Save route image if goal is drawn
        if action == "none none goal":
            out_path = f"minimaps/{self.args.new_map}/route{self.idx_routes+1}.png"
            cv2.imwrite(out_path, self.img_route)
            self.idx_routes += 1
            self.img_route = self.img_map.copy()
            logger.info(f"Save route image to {out_path}")

        # Save img_map to map.png
        if self.kb.is_pressed_func_key[3]: # 'F4' is pressed
            out_path = f"minimaps/{self.args.new_map}/map.png"
            cv2.imwrite(out_path, self.img_map)
            self.kb.is_pressed_func_key[3] = False
            logger.info(f"Save map image to {out_path}")

        #####################
        ### Debug Windows ###
        #####################
        # Print text on debug image
        self.update_info_on_img_frame_debug()

        # Show debug image on window
        self.update_img_frame_debug()

        # Check if need to save screenshot
        if self.kb.is_pressed_func_key[1]: # 'F2' is pressed
            screenshot(self.img_frame)
            self.kb.is_pressed_func_key[1] = False

        # Resize img_route_debug for better visualization
        self.img_route_debug = cv2.resize(
                    self.img_route_debug, (0, 0),
                    fx=self.cfg["minimap"]["debug_window_upscale"],
                    fy=self.cfg["minimap"]["debug_window_upscale"],
                    interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Route Map Debug", self.img_route_debug)

        # Enable cached location since second frame
        self.is_first_frame = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Argument to specify map name
    parser.add_argument(
        '--new_map',
        type=str,
        default='new_map',
        help='Specify the new map name'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='custom',
        help='Choose customized config yaml file in config/'
    )

    parser.add_argument(
        '--map',
        type=str,
        default='',
        help='use this map instead of creating a new one'
    )

    try:
        routeRecorder = RouteRecorder(parser.parse_args())
    except Exception as e:
        logger.error(f"RouteRecorder Init failed: {e}")
        sys.exit(1)
    else:
        while True:
            t_start = time.time()

            # Process one game window frame
            routeRecorder.run_once()

            # Exit if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Cap FPS to save system resource
            frame_duration = time.time() - t_start
            target_duration = 1.0 / routeRecorder.fps_limit
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

        cv2.destroyAllWindows()
