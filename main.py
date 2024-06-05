import random

import mss
import pygetwindow as pw
import pyscreenshot as ps
import pyautogui as pg
import cv2
import numpy as np

class fisherman:
    def __init__(self, total_carry, total_silver, window_title):
        self.total_carry = total_carry
        self.silver = total_silver
        self.window_title = window_title
        self.path = []
        self.current_location = 0
        self.last_fishing_location = None

    def run(self):
        running = True
        while running:
            self.find_water()


    def change_location(self):
        if self.current_location == len(self.path):
            self.current_location += 1


    def get_image(self):
        window = pw.getWindowsWithTitle(self.window_title)[0]
        window = {
            "left": window.left,
            "top": window.top,
            "width": window.width,
            "height": window.height
        }

        with mss.mss() as sct:
            return np.array(sct.grab(window))

    def find_water(self, image):
        hue_channel = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 0]

        lower_blue = 180
        upper_blue = 260

        mask = cv2.inRange(hue_channel, lower_blue, upper_blue)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest_contour)
            water_position = (x + w // 2, y - h // 2)
        else:
            water_position = None

        return water_position

    def find_intersection(self, contour, viewport):
        intersection = cv2.intersectConvexConvex(viewport.astype(np.float32), contour.astype(np.float32))

        if intersection[1] > 0:
            intersection_points = intersection[0]
            intersection_points = intersection_points.astype(np.int32)
            return intersection_points
        else:
            return None

    def throw_hook(self, position):
        pg.moveTo(position[0], position[1], random.uniform(0.6, 2.7), pg.easeOutBack)
        pg.leftClick()

    def check_float(self, float_image):
        red_float_img = float_image[:, 2]
        return not np.any(red_float_img.flat >= 128)

    def catch_fish(self):
        pg.leftClick()

    def play_minigame(self, minigame_image):
        ...

    def start_fishing(self):
        image = self.get_image()
        water_position = self.find_water(image)
        self.throw_hook(water_position)
        bias_x = 50 - random.randint(-10, 10)
        bias_y = 50 - random.randint(-10, 10)
        pg.moveTo(water_position[0] + bias_x, water_position[1] + bias_y, random.uniform(0.6, 2.7), pg.easeOutBack)

        fishing = True
        while fishing:
            is_fish = False
            while not is_fish:
                image = self.get_image()
                float_image = image[
                    water_position[0] - 30:water_position[0] + 30,
                    water_position[1] - 30: water_position[1] + 30
                ]
                is_fish = self.check_float(float_image)
                if is_fish:
                    self.catch_fish()
                    minigame_is_on = True
                    while minigame_is_on:
                        minigame_image = image[
                            water_position[0] - 30:water_position[0] + 30,
                            water_position[1] - 30: water_position[1] + 30
                        ]
                        result = self.play_minigame(minigame_image)
                        if result == True:
                            minigame_is_on = False
                            pg.moveTo(water_position[0] - bias_x, water_position[1] - bias_y, random.uniform(0.6, 2.7),
                                      pg.easeOutBack)
                            return True
                    fishing = False

        pg.moveTo(water_position[0] - bias_x, water_position[1] - bias_y, random.uniform(0.6, 2.7),
                  pg.easeOutBack)
