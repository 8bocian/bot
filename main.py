import math
import random
import time

import matplotlib.pyplot as plt
import mss
import pygetwindow as pw
import pyscreenshot as ps
import pyautogui as pg
import cv2
import numpy as np

class Fisherman:
    def __init__(self, total_carry=800, total_silver=0, window_title=""):
        self.total_carry = total_carry
        self.silver = total_silver
        self.window_title = window_title
        self.path = []
        self.current_location = 0
        self.last_fishing_location = None

    def get_window(self):
        return pw.getWindowsWithTitle(self.window_title)[0]

    def get_player_position(self):
        window = self.get_window()
        return ((window.width // 2) - 5, (window.height // 2) - 70)

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
            img = np.array(sct.grab(window))

            img_h, img_w, _ = img.shape
            img = img[7:img_h - 7, 7:img_w - 7, :]
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def find_water(self, image):
        hue_channel = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 0]

        lower_blue = 90
        upper_blue = 120

        mask = cv2.inRange(hue_channel, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            M = cv2.moments(largest_contour)

            # Calculate the x and y coordinates of the centroid
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                water_position = (cX, cY)
        else:
            water_position = None
            largest_contour = None

        return water_position, largest_contour

    def find_intersection(self, contour, viewport):
        intersection = cv2.intersectConvexConvex(viewport.astype(np.float32), contour.astype(np.float32))

        if intersection[1] > 0:
            intersection_points = intersection[0]
            intersection_points = intersection_points.astype(np.int32)
            return intersection_points
        else:
            return None

    def throw_hook(self, position):
        center = self.get_player_position()
        center = self.win2glob(center)
        position = self.win2glob(position)
        pg.moveTo(center[0], center[1])#, random.uniform(0.6, 2.7), pg.easeOutBack)
        pg.moveTo(position[0], position[1])#, random.uniform(0.6, 2.7), pg.easeOutBack)
        pg.mouseDown( button='left')


    def win2glob(self, position):
        window = self.get_window()
        return (position[0] + window.left, position[1] + window.top)

    def glob2win(self, position):
        ...

    def check_hook(self, hook_image):
        hue_channel = cv2.cvtColor(hook_image, cv2.COLOR_RGB2HSV)[:, :, 0]
        # plt.imshow(hue_channel, cmap='gray')
        # plt.show()
        # Define the range for red hues in HSV
        lower_red1 = np.array(0)
        upper_red1 = np.array(10)
        lower_red2 = np.array(347)
        upper_red2 = np.array(260)

        # Create masks for the red hue ranges
        mask1 = cv2.inRange(hue_channel, lower_red1, upper_red1)
        mask2 = cv2.inRange(hue_channel, lower_red2, upper_red2)

        # Combine the masks
        red_mask = cv2.bitwise_or(mask1, mask2)
        # Count the number of red pixels
        count_red = np.count_nonzero(red_mask)

        return count_red



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

        is_fish = False
        while not is_fish:
            image = self.get_image()
            float_image = image[
                water_position[0] - 100:water_position[0] + 100,
                water_position[1] - 100: water_position[1] + 100
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

        pg.moveTo(water_position[0] - bias_x, water_position[1] - bias_y, random.uniform(0.6, 2.7),
                  pg.easeOutBack)

def calculate_hook_position(p1, p2, X):
    x1, y1 = p1
    x2, y2 = p2
    delta_x = x2 - x1
    delta_y = y1 - y2
    angle = math.atan2(delta_y, delta_x)

    new_x = x1 + X * math.cos(angle)
    new_y = y1 - X * math.sin(angle)

    return int(new_x), int(new_y)


if __name__ == "__main__":
    fisherman = Fisherman(window_title="Albion Online Client")
    while True:
        img = fisherman.get_image()
        water_position, contour = fisherman.find_water(img)
        fisherman.throw_hook(water_position)
        player_position = fisherman.get_player_position()
        time.sleep(2.5)
        avg_colors = 0
        iterations = 0
        while True:
            iterations += 1
            img = fisherman.get_image()

            hook_position = calculate_hook_position(player_position, water_position, 440)
            hook_image = img[
                hook_position[1] - 60:hook_position[1] + 60,
                hook_position[0] - 60: hook_position[0] + 60
            ]
            hook_color = fisherman.check_hook(hook_image)
            avg_colors += hook_color
            mean_hook_color = (avg_colors) / iterations
            # if hook_color < 0.4 * mean_hook_color:
            print(hook_color, mean_hook_color)
            cv2.circle(img, player_position, 5, (255, 0, 0), -1)
            cv2.circle(img, hook_position, 5, (0, 255, 0), -1)
            cv2.circle(img, water_position, 5, (255, 255, 255), -1)

            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
            # plt.imshow(img)
            # plt.show()
            cv2.imshow('Hook', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

#TODO: TRZEBA JAKOS WYKRYWAC SPLAWIK CZY SIE ZANURZA, OGARNAC MINIGIERKE I CHODZENIE
