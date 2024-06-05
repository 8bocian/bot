import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_red_content(hue_channel, red_range_low, red_range_high=):
    red_mask = (hue_channel >= red_range_low) & (hue_channel <= red_range_high)

    # Calculate the percentage of red pixels
    red_pixels = np.sum(red_mask)
    total_pixels = hue_channel.size
    red_percentage = (red_pixels / total_pixels) * 100
img = cv2.imread('img.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hue = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
float_img = img[45:, 335:, :]
water_img = img[45:, 255:336, :]



plt.imshow(float_img)
plt.show()
red_float_img = float_img[:, :, 0]
total_red_float = sum(red_float_img.flatten())

red_water_img = water_img[:, :, 0]
total_red_water = sum(red_water_img.flatten())

print("float", total_red_float / (float_img.shape[0] * float_img.shape[1]))
print(np.any(red_float_img.flat >= 128))

print("water", total_red_water / (water_img.shape[0] * water_img.shape[1]))
print(np.any(red_water_img.flat >= 128 ))

plt.imshow(red_float_img, cmap='gray')
plt.show()

plt.imshow(red_water_img, cmap='gray')
plt.show()
