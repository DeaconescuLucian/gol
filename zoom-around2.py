import cv2
import numpy as np
import random
import math
from numba import jit, cuda
from numba import prange
import os

gen = 0


size = 1000
canvas = np.zeros((size, size, 3), dtype=np.uint8)


frame = canvas.copy()
cropped_image = frame

gen = 0
global scale, mouseX, mouseY, centerX, centerY, oldCenterX, oldCenterY
mouseX = size // 2
mouseY = size // 2
centerX = size // 2
centerY = size // 2
oldCenterX = size // 2
oldCenterY = size // 2
scale = 1.0
scale_counter = 0.1 #0.005
cv2.imshow('Animation', frame)

f_gen = 0
counter = 1
y_move = False
x_move = True
x_move_incr = -12
y_move_incr = 12
# 3by3
# x_checkpoints = [(size * 3) // 6, (size * 15)//6, (size * 3) // 2]
# y_checkpoints = [(size * 15) // 6, (size * 3) // 2]

# 6by6
x_checkpoints = [(size * 6) // 12, (size * 66)//12, (size * 6) // 2]
y_checkpoints = [(size * 66) // 12, (size * 6) // 2]
x_c_p = 0
y_c_p = 0
x_c_p_s = -1
y_c_p_s = 1
end = False
while True:
    # new_frame1 = compute_new_frame(frame, gen, kernel,red_factor,green_factor,blue_factor)
    #
    # frame = new_frame1
    frame = cv2.imread(f"infinite_loop_edge/{f_gen}.jpg")
    # frame = cv2.imread("psihedelic/{f}.png".format(f=f_gen))

    scaled_size = int(size * scale)

    displayed_frame = cv2.resize(frame, (scaled_size, scaled_size))
    # centerX = int(oldCenterX + (mouseX - (size//2)))
    # centerY = int(oldCenterY + (mouseY - (size//2)))
    centerX = scaled_size // 2
    centerY = scaled_size // 4


    x_top_left = (centerX - size//2)
    y_top_left = (centerY - size//2)

    if x_top_left < 0:
        x_top_left = 0
        x_bottom_right = size
        # x_top_left_bonus = 0 - x_top_left
    else:
        x_bottom_right = x_top_left + size
    #
    if y_top_left < 0:
        y_top_left = 0
        y_bottom_right = size
        # y_top_left_bonus = 0 - y_top_left
    else:
        y_bottom_right = y_top_left + size

    if x_bottom_right > scaled_size:
        x_top_left = scaled_size - size
        x_bottom_right = scaled_size

    if y_bottom_right > scaled_size:
        y_top_left = scaled_size - size
        y_bottom_right = scaled_size

    top_left = (x_top_left, y_top_left)
    bottom_right = (x_bottom_right, y_bottom_right)


    cropped_image = displayed_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    cv2.imshow('Animation', cropped_image)

    # cv2.resizeWindow('Animation', scaled_width, scaled_height)
    # cv2.imwrite('edited-clip-collage-cell128-p3-zoom/{f}.png'.format(f=gen), cropped_image)
    gen += 1
    f_gen += counter
    scale += scale_counter
    if gen % 574 == 0:
        counter *= -1
        f_gen += counter

    if scale >= 5.99:
        scale_counter = -0.1
    if scale <= 1:
        scale_counter = 0.1

    # Wait for 25 milliseconds
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()