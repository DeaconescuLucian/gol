import cv2
import numpy as np
import random
import math
from numba import jit, cuda
from numba import prange
import os

gen = 0



size = 512
canvas = np.zeros((size, size, 3), dtype=np.uint8)



def zoom(x,y,flags):
    global scale, mouseX, mouseY, centerX, centerY, oldCenterX, oldCenterY, scaled_size
    mouseX = int(x)
    mouseY = int(y)
    absCenterX = centerX // scale
    absCenterY = centerY // scale
    signX = 1 if mouseX > absCenterX else - 1 if mouseX < absCenterX else 0
    signY = 1 if mouseY > absCenterY else - 1 if mouseY < absCenterY else 0

    # Scroll up
    if flags > 0:
        addedDim = size * 0.05
        scale += 0.1

        centerX = int(centerX + signX * addedDim + signX * ((mouseX - size // 2) / (size // 2)) * addedDim)
        centerY = int(centerY + signY * addedDim + signY * ((mouseY - size // 2) / (size // 2)) * addedDim)
    # Scroll down
    else:
        substractedDim = size * 0.05
        scale -= 0.1
        if scale < 1.0:
            scale = 1.0
        centerX = int(centerX - signX * substractedDim - ((mouseX - size // 2) / (size // 2)) * substractedDim)
        centerY = int(centerY - signY * substractedDim - ((mouseY - size // 2) / (size // 2)) * substractedDim)

    scaled_size = int(size * scale)
    verifiedx = False
    verifiedy = False
    if centerX < size // 2 and verifiedx is False:
        centerX = size // 2
        verifiedx = True
    if centerX > scaled_size - size // 2 and verifiedx is False:
        centerX = scaled_size - size // 2

    if centerY < size // 2 and verifiedy is False:
        centerY = size // 2
        verifiedy = True

    if centerY > scaled_size - size // 2 and verifiedy is False:
        centerY = scaled_size - size // 2


def mouse_callback(event, x, y, flags, param):
    global scale, mouseX, mouseY, centerX, centerY, oldCenterX, oldCenterY
    if event == cv2.EVENT_MOUSEWHEEL:
        zoom(x,y,flags)


frame = canvas.copy()
cropped_image = frame

gen = 0
global scale, mouseX, mouseY, centerX, centerY, oldCenterX, oldCenterY, scaled_size
mouseX = size // 2
mouseY = size // 2
centerX = size // 2
centerY = size // 2
oldCenterX = size // 2
oldCenterY = size // 2
scale = 1.0
scale_counter = 0.01 #0.005
cv2.imshow('Animation', frame)
cv2.setMouseCallback("Animation", mouse_callback)

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
    frame = cv2.imread("generated_img/me.jpg")

    scaled_size = int(size * scale)

    displayed_frame = cv2.resize(frame, (scaled_size, scaled_size))
    # centerX = int(oldCenterX + (mouseX - (size//2)))
    # centerY = int(oldCenterY + (mouseY - (size//2)))

    # if scale <= 6:
    #     if not end:
    #         centerX = scaled_size // 2 + int(5/6 * (scaled_size // 2) )
    #         centerY = scaled_size // 2 - int(5/6 * (scaled_size // 2) )
    #     else:
    #         centerX = scaled_size // 2
    #         centerY = scaled_size // 2
    # else:
    #     if x_move and x_c_p < len(x_checkpoints):
    #         centerX += x_move_incr
    #         if x_c_p < len(x_checkpoints):
    #             if centerX < x_checkpoints[x_c_p] and x_c_p_s == -1:
    #                 x_move = False
    #                 y_move = True
    #                 x_c_p += 1
    #                 x_c_p_s = 1
    #                 x_move_incr *= -1
    #         if x_c_p < len(x_checkpoints):
    #             if centerX > x_checkpoints[x_c_p] and x_c_p_s == 1:
    #                 x_move = False
    #                 y_move = True
    #                 x_c_p += 1
    #                 x_c_p_s = -1
    #                 x_move_incr *= -1
    #     if y_move and y_c_p < len(y_checkpoints):
    #         centerY += y_move_incr
    #         if y_c_p < len(y_checkpoints):
    #             if centerY < y_checkpoints[y_c_p] and y_c_p_s == -1:
    #                 y_move = False
    #                 x_move = True
    #                 y_c_p += 1
    #                 y_c_p_s = 1
    #                 y_move_incr *= -1
    #         if y_c_p < len(y_checkpoints):
    #             if centerY > y_checkpoints[y_c_p] and y_c_p_s == 1:
    #                 y_move = False
    #                 x_move = True
    #                 y_c_p += 1
    #                 y_c_p_s = -1
    #                 y_move_incr *= -1
    #
    #     if x_c_p == len(x_checkpoints) and y_c_p == len(y_checkpoints) and not end:
    #         scale = 5.99
    #         scale_counter = -0.01
    #         end = True

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

    # cv2.resizeWindow('Animation', scaled_size, scaled_size)
    # cv2.imwrite('edited-clip-collage-cell128-multi/{f}.png'.format(f=gen), cropped_image)
    # gen += 1
    # f_gen+=counter
    #
    # if gen % 347 == 0:
    #     counter *= -1
    #
    # if scale >= 6 or scale <= 1:
    #     # scale_counter *= -1
    #     scale_counter = 0
    #     if scale < 1:
    #         scale = 1

    # Wait for 25 milliseconds
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()