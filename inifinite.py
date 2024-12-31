import cv2
import numpy as np
import random
import math
from numba import jit, cuda
from numba import prange
import os

gen = 0

video_length = 300
width = 205
height = 128
no_clips = 4
clip_index = 0
small_width = 2
small_height = 1

scale = 1.0
max_scale = 6.0
scale_counter = max_scale / video_length

gen = 0
f_gen = 0
counter = -1

f_gen1 = 0


def pasteOver(big_image, small_image, small_width, small_height, big_width, big_height):
    x_offset = (big_width - small_width) // 2
    y_offset = (big_height - small_height) // 2
    new_image = big_image
    new_image[y_offset:y_offset + small_height, x_offset:x_offset + small_width] = small_image
    return new_image

while True:
    frame = cv2.imread(f"wide/{clip_index}/{f_gen}.png")

    if scale < max_scale:
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        displayed_frame = cv2.resize(frame, (scaled_width, scaled_height))
        centerX = scaled_width // 2
        centerY = scaled_height // 2

        x_top_left = (centerX - width//2)
        y_top_left = (centerY - height//2)

        if x_top_left < 0:
            x_top_left = 0
            x_bottom_right = width
        else:
            x_bottom_right = x_top_left + width
        #
        if y_top_left < 0:
            y_top_left = 0
            y_bottom_right = height
        else:
            y_bottom_right = y_top_left + height

        if x_bottom_right > scaled_width:
            x_top_left = scaled_width - width
            x_bottom_right = scaled_width

        if y_bottom_right > scaled_height:
            y_top_left = scaled_height - height
            y_bottom_right = scaled_height

        top_left = (x_top_left, y_top_left)
        bottom_right = (x_bottom_right, y_bottom_right)


        cropped_image = displayed_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        if scale <= 3.0:
            cv2.imshow('Animation', cropped_image)
            cv2.imwrite('infinite_loop/{f}.png'.format(f=gen), cropped_image)
        scale += scale_counter

    if scale > 3.0:
        if small_width <= width and small_height <= height:
            frame2 = cv2.imread(f"wide/{clip_index+1}/{f_gen1}.png")
            resized_frame2 = cv2.resize(frame2, (small_width, small_height))
            overlayed_img = pasteOver(cropped_image, resized_frame2, small_width, small_height, width, height)
            cv2.imshow('Animation', overlayed_img)
            cv2.imwrite('infinite_loop/{f}.png'.format(f=gen), overlayed_img)
            small_width += int(width * scale_counter)
            small_height += int(height * scale_counter)
            f_gen1 += 1
        else:
            if clip_index < no_clips:
                clip_index += 1
                scale = 1.0
                small_width = 2
                small_height = 1
                f_gen = f_gen1
                f_gen1 = 0
            else:
                break
                clip_index=0
                f_gen1 = 0

    f_gen += counter
    gen += 1
    if f_gen == video_length + 1 or f_gen == -1:
        counter *= -1
        f_gen += counter

    # Wait for 25 milliseconds
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()