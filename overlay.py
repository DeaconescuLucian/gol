import cv2
import numpy as np
import math
import random
from numba import jit, cuda
from numba import prange
import time
import os

# cap = cv2.VideoCapture(0)

@jit
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

@jit(nopython=True, parallel=True,nogil=True,target_backend=cuda)
def add_effect(frame, filtered_image, mask, overlay_factor):
    global penPressed, penX, penY, pen
    new_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    for i in prange(frame.shape[0]):
        for j in prange(frame.shape[1]):
            # new_frame[i, j, 0] = (1 - overlay_factor) * frame[i, j, 0] + overlay_factor * filtered_image[i, j, 0]
            # new_frame[i, j, 1] = (1 - overlay_factor) * frame[i, j, 1] + overlay_factor * filtered_image[i, j, 1]
            # new_frame[i, j, 2] = (1 - overlay_factor) * frame[i, j, 2] + overlay_factor * filtered_image[i, j, 2]
            if mask[i, j, 0] == 255 and mask[i, j, 1] == 0 and mask[i, j, 2] == 0:
                new_frame[i, j, 0] = frame[i, j, 0]-(overlay_factor * filtered_image[i, j, 0]) * (np.max(frame[i, j, 0:3]) / 255)
                new_frame[i, j, 1] = frame[i, j, 1]-(overlay_factor * filtered_image[i, j, 1]) * (np.max(frame[i, j, 0:3]) / 255)
                new_frame[i, j, 2] = frame[i, j, 2]-(overlay_factor * filtered_image[i, j, 2]) * (np.max(frame[i, j, 0:3]) / 255)
            else:
                new_frame[i, j, 0] = frame[i, j, 0]
                new_frame[i, j, 1] = frame[i, j, 1]
                new_frame[i, j, 2] = frame[i, j, 2]

    return new_frame


def draw(frame, color):
    for i in range(penY - pen//2, penY + pen//2, 1):
        for j in range(penX - pen//2, penX + pen//2, 1):
            if calculate_distance(penX, penY, j, i) <= pen//2:
                frame[i, j, 0] = color[0]
                frame[i, j, 1] = color[1]
                frame[i, j, 2] = color[2]


def mouse_callback(event, x, y, flags, param):
    global penPressed, penX, penY, new
    penX = int(x)
    penY = int(y)

    if event == cv2.EVENT_LBUTTONDOWN:
        penPressed = True

    if event == cv2.EVENT_LBUTTONUP:
        penPressed = False

    if penPressed:
        draw(new, np.array([255, 0, 0]))


# _, frame = cap.read()
frame = cv2.imread("clip1/0.jpg")
frame = cv2.resize(frame, (256, 256))
filtered_image = cv2.imread("clip1/0.png")
filtered_image = cv2.resize(filtered_image, (256, 256))
# mask = cv2.imread("me.png")
mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        mask[i, j, 0] = 255
#
# mask = cv2.resize(mask, (1024, 1024))



gen = 0
gen_counter = 1
f_gen=0
f_gen_counter=1

animation_gen = 0
overlay_factor = 0.6
overlay_increment = 0.01
pen = 30

global penPressed, penX, penY, new
penPressed = False
penX = 0
penY = 0

cv2.imshow('frame', frame)
new = frame

#need this for masking
# cv2.setMouseCallback("frame", mouse_callback)


# while True:
#     new = add_effect(frame, filtered_image, mask, overlay_factor)
#     new = cv2.GaussianBlur(new, (5, 5), 0)
#     cv2.imshow('frame', new)
#     overlay_factor += overlay_increment
#
#     if overlay_factor >= 1 or overlay_factor <= 00:
#         overlay_increment *= -1
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to exit
#         break
#
#     if cv2.waitKey(1) & 0xFF == ord('w'):  # press 'w' to make pen bigger
#         pen += 1
#
#     if cv2.waitKey(1) & 0xFF == ord('e'):  # press 'e' to make pen smaller
#         pen -= 1
#
#     if cv2.waitKey(1) & 0xFF == ord('s'):  # press 's' to exit
#         cv2.imwrite('me.png'.format(f=gen), new)

# overlay_factor = 0.45
for clip in range(9):
    print(f"Clip{clip}")
    gen = 0
    try:
        os.mkdir(f"edited-clip/{clip}")
    except FileExistsError:
        pass
    while True:
        frame = cv2.imread("clip{clip}/{f}.png".format(f=gen, clip=clip))
        frame = cv2.resize(frame, (256, 256))
        filtered_image = cv2.imread("clip{clip}/{f}.jpg".format(f=gen, clip=clip))
        filtered_image = cv2.resize(filtered_image, (256, 256))
        new = add_effect(frame, filtered_image, mask, overlay_factor)
        new = cv2.GaussianBlur(new, (5, 5), 0)
        cv2.imshow('frame', new)
        print(overlay_factor)
        # cv2.imwrite('edited-clip/{clip}/{f}.png'.format(f=gen,clip=clip), new)
        overlay_factor += overlay_increment

        if overlay_factor >= 0.7 or overlay_factor <= 0.3:
            overlay_increment *= -1

        gen += gen_counter
        f_gen +=f_gen_counter
        animation_gen += 1

        # if f_gen == 198:
        #     f_gen_counter *= -1
        #     f_gen = 1477
        #     break
        #
        # if f_gen == 0:
        #     f_gen_counter *= -1
        #     f_gen = 0
        #     # break

        if gen == 199:
            break
            # gen_counter *= -1
            # gen = 226

        # if gen == 0:
        #     gen_counter *= -1
        #     gen = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to exit
            break

        if cv2.waitKey(1) & 0xFF == ord('w'):  # press 'w' to make pen bigger
            pen += 1

        if cv2.waitKey(1) & 0xFF == ord('e'):  # press 'e' to make pen smaller
            pen -= 1

        if cv2.waitKey(1) & 0xFF == ord('s'):  # press 's' to exit
            cv2.imwrite('me.png', new)

cv2.destroyAllWindows()



