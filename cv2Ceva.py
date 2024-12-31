import cv2
import numpy as np
import math
import random
from numba import jit, cuda
from numba import prange
import time
import os

cap = cv2.VideoCapture(0)

edge_detection_kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

edge_detection_kernel_destructor = np.array([[4, 2, 4],
                   [2, 1, 2],
                   [4, 2, 4]])

big_edge_detection_kernel = np.array([[-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, 24, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1]])

big_edge_detection_kernel_destructor = np.array([[-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [0, 0, 1, 0, 0],
                    [-1, -1, -1, -1, -1],
                    [16, 8, 0, 8, 16]])


big_edge_detection_kernel_destructor = np.array([[16, 8, 4, 8, 16],
                    [8, 4, 2, 4, 8],
                    [4, 2, 1, 2, 4],
                    [8, 4, 2, 4, 8],
                    [16, 8, 4, 8, 16]])


one_kernel = np.array([[0.11, 0.11, 0.11],
               [0.11, 0.11, 0.11],
               [0.11, 0.11, 0.11]])


@jit(nopython=True, parallel=True,nogil=True,target_backend=cuda)
def add_effect(frame, filtered_image, r, g, b):
    new_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    max_c = 250
    for i in prange(frame.shape[0]):
        for j in prange(frame.shape[1]//2 + 1):
            if j < 10000: #10000 - ca sa se aplice pe toata imaginea daca semnul e >= / -1 daca semnul e >
                if filtered_image[i, j, 0] > visible_endge_min_value or filtered_image[i, j, 1] > visible_endge_min_value or filtered_image[i, j, 2] > visible_endge_min_value:
                    new_frame[i, j] = [0,0,0]
                else:
                    new_frame[i,j] = [b,g,r]
                if filtered_image[i, frame.shape[1] - j, 0] > visible_endge_min_value or filtered_image[
                    i, frame.shape[1] - j, 1] > visible_endge_min_value or filtered_image[i, frame.shape[1] - j, 2] > visible_endge_min_value:
                    new_frame[i, frame.shape[1] - j] = [0,0,0]
                else:
                    new_frame[i, frame.shape[1] - j] = [b,g,r]
            else:
                new_frame[i, j] = frame[i, j]
                new_frame[i, frame.shape[1] - j] = frame[i, frame.shape[1] - j]

    return new_frame

@jit(nopython=True, parallel=True,nogil=True,target_backend=cuda)
def copy_non_black(frame, new):
    # new_frame = np.copy(frame)
    x = (frame.shape[1] - new.shape[1]) // 2
    y = (frame.shape[0] - new.shape[0]) // 2
    frame[y:y+new.shape[0], x: x+new.shape[1]] = new
    # for i in prange(y, y + new.shape[0], 1):
    #     for j in prange(x, x + new.shape[1], 1):
    #         if new[i,j, 0] + new[i,j,1] + new[i,j,2] > 50:
    #             new_frame[i, j] = new[i,j]
    return frame


# _, frame = cap.read()
#resize image if needed
#frame = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_AREA)

#region coef
# 0.1 and 0.025 preety wild / and 0.002 nice

coef = 0.1
destructor_coef = 0.001
#endregion
visible_endge_min_value = 1

gen = 0
f_gen = 0
counter = 1

alpha = 0
visible_coef = 1



clip = 5
colors = [{'r': 180, 'g': 70, 'b': 70}, {'r': 80, 'g': 200, 'b': 10}, {'r': 121, 'g': 10, 'b': 200}, {'r': 10, 'g': 200, 'b': 190}]
color_index = 0
_r = colors[color_index]['r']
_g = colors[color_index]['g']
_b = colors[color_index]['b']
blur_factor = 9 # 131
blur_factor_coef = 0 #-2
while True:
    # current_time = time.time()  # Get the current time
    # elapsed_time = current_time - start_time

    # if elapsed_time >= 1:
    #     #print(gen-last_gen)
    #     last_gen = gen
    #     start_time = time.time()  # Reset the start time
    frame = cv2.imread(f"wide/0/{gen}.png")
    # cv2.imshow('frame', frame)
    # _, frame = cap.read()
    # frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
    #cv2.imshow('frame', frame)
    r = _r * math.fabs(np.sin(gen/45))
    g = _g * math.fabs(np.sin(gen/90))
    b = _b * math.fabs(np.sin(gen/120))
    frame = cv2.resize(frame, (817, 512), interpolation=cv2.INTER_LANCZOS4)

    #Cool effect
    filtered_image = cv2.GaussianBlur(frame, (blur_factor, blur_factor), 0)
    filtered_image = cv2.filter2D(filtered_image, -1, big_edge_detection_kernel)
    # filtered_image1 = cv2.filter2D(filtered_image, -1, edge_detection_kernel)
    # cv2.imshow('frame', filtered_image)

    # size += sizeSign
    # if size == frame.shape[1] // 2 or size == 0:
    #     sizeSign *= -1


    #Apply edged over the original image
    new = add_effect(frame, filtered_image, r, g, b)

    # final = cv2.fastNlMeansDenoisingColored(new, None, 50, 50, 3, 9)

    # new1 = add_effect(frame, new, 80, 200, 10, left, top)
    # baseFrame = cv2.resize(filtered_image, (256, 256), interpolation=cv2.INTER_AREA)
    # baseFrame = copy_non_black(baseFrame, new)
    # # if start == False and leftSign == -1:
    #     start = True
    # if start == True:
    #     cv2.imshow('frame', new)
    cv2.imshow('frame', new)
    # cv2.imwrite(f'infinite_loop_edge/{f_gen}.png', final)
    destructor = big_edge_detection_kernel_destructor * coef
    # big_edge_detection_kernel = big_edge_detection_kernel + destructor_coef * destructor * (gen % 100) / 90
    #
    # destructor = edge_detection_kernel_destructor * coef
    # edge_detection_kernel = edge_detection_kernel + 0.0005 * destructor * (gen % 100) / 90
    #print("Frame {f} of 226.".format(f=gen))
    gen +=counter
    f_gen+=1
    blur_factor += blur_factor_coef

    if blur_factor < 9 or blur_factor > 131:
        blur_factor_coef *= -1
        blur_factor += blur_factor_coef

    if gen % 20 == 0:
        big_edge_detection_kernel_destructor = big_edge_detection_kernel_destructor * (-1)

    # if gen % 120 == 0:
    #     visible_coef *=  -1

    if gen % 300 == 0:
        color_index += 1
        if color_index > 3:
            break
        _r = colors[color_index]['r']
        _g = colors[color_index]['g']
        _b = colors[color_index]['b']
        counter *= -1
        # break

    # left+= leftSign
    # if left == frame.shape[1] // 2 or left == 0:
    #     leftSign*=-1


    # top += topSign
    # if top == frame.shape[0] or top == 0:
    #     topSign *= -1

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to exit
        # cv2.imwrite('me.png'.format(f=gen), frame)
        break


cv2.destroyAllWindows()


#devil, boobs
# coef = 0.01
# destructor_coef = 0.025
#visible_endge_min_value = 200


#angel
# coef = 0.01
# destructor_coef = 0.025
#visible_endge_min_value = 150

