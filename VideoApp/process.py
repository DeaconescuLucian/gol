import cv2
import numpy as np
import math
import random
from numba import jit, cuda
from numba import prange
import time

# cap = cv2.VideoCapture(0)

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

# big_edge_detection_kernel_destructor = np.array([[-1, -1, -1, -1, -1],
#                     [-1, -1, -1, -1, -1],
#                     [0, 0, 1, 0, 0],
#                     [-1, -1, -1, -1, -1],
#                     [16, 8, 0, 8, 16]])


big_edge_detection_kernel_destructor = np.array([[16, 8, 4, 8, 16],
                    [8, 4, 2, 4, 8],
                    [4, 2, 1, 2, 4],
                    [8, 4, 2, 4, 8],
                    [16, 8, 4, 8, 16]])

visible_endge_min_value = 20
@jit(nopython=True, parallel=True,nogil=True,target_backend=cuda)
def add_effect(frame, filtered_image, r, g, b, left):
    new_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    for i in prange(frame.shape[0]):
        for j in prange(frame.shape[1]//2 + 1):
            if j < left: #10000 - ca sa se aplice pe toata imaginea daca semnul e >= / -1 daca semnul e >
                if filtered_image[i, j, 0] > visible_endge_min_value and filtered_image[i, j, 1] > visible_endge_min_value and filtered_image[i, j, 2] > visible_endge_min_value:
                    new_frame[i, j] = [r,g,b]
                # else:
                #     new_frame[i, j] = frame[i, j]
                if filtered_image[i, frame.shape[1] - j, 0] > visible_endge_min_value and filtered_image[
                    i, frame.shape[1] - j, 1] > visible_endge_min_value and filtered_image[i, frame.shape[1] - j, 2] > visible_endge_min_value:
                    new_frame[i, frame.shape[1] - j] = [r,g,b]
                # else:
                #     new_frame[i, frame.shape[1] - j] = frame[i, frame.shape[1] - j]
            else:
                new_frame[i, j] = frame[i, j]
                new_frame[i, frame.shape[1] - j] = frame[i, frame.shape[1] - j]

    return new_frame


#resize image if needed
#frame = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_AREA)


def animate(frame, coef, destructor_coef):
    gen = 0
    f_gen = 0
    counter = 1
    left = 0
    leftSign = 1
    # coef = 0.01
    # destructor_coef = 0.025
    # frame = cv2.imread("../images/boobs.png")
    global big_edge_detection_kernel
    global big_edge_detection_kernel_destructor
    while True:
        # print("here")

        r = 255 * math.fabs(np.sin(gen/60))
        g = 200 * math.fabs(np.sin(gen/120))
        b = 200 * math.fabs(np.sin(gen/260))

        #Cool effect
        filtered_image = cv2.GaussianBlur(frame, (9, 9), 0)
        filtered_image = cv2.filter2D(filtered_image, -1, big_edge_detection_kernel)


        #Apply edged over the original image
        new = add_effect(frame, filtered_image, r, g, b, left)
        # cv2.imwrite('devil/{f}.jpg'.format(f=gen), new)
        destructor = big_edge_detection_kernel_destructor * coef
        big_edge_detection_kernel = big_edge_detection_kernel + destructor_coef * destructor * (gen % 100) / 90

        #print("Frame {f} of 226.".format(f=gen))
        gen += 1
        f_gen+=counter

        if gen % 20 == 0:
            big_edge_detection_kernel_destructor = big_edge_detection_kernel_destructor * (-1)

        left+= leftSign
        if left == frame.shape[1] // 2 or left == 0:
            leftSign*=-1

        # cv2.imshow('Window', new)
        #
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        yield new
    # cv2.destroyAllWindows()

# animate(None, None, None)






