import cv2
import numpy as np
import random
import math
from numba import jit, cuda
from numba import prange

gen = 0
'''
    kernels
'''

#worms
kernel = np.array([[0.68, -0.9, 0.68],
                   [-0.9, -0.66, -0.9],
                   [0.68, -0.9, 0.68]])

#waves
# kernel = np.array([[0.565, -0.716, 0.565],
#                    [-0.716, 0.627, -0.716],
#                    [0.565, -0.716, 0.565]])

#game of life
# kernel = np.array([[1, 1, 1],
#                    [1, 9, 1],
#                    [1, 1, 1]])

#fabric
# kernel = np.array([[0.037, 0.43, -0.737],
#                    [0.406, -0.321, -0.319],
#                    [-0.458, 0.416, 0.478]])

#mitosis
# kernel = np.array([[-0.939, 0.88, -0.939],
#                    [0.88, 0.4, 0.88],
#                    [-0.939, 0.88, -0.939]])

#gaussian
# kernel = np.array([[0, 1, 0],
#                    [1, 1, 1],
#                    [0, 1, 0]])

#slime
# kernel = np.array([[0.8, -0.85, 0.8],
#                    [-0.85, -0.2, -0.85],
#                    [0.8, -0.85, 0.8]])

#smth
# kernel = np.array([[0.292, 0.577, -0.93],
#                    [-0.38, -0.629, -0.021],
#                    [0.391, -0.458, 0.524]])

#spread
# kernel = np.array([[0.2, 0.1, 0.2],
#                    [0.1, 0, 0.1],
#                    [0.2, 0.1, 0.2]])

#spread_reversed
# kernel = np.array([[-0.16, -0.1, -0.16],
#                    [-0.1, 0, -0.1],
#                    [-0.16, -0.1, -0.16]])
'''
    Filter kernels
'''


'''
    functions
'''
@jit(nopython=True,nogil=True,target_backend=cuda)
def compute_dot_product(y_top,y_bottom,x_left,x_right, x, y, neighbours, kernel):
    return np.sum(np.multiply(kernel[abs(y_top - (y - 1)):3 - abs(y_bottom - (y + 2)), abs(x_left - (x - 1)):3 - abs(x_right - (x + 2))],neighbours))


@jit(nopython=True, parallel=True,nogil=True,target_backend=cuda)
def compute_new_frame(frame, generation, kernel, slime_factor,red_factor,green_factor,blue_factor):
    new_frame = np.copy(frame)
    if generation < 1000000:
        for y in prange(frame.shape[0]//2):
            for x in prange(frame.shape[1]//2):
                y_top = max(y - 1, 0)
                y_bottom = min(y + 2, size//2)
                x_left = max(x - 1, 0)
                x_right = min(x + 2, size//2)
                #neighbours = frame[y_top:y_bottom, x_left:x_right, 2]
                neighbours = frame[y_top:y_bottom, x_left:x_right, (generation - 1) % 3 if generation > 0 else 0]
                dot_product = compute_dot_product(y_top,y_bottom,x_left,x_right,x,y,neighbours, kernel)
                if generation % 3 == 2:
                    new_frame[y, x, 2] = slime_activation(dot_product / 255, slime_factor)*red_factor
                if generation % 3 == 1:
                    new_frame[y, x, 1] = slime_activation(dot_product / 255, slime_factor)*green_factor
                if generation % 3 == 0:
                    new_frame[y, x, 0] = slime_activation(dot_product / 255, slime_factor)*blue_factor
                # new_frame[y, x, 2] = slime_activation(dot_product / 255,slime_factor) * 255
        for y in prange(frame.shape[0] // 2 + 1, frame.shape[0], 1):
            for x in prange(frame.shape[1] // 2 + 1, frame.shape[1], 1):
                y_top = max(y - 1, 0)
                y_bottom = min(y + 2, size - 1)
                x_left = max(x - 1, 0)
                x_right = min(x + 2, size - 1)
                # neighbours = frame[y_top:y_bottom, x_left:x_right, 2]
                neighbours = frame[y_top:y_bottom, x_left:x_right, (generation - 1) % 3 if generation > 0 else 0]
                dot_product = compute_dot_product(y_top, y_bottom, x_left, x_right, x, y, neighbours, kernel)
                if generation % 3 == 2:
                    new_frame[y, x, 2] = slime_activation(dot_product / 255, slime_factor) * red_factor
                if generation % 3 == 1:
                    new_frame[y, x, 1] = slime_activation(dot_product / 255, slime_factor) * red_factor
                if generation % 3 == 0:
                    new_frame[y, x, 0] = slime_activation(dot_product / 255, slime_factor) * blue_factor
                # new_frame[y, x, 2] = slime_activation(dot_product / 255,slime_factor) * 255
        for y in prange(frame.shape[0] // 2 + 1, frame.shape[0], 1):
            for x in prange(0, frame.shape[1] // 2, 1):
                y_top = max(y - 1, 0)
                y_bottom = min(y + 2, size - 1)
                x_left = max(x - 1, 0)
                x_right = min(x + 2, size//2)
                # neighbours = frame[y_top:y_bottom, x_left:x_right, 2]
                neighbours = frame[y_top:y_bottom, x_left:x_right, (generation - 1) % 3 if generation > 0 else 0]
                dot_product = compute_dot_product(y_top, y_bottom, x_left, x_right, x, y, neighbours, kernel)
                if generation % 3 == 2:
                    new_frame[y, x, 2] = slime_activation(dot_product / 255, slime_factor) * blue_factor
                if generation % 3 == 1:
                    new_frame[y, x, 1] = slime_activation(dot_product / 255, slime_factor) * red_factor
                if generation % 3 == 0:
                    new_frame[y, x, 0] = slime_activation(dot_product / 255, slime_factor) * red_factor
                # new_frame[y, x, 2] = slime_activation(dot_product / 255,slime_factor) * 255
        for y in prange(0, frame.shape[0] // 2, 1):
            for x in prange(frame.shape[1] // 2 + 1, frame.shape[1], 1):
                y_top = max(y - 1, 0)
                y_bottom = min(y + 2, size//2)
                x_left = max(x - 1, 0)
                x_right = min(x + 2, size - 1)
                # neighbours = frame[y_top:y_bottom, x_left:x_right, 2]
                neighbours = frame[y_top:y_bottom, x_left:x_right, (generation - 1) % 3 if generation > 0 else 0]
                dot_product = compute_dot_product(y_top, y_bottom, x_left, x_right, x, y, neighbours, kernel)
                if generation % 3 == 2:
                    new_frame[y, x, 2] = slime_activation(dot_product / 255, slime_factor) * red_factor
                if generation % 3 == 1:
                    new_frame[y, x, 1] = slime_activation(dot_product / 255, slime_factor) * blue_factor
                if generation % 3 == 0:
                    new_frame[y, x, 0] = slime_activation(dot_product / 255, slime_factor) * red_factor
                # new_frame[y, x, 2] = slime_activation(dot_product / 255,slime_factor) * 255
    else:
        for y in prange(frame.shape[0]):
            for x in prange(frame.shape[1]):
                y_top = max(y - 1, 0)
                y_bottom = min(y + 2, size)
                x_left = max(x - 1, 0)
                x_right = min(x + 2, size)
                #neighbours = frame[y_top:y_bottom, x_left:x_right, 2]
                neighbours = frame[y_top:y_bottom, x_left:x_right, (generation - 1) % 3 if generation > 0 else 0]
                dot_product = compute_dot_product(y_top,y_bottom,x_left,x_right,x,y,neighbours, kernel)
                if generation % 3 == 2:
                    new_frame[y, x, 2] = slime_activation(dot_product / 255, slime_factor)*red_factor
                if generation % 3 == 1:
                    new_frame[y, x, 1] = slime_activation(dot_product / 255, slime_factor)*green_factor
                if generation % 3 == 0:
                    new_frame[y, x, 0] = slime_activation(dot_product / 255, slime_factor)*blue_factor
                # new_frame[y, x, 2] = slime_activation(dot_product / 255,slime_factor) * 255
    return new_frame

@jit(nopython=True)
def inverse_gaussian(x):
    return -1.0 / (0.89 * math.pow(x, 2.0) + 1.0) + 1.0

@jit(nopython=True)
def activation(x):
    if x == 3 or x == 11 or x == 12:
        return 1
    return 0

@jit(nopython=True,target_backend=cuda)
def wave_activation(x):
    return math.fabs(1.2*x)

e = 2.718
@jit(nopython=True,parallel=True)
def tanh_activation(x):
    return (np.power(2.718,2*x)-1)/(np.power(2.718,2*x)+1)

@jit(nopython=True,parallel=True)
def mitosis_activation(x):
    return -1/(0.9*np.power(x,2)+1)+1

@jit(nopython=True,parallel=True)
def gaussian_activation(x,b):
    return 1/np.power(2, (np.power(x-b,2)))


@jit(nopython=True)
def basic_activation(x):
    return x


@jit(nopython=True,parallel=True)
def slime_activation(x,b):
    return 1/np.power(2, (np.power(x-b,2))) #0.5, 0.6 very cool


'''
    Run code
'''


size = 1024
canvas = np.zeros((size, size, 3), dtype=np.uint8)
red_factor = 250
green_factor = 100
blue_factor = 150
#nice 255 200 200
#nice 255 185 200

#canvas[4, 4, 2] = 255
# canvas[size - 4, 4, 2] = 255
# canvas[4, size - 4, 2] = 255
# canvas[size - 4, size - 4, 2] = 255
# canvas[canvas.shape[0]//2 - 9 : canvas.shape[0]//2 + 9, canvas.shape[1]//2 - 9 : canvas.shape[1]//2 + 9, 2] = red_factor
# canvas[canvas.shape[0]//2 - 9 : canvas.shape[0]//2 + 9, canvas.shape[1]//2 - 9 : canvas.shape[1]//2 + 9, 1] = green_factor
# canvas[canvas.shape[0]//2 - 9 : canvas.shape[0]//2 + 9, canvas.shape[1]//2 - 9 : canvas.shape[1]//2 + 9, 0] = blue_factor
#canvas[0 : canvas.shape[0], 0 : canvas.shape[1], 2] = 255
#canvas[canvas.shape[0]//2, canvas.shape[1]//2] = [0,0,red_factor]



# for y in range(canvas.shape[0]):
#     for x in range(canvas.shape[1]):
#         canvas[y, x] = [0,0,random.randint(0, 1)*255]


frame = canvas.copy()
# Create a loop to animate the canvas
#frame = cv2.imread("devil.png")
kernelSign = 1
slime_factor = 0.5
slime_increment = -0.1

while True:
    # Copy the original canvas to work wit
    # Change the color of each pixel randomly
    new_frame1 = compute_new_frame(frame, gen, kernel,slime_factor,red_factor,green_factor,blue_factor)

    # Display the canvas
    frame = new_frame1

    cv2.imshow('Animation', frame)
    # cv2.imwrite('psihedelic4/{f}.png'.format(f=gen), frame)
    gen += 1

    # if gen == 500 :
    #     slime_factor += slime_increment
    #     slime_increment *= -1
        # break

    # if gen%5 == 0:
    #     slime_factor += slime_increment
    #     if gen%500 == 0:
    #         slime_increment *= -1
    #     print(slime_factor)

    # Wait for 25 milliseconds
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # if gen == 2000:
    #     break

# Clean up and exit
cv2.destroyAllWindows()