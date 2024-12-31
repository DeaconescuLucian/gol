import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import keyboard
import cv2
from tqdm import tqdm

def check_ones_zeros(rule_number, no_ones):
    binary = format(rule_number, '032b')
    ones = binary.count('1')
    return ones == no_ones

def calculate_percent_ones(arr):
    size = arr.size
    num_ones = np.sum(arr == 1)
    percent_ones = num_ones / size * 100
    return percent_ones

def check_if_arr_is_usefull(arr):
    if calculate_percent_ones(arr) < 1:
        return False
    else:
        return True


def wolfram_rule(rule_number, iterations, no_ones, neighborhood_size=5):
    # Convert rule number to binary and pad with zeros
    rule_binary = format(rule_number, '0{}b'.format(2**neighborhood_size))
    patterns = {}
    # Create dictionary of neighborhood patterns and their corresponding output
    for i in range(2 ** neighborhood_size):
        pattern_binary = format(i, '0{}b'.format(neighborhood_size))
        pattern = ''.join(['1' if x == '1' else '0' for x in pattern_binary])
        output = rule_binary[i]
        patterns[pattern] = output

    # Initialize array with first row as all zeros except for the center which is 1
    size = (iterations, iterations * 2 + 1)
    arr = np.zeros(size)
    arr[0, iterations] = 1

    for i in range(1, iterations):
        for j in range(neighborhood_size//2, iterations * 2 - neighborhood_size//2):
            neighborhood = ''.join(str(int(x)) for x in arr[i - 1, j - neighborhood_size//2:j + neighborhood_size//2 + 1])
            arr[i, j] = int(patterns[neighborhood])
    if check_if_arr_is_usefull(arr):
        img = np.uint8(arr * 255)
        cv2.imwrite('wolfram{}.{}/rule{}.jpg'.format(neighborhood_size, no_ones, rule_number), img)

r = 15
while r <= 30:
    print("Generating for r={}".format(r))
    for k in tqdm(range(2**25 - 2**8,2**28, 2**8)):
        if check_ones_zeros(k, r):
            wolfram_rule(k, 100, r)
    r+=1

