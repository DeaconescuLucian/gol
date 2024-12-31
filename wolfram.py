import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def wolfram_rule(rule_number, iterations, start, startDecimal):
    # Convert rule number to binary and pad with zeros
    rule_binary = format(rule_number, '08b')

    # Create dictionary of neighborhood patterns and their corresponding output
    patterns = {'111': rule_binary[0], '110': rule_binary[1], '101': rule_binary[2],
                '100': rule_binary[3], '011': rule_binary[4], '010': rule_binary[5],
                '001': rule_binary[6], '000': rule_binary[7]}

    # Initialize array with first row as all zeros except for the center which is 1
    size = (iterations, iterations * 2 + 1)
    arr = np.zeros(size)
    arr[0, iterations-1:iterations+2] = start

    #Create a function to generate the next frame of the animation
    for i in range(1, iterations):
        for j in range(1, iterations * 2):
            neighborhood = ''.join(str(int(x)) for x in arr[i-1, j - 1:j + 2])
            arr[i, j] = int(patterns[neighborhood])
    img = np.uint8(arr * 255)
    # Plot the Wolfram rule
    cv2.imwrite('wolfram{}.{}/rule{}.jpg'.format(3,startDecimal,rule_number), img)

# Test the function
for j in range (8):
    start = np.zeros(3)
    start[0] = int(format(j, '03b')[0])
    start[1] = int(format(j, '03b')[1])
    start[2] = int(format(j, '03b')[2])
    for i in tqdm(range(256)):
        wolfram_rule(30, 100,start, j)

