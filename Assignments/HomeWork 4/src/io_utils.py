import numpy as np
from PIL import Image
import os
import random
import sys


def generate_input_vector(folder_name, class_letter):
    """This Function generated the input data set by reading the files in TRAIN_FOLDER_NAME"""
    inp = []
    output = []
    class_count = 0
    if os.path.exists(folder_name):
        for f in os.listdir(folder_name):
            if f.endswith('.png'):
                numpy_array = np.array(Image.open(folder_name + "/" + f).convert('L'), 'f')
                inp.append(numpy_array.flatten())
                if f.endswith(class_letter+'.png'):
                    output.append([1, 0])
                    class_count += 1
                else:
                    output.append([0, 1])
    print "Class Count:", class_count, "out of", len(output)
    return np.array(inp), np.array(output)


def __swap(index_i, index_j, arr):
    """The function swaps elements at index ix & ij for both arrays ax & ay"""
    arr[index_i], arr[index_j] = arr[index_j].copy(), arr[index_i].copy()


def randomize_data(arr_a, arr_b):
    """The function randomizes the array"""
    for ix in range(0, len(arr_a)-1):
        j = random.randint(ix+1, len(arr_a)-1)
        __swap(ix, j, arr_a)
        __swap(ix, j, arr_b)


def read_network_file(file_name):
    if os.path.isfile(file_name):
        network_file = open(file_name, "r")
        layers = []
        for line in network_file:
            layers.append(line.rstrip('\n'))
        return layers
    else:
        print "Invalid network file."
        sys.exit(0)


def write_results_to_file(file_name, results):
    train_file = open(file_name, "w")
    train_file.write('\n'.join(map(str, results)))
    train_file.close()
