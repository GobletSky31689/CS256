from PIL import Image
import os
import math
import sys
from helpful_Functions import *

EPSILON = float(sys.argv[1])  # 0.5
MAX_UPDATES = int(sys.argv[2])  # 1000000
CLASS_LETTER = sys.argv[3]  # "C"
MODEL_FILE_NAME = sys.argv[4]  # "plus_model.txt"
TRAIN_FOLDER_NAME = sys.argv[5]  # "test1"


INPUT_DATA_SET = []
POSITIVE_INPUT_INDICES = []
NEGATIVE_INPUT_INDICES = []

CENTROID_POSITIVE = []
CENTROID_NEGATIVE = []
LAMBDA_VALUE = 0


def generate_input_vector():
    """This Function generated the input data set by reading the files in TRAIN_FOLDER_NAME"""
    global POSITIVE_INPUT_INDICES, NEGATIVE_INPUT_INDICES, INPUT_DATA_SET
    if os.path.exists(TRAIN_FOLDER_NAME):
        for f in os.listdir(TRAIN_FOLDER_NAME):
            if f.endswith('.png'):
                numpy_array = numpy.array(Image.open(TRAIN_FOLDER_NAME + "/" + f).convert('L'), 'f')
                INPUT_DATA_SET.append(numpy_array.flatten())
                if f.endswith(CLASS_LETTER+'.png'):
                    POSITIVE_INPUT_INDICES.append(int(f.split('_')[0])-1)
                else:
                    NEGATIVE_INPUT_INDICES.append(int(f.split('_')[0])-1)
generate_input_vector()
if len(INPUT_DATA_SET) == 0:
    print "NO DATA"
    sys.exit(0)
print 'here'


def calculate_reduced_vectors():
    """This Function calculates the x_prime or reduced vector, using helpful_functions.py"""
    global CENTROID_POSITIVE, CENTROID_NEGATIVE, LAMBDA_VALUE, INPUT_DATA_SET
    global POSITIVE_INPUT_INDICES, NEGATIVE_INPUT_INDICES
    CENTROID_POSITIVE = get_centroid_vector(INPUT_DATA_SET, POSITIVE_INPUT_INDICES)
    CENTROID_NEGATIVE = get_centroid_vector(INPUT_DATA_SET, NEGATIVE_INPUT_INDICES)
    LAMBDA_VALUE = calculate_lambda(CENTROID_POSITIVE, CENTROID_NEGATIVE, POSITIVE_INPUT_INDICES,
                                    NEGATIVE_INPUT_INDICES, INPUT_DATA_SET)
    reduced_set = calculate_reduced_input_data_set(INPUT_DATA_SET, POSITIVE_INPUT_INDICES,
                                                   NEGATIVE_INPUT_INDICES, CENTROID_POSITIVE, CENTROID_NEGATIVE)
    return reduced_set
REDUCED_INPUT_DATA_SET = calculate_reduced_vectors()


def kernel_function(vector_1, vector_2):
    """This Function calculates the polynomial kernel"""
    x = vector_1.dot(vector_2) + 1.0
    return math.pow(x, 4)


# Here we initialize the A, B, C, D, E and alpha values
first_positive_reduced_vector = REDUCED_INPUT_DATA_SET[POSITIVE_INPUT_INDICES[0]]
first_negative_reduced_vector = REDUCED_INPUT_DATA_SET[NEGATIVE_INPUT_INDICES[0]]
A = kernel_function(first_positive_reduced_vector, first_positive_reduced_vector)
B = kernel_function(first_negative_reduced_vector, first_negative_reduced_vector)
C = kernel_function(first_positive_reduced_vector, first_negative_reduced_vector)
D_VECTOR = [0] * len(REDUCED_INPUT_DATA_SET)
E_VECTOR = [0] * len(REDUCED_INPUT_DATA_SET)
for ind in xrange(len(REDUCED_INPUT_DATA_SET)):
    D_VECTOR[ind] = kernel_function(REDUCED_INPUT_DATA_SET[ind], first_positive_reduced_vector)
    E_VECTOR[ind] = kernel_function(REDUCED_INPUT_DATA_SET[ind], first_negative_reduced_vector)
alpha_values = [0] * len(REDUCED_INPUT_DATA_SET)
alpha_values[POSITIVE_INPUT_INDICES[0]] = 1
alpha_values[NEGATIVE_INPUT_INDICES[0]] = 1


def do_positive_adaptation_step(t):
    """This Function performs the adaption when t index is in POSITIVE_INPUT_INDICES"""
    global A, C, D_VECTOR, REDUCED_INPUT_DATA_SET, E_VECTOR, POSITIVE_INPUT_INDICES, alpha_values
    new_support_vector = REDUCED_INPUT_DATA_SET[t]
    numerator = A - D_VECTOR[t] + E_VECTOR[t] - C
    kernel_min_val = kernel_function(new_support_vector, new_support_vector)
    den = A + kernel_min_val - (2.0 * (D_VECTOR[t] - E_VECTOR[t]))
    q = min(1.0, numerator / den)
    C = (1.0 - q) * C + q * E_VECTOR[t]
    A = (A * math.pow(1.0 - q, 2.0)) + (2.0 * (1.0 - q) * q * D_VECTOR[t]) + (math.pow(q, 2.0) * kernel_min_val)
    for i in POSITIVE_INPUT_INDICES:
        alpha_values[i] = (1.0 - q) * alpha_values[i]
    alpha_values[t] += q
    D_VECTOR = [(1 - q) * D_VECTOR[i] + q * kernel_function(REDUCED_INPUT_DATA_SET[i], new_support_vector) for i in xrange(len(REDUCED_INPUT_DATA_SET))]


def do_negative_adaptation_step(t):
    """This Function performs the adaption when t index is in NEGATIVE_INPUT_INDICES"""
    global B, C, E_VECTOR, REDUCED_INPUT_DATA_SET, D_VECTOR, alpha_values, NEGATIVE_INPUT_INDICES
    new_support_vector = REDUCED_INPUT_DATA_SET[t]
    numerator = B - E_VECTOR[t] + D_VECTOR[t] - C
    kernel_min_val = kernel_function(new_support_vector, new_support_vector)
    den = B + kernel_min_val - 2.0 * (E_VECTOR[t] - D_VECTOR[t])
    q = min(1.0, numerator / den)
    C = (1.0 - q) * C + q * D_VECTOR[t]
    B = (B * math.pow(1.0 - q, 2.0)) + (2.0 * (1.0 - q) * q * E_VECTOR[t]) + (math.pow(q, 2.0) * kernel_min_val)
    for i in NEGATIVE_INPUT_INDICES:
        alpha_values[i] = (1.0 - q) * alpha_values[i]
    alpha_values[t] += q
    E_VECTOR = [(1 - q) * E_VECTOR[i] + q * kernel_function(REDUCED_INPUT_DATA_SET[i], new_support_vector) for i in xrange(len(REDUCED_INPUT_DATA_SET))]





# Here, we start the training of our SVM
# We train till our SVM is EPSILON accurate
# or till MAX_UPDATES are performed, whichever earlier

num_updates = 0
while num_updates < MAX_UPDATES:
    print num_updates
    denominator = math.sqrt(A + B - 2.0 * C)
    min_positive = (D_VECTOR[POSITIVE_INPUT_INDICES[0]] - E_VECTOR[POSITIVE_INPUT_INDICES[0]] + B - C) / denominator
    min_positive_index = 0
    min_negative = (E_VECTOR[NEGATIVE_INPUT_INDICES[0]] - D_VECTOR[NEGATIVE_INPUT_INDICES[0]] + A - C) / denominator
    min_negative_index = 0
    for index in POSITIVE_INPUT_INDICES:
        if denominator != 0:
            new_val = (D_VECTOR[index] - E_VECTOR[index] + B - C) / denominator
            if min_positive > new_val:
                min_positive = new_val
                min_positive_index = index
    for index in NEGATIVE_INPUT_INDICES:
        if denominator != 0:
            new_val = (E_VECTOR[index] - D_VECTOR[index] + A - C) / denominator
            if min_negative > new_val:
                min_negative = new_val
                min_negative_index = index
    min_val = min(min_negative, min_positive)
    if denominator - min_val < EPSILON:
        print A, B, alpha_values
        break
    else:
        num_updates += 1
        if min_val == min_positive:
            do_positive_adaptation_step(min_positive_index)
        else:
            do_negative_adaptation_step(min_negative_index)

# Here, our SVM has been trained
# We will now write this trained model in MODEL_FILE_NAME
out_file = file(MODEL_FILE_NAME, 'w')
out_file.write(CLASS_LETTER+"\n")
out_file.write(str(LAMBDA_VALUE) + "\n")
out_file.write(str(CENTROID_POSITIVE) + "\n")
out_file.write(str(CENTROID_NEGATIVE) + "\n")
out_file.write(str(((B-A)/2))+"\n")
FINAL_WEIGHTS = []
for ind in xrange(len(alpha_values)):
    if alpha_values[ind] != 0:
        FINAL_WEIGHTS.append((alpha_values[ind], ind))
out_file.write(str(FINAL_WEIGHTS)+"\n")
out_file.close()
