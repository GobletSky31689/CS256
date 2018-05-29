from PIL import Image
import os
import math
import sys
from helpful_Functions import *

MODEL_FILE_NAME = sys.argv[1]  # "plus_model.txt"
TRAIN_FOLDER_NAME = sys.argv[2]  # "test1"
TEST_FOLDER_NAME = sys.argv[3]  # "test2"


try:

    model_file = file(MODEL_FILE_NAME)
    CLASS_LETTER = model_file.readline().rstrip("\n").strip(" ")
    LAMBDA_VALUE = float(model_file.readline())
    CENTROID_POSITIVE = numpy.asarray(eval(model_file.readline()))
    CENTROID_NEGATIVE = numpy.asarray(eval(model_file.readline()))
    CONSTANT = float(model_file.readline())
    WEIGHTS = eval(model_file.readline())
except IOError:
    print "CAN'T FIND MODEL FILE"
    sys.exit(0)
except ValueError:
    print "MODEL FILE IS NOT OF THE CORRECT FORMAT"
    sys.exit(0)
except:
    print "MODEL FILE IS NOT OF THE CORRECT FORMAT"
    sys.exit(0)


INPUT_DATA_SET = []
POSITIVE_INPUT_INDICES = []
NEGATIVE_INPUT_INDICES = []

TRAIN_DATA_SET = []
POSITIVE_TRAIN_INDICES = []
NEGATIVE_TRAIN_INDICES = []


def generate_train_data():
    """This Function reads the training data set from the TRAIN_FOLDER_NAME"""
    global TRAIN_DATA_SET, POSITIVE_TRAIN_INDICES, NEGATIVE_TRAIN_INDICES, LAMBDA_VALUE, TRAIN_FOLDER_NAME
    if os.path.exists(TRAIN_FOLDER_NAME):
        for f in os.listdir(TRAIN_FOLDER_NAME):
            if f.endswith('.png'):
                numpy_array = numpy.array(Image.open(TRAIN_FOLDER_NAME + '/' + f).convert('1'))
                numpy_array_inverted = numpy.invert(numpy_array.flatten())
                if f.endswith(CLASS_LETTER+'.png'):
                    TRAIN_DATA_SET.append(numpy_array_inverted)
                    POSITIVE_TRAIN_INDICES.append(int(f.split('_')[0])-1)
                else:
                    TRAIN_DATA_SET.append(numpy_array_inverted)
                    NEGATIVE_TRAIN_INDICES.append(int(f.split('_')[0])-1)
generate_train_data()
if len(TRAIN_DATA_SET) == 0:
    print "NO TRAINING DATA"
    sys.exit(0)

# calculate the x_prime vector. We are only interested in that
REDUCED_TRAINED_DATA = calculate_reduced_input_data_set(TRAIN_DATA_SET, POSITIVE_TRAIN_INDICES, NEGATIVE_TRAIN_INDICES,
                                                        CENTROID_POSITIVE, CENTROID_NEGATIVE)


def generate_input_vector():
    """This Function reads the test data set from the TEST_FOLDER_NAME"""
    global POSITIVE_INPUT_INDICES, NEGATIVE_INPUT_INDICES, INPUT_DATA_SET
    if os.path.exists(TEST_FOLDER_NAME):
        for f in os.listdir(TEST_FOLDER_NAME):
            if f.endswith('.png'):
                numpy_array = numpy.array(Image.open(TEST_FOLDER_NAME + '/' + f).convert('1'))
                INPUT_DATA_SET.append(numpy.invert(numpy_array.flatten()))
                if f.endswith(CLASS_LETTER+'.png'):
                    POSITIVE_INPUT_INDICES.append(int(f.split('_')[0])-1)
                else:
                    NEGATIVE_INPUT_INDICES.append(int(f.split('_')[0])-1)
generate_input_vector()
if len(INPUT_DATA_SET) == 0:
    print "NO TESTING DATA"
    sys.exit(0)


def kernel_function(vector_1, vector_2):
    """This Function calculates the polynomial kernel with degree 4"""
    x = vector_1.dot(vector_2) + 1.0
    return math.pow(x, 4)


# print "Positive Indices Length:", len(POSITIVE_INPUT_INDICES)
# print "Negative Indices Length", len(NEGATIVE_INPUT_INDICES)
# print "Total Data set Length : ", len(INPUT_DATA_SET)
# Here, we start testing out SVM model
correct_predictions = 0
false_negatives = 0
false_positives = 0
for index in xrange(len(INPUT_DATA_SET)):
    sum_value = 0
    for i in xrange(len(WEIGHTS)):
        if WEIGHTS[i][1] in POSITIVE_TRAIN_INDICES:
            sum_value = sum_value + WEIGHTS[i][0] * kernel_function(REDUCED_TRAINED_DATA[WEIGHTS[i][1]],
                                                                    INPUT_DATA_SET[index])
        else:
            sum_value = sum_value - WEIGHTS[i][0] * kernel_function(REDUCED_TRAINED_DATA[WEIGHTS[i][1]],
                                                                    INPUT_DATA_SET[index])
    if sum_value + CONSTANT > 0:
        prediction = True
    else:
        prediction = False
    actual = index in POSITIVE_INPUT_INDICES
    if prediction == actual:
        print str(index+1) + " Correct"
        correct_predictions += 1
    elif prediction:
        print str(index+1) + " False Positive"
        false_positives += 1
    else:
        print str(index+1) + " False Negative"
        false_negatives += 1


print "Fraction Correct: " + str(1.0*correct_predictions/len(INPUT_DATA_SET))
print "Fraction False Positive: " + str(1.0*false_positives/len(INPUT_DATA_SET))
print "Fraction False Negative: " + str(1.0*false_negatives/len(INPUT_DATA_SET))
