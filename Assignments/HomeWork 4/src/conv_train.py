# python conv_train.py cost network_description epsilon max_updates class_letter model_file_name train_folder_name
import tensorflow as tf
import numpy as np
from io_utils import generate_input_vector, randomize_data, read_network_file  # , write_results_to_file
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CROSS_ONLY = "cross"
CROSS_L1 = "cross-l1"
CROSS_L2 = "cross-l2"
CROSS_TEST = "ctest"


COST_FUNC = sys.argv[1]
NETWORK_DESC_FILE = sys.argv[2]
EPSILON = float(sys.argv[3])
MAX_UPDATES = int(sys.argv[4])
CLASS_LETTER = sys.argv[5]
MODEL_FILE = sys.argv[6]
FOLDER_NAME = sys.argv[7]

# EXPERIMENT_FILE_NAME = "Experiment 3/temp/cross-l2.txt"


TRAIN_FOLDER_NAME = FOLDER_NAME
# TEST_FOLDER_NAME = "DataSets/testCircle1000"

batch_size = 128
x = tf.placeholder('float', [None, 625])
y = tf.placeholder('float')

regularization_alpha = 1


def conv2d(x_image, w_con):
    return tf.nn.conv2d(x_image, w_con, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x_image):
    return tf.nn.max_pool(x_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(layers):

    conv = tf.reshape(x, [-1, 25, 25, 1])
    input_size = 1
    regularizers = 0
    image_size = 25
    layer_count = '1'
    for item in layers:
        layer = item.split(" ")
        filter_size = int(layer[0])
        if len(layer) == 2:
            feature_size = int(layer[1])
            weight = tf.Variable(tf.random_normal([filter_size, filter_size, input_size, feature_size],
                                                  name='w'+layer_count))
            bias = tf.Variable(tf.random_normal([feature_size]),
                               name='b'+layer_count)
            conv = tf.nn.relu(conv2d(conv, weight) + bias)
            # conv = maxpool2d(conv)
            input_size = feature_size
            # image_size = (image_size + 1) / 2
            layer_count = str(int(layer_count) + 1)
            if COST_FUNC == CROSS_L2:
                regularizers += tf.nn.l2_loss(weight)
            elif COST_FUNC == CROSS_L1:
                regularizers += tf.reduce_sum(tf.abs(weight))
        else:
            break

    num_units = int(layers[-1])
    input_fc = tf.reshape(conv, [-1, image_size * image_size * input_size])
    weight = tf.Variable(tf.random_normal([image_size * image_size * input_size, num_units]), name='wfc')
    bias = tf.Variable(tf.random_normal([num_units]), name='bfc')
    fc = tf.nn.relu(tf.matmul(input_fc, weight) + bias)
    # fc = tf.nn.dropout(fc, keep_rate)
    if COST_FUNC == CROSS_L2:
        regularizers += tf.nn.l2_loss(weight)
    elif COST_FUNC == CROSS_L1:
        regularizers += tf.reduce_sum(tf.abs(weight))

    weight = tf.Variable(tf.random_normal([num_units, 2]), name='wo')
    bias = tf.Variable(tf.constant([0.5, 0.5]), name='bo')
    if COST_FUNC == CROSS_L2:
        regularizers += tf.nn.l2_loss(weight)
    elif COST_FUNC == CROSS_L1:
        regularizers += tf.reduce_sum(tf.abs(weight))
    output = tf.matmul(fc, weight) + bias

    return output, regularizers


def train_neural_network():
    layers = read_network_file(NETWORK_DESC_FILE)

    print "Building network"
    prediction, regularizers = convolutional_neural_network(layers)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    regularized_cost = cost + regularization_alpha * regularizers
    optimizer = tf.train.AdamOptimizer().minimize(regularized_cost)
    print "Network built"

    print "Reading data"
    training_input, training_output = generate_input_vector(FOLDER_NAME, CLASS_LETTER)
    # testing_input, testing_output = generate_input_vector(TEST_FOLDER_NAME, CLASS_LETTER)
    # print "Data read", "Training size:", len(training_input), "Testing size:", len(testing_input)
    # results = [("Training size:", len(training_input),
    #             "Testing size:", len(testing_input))]
    print "Start training"
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(MAX_UPDATES):
            epoch_train_cost = 0
            epoch_validation_cost = 0
            randomize_data(training_input, training_output)
            block = len(training_input) / 5
            for i in range(5):
                print '.',
                x_validate = training_input[i * block: (i + 1) * block]
                y_validate = training_output[i * block: (i + 1) * block]
                x_train = np.delete(training_input, np.s_[i * block: (i + 1) * block], axis=0)
                y_train = np.delete(training_output, np.s_[i * block: (i + 1) * block], axis=0)
                for k in range(0, len(x_train), batch_size):
                    current_batch_x = x_train[k:k+batch_size]
                    current_batch_y = y_train[k:k+batch_size]
                    sess.run(optimizer, {x: current_batch_x, y: current_batch_y})
                remainder = len(x_train) % batch_size
                if remainder != 0:
                    current_batch_x = x_train[len(x_train)-remainder:]
                    current_batch_y = y_train[len(y_train)-remainder:]
                    sess.run(optimizer, {x: current_batch_x, y: current_batch_y})
                epoch_train_cost += sess.run(cost, {x: x_train, y: y_train})
                epoch_validation_cost += sess.run(cost, {x: x_validate, y: y_validate})
            print

            # d, dd = sess.run([prediction, tf.nn.softmax(prediction)],  {x: x_validate[0:3], y: y_validate[0:3]})
            # print "Prediction:", d[0], dd[0], y_validate[0]
            # print "Prediction:", d[1], dd[1], y_validate[1]
            # print "Prediction:", d[2], dd[2], y_validate[2]

            print('Epoch', epoch+1, 'completed out of', MAX_UPDATES,
                  'loss:', (epoch_train_cost/5, epoch_validation_cost/5))
            # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            # correct = tf.equal(tf.nn.softmax(prediction), y)
            # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            # test_accuracy = accuracy.eval({x: testing_input, y: testing_output})
            # train_accuracy = accuracy.eval({x: training_input, y: training_output})
            # print('Accuracy on train:', train_accuracy)
            # print('Accuracy on test:', test_accuracy)
            # results.append((epoch+1, epoch_train_cost/5, epoch_validation_cost/5, train_accuracy, test_accuracy))
            if float(epoch_validation_cost/5) < 0.1:
                break

        saver.save(sess, MODEL_FILE)
        sess.close()

    # write_results_to_file(EXPERIMENT_FILE_NAME, results)
    print "Training Complete"


def test_neural_network():
    testing_input, testing_output = generate_input_vector(FOLDER_NAME, CLASS_LETTER)

    layers = read_network_file(NETWORK_DESC_FILE)

    print "Building network"
    prediction, _ = convolutional_neural_network(layers)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print "Network built"

    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, MODEL_FILE)
        print "Model restored"
        print "Start testing"
        acc = accuracy.eval({x: testing_input, y: testing_output})
        print "Testing Complete"
        print "Overall Accuracy:", acc


if COST_FUNC is not CROSS_TEST:
    train_neural_network()
else:
    test_neural_network()
