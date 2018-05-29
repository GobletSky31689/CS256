import tensorflow as tf
import os
from helpful_functions import randomize_data, get_sticky_class
import numpy as np
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


A = 1
B = 2
C = 3
D = 4

GENOME_DICT = {
    'A': A,
    'B': B,
    'C': C,
    'D': D
}

y_mapping = {
    0: [1, 0, 0, 0, 0, 0],
    1: [0, 1, 0, 0, 0, 0],
    2: [0, 0, 1, 0, 0, 0],
    3: [0, 0, 0, 1, 0, 0],
    4: [0, 0, 0, 0, 1, 0],
    5: [0, 0, 0, 0, 0, 1]
}

INPUT_DATA_SET = []
OUTPUT_DATA_SET = []
CLASS_COUNT = [0]*6
DATA_FOLDER = ""
MODEL_FILE = ""


def get_input_data():
    """This Function generates the input data set by reading the files in DATA_FOLDER"""
    global INPUT_DATA_SET, OUTPUT_DATA_SET
    if os.path.exists(DATA_FOLDER):
        for f in os.listdir(DATA_FOLDER):
            if f.endswith('.txt'):
                data_file = file(DATA_FOLDER + "/" + f)
                for line in data_file.readlines():
                    INPUT_DATA_SET.append([GENOME_DICT[bit] for bit in line.strip("\n")])
                    OUTPUT_DATA_SET.append(y_mapping[get_sticky_class(line.strip("\n"))])
                    CLASS_COUNT[get_sticky_class(line.strip("\n"))] += 1


def perceptron(wts, inputs, bias, activation):
    nodes = tf.matmul(inputs, wts) + bias
    return activation(nodes)


learning_rate = 0.05
learning_algo = tf.train.GradientDescentOptimizer
weight_initializer = tf.contrib.layers.xavier_initializer()
bias_initializer = tf.constant_initializer(0.1)

num_of_input_bits = 40
num_of_output_bits = 6
num_neurons_layer_1 = 100
num_neurons_layer_2 = 100
num_neurons_layer_3 = 100
num_neurons_layer_4 = 100


with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, [None, num_of_input_bits])


with tf.variable_scope('layer_1'):
    weights = tf.get_variable("weights_1", shape=[num_of_input_bits, num_neurons_layer_1],
                              initializer=weight_initializer)
    biases = tf.get_variable("biases_1", shape=[num_neurons_layer_1],
                             initializer=tf.zeros_initializer)
    output_layer_1 = perceptron(weights, X, biases, tf.nn.relu)


with tf.variable_scope('layer_2'):
    weights = tf.get_variable("weights_2", shape=[num_neurons_layer_1, num_neurons_layer_2],
                              initializer=weight_initializer)
    biases = tf.get_variable("biases_2", shape=[num_neurons_layer_2],
                             initializer=tf.zeros_initializer)
    output_layer_2 = perceptron(weights, output_layer_1, biases, tf.nn.relu)


with tf.variable_scope('layer_3'):
    weights = tf.get_variable("weights_3", shape=[num_neurons_layer_2, num_neurons_layer_3],
                              initializer=weight_initializer)
    biases = tf.get_variable("biases_3", shape=[num_neurons_layer_3],
                             initializer=tf.zeros_initializer)
    output_layer_3 = perceptron(weights, output_layer_2, biases, tf.nn.relu)


with tf.variable_scope('layer_4'):
    weights = tf.get_variable("weights_4", shape=[num_neurons_layer_3, num_neurons_layer_4],
                              initializer=weight_initializer)
    biases = tf.get_variable("biases_4", shape=[num_neurons_layer_4],
                             initializer=tf.zeros_initializer)
    output_layer_4 = perceptron(weights, output_layer_3, biases, tf.nn.relu)


with tf.variable_scope('layer_5'):
    weights = tf.get_variable("weights_5", shape=[num_neurons_layer_4, num_of_output_bits],
                              initializer=weight_initializer)
    biases = tf.get_variable("biases_5", shape=[num_of_output_bits],
                             initializer=tf.zeros_initializer)
    final_output = perceptron(weights, output_layer_4, biases, tf.identity)


with tf.variable_scope('loss'):
    Y = tf.placeholder(tf.float32, [None, num_of_output_bits])
    # loss = tf.reduce_mean(tf.squared_difference(final_output, Y))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=final_output))


with tf.variable_scope('train'):
    train_step = learning_algo(learning_rate).minimize(loss)

prediction = tf.nn.softmax(final_output)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

mini_batch_size = 100
epoch_count = 20


def train(folder, output_file=None):
    global DATA_FOLDER, MODEL_FILE
    DATA_FOLDER = folder
    MODEL_FILE = output_file
    get_input_data()
    print "Class Distribution", CLASS_COUNT

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        total_items_processed = 0

        for ep in range(epoch_count):
            randomize_data(INPUT_DATA_SET, OUTPUT_DATA_SET)

            for k in range(0, len(INPUT_DATA_SET), mini_batch_size):
                current_batch_x = INPUT_DATA_SET[k:k+mini_batch_size]
                current_batch_y = OUTPUT_DATA_SET[k:k+mini_batch_size]
                session.run(train_step, {X: current_batch_x, Y: current_batch_y})
                total_items_processed += mini_batch_size
                if total_items_processed % 1000 == 0 and output_file:
                    print total_items_processed, "items processed"
            remainder = len(INPUT_DATA_SET) % 50
            if remainder != 0:
                current_batch_x = INPUT_DATA_SET[len(INPUT_DATA_SET)-remainder:]
                current_batch_y = OUTPUT_DATA_SET[len(INPUT_DATA_SET)-remainder:]
                session.run(train_step, {X: current_batch_x, Y: current_batch_y})
                total_items_processed += remainder
                if total_items_processed % 1000 == 0 and output_file:
                    print total_items_processed, "items processed"
            # print "Epoch", ep+1, "Processing complete!"
        if output_file:
            print "Processing complete!"
            print "No of items trained on:", total_items_processed

            # Fetch and print final weights from the global scope
            with tf.variable_scope('', reuse=True):
                saver = tf.train.Saver()
                saver.save(session, MODEL_FILE)


def test(model, folder):
    global DATA_FOLDER, MODEL_FILE
    DATA_FOLDER = folder
    MODEL_FILE = model
    get_input_data()
    print "Class Distribution", CLASS_COUNT

    with tf.Session() as session:
        with tf.variable_scope('', reuse=True):
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(session, MODEL_FILE)
            start_time = datetime.now()

            acc, pred, out = session.run([accuracy,
                                          tf.argmax(prediction, 1),
                                          tf.argmax(Y, 1)],
                                         {X: INPUT_DATA_SET, Y: OUTPUT_DATA_SET})
            print "Overall Accuracy:",  acc
            print "Confusion Matrix:\n", tf.confusion_matrix(out, pred).eval(), "seconds"
            print "Total testing time:", (datetime.now() - start_time).total_seconds(), "seconds"


def cross_validate_and_train(folder, output_file=None):
    global DATA_FOLDER, MODEL_FILE
    DATA_FOLDER = folder
    MODEL_FILE = output_file
    get_input_data()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        block = len(INPUT_DATA_SET) / 5

        accuracy_log = []

        total_items_processed = 0

        prediction_log = []
        output_log = []
        start_time = datetime.now()

        for ep in range(epoch_count):
            randomize_data(INPUT_DATA_SET, OUTPUT_DATA_SET)
            for i in range(5):
                x_test = INPUT_DATA_SET[i * block: (i + 1) * block]
                y_test = OUTPUT_DATA_SET[i * block: (i + 1) * block]
                x_train = np.delete(INPUT_DATA_SET, np.s_[i * block: (i + 1) * block], axis=0)
                y_train = np.delete(OUTPUT_DATA_SET, np.s_[i * block: (i + 1) * block], axis=0)

                for k in range(0, len(x_train), mini_batch_size):
                    current_batch_x = x_train[k:k+mini_batch_size]
                    current_batch_y = y_train[k:k+mini_batch_size]
                    session.run(train_step, {X: current_batch_x, Y: current_batch_y})
                    total_items_processed += mini_batch_size
                    if total_items_processed % 1000 == 0 and output_file:
                        print total_items_processed, "items processed"
                remainder = len(x_train) % 50
                if remainder != 0:
                    current_batch_x = x_train[len(x_train)-remainder:]
                    current_batch_y = y_train[len(y_train)-remainder:]
                    session.run(train_step, {X: current_batch_x, Y: current_batch_y})
                    total_items_processed += remainder
                    if total_items_processed % 1000 == 0 and output_file:
                        print total_items_processed, "items processed"
                fold_accuracy, fold_pred, fold_out = session.run([accuracy,
                                                                  tf.argmax(prediction, 1),
                                                                  tf.argmax(Y, 1)],
                                                                 {X: x_test, Y: y_test})
                accuracy_log.append(fold_accuracy)
                prediction_log.extend(fold_pred)
                output_log.extend(fold_out)

        print "Total training time:", (datetime.now() - start_time).total_seconds(), "seconds"

        start_time = datetime.now()

        total_accuracy = 0
        for acc in accuracy_log:
            total_accuracy += acc
        total_accuracy = total_accuracy / len(accuracy_log)
        print "Overall Accuracy:", total_accuracy
        print "Confusion Matrix:\n", tf.confusion_matrix(output_log, prediction_log).eval()

        print "Total testing time:", (datetime.now() - start_time).total_seconds(), "seconds"

        if output_file:
            # Fetch and print final weights from the global scope
            with tf.variable_scope('', reuse=True):
                saver = tf.train.Saver()
                saver.save(session, MODEL_FILE)
