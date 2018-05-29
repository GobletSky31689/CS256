# Sample Command: python sticky_snippet_net.py mode model_file data_folder

import sys
import tensorflow as tf
import os
import helpful_functions
import random
import numpy as np;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODE = sys.argv[1]
MODE_TRAIN = 'train'
MODE_FOLD = '5fold'
MODE_TEST = 'test'

MODEL_FILE = sys.argv[2]

DATA_FOLDER = sys.argv[3]

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


def get_input_data():
    """This Function generates the input data set by reading the files in DATA_FOLDER"""
    global INPUT_DATA_SET, OUTPUT_DATA_SET
    if os.path.exists(DATA_FOLDER):
        for f in os.listdir(DATA_FOLDER):
            if f.endswith('.txt'):
                data_file = file(DATA_FOLDER + "/" + f)
                for line in data_file.readlines():
                    INPUT_DATA_SET.append([GENOME_DICT[bit] for bit in line.strip("\n")])
                    OUTPUT_DATA_SET.append(y_mapping[helpful_functions.get_sticky_class(line.strip("\n"))])
                    CLASS_COUNT[helpful_functions.get_sticky_class(line.strip("\n"))] += 1
get_input_data()
print "Class Distribution", CLASS_COUNT
print INPUT_DATA_SET
print OUTPUT_DATA_SET
def perceptron(weights, inputs, biases, activation):
    nodes = tf.matmul(inputs, weights) + biases
    return activation(nodes)


# We initialize W as [[0, 0], [0, 0]]
# First 40 means length of input vector is 40, second 6 means no of classes are six
W1 = tf.get_variable("W1",shape=[40,6], dtype=tf.float32, initializer=tf.zeros_initializer)
W_R  = tf.get_variable("W2", shape=[4,6,6], dtype=tf.float32, initializer=tf.random_uniform_initializer)
B_R = tf.get_variable("B2", shape=[4,1,6], dtype=tf.float32, initializer=tf.zeros_initializer)
# We initialize THETA as [0, 0]
# Here, 6 means no of classes are two (STICKY, NON-STICKY)
# [None, 40] means length of each input vector is 40 and there can be any no of vectors
# x = tf.placeholder(tf.float32, [None, 40])
x = tf.placeholder(dtype=tf.float32, shape=[1,40]);
# [None, 6] means length of each target vector is 6 because there can be six classes
B1 = tf.get_variable("B1", shape=[1,6], dtype=tf.float32, initializer=tf.random_uniform_initializer)
init  = tf.global_variables_initializer()
y = tf.placeholder(dtype=tf.float32,shape=[1,6]);
sess = tf.Session();
sess.run(init)
def calculatePercetron(Input,Output):
    global W_R, W1, B_R, B1,sess,x,y;
    layer1 = perceptron(W1, x, biases=B1, activation=tf.nn.relu);
    # print "Layer 1 : ",sess.run(layer1)
    layer2 = perceptron(W_R[0], layer1, B_R[0], tf.nn.relu)
    # print "Layer 2 : ",sess.run(layer2)
    layer3 = perceptron(W_R[1], layer2, B_R[1], tf.nn.relu)
    # print "Layer 3 : ",sess.run(layer3)
    layer4 = perceptron(W_R[2], layer3, B_R[2], tf.nn.relu)
    # print "Layer 4 : ",sess.run(layer4)
    layer5 = perceptron(W_R[3], layer4, B_R[3], tf.nn.softmax)
    # print "Layer 5 : ",sess.run(layer5)
    loss = tf.reduce_mean(tf.square(tf.subtract(tf.argmax(layer5,1), tf.argmax(y,1))))
    # print "Loss : ", sess.run(loss)
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    for i in range(len(Input)):
        # print (Input[i]).reshape(1,40).shape
        print
        print
        print "Iteration :",i+1
        print sess.run([train_step],feed_dict={x:(Input[i]).reshape(1,40), y:(Output[i]).reshape(1,6)})

    # sess.run(train_step)

    print ""
    # curr_W, curr_b, curr_loss = sess.run([W, THETA, loss], {x: x_train, y: y_train})
    # print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

calculatePercetron(np.asarray(INPUT_DATA_SET).astype(float), np.asarray(OUTPUT_DATA_SET).astype(float));
# for i in range(len(INPUT_DATA_SET)):
#     print
#     print
#     print "Iteration ", i
#     print np.asarray(INPUT_DATA_SET[i]).reshape(1,40)
#     calculatePercetron(np.asarray(INPUT_DATA_SET[i]).astype(float).reshape(1,40), np.asarray(OUTPUT_DATA_SET[i]).astype(int).reshape(1,6))
    # print (np.asarray(INPUT_DATA_SET[1])).shape
    # print len(INPUT_DATA_SET)

        # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("/Users/Gaurav/Desktop/temp", sess.graph)




# print sess.run(layer_5, {x: x_train, y: y_train})
# correct_prediction = tf.equal(tf.argmax(layer_5, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, {x: x_train, y: y_train}))

