import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def perceptron(weights, inputs, biases, activation):
    nodes = tf.reduce_sum(weights * inputs, 1) + biases
    return activation(nodes)


def step(nodes):
    return tf.ceil(tf.clip_by_value(nodes, 0, 1))

x = tf.placeholder(tf.float32, shape=3)
W = tf.constant([
    [1.0, 1.0, 1.0],
    [-1.0, -1.0, -1.0],
    [1.0, 1.0, 1.0]
])
B = tf.constant([
    -0.5,
    1.5,
    -2.5
])

layer_1 = perceptron(W, x, B, step)
W4 = tf.constant([
    [1.0, 1.0, 2.0]
])
B4 = tf.constant([
    -1.99
])
layer_2 = perceptron(W4, layer_1, B4, step)
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
print session.run(layer_2, {x: [0, 0, 1]})
print session.run(layer_2, {x: [0, 1, 1]})
print session.run(layer_2, {x: [1, 1, 1]})
