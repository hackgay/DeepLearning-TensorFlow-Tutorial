

# coding: utf-8


# First, we do the basic setup.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# config
logs_path = "/tmp/mnist/2"

# Load mnist data set
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.name_scope('input'):
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, [None, 784], name="x-input")

    # target 10 output classes