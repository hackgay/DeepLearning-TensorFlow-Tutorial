
from __future__ import absolute_import, division, print_function
from builtins import range
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import urllib2
import time

def bias_variable(shape):
    # Here we just choose to initialize our biases to 0.
    # However, this is not an agreed-upon standard and
    # some initialize the biases to 0.01 to ensure
    # that all ReLU units fire in the beginning.
    initial = tf.constant(0.00, shape=shape)
    return tf.Variable(initial)

def one_hot(lst, num_elements):
    out = np.zeros((len(lst), num_elements))
    out[np.arange(len(lst)), lst] = 1
    return out

print ('Downloading Shakespeare data')
source = urllib2.urlopen("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
shakespeare = source.read()
print ('Download complete')

# First we need to generate a mapping between unique
# characters 
num_chars = len(set(shakespeare))
i2c_map = {i: c for i, c in enumerate(set(shakespeare))}
c2i_map = {c: i for i, c in i2c_map.iteritems()}

tf.reset_default_graph()

num_timesteps = 30

# [num inputs per timestep, num neurons in RNN Cell, num outputs for fully connected layer]
num_neurons = 150 # [num_chars, 150, num_chars] 
batch_size  = 1

x = tf.placeholder(tf.float32, [batch_size, None, num_chars])
y = tf.placeholder(tf.float32, shape=[None, num_chars])

state = tf.placeholder(tf.float32, shape=[batch_size, num_neurons])
basic_cell = tf.contrib.rnn.GRUCell(num_units=num_neurons)
outputs, final_state = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32, initial_state=state)

# outputs :: [batch_size, timesteps, 150]
# logits  :: [batch_size, timesteps, num_chars]

w = tf.get_variable("w", shape=[num_neurons, num_chars])
b = bias_variable([num_chars])
logits = tf.tensordot(outputs, w, [[2], [0]]) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits,2), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

config = tf.ConfigProto()

with tf.Session(config=config) as sess:
    num_epochs = 30
        
    shakespeare_trim = shakespeare * num_epochs

    num_train = len(shakespeare_trim)
    
    print("Training for %d epochs (%d characters)" % (num_epochs, num_train))
    
    current_idx = 0
    
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    rnn_state = tf.zeros((batch_size, num_neurons)).eval() # np.load('rnn_state.npy') # 

    # Train
    start_time = time.time()
    old_time = start_time
    # for i in range(num_epochs):
    
    chars_per_iter = batch_size * num_timesteps
    num_iterations = (num_train - 1) / chars_per_iter
    print("At %d characters per iteration, this will take %d iterations." % (chars_per_iter, num_iterations))
    for j in range(num_iterations):
        x_data = shakespeare_trim[current_idx:(current_idx + chars_per_iter)]
        y_data = shakespeare_trim[(current_idx + 1):(current_idx + chars_per_iter + 1)]

        current_idx += chars_per_iter

        x_data = [c2i_map[c] for c in x_data]
        x_batch = np.reshape(one_hot(x_data, num_chars), (batch_size, num_timesteps, num_chars))

        y_data = [c2i_map[c] for c in y_data]
        y_batch = one_hot(y_data, num_chars)

        _, rnn_state = sess.run([train_step, final_state], 
                                feed_dict={x: x_batch, y: y_batch, state: rnn_state})
        if j % 50 == 0:
            train_accuracy, loss = sess.run([accuracy, cross_entropy], 
                                            feed_dict={x: x_batch, y: y_batch, state: rnn_state})
            curr_time = time.time()
            print("iter %d / %d completed: training accuracy %g, loss %g, elapsed time %d sec, time delta %d sec"
                  %(j, num_iterations, train_accuracy, loss, (curr_time - start_time), (curr_time - old_time)))
            old_time = curr_time
    
    print("Training finished.")
    # Save the model
    save_path = saver.save(sess, "./ShakespeareRNN.ckpt")
    np.save('rnn_state', rnn_state)
    print("Model saved in file: %s" % save_path)
    
    num_chars_to_generate = 1000