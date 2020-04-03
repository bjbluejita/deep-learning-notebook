'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年08月01日 15:46
@Description: 
@URL: https://blog.csdn.net/junjun150013652/article/details/81331448
@version: V1.0
'''

import numpy as np
import tensorflow as tf



n_steps = 2
n_inputs = 3
n_neurons = 5

X = tf.placeholder( tf.float32, [ None, n_steps, n_inputs ] )
basic_cell = tf.nn.rnn_cell.BasicRNNCell( num_units=n_neurons )

seq_length = tf.placeholder( tf.int32, [None] )
outputs, states = tf.nn.dynamic_rnn( basic_cell, X, dtype=tf.float32,
                                     sequence_length=seq_length )
print( 'outputs:', outputs )
print( 'states:', states )

init = tf.global_variables_initializer()

X_batch = np.array(
    [
    #step 0       step1
    [[0, 1, 2], [9, 8, 7]], # instance 1
    [[3, 4, 5], [0, 0, 0]], # instance 2 (padded with zero vectors)
    [[6, 7, 8], [6, 5, 4]], # instance 3
    [[9, 0, 1], [3, 2, 1]], # instance 4
    [[9, 0, 1], [3, 2, 1]], # instance 5
    [[3, 2, 1], [3, 3, 1]], # instance 6
    ]
)
seq_length_batch = np.array( [ 2, 1, 2, 2, 3, 9 ] )

with tf.Session() as sess:
    init.run()
    outputs_val, states_val = sess.run(
        [ outputs, states ], feed_dict={ X: X_batch, seq_length: seq_length_batch }
    )

print( "outputs_val.shape:", outputs_val.shape, "states_val.shape:",  states_val.shape )
print( 'output:', outputs_val[0] )
print( 'states_val:', states_val[0] )