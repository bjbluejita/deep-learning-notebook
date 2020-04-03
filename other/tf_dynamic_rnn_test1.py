'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年08月02日 15:10
@Description: 
@URL: https://blog.csdn.net/junjun150013652/article/details/81331448
@version: V1.0
'''
import tensorflow as tf
import numpy as np

n_steps = 2
n_inputs = 3
n_neurons = 5
n_layers = 3

X = tf.placeholder( tf.float32, [ None, n_steps, n_inputs ] )
seq_length = tf.placeholder( tf.int32, [None] )

layers =[ tf.nn.rnn_cell.BasicRNNCell( num_units=n_neurons,
                                      activation=tf.nn.relu )
          for _ in range( n_layers )
          ]
multi_layers = tf.nn.rnn_cell.MultiRNNCell( layers )
outputs, states = tf.nn.dynamic_rnn( multi_layers, X, dtype=tf.float32, sequence_length=seq_length )
print( 'outputs:', outputs )
print( 'states:', states )

init = tf.global_variables_initializer()

X_batch = np.array([
    # step 0     step 1
    [[0, 1, 2], [9, 8, 7]], # instance 1
    [[3, 4, 5], [0, 0, 0]], # instance 2 (padded with zero vectors)
    [[6, 7, 8], [6, 5, 4]], # instance 3
    [[9, 0, 1], [3, 2, 1]], # instance 4
])
seq_length_batch = np.array([2, 1, 2, 2])

with tf.Session() as sess:
    sess.run( init )
    outputs_val, states_val = sess.run( [ outputs, states ],
                                        feed_dict={ X: X_batch, seq_length : seq_length_batch } )

    print( 'outputs_val shape: ', outputs_val.shape, ' states_val size:', len(states_val) )
    print( 'outputs_val:', outputs_val[-1] )
    print( 'states_val:', states_val[2] )