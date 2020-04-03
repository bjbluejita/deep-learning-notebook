'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年08月02日 10:18
@Description: 
@URL: 
@version: V1.0
'''
import tensorflow as tf
import numpy as np

lstm_cell = tf.nn.rnn_cell.LSTMCell( num_units=128 )
inputs = tf.placeholder( np.float32, shape=( 32, 100) ) #32是batch_size
input_feeds = np.random.standard_normal( ( 32, 100 )  )
h0 = lstm_cell.zero_state( 32, np.float32 )
outputs, h1 = lstm_cell( inputs, h0 )
print( 'lstm_cell.state_size:',lstm_cell.state_size )
print( 'outputs:\t',outputs )
print( 'h1.h:\t\t', h1.h )
print( 'h1.c:\t\t', h1.c )

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    outputs_val, h1_val = sess.run( [ outputs, h1 ], feed_dict={ inputs: input_feeds } )
    print( 'output:', outputs_val[0] )
    print( 'h1.h:', h1_val.h[0] )
    print( 'h1.c:', h1_val.c[0] )