'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年08月02日 10:43
@Description: 
@URL: 
@version: V1.0
'''

import tensorflow as tf
import numpy as np

def get_a_cell():
    return tf.nn.rnn_cell.LSTMCell( num_units=128 )

cells = tf.nn.rnn_cell.MultiRNNCell( [ get_a_cell() for _ in range(3) ] ) #get 3 层 LSTMCELL
'''
cells也是RNNCell的子类
它的state_sizeshi(128, 128, 128)
( 128, 128, 128) 不是128X128X128的意思， 而是表示共有3个隐含层状态，每个隐含层状态的大小是128
'''
print( cells.state_size )
inputs = tf.placeholder( tf.float32, shape=( 32, 100) )
input_feeds = np.random.standard_normal( ( 32, 100 )  )
h0 = cells.zero_state( 32, np.float32 )
outputs, h1 = cells( inputs, h0 )
print( 'outputs:\t',outputs )
print( 'h1:\t', h1 )
print( 'h1.h:\t\t', h1[2].h )
print( 'h1.c:\t\t', h1[2].c )

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    output_values, h1_values = sess.run( [ outputs, h1 ], feed_dict={ inputs: input_feeds })
    print( 'output_values:', output_values )
    print( 'h1_values[2].h:', h1_values[2].h )