'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年11月19日 10:46
@Description: 
@URL: https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_20/networks/mlp.py
@version: V1.0
'''
import tensorflow as tf

from .layers import *
from .network import Network

class MLPSmall( Network ):
    def __init__(self,
                 sess,
                 observation_dims,
                 history_length,
                 output_size,
                 trainable=True,
                 batch_size=None,
                 weights_initializer=initializers.xavier_initializer(),
                 biases_initializer=tf.zeros_initializer,
                 hidden_activation_fn=tf.nn.relu,
                 output_activation_fn=None,
                 hidden_sizes=[ 50, 50, 50 ],
                 value_hidden_size=[25],
                 advantage_hidden_sizes=[25],
                 network_output_type='dueling',
                 name='MLPSmall'
                 ):
        super( MLPSmall, self ).__init__( sess, name )

        with tf.variable_scope( name ):
            layer = self.inputs = tf.placeholder( 'float32', [ batch_size, history_length ] + observation_dims, 'inputs' )

            if len( layer.get_shape().as_list() ) == 3:
                assert layer.get_shape().as_list()[1] == 1
                layer = tf.reshape( layer, [-1] + layer.get_shape().as_list()[2:] )

            for idx, hidden_size in enumerate( hidden_sizes ):
                w_name, b_name = 'w_%d' % idx, 'b_%d' % idx

                layer, self.var[w_name], self.var[b_name] = \
                    linear( layer, hidden_size, weights_initializer,
                            biases_initializer, hidden_activation_fn, trainable, name='lin_%d' % idx )

            self.build_output_ops( layer, network_output_type,
                                   value_hidden_size, advantage_hidden_sizes, output_size,
                                   weights_initializer, biases_initializer, hidden_activation_fn,
                                   output_activation_fn, trainable )