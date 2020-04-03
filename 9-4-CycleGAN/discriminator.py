'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年06月18日 11:05
@Description: 
@URL: https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_11/discriminator.py
@version: V1.0
'''
import tensorflow as tf
import ops

class Discriminator:

    def __init__(self, name, is_training, norm='instance', use_sigmoid=False ):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False
        self.use_sigmoid = use_sigmoid

    def __call__(self, input ):
        '''

        :param input: batch_size x image_size x image_size x 3
        :return: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
              filled with 0.9 if real, 0.0 if fake
        '''
        with tf.variable_scope( self.name ):
            # convolution layers
            C64 = ops.Ck( input, 64, reuse=self.reuse, norm=None,
                          is_training=self.is_training, name='C64' )
            C128 = ops.Ck( C64, 128, reuse=self.reuse, norm=self.norm,
                           is_training=self.is_training, name='C128' )
            C256 = ops.Ck( C128, 256, reuse=self.reuse, norm=self.norm,
                           is_training=self.is_training, name='C256' )
            C512 = ops.Ck( C256, 512, reuse=self.reuse, norm=self.norm,
                           is_training=self.is_training, name='C512' )

            # apply a convolution to produce a 1 dimensional output (1 channel?)
            # use_sigmoid = False if use_lsgan = True
            output = ops.last_conv( C512, reuse=self.reuse,
                                    use_sigmoid=self.use_sigmoid, name='output' )

        self.reuse = True
        self.variables = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name )

        return output