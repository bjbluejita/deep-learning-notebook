'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年05月16日 15:45
@Description: 
@URL: https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_7/preprocessing/cifarnet_preprocessing.py
@version: V1.0
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_PADDING = 4

slim = tf.contrib.slim

def preprocess_for_train( image,
                          output_height,
                          output_width,
                          padding=_PADDING ):
    '''
    Preprocesses the given image for training.
    :param image:  A `Tensor` representing an image of arbitrary size.
    :param output_height:  The height of the image after preprocessing.
    :param output_width:  The width of the image after preprocessing.
    :param padding: The amound of padding before and after each dimension of the image.
    :return:  A preprocessed image.
    '''
    tf.summary.image( 'image', tf.expand_dims( image, 0 ) )

    image = tf.to_float( image )
    if padding > 0:
        image = tf.pad( image, [ [ padding, padding ], [padding, padding], [0,0] ] )
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop( image,
                                      [ output_height, output_width, 3 ] )
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right( distorted_image )
    tf.summary.image( 'distorted_image', tf.expand_dims( distorted_image, 0) )

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness( distorted_image,
                                                  max_delta=63 )
    distorted_image = tf.image.random_contrast( distorted_image,
                                                lower=0.2, upper=1.8 )
    # Subtract off the mean and divide by the variance of the pixels.
    return tf.image.per_image_standardization( distorted_image )

def preprocess_for_eval( image, output_height, output_width ):
    '''
    Preprocesses the given image for evaluation
    :param image: A `Tensor` representing an image of arbitrary size.
    :param output_height: The height of the image after preprocessing.
    :param output_width: The width of the image after preprocessing.
    :return:   A preprocessed image.
    '''
    tf.summary.image( 'image', tf.expand_dims( image, 0 ) )

    image = tf.to_float( image )

    # Resize and crop if needed.
    resized_image = tf.image.resize_image_with_crop_or_pad( image,
                                                            output_height,
                                                            output_width )
    tf.summary.image( 'resize_image', tf.expand_dims( resized_image, 0 ) )

    # Subtract off the mean and divide by the variance of the pixels.
    return tf.image.per_image_standardization( resized_image )

def preprocess_image( image, output_height, output_width, is_training=False ):
    '''
    Preprocesses the given image
    :param image:  A `Tensor` representing an image of arbitrary size.
    :param output_height:  The height of the image after preprocessing.
    :param output_width:   The width of the image after preprocessing.
    :param is_training:   `True` if we're preprocessing the image for training and
                          `False` otherwise.
    :return:   A preprocessed image.
    '''
    if is_training:
        return preprocess_for_train( image, output_height, output_width )
    else:
        return preprocess_for_eval( image, output_height, output_width )
