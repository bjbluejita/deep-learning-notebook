'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年05月16日 14:56
@Description: 
@URL:https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_7/preprocessing/preprocessing_factory.py
@version: V1.0
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import cifarnet_preprocessing
from . import inception_preprocessing
from . import lenet_preprocessing
from . import vgg_preprocessing

slim = tf.contrib.slim

def get_preprocessing( name, is_training=False ):
    '''
    Returns preprocessing_fn(image, height, width, **kwargs).
    :param name: The name of the preprocessing function.
    :param is_training:  `True` if the model is being used for training and `False`
                        otherwise.
    :return:  preprocessing_fn: A function that preprocessing a single image (pre-batch).
              It has the following signature:
              image = preprocessing_fn(image, output_height, output_width, ...).
    '''
    preprocessing_fn_map = {
        'cifarnet' : cifarnet_preprocessing,
        'inception' : inception_preprocessing,
        'inception_v1' : inception_preprocessing,
        'inception_v2': inception_preprocessing,
        'inception_v3': inception_preprocessing,
        'inception_v4': inception_preprocessing,
        'inception_resnet_v2': inception_preprocessing,
        'lenet': lenet_preprocessing,
        'resnet_v1_50': vgg_preprocessing,
        'resnet_v1_101': vgg_preprocessing,
        'resnet_v1_152': vgg_preprocessing,
        'vgg': vgg_preprocessing,
        'vgg_a': vgg_preprocessing,
        'vgg_16': vgg_preprocessing,
        'vgg_19': vgg_preprocessing,
    }

    if name not in preprocessing_fn_map:
        raise  ValueError( 'Preprocess name [%s] was not recgnized' % name )

    def preprocessing_fn( image, output_height, output_width, **kwargs ):
        return preprocessing_fn_map[ name ].preprocess_image( image, output_height, output_width, is_training=is_training, **kwargs )

    def unprocessing_fn( image, **kwargs ):
        return preprocessing_fn_map[ name ].unprocess_image( image, **kwargs )

    return preprocessing_fn, unprocessing_fn