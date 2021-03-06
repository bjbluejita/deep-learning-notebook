'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年05月21日 16:34
@Description: 
@URL: https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_7/nets/nets_factory.py
@version: V1.0
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from nets import alexnet
from nets import cifarnet
from nets import inception
from nets import lenet
from nets import overfeat
from nets import resnet_v1
from nets import resnet_v2
from nets import vgg

slim = tf.contrib.slim

networks_map = { 'alexnet_v2': alexnet.alexnet_v2,
                 'cifarnet': cifarnet.cifarnet,
                 'overfeat': overfeat.overfeat,
                 'vgg_a': vgg.vgg_a,
                 'vgg_16': vgg.vgg_16,
                 'vgg_19': vgg.vgg_19,
                 'inception_v1': inception.inception_v1,
                 'inception_v2': inception.inception_v2,
                 'inception_v3': inception.inception_v3,
                 'inception_v4': inception.inception_v4,
                 'inception_resnet_v2': inception.inception_resnet_v2,
                 'lenet': lenet.lenet,
                 'resnet_v1_50': resnet_v1.resnet_v1_50,
                 'resnet_v1_101': resnet_v1.resnet_v1_101,
                 'resnet_v1_152': resnet_v1.resnet_v1_152,
                 'resnet_v1_200': resnet_v1.resnet_v1_200,
                 'resnet_v2_50': resnet_v2.resnet_v2_50,
                 'resnet_v2_101': resnet_v2.resnet_v2_101,
                 'resnet_v2_152': resnet_v2.resnet_v2_152,
                 'resnet_v2_200': resnet_v2.resnet_v2_200,
                 }

arg_scope_map = {'alexnet_v2': alexnet.alexnet_v2_arg_scope,
                 'cifarnet': cifarnet.cifarnet_arg_scope,
                 'overfeat': overfeat.overfeat_arg_scope,
                 'vgg_a': vgg.vgg_arg_scope,
                 'vgg_16': vgg.vgg_arg_scope,
                 'vgg_19': vgg.vgg_arg_scope,
                 'inception_v1': inception.inception_v3_arg_scope,
                 'inception_v2': inception.inception_v3_arg_scope,
                 'inception_v3': inception.inception_v3_arg_scope,
                 'inception_v4': inception.inception_v4_arg_scope,
                 'inception_resnet_v2':
                     inception.inception_resnet_v2_arg_scope,
                 'lenet': lenet.lenet_arg_scope,
                 'resnet_v1_50': resnet_v1.resnet_arg_scope,
                 'resnet_v1_101': resnet_v1.resnet_arg_scope,
                 'resnet_v1_152': resnet_v1.resnet_arg_scope,
                 'resnet_v1_200': resnet_v1.resnet_arg_scope,
                 'resnet_v2_50': resnet_v2.resnet_arg_scope,
                 'resnet_v2_101': resnet_v2.resnet_arg_scope,
                 'resnet_v2_152': resnet_v2.resnet_arg_scope,
                 'resnet_v2_200': resnet_v2.resnet_arg_scope,
                 }

def get_network_fn( name, num_classes, weight_decay=0.0, is_training=False ):
    '''
    Returns a network_fn such as `logits, end_points = network_fn(images)
    :param name:  The name of the network.
    :param num_classes: The number of classes to use for classification.
    :param weight_decay: The l2 coefficient for the model weights.
    :param is_training:  `True` if the model is being used for training and `False`
                      otherwise.
    :return: A function that applies the model to a batch of images. It has
        the following signature:
          logits, end_points = network_fn(images)
    '''
    if name not in networks_map:
        raise  ValueError( 'Name of network unknown %s' % name )
    arg_scope = arg_scope_map[ name ]( weight_decay=weight_decay )
    func = networks_map[ name ]
    @functools.wraps( func )
    def network_fn( image, **kwargs ):
        with slim.arg_scope( arg_scope ):
            return func( image, num_classes, is_training=is_training, **kwargs )
        if hasattr( func, 'default_image_size' ):
            network_fn.default_image_size = func.default_image_size

    return network_fn