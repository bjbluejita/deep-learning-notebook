'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年06月18日 9:57
@Description: 
@URL: https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_11/utils.py
@version: V1.0
'''
import tensorflow as tf
import random

def convert2int( image ):
    return tf.image.convert_image_dtype( ( image + 1.0 ) / 2.0, tf.uint8 )

def convert2float( image ):
    image = tf.image.convert_image_dtype( image, tf.float32 )
    return ( image / 127.5 ) - 1.0

def batch_convert2int( images ):
    return tf.map_fn( convert2int, images, dtype=tf.uint8 )

def batch_convert2float( images ):
    return tf.map_fn( convert2float, images, dtype=tf.float32 )

class ImagePool:
    '''
    History of generated images
    Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
    '''
    def __init__(self, pool_size ):
        self.pool_size = pool_size
        self.images = []

    def query(self, image ):
        if self.pool_size == 0:
            return image

        if len( self.images ) < self.pool_size :
            self.images.append( image )
            return image
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randrange( 0, self.pool_size )
                tmp = self.images[ random_id ].copy()
                self.images[ random_id ] = image.copy()
                return tmp
            else:
                return image