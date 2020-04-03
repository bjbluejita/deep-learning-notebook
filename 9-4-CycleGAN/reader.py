'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年06月18日 13:51
@Description: 
@URL: https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_11/reader.py
@version: V1.0
'''
import tensorflow as tf
import utils

class Reader():

    def __init__(self, tfrecords_file, image_size=256,
                 min_queue_examples=1000, batch_size=1, num_threads=8, name='' ):
        '''

        :param tfrecord_file: string, tfrecords file path
        :param image_size:
        :param min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
        :param batch_size: integer, number of images per batch
        :param num_threads:
        :param name:
        '''
        self.tfrecords_file = tfrecords_file
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.name = name


    def feed(self ):
        '''

        :return:images: 4D tensor [batch_size, image_width, image_height, image_depth]
        '''
        with tf.name_scope( self.name ):
            filename_queue = tf.train.string_input_producer( [ self.tfrecords_file ] )
            reader = tf.TFRecordReader()

            _, serialized_example = self.reader.read( filename_queue )
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/file_name' : tf.FixedLenFeature( [], tf.string ),
                    'image/encoded_image' : tf.FixedLenFeature( [], tf.string ),
                }
            )

            image_buffer = features[ 'image/encoded_image' ]
            image = tf.image.decode_jpeg( image_buffer, channels=3 )
            image = self._preprocess( image )
            images = tf.train.shuffle_batch(
                [ image ], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=self.min_queue_examples
            )
            tf.summary.image( '_input', images )

        return images


    def _preprocess(self, image ):
        image = tf.image.resize_images( image, size=( self.image_size, self.image_size ) )
        image = utils.convert2float( image )
        image.set_shape( [ self.image_size, self.image_size, 3 ] )
        return image
