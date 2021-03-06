'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年05月31日 13:53
@Description: 
@URL: https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_8/main.py
@version: V1.0
'''
import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import show_all_variables, pp, visualize

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer( 'epoch', 200, 'Epoch to train [25]' )
flags.DEFINE_float( 'learning_rate', 0.0002, 'Learning rate of for adam [0.0002]' )
flags.DEFINE_float( 'beta1', 0.5, 'Momentum term of adam[0.5]' )
flags.DEFINE_integer( 'train_size',600, 'The size of train images[np.inf]' )
flags.DEFINE_integer( 'batch_size', 64, 'The size batch images [64]' )
flags.DEFINE_integer( 'input_height', 108, 'The size of image to use (will be center cropped) [108]' )
flags.DEFINE_integer( 'input_width', None, 'The size of image to use (will be center cropped). If None, same value as input_height' )
flags.DEFINE_integer( 'output_height', 64, 'The size of the output images to produce [64]' )
flags.DEFINE_integer( 'output_width', None, 'The size of the output images to produce. If None, same value as output_height' )
flags.DEFINE_string( 'dataset', 'mnist', 'The name of dataset [celebA, mnist, lsun]' )
flags.DEFINE_string( 'input_fname_pattern', '*.jpg', 'Glob pattern of filename of input images [*]' )
flags.DEFINE_string( 'checkpoint_dir', 'checkpoint', 'Directory name to save the checkpoints [checkpoint]' )
flags.DEFINE_string( 'sample_dir', 'samples', 'Directory name to save the image samples' )
flags.DEFINE_boolean( 'train', False, 'True for training, False for testing [False]' )
flags.DEFINE_boolean( 'crop', False, 'True for training, False for testing [False]' )
flags.DEFINE_boolean( 'visualize', False, 'True for visualizing, False for nothing' )
FLAGS = flags.FLAGS

def main(_):
    pp.pprint( flags.FLAGS.__flags )

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists( FLAGS.checkpoint_dir ):
        os.makedirs( FLAGS.checkpoint_dir )
    if not os.path.join( FLAGS.sample_dir ):
        os.makedirs( FLAGS.sample_dir )

    run_config = tf.ConfigProto()
    #run_config.gpu_options.allow_growth = True

    with tf.Session( config=run_config ) as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_height=FLAGS.output_height,
                output_width=FLAGS.output_width,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                y_dim=10,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir
            )
        else:
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_height=FLAGS.output_height,
                output_width=FLAGS.output_width,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir
            )
        show_all_variables()

        if FLAGS.train:
            dcgan.train( FLAGS )
        else:
            if not dcgan.load( FLAGS.checkpoint_dir )[0]:
                raise  Exception( '[!] Train a model first, then run test mode' )

        OPTION = 2
        visualize( sess, dcgan, FLAGS, OPTION )

if __name__ == '__main__':
    tf.app.run()