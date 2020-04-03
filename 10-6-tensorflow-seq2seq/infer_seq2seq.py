'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年08月06日 14:32
@Description: 
@URL: https://github.com/wavewangyue/tensorflow_seq2seq/blob/master/infer_seq2seq.py
@version: V1.0
'''
#change for git
import tensorflow as tf
import numpy as np
import random
import time

from train_seq2seq import load_data, make_vocab, get_batch, Config
from model_seq2seq_contrib import Seq2seq

tf_config = tf.ConfigProto( allow_soft_placement=True )
tf_config.gpu_options.allow_growth = True

model_path = 'checkpoint/model.ckpt'

if __name__ == '__main__':
    print( '(1)load data......' )
    docs_source, docs_target = load_data( '' )
    w2i_source, i2w_source = make_vocab( docs_source )
    w2i_target, i2w_target = make_vocab( docs_target )

    print( '(2)build model......' )
    config = Config()
    config.source_vocab_size = len( w2i_source )
    config.target_vocab_size = len( w2i_target )
    model = Seq2seq( config=config,
                     w2i_target=w2i_target,
                     useTeacherForcing=False,
                     useAttention=True,
                     useBeamSearch=1 )

    print( '(3) run model......' )
    print_every = 100
    max_target_len = 20

    with tf.Session( config=tf_config ) as sess:
        saver = tf.train.Saver()
        saver.restore( sess, model_path )

        source_batch, source_lens, target_batch, target_lens = get_batch( docs_source, w2i_source, docs_target, w2i_target, config.batch_size )

        feed_dict={
            model.seq_inputs : source_batch,
            model.seq_inputs_length : source_lens,
            model.seq_targets : [ [0] * max_target_len ] * len( source_batch ),
            model.seq_targets_length : [ max_target_len ] * len( source_batch )
        }

        print( 'sample:\n' )
        predict_batch = sess.run( model.out, feed_dict=feed_dict )
        for i in range(7):
            print( 'in:', [ i2w_source[num] for num in source_batch[i]   if i2w_source[num] != '_PAD' ] )
            print( 'out:', [ i2w_target[num] for num in predict_batch[i] if i2w_target[num] != '_PAD' ] )
            print( 'tar:', [ i2w_target[num] for num in target_batch[i]  if i2w_target[num] != '_PAD' ] )