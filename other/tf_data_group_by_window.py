'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月22日 15:57
@Description: 
@URL: https://www.e-learn.cn/content/wangluowenzhang/694648
@version: V1.0
'''
import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.python.data.experimental import  group_by_window
import numpy as np

componets = np.arange( 100 ).astype( np.int64 )
dataset = Dataset.from_tensor_slices( componets )
#dataset = dataset.apply( group_by_window( key_func=lambda x: x%2, reduce_func=lambda _, els: els.batch(10), window_size=100 ) )

iterator = dataset.make_one_shot_iterator()
features = iterator.get_next()
with tf.Session( ) as sess:
    try:
        while True:
            print(sess.run(features))
    except tf.errors.OutOfRangeError:
        print("end!")
