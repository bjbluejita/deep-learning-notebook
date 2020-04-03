'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月05日 11:28
@Description: 
@URL: https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_15/test_input_csv.py
@version: V1.0
'''
from __future__ import print_function
import  tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import CSVReader
from tensorflow.contrib.timeseries.python.timeseries import RandomWindowInputFn

csv_file_name = './data/period_trend.csv'
reader = CSVReader( csv_file_name )
with tf.Session() as sess:
    data = reader.read_full()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners( sess=sess, coord=coord )
    print( sess.run( data ) )
    coord.request_stop()

train_input_fn = RandomWindowInputFn( reader, batch_size=2, window_size=10 )

with tf.Session() as sess:
    data = train_input_fn.create_batch()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners( sess=sess, coord=coord )
    batch1 = sess.run( data[0] )
    batch2 = sess.run( data[0] )
    coord.request_stop()

print( 'batch1 :', batch1 )
print( 'batch2 :', batch2 )