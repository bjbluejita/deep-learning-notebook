'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年06月19日 16:04
@Description: 
@URL: https://blog.csdn.net/dcrmg/article/details/79780331
@version: V1.0
'''
import tensorflow as tf
import numpy as np

sample_num = 5
epoch_num = 200
batch_size = 3
batch_total = int( sample_num / batch_size ) + 1

callback_num = 0

# 生成4个数据和标签
def generate_data( sample_num=sample_num ):
    global callback_num

    labels = np.asarray( range( 0, sample_num ) )
    images = np.random.random( [ sample_num, 224, 224, 3 ] )
    print( '[ {} ]Image size {} label size {}'.format( callback_num, images.shape, labels.shape ) )
    callback_num += 1
    return  images, labels

def get_batch_data( batch_size=batch_size ):
    images, labels = generate_data()
    images = tf.cast( images, tf.float32 )
    labels = tf.cast( labels, tf.int32 )

    #从tensor列表中按顺序或随机抽取一个tensor准备放入文件名称队列
    input_queue = tf.train.slice_input_producer( [ images, labels ], num_epochs=epoch_num, shuffle=False )

    #从文件名称队列中读取文件准备放入文件队列
    image_batch, label_batch = tf.train.batch( input_queue, batch_size=batch_size, num_threads=5, capacity=64, allow_smaller_final_batch=False )

    return image_batch, label_batch

image_batch, label_batch = get_batch_data( batch_size=batch_size )

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    sess.run( tf.local_variables_initializer() )

    # 开启一个协调器
    coord = tf.train.Coordinator()
    # 使用start_queue_runners 启动队列填充
    threads = tf.train.start_queue_runners( sess=sess, coord=coord )

    try:
        while not coord.should_stop():
            print( '*********************' )
            image_batch_v, label_batch_v = sess.run( [image_batch, label_batch] )
            print( image_batch_v.shape, label_batch_v.shape )
    except tf.errors.OutOfRangeError:
        print( 'done now kill all threads' )
    finally:
        coord.request_stop()
        print( 'all thread are asked to kill' )

    coord.join( threads ) #把开启的线程加入主线程，等待threads结束


    print( 'Done' )
