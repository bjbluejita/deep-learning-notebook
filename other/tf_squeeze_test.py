'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年08月06日 15:37
@Description: tf.squeeze()用于压缩张量中为1的轴
@URL: https://blog.csdn.net/LoseInVain/article/details/78994695
@version: V1.0
'''
import tensorflow as tf

raw = tf.Variable( tf.random_normal( shape=( 3, 1, 3, 1 ) ) )
squeezed = tf.squeeze( raw )

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    print( raw.shape )
    print( '------------------------' )
    print( sess.run( squeezed ).shape )
    print( sess.run( squeezed ) )