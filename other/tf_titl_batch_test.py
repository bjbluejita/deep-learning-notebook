'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年08月05日 16:50
@Description: 
@URL: https://blog.csdn.net/zjm750617105/article/details/85709286
@version: V1.0
'''

import tensorflow as tf

t = tf.constant( [ [ 1, 1, 1, 9], [2, 2, 2, 9], [7, 7, 7, 9] ] )
# 第一维度和第二维度都保持不变
z0 = tf.tile( t, multiples=[ 1, 1 ] )
# 第1维度不变, 第二维度复制为2份
z1 = tf.tile(t, multiples=[1, 2])
# 第1维度复制为两份, 第二维度不变
z2 = tf.tile(t, multiples=[2, 1])

encoder_outputs = tf.constant([[[1, 3, 1], [2, 3, 2]], [[2, 3, 4], [2, 3, 2]]])
print(encoder_outputs.get_shape())  # (2, 2, 3)
# 将batch内的每个样本复制3次, tile_batch() 的第2个参数是一个 int 类型数据
z4 = tf.contrib.seq2seq.tile_batch( encoder_outputs, multiplier=3 )

with tf.Session() as sess:
    print( sess.run( z0 ) )
    print( '-------------------' )
    print( sess.run( z1 ) )
    print( '-------------------' )
    print( sess.run( z2 ) )
    print( '-------------------' )
    print( sess.run( z4 ) )