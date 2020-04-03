'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年06月18日 16:32
@Description: 
@URL: 
@version: V1.0
'''
import unittest
import tensorflow as tf
import ops

class testOpsMethods( unittest.TestCase ):

    def test_c7s1_k(self ):
        input = tf.placeholder( tf.float32,
                                shape=[64, 256, 256,3] )
        result = ops.c7s1_k( input, 32 )
        print( result )

    def test_dk(self ):
        input = tf.placeholder( tf.float32,
                                shape=[64, 256, 256,3] )
        result = ops.dk( input, 32, name='d64')
        print( result )

    def test_Rk(self ):
        input = tf.placeholder( tf.float32,
                                shape=[64, 256, 256,32] )
        result = ops.Rk( input, 32, name='Rk64')
        print( result )

    def test_n_res_blocks(self ):
        input = tf.placeholder( tf.float32,
                                shape=[64, 256, 256,3] )
        result = ops.n_res_blocks( input, reuse=False )
        print( result )

    def test_uk(self ):
        input = tf.placeholder( tf.float32,
                                shape=[64, 256, 256,32] )
        result = ops.uk( input, 32, name='Uk64')
        print( result )

    def test_Ck(self ):
        input = tf.placeholder( tf.float32,
                                shape=[64, 256, 256,32] )
        result = ops.Ck( input, 32, name='Ck64')
        print( result )

    def test_last_conv(self ):
        input = tf.placeholder( tf.float32,
                                shape=[64, 256, 256,32] )
        result = ops.last_conv( input, name='LAST_CONV')
        print( result )



if __name__ == '__main__':
    unittest.main()
