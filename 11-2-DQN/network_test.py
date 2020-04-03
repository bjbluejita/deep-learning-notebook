'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年11月20日 16:29
@Description: 
@URL: 
@version: V1.0
'''
import tensorflow as tf
from networks.cnn import  CNN
from networks.mlp import MLPSmall

class NetworkTest( tf.test.TestCase ):

    def calc_gpu_fraction(fraction_string):
        idx, num = fraction_string.split('/')
        idx, num = float(idx), float(num)

        fraction = 1 / (num - idx + 1)
        print (" [*] GPU : %.4f" % fraction)
        return fraction

    def testCNN(self):
        with tf.Session(config=tf.ConfigProto( )) as sess:
            pred_network = CNN(sess=sess,
                               data_format='NCHW',
                               history_length=4,
                               observation_dims=[80, 80],
                               output_size=18,
                               network_header_type='nature',
                               name='pred_network', trainable=True)

    def testMLPSmall(self):
        with tf.Session(config=tf.ConfigProto( )) as sess:
            MLPSmall(sess=sess,
                     observation_dims=[80, 80],
                     history_length=4,
                     output_size=18,
                     hidden_activation_fn=tf.sigmoid,
                     network_output_type='normal',
                     name='pred_network', trainable=True)


if __name__ == '__main__':
    tf.test.main()
