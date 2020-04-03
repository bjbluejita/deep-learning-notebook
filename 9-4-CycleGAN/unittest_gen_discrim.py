'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年06月19日 10:15
@Description: 
@URL: 
@version: V1.0
'''
import unittest
import tensorflow as tf
from  generator import  Generator
from  discriminator import Discriminator
from model import  CycleGAN

class testGeneratorDiscriminator( unittest.TestCase ):

    def test_Generator(self):
        input = tf.placeholder( tf.float32,
                                shape=[64, 256, 256,3] )
        G = Generator( 'GE', True )
        output = G( input )
        print( output )

    def test_discriminator(self ):
        input = tf.placeholder( tf.float32,
                                shape=[64, 256, 256,3] )
        D = Discriminator( 'D', True )
        output = D( input )
        print( output )


    def test_CycleGAN(self):
        cycleGan = CycleGAN(
            X_train_file='data/tfrecords/apple.tfrecord',
            Y_train_file = 'data/tfrecords/orange.tfrecord',
        )
        G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x = cycleGan.model()
        print( G_loss )

if __name__ == '__main__':
    unittest.main()