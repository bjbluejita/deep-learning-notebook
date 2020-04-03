'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年11月19日 15:57
@Description: 
@URL: 
@version: V1.0
'''
import tensorflow as tf
from environments.environment import ToyEnvironment, AtariEnvironment

class EnvironmentTest( tf.test.TestCase ):

    def testToyEnvironment(self):
        env = ToyEnvironment( 'Breakout-v0', 1,
                              30, [80, 80],
                              'NHWC', False, False )



    def testAtariEnvironment(self):
        env = AtariEnvironment( 'Breakout-v0', 1,
                                30, [80, 80],
                                'NHWC', False, False )

if __name__ == '__main__':
    tf.test.main()
