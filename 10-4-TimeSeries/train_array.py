'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月05日 16:34
@Description: 
@URL: https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_15/train_array.py
@version: V1.0
'''
from __future__ import print_function
import numpy as np
import matplotlib
#matplotlib.use( 'agg' )
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader
from tensorflow.contrib.timeseries.python.timeseries import RandomWindowInputFn
from tensorflow.contrib.timeseries.python.timeseries import ARRegressor
from tensorflow.contrib.timeseries.python.timeseries import ARModel
from tensorflow.contrib.timeseries.python.timeseries import WholeDatasetInputFn, predict_continuation_input_fn

def main( _ ):
    x = np.array( range(1000) )
    noise = np.random.uniform( -0.2, 0.2, 1000 )
    #y = np.sin( np.pi * x / 100 ) -  np.cos( np.pi * ( x + noise * 1000 ) / 100 ) + x / 200 + noise
    y = np.sin(np.pi * x / 50 ) + np.cos(np.pi * x / 50) + np.sin(np.pi * x / 25) + noise  + x / 200
    plt.plot( x, y )
    plt.show()
    plt.savefig('timeseries_y.jpg')

    data = {
        tf.contrib.timeseries.TrainEvalFeatures.TIMES : x,
        tf.contrib.timeseries.TrainEvalFeatures.VALUES : y,
    }
    reader = NumpyReader( data )

    train_input_fn = RandomWindowInputFn( reader, batch_size=16, window_size=40 )

    ar = ARRegressor(
        periodicities=200, input_window_size=30, output_window_size=10,
        num_features=1,
        loss=ARModel.NORMAL_LIKELIHOOD_LOSS
    )
    ar.train( input_fn=train_input_fn, steps=6000 )

    evaluation_input_fn = WholeDatasetInputFn( reader )
    # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
    evaluation = ar.evaluate( input_fn=evaluation_input_fn, steps=1 )

    ( predictions, ) = tuple( ar.predict(
        input_fn=predict_continuation_input_fn(
            evaluation, steps=250
            )  ) )

    plt.figure( figsize=( 15, 15 ) )
    plt.plot( data[ 'times' ].reshape( -1 ), data[ 'values' ].reshape( -1 ), label='origin' )
    plt.plot( evaluation[ 'times' ].reshape( -1 ), evaluation[ 'mean' ].reshape( -1 ), label='evaluation' )
    plt.plot( predictions[ 'times' ].reshape( -1 ), predictions[  'mean' ].reshape( -1 ), label='prediction' )
    plt.xlabel( 'time_step' )
    plt.ylabel( 'values' )
    plt.legend( loc=4 )
    plt.savefig( 'predict_result_200.jpg' )


if __name__ == '__main__':
    tf.logging.set_verbosity( tf.logging.INFO )
    tf.app.run()
