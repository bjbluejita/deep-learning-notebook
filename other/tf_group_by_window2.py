'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月23日 11:23
@Description: 
@URL: https://xbuba.com/questions/47588312
@version: V1.0
'''
import  numpy as np
import  tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.contrib.data import batch_and_drop_remainder, group_by_window

def dump_dataset( dataset ):
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    with tf.Session( ) as sess:
        try:
            while True:
                print(sess.run(features))
        except tf.errors.OutOfRangeError:
            print("end!")

length = 32
componets = np.array( [ [i] for i in range( length ) ], dtype=np.int64 )
#print( componets )
dataset = Dataset.from_tensor_slices( componets )
dump_dataset( dataset )
window_size = 4

dataset = dataset.apply( batch_and_drop_remainder( window_size) )
dump_dataset( dataset )
# [[0][1][2][3]]
# [[4][5][6][7]]
# [[8][9][10][11]]

# Skip first row and duplicate all rows, this allows the creation of overlapping window
dataset1 = dataset.apply( group_by_window( key_func=lambda x: 3,
                                           reduce_func= lambda k, d: d.shuffle( 3 ),
                                           window_size=2 ))
dump_dataset( dataset1 )