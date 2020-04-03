'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月02日 10:53
@Description: 
@URL: 
@version: V1.0
'''
import unittest
from series_classification import ToSequenceData
import tensorflow as tf
import numpy as np

class test_ToSequenceData( unittest.TestCase ):

    def test_create_ToSequenceData(self):
        sequenceData = ToSequenceData( max_seq_len=50 )
        print(  sequenceData.data[0] )
        print( sequenceData.labels[0] )

    def test_next_ToSequenceData(self):
        sequenceData = ToSequenceData( max_seq_len=50 )
        data, labels, lens = sequenceData.next( batch_size=20)
        data, labels, lens = sequenceData.next( batch_size=20 )
        print( np.array( data ).shape )
        print( np.array( labels ).shape )
        print( np.array( lens ).shape )


if __name__ == '__main__':
    unittest.main()