'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月12日 17:21
@Description: 
@URL: https://blog.csdn.net/kylin_learn/article/details/85135453
@version: V1.0
'''
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

def split_sequences( sequences, n_steps ):
    X, y = list(), list()
    for i in range( len( sequences ) ):
        end_ix = i + n_steps
        if end_ix > len( sequences ) - 1 :
            break
        seq_x, seq_y = sequences[ i:end_ix, : ], sequences[ end_ix, : ]
        X.append( seq_x )
        y.append( seq_y )
    return array( X ), array( y )

in_seq1 = array( [ 10, 20, 30, 40, 50, 60, 70, 80, 90 ] )
in_seq2 = array( [ 15, 25, 35, 45, 55, 65, 75, 85, 95 ] )
out_seq = array( [ in_seq1[i] + in_seq2[i] for i in range( len( in_seq1 ) ) ] )
in_seq1 = in_seq1.reshape(( len( in_seq1 ), 1 ) )
in_seq2 = in_seq2.reshape( ( len( in_seq2 ), 1 ) )
out_seq = out_seq.reshape( ( len( out_seq ), 1 ) )
dataset = hstack( ( in_seq1, in_seq2, out_seq ) )
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences( dataset, n_steps=n_steps )
print( X.shape, y.shape )
for i in range( len( X ) ):
    print( X[i], y[i] )
