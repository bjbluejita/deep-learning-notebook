'''
多个x时间变量用于预测y的时间
但是y不作为x的一份子
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月12日 16:15
@Description: 
@URL:https://blog.csdn.net/kylin_learn/article/details/85135453
@version: V1.0
'''
from numpy import array, hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

in_seq1 = array( [ 10, 20, 30, 40, 50, 60, 70, 80, 90 ] )
in_seq2 = array( [15, 25, 35, 45, 55, 65, 75, 85, 95] )
out_seq = array( [ in_seq1[i]  * in_seq2[i] for i in range( len( in_seq1 ))] )
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape( ( len( in_seq1), 1) )
in_seq2 = in_seq2.reshape( ( len( in_seq2), 1 ) )
out_seq = out_seq.reshape( ( len( out_seq ), 1 ) )
# horizontally(水平) stack columns
dataset = hstack( ( in_seq1, in_seq2, out_seq ) )
print( dataset.shape )
print( dataset )
'''
现在定为时间步长为3
举个例子，就是用
10 15
20 25
30 35
预测65
'''
# split a multivariate sequence into samples
def split_sequences( sequences, n_steps ):
    X, y = list(), list()
    for i in range( len( sequences ) ):
        end_ix = i + n_steps
        if end_ix > len( sequences ):
            break
        seq_x, seq_y = sequences[ i:end_ix, :-1 ], sequences[ end_ix-1, -1 ]
        X.append( seq_x )
        y.append( seq_y )
    return array( X ), array( y )

n_steps = 3
X, y = split_sequences( dataset, n_steps=n_steps )
print( X.shape, y.shape )
for i in range( len( X ) ):
    print( X[i], y[i] )
n_features = X.shape[ 2 ]

model = Sequential()
model.add( LSTM( 128, activation='relu', input_dtype=( n_steps, n_features )))
model.add( Dense( 64 ) )
model.add( Dense( 1 ) )
model.compile( optimizer='adam', loss='mse' )
model.fit( X, y, epochs=900, verbose=1 )

x_input = array( [ [ 80, 85 ], [ 90, 95 ], [ 100, 105 ] ])
x_input = x_input.reshape( ( 1, n_steps, n_features ) )
yhat = model.predict( x_input, verbose=0 )
print( 'predict:', yhat )