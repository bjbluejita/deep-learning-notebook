'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年02月21日 14:37
@Description: 
@URL: https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/1.a-rnn-introduction.ipynb
@version: V1.0
'''
import keras
print( 'keras version {}'.format( keras.__version__ ))
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add( Embedding( 10000, 32) )
model.add( SimpleRNN(32) )# return_sequences預設為False
model.summary()

model1 = Sequential()
model1.add( Embedding( 10000, 32 ))
model1.add( SimpleRNN( 32, return_sequences=True ) )
model1.add( SimpleRNN( 32, return_sequences=True ) )
model1.add( SimpleRNN( 32, return_sequences=True ) )
model1.add( SimpleRNN(32) ) # 只有最後的RNN層只需要最後的output, 因此不必特別去設置"return_sequences"
model1.summary()