'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年02月21日 14:49
@Description: 
@URL:https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/1.a-rnn-introduction.ipynb
@version: V1.0
'''
import keras
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense
from IPython.display import Image
from keras.layers import LSTM

print( 'keras version:' , keras.__version__ )
max_features = 10000  #要考慮作為特徵的語詞數量
maxlen = 500 # 當句子的長度超過500個語詞的部份,就把它刪除掉
batch_size = 32

## 載入IMDB的資料
print( 'Loading data...' )
( input_train, y_train ), ( input_test, y_test ) = imdb.load_data( num_words=max_features )
print( 'input_train shape:', input_train.shape )
print( len( input_train ), ': train sequence' )
print( len( input_test), ': test sequence' )

# 如果長度不夠的話就補空的
print( 'Pad sequence (sample x time)' )
input_train = sequence.pad_sequences( input_train, maxlen=maxlen )
input_test = sequence.pad_sequences( input_test, maxlen=maxlen )
print( 'input_train shape:', input_train.shape )
print( 'input_test shape:', input_test.shape )

#一個Embedding層和一個SimpleRNN層來訓練一個簡單的循環網絡(RNN)
model = Sequential()
model.add( Embedding( max_features, 32 ) )
model.add( LSTM( 32 ) )
model.add( Dense( 1, activation='sigmoid') )
model.summary()

model.compile( optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'] )
history = model.fit( input_train, y_train,
                     epochs=10,
                     batch_size=128,
                     validation_data=( input_test, y_test ))

#圖表显示
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range( len(acc) )
plt.plot( epochs, acc, label='Trainning acc' )
plt.plot( epochs, val_acc, label='Validation acc' )
plt.title( 'Training and validation accuracy' )
plt.legend()

plt.figure()
plt.plot( epochs, loss, label='Trainning loss' )
plt.plot( epochs, val_loss, label='Validation loss' )
plt.title( 'Training and Validation loss')
plt.legend()
plt.imsave( 'acc_loss.png' )

Image( 'acc_loss.png' )
#plt.show()


