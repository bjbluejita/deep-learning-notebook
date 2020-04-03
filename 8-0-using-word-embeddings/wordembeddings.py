'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年02月25日 10:50
@Description: 
@URL: https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/8.0-using-word-embeddings.ipynb
@version: V1.0
'''
import warnings
warnings.filterwarnings( 'ignore' )

import keras
from keras.layers import Embedding

# 嵌入層( Embedding layer)的構建至少需要兩個參數：
# 可能的符標(token)的數量，這裡是 1000(1 + maximum word index),
# 和嵌入(embedding)的維度，這裡是 64。
embedding_layers = Embedding( 1000, 64 )
#print( embedding_layers )

from keras.datasets import imdb
from keras import preprocessing

# 要考慮作為特徵的單詞數
max_features = 10000
# 在此單詞數量之後剪切文本
maxlen = 20

# 將數據加載為整數列表
( x_train, y_train ), ( x_test, y_test ) = imdb.load_data( num_words=max_features )

# 這將我們的整數列表變成一個2D整個張量 (samples, maxlen)
x_train = preprocessing.sequence.pad_sequences( x_train, maxlen=maxlen )
x_test = preprocessing.sequence.pad_sequences( x_test, maxlen=maxlen )

print( x_train.shape )
print( x_test.shape )

from keras.models import  Sequential
from keras.layers import  Flatten, Dense
from keras.layers import LSTM

model = Sequential()
# 我們為嵌入層指定最大輸入長度，以便稍後將嵌入式輸入平坦化
# 參數：
#      符標(token)的數量，這裡是 1000
#      嵌入(embedding)的維度，這裡是 8
model.add( Embedding( 10000, 8, input_length=maxlen ) )

# 在嵌入層之後，我們的張量形狀轉換成為 `(samples, maxlen, 8)`.
# 我們將3D嵌入張量變成2D張量形狀 `(samples, maxlen * 8)`
model.add( Flatten() )

# 我們添加一個二元分類層
model.add( Dense( 1, activation='sigmoid' ) )

model.compile( optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'] )
model.summary()
history = model.fit( x_train, y_train,
                     epochs=10,
                     batch_size=32,
                     validation_split=0.2 )

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
#plt.imsave( 'acc_loss.png' )

#Image( 'acc_loss.png' )
plt.show()
