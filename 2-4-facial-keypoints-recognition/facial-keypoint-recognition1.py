'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年03月05日 10:36
@Description: 
@URL: https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/2.4-facial-keypoints-recognition.ipynb
@version: V1.0
'''

#資料預處理
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# 資料路徑
FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'
FLOOKUP = 'data/IdLookupTable.csv'

def load( test=False, cols=None ):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = pd.read_csv( fname )

    # Image欄位有像素的資料(pixel values)並轉換成 numpy arrays
    df[ 'Image' ] = df[ 'Image'].apply( lambda im: np.fromstring( im, sep=' ' ) )

    if cols:
        df = df[ list( cols ) ] + [ 'Image' ]

    print( df.count() )
    df = df.dropna()

    X = np.vstack( df['Image'].values ) / 255 # 將像素值進行歸一化 [0, 1]
    X = X.astype( np.float32 ) # 轉換資料型態

    if not test:    # 只有 FTRAIN有目標的標籤(label)
        y = df[ df.columns[:-1] ].values
        y = ( y - 48 ) / 48
        X, y = shuffle( X, y , random_state=42 )  # 對資料進行洗牌
        y = y.astype( np.float32 )
    else:
        y = None

    return  X, y

def load2( test=False, cols=None ):
    X, y = load( test=test, cols=cols )
    X = X.reshape( -1, 96, 96, 1 )  # 轉換成Conv2D的卷積層的input shape

    return X, y

#模型1:只有單一的隱藏層
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential()
model.add( Dense( 100, activation='relu', input_shape=( 9216, ) ) )
model.add( Dense( 30 ) )

sgd = SGD( lr=0.01, momentum=0.9, nesterov=True )
model.compile( loss='mean_squared_error', optimizer=sgd )  # 使用"MSE"來做為loss function
model.summary()

# 載入模型訓練資料
X, y = load()
print( 'X.shape={}, X.min={:.3f}  X.max={:.3f}'.format( X.shape, X.min(), X.max() ) )
print( 'y.shape={}, y.min={:.3f} y.max={:.3f}'.format( y.shape, y.min(), y.max() ))

# 設定訓練參數
batch_size = 32
epochs = 100

# 開始訓練
history = model.fit( X, y,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_split=0.2)

X_test, _ = load(test=True) # 取得測試資料集
y_pred = model.predict(X_test) # 進行預測

def plot_sample( x, y, axis ):
    img = x.reshape( 96, 96 )
    axis.imshow( img, cmap='gray' )
    # 把模型預測出來的15個臉部關鍵點打印在圖像上
    axis.scatter( y[0::2] *48 + 48, y[1::2]*48+48, marker='x', s=10 )

# 打印一個6x6的圖像框格
fig = plt.figure( figsize=(6,6))
fig.subplots_adjust( left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05 )

# 選出測試圖像的前16個進行視覺化
for i in range(16):
    ax = fig.add_subplot( 4, 4, i+1, xticks=[], yticks=[] )
    plot_sample( X_test[i], y_pred[i], ax  )

plt.show()