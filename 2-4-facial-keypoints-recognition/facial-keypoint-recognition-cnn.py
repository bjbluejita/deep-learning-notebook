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

#模型2:CNN
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from collections import OrderedDict
from sklearn.model_selection import  train_test_split

# 擴展keras的ImageDataGenerator來產生更多的圖像資料
class FlippedImageGenerator( ImageDataGenerator ):
    # 由於臉部的關鍵點是左右對應的, 我們將使用鏡像(flip)的手法來產生圖像
    flip_indics = [ (0,2), (1,3),( 4, 8) , (5, 9),
                    (6, 10), (7, 11), (12, 16), (13, 17),
                    (14, 18), (15, 19), (22, 24), (23, 25)
                    ]

    def next(self):
        X_batch, y_batch = super( FlippedImageGenerator, self ).next()
        batch_size = X_batch.shape[0]
        # 隨機選擇一些圖像來進行水平鏡像(flip)
        indices = np.random.choice( batch_size, batch_size/2, replace=False )
        X_batch[ indices ] = X_batch[ indices, :, :, ::-1 ]

        # 對於有進行過水平鏡像的圖像, 也把臉部關鍵座標點進行調換
        if y_batch is not  None:
            y_batch[ indices, ::2 ] = y_batch[ indices, ::2 ] * -1

            for a, b in self.flip_indices:
                y_batch[ indices, a ] , y_batch[ indices, b ] = (
                    y_batch[ indices, b ], y_batch[ indices, a ]
                )

        return X_batch, y_batch

#網絡模型構建
def cnn_model():
    model = Sequential()
    model.add( Conv2D( 32, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', input_shape=(96,96,1) ))
    model.add( Conv2D( 32, (3,3), activation='relu' ) )
    model.add( MaxPooling2D( pool_size=(2,2) ))
    model.add( Dropout(0.2) )

    model.add( Conv2D( 64, (3,3), padding='same', activation='relu' ))
    model.add( Conv2D( 64, (3,3), activation='relu' ) )
    model.add( MaxPooling2D( pool_size=(2,2) ) )
    model.add( Dropout(0.2) )

    model.add( Conv2D( 128, (3,3), padding='same', activation='relu' ) )
    model.add( Conv2D( 128, (3,3), activation='relu' ) )
    model.add( MaxPooling2D( pool_size=(2, 2) ) )
    model.add( Dropout(0.2) )

    model.add( Flatten() )
    model.add( Dense( 128, activation='relu' ) )
    model.add( Dropout(0.5) )
    model.add( Dense( 30) ) # 因為有15個關鍵座標(x,y), 共30個座標點要預測

    return model

model2 = cnn_model()
model2.summary()

# 使用與第1個模型相同的optimer與loss function
sgd = SGD( lr=0.01, momentum=0.9, nesterov=True )
model2.compile( loss='mean_squared_error',
                optimizer=sgd )



# 載入模型訓練資料
X, y = load2()

# 進行資料拆分
X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.2, random_state=44 )

# 產生一個圖像產生器instance
flipgen = FlippedImageGenerator()

# 設定訓練參數
batch_size = 32
epochs = 1

# 開始訓練
history = model2.fit_generator( flipgen.flow( X_train, y_train, batch_size=batch_size),
                     steps_per_epoch=len( X_train )/batch_size,
                     epochs=epochs,
                     validation_data=( X_val, y_val ) )

X_test, _ = load2(test=True) # 取得測試資料集
y_pred = model2.predict(X_test) # 進行預測

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