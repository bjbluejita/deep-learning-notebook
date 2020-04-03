'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年02月28日 15:36
@Description: 
@URL: https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/2.0-first-steps-with-julia.ipynb
@version: V1.0
'''
#資料預處理
import os
import glob
import pandas as pd
import math
import numpy as np
from scipy.misc import imread, imsave, imresize
from natsort import natsorted

# 圖像資料的檔案路徑
path = os.path.join( os.getcwd(), 'data')

# 圖像轉換後的目標大小 (32像素 x 32像素)
img_height, img_width = 32, 32

# 轉換圖像後的儲存目錄
suffix = 'Preproc'
trainDataPath = path + '\\train' + suffix
testDataPath = path + '\\test' + suffix

# 產生目錄
if not os.path.exists( trainDataPath ):
    os.makedirs( trainDataPath )

if not os.path.exists( testDataPath ):
    os.makedirs( testDataPath )

### 圖像大小與圖像的色彩的預處理 ###
for datasetType in [ 'train', 'test' ]:
    # 透過natsorted可以讓回傳的檔案名稱的排序
    imgFiles = natsorted( glob.glob( path + '\\' + datasetType + '\\*' ) )
    print( '')
    # 初始一個ndarray物件來暫存讀進來的圖像資料
    imgData = np.zeros( ( len(imgFiles), img_height, img_width ))

    # 使用迴圈來處理每一筆圖像檔
    for i, imgFilePath in enumerate( imgFiles ):
        # 圖像的色彩 (Image Color)處理
        img = imread( imgFilePath, True ) # True: 代表讀取圖像時順便將多階圖像, 打平成灰階(單一通道:one channel)

        # 圖像大小的修改 (Image Resizing)
        imgResized = imresize( img, ( img_height, img_width ) )

        # 把圖像資料儲放在暫存記憶體中
        imgData[i] = imgResized

        # 將修改的圖像儲存到檔案系統 (方便視覺化了解)
        filename = os.path.basename( imgFilePath )
        filenameDotSplit = filename.split( '.' )
        newFileName = str( int(filenameDotSplit[0]) ).zfill(5) + '.' + filenameDotSplit[-1].lower()
        newFileName = path + '\\' + datasetType + suffix + '\\' + newFileName
        imsave( newFileName, imgResized )

    # 新增加"Channel"的維度
    print( 'before:', imgData.shape )
    imgData = imgData[ :, :, :, np.newaxis ]  # 改變前: []
    print( 'after:', imgData.shape )

    # 進行資料(pixel值)標準化
    imgData = imgData.astype('float32') / 256

    # 以numpy物件將圖像轉換後的ndarray物件保存在檔案系統中
    np.save( path + '\\' + datasetType + suffix + '.npy', imgData )

#標籤轉換
#字符的標籤進行one-hot編碼的轉換
#首先，我們將字符轉換為連續整數。由於要預測的字符
# 是[0~9],[a~z]及[A~Z]共有62個字符, 所以我們將把每
# 個字符 對應到[0~61]的整數
# 標籤轉換 (Label Conversion)
import keras

def label2int( ch ):
    asciiVal = ord( ch )
    if( asciiVal <= 57 ):   #0-9
        asciiVal -= 48
    elif ( asciiVal <=90 ):  #A-Z
        asciiVal -= 55
    else:  #a-z
        asciiVal -= 61

    return asciiVal

def int2label( i ):
    if(i<=9): #0-9
        i+=48
    elif(i<=35): #A-Z
        i+=55
    else: #a-z
        i+=61
    return chr(i)

# 載入標籤資料
y_train =  pd.read_csv(  path + '\\' + 'trainLabels.csv').values[:,1] #只保留"標籤資料"欄

# 對標籤(Label)進行one-hot編碼
Y_train = np.zeros( ( y_train.shape[0], 62 ) ) # A-Z, a-z, 0-9共有62個類別

for i in range( y_train.shape[0] ):
    Y_train[i][ label2int( y_train[i] ) ] = 1  # One-hot

#把轉換過的標籤(Label)資料保存在檔案系統便於後續的快速載入與處理
np.save( path + "\\" + 'labelsPreproc.npy', Y_train )

import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

batch_size = 64 # 訓練批次量 (Batch Size)
nb_classes = 62  # A-Z, a-z, 0-9共有62個類別
nb_epoch = 500   # 進行500個訓練循環

# Input image dimensions
# 要輸入到第一層網絡的圖像大小 (32像素 x 32像素)
img_height, img_width = 32, 32

# 相關資料的路徑
path = os.path.join( os.getcwd(), 'data')
# 載入預處理好的訓練資料與標籤
X_train_all = np.load( path + '\\' + 'trainPreproc.npy' )
Y_train_all = np.load( path + '\\' + 'labelsPreproc.npy' )

# 將資料區分為訓練資料集與驗證資料集
X_train, X_val, Y_train, Y_val = train_test_split( X_train_all, Y_train_all,
                                                   test_size=0.25, stratify=np.argmax( Y_train_all, axis=1) )
# 設定圖像增強(data augmentation)的設定
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.4,
    zoom_range=0.3,
    channel_shift_range=0.1
)

### 卷積網絡模型架構
model = Sequential()

model.add( Convolution2D( 128, (3,3,), padding='same', kernel_initializer='he_normal', activation='relu',
                          input_shape=( img_height, img_width, 1 ) ) )
model.add( Convolution2D( 128, (3,3), padding='same', kernel_initializer='he_normal', activation='relu' ))
model.add( MaxPooling2D( pool_size=(2,2) ) )

model.add( Convolution2D( 256, (3,3), padding='same', kernel_initializer='he_normal', activation='relu' ))
model.add( Convolution2D( 256, (3,3), padding='same', kernel_initializer='he_normal', activation='relu' ))

model.add( MaxPooling2D( pool_size=(2,2) ) )

model.add( Convolution2D( 512, (3,3), padding='same', kernel_initializer='he_normal', activation='relu' ))
model.add( Convolution2D( 512, (3,3), padding='same', kernel_initializer='he_normal', activation='relu' ))
model.add( Convolution2D( 512, (3,3), padding='same', kernel_initializer='he_normal', activation='relu' ))

model.add( MaxPooling2D( pool_size=(2,2) ) )

model.add( Flatten() )
model.add( Dense( 4096, kernel_initializer='he_normal', activation='relu' ) )
model.add( Dropout( 0.5 ) )

model.add( Dense( 4096, kernel_initializer='he_normal', activation='relu' ) )
model.add( Dropout(0.5) )

model.add( Dense( nb_classes, kernel_initializer='he_normal', activation='softmax' ) )
model.summary()

### 模型訓練學習 ###
# 首先使用AdaDelta來做第一階段的訓練, 因為AdaMax會無卡住
model.compile( loss='categorical_crossentropy',
               optimizer='adadelta',
               metrics=['accuracy'] )
model.fit( X_train, Y_train,
           epochs=20,
           batch_size=batch_size,
           validation_data=( X_val, Y_val) )

# 接著改用AdaMax
model.compile( loss='categorical_crossentropy',
               optimizer='adamax',
               metrics=['accuracy'] )
# 我們想要保存在訓練過程中驗證結果比較好的模型
modelWeightFile = 'best.kerasModelWeight.hdf5'
saveBestModel = ModelCheckpoint( modelWeightFile, monitor='val_acc',
                                 save_best_only=True, save_weights_only=True,
                                 verbose=1 )
# 在訓練的過程透過ImageDataGenerator來持續產生圖像資料
history = model.fit_generator( datagen.flow( X_train, Y_train, batch_size=batch_size ),
                               steps_per_epoch=len(X_train) / batch_size,
                               epochs=nb_epoch,
                               validation_data=( X_val, Y_val),
                               callbacks=[saveBestModel],
                               verbose=1 )

### 進行預測 ###
# 載入訓練過程中驗證結果最好的模型
model.load_weights( modelWeightFile )

#載入Kaggle測試資料集
X_test = np.load( path + '\\' + 'testPreproc.npy' )

#預測字符的類別
Y_test_pred = model.predict_classes( X_test )

# 從類別的數字轉換為字符
vInt2label = np.vectorize( int2label )
Y_test_pred = vInt2label( Y_test_pred )

# 保存預測結果到檔案系統
np.savetxt(path+"/jular_pred" + ".csv", np.c_[range(6284,len(Y_test_pred)+6284),Y_test_pred], delimiter=',', header = 'ID,Class', comments = '', fmt='%s')


# 透過趨勢圖來觀察訓練與驗證的走向 (特別去觀察是否有"過擬合(overfitting)"的現象)
import matplotlib.pyplot as plt

# 把每個訓練循環(epochs)的相關重要的監控指標取出來
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 取得整個訓練循環(epochs)的總次數
epochs = rang( len(acc) )

# 把"訓練準確率(Training acc)"與"驗證準確率(Validation acc)"的趨勢線形表現在圖表上
plt.plot( epochs, acc, 'bo', label='Trainning acc' )
plt.plot( epochs, val_acc, 'b', label='Validation acc' )
plt.title( 'Trainning and validation accuracy' )
plt.legend()

plt.figure()

# 把"訓練損失(Training loss)"與"驗證損失(Validation loss)"的趨勢線形表現在圖表上
plt.plot( epochs, loss, 'bo', label='Trainning loss' )
plt.plot( epochs, val_loss, 'b', label='Validation loss' )
plt.title( 'Trainning and validation loss' )
plt.legend

plt.show()
