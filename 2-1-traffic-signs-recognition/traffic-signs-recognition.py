'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年03月04日 10:44
@Description: 
@URL: https://github.com/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/2.1-traffic-signs-recognition.ipynb
@version: V1.0
'''
#資料預處理 (Data Preprocessing)
import warnings
warnings.filterwarnings( 'ignore' )

import numpy as np
import skimage.io as io
import skimage.color  as color
import skimage.exposure as exposure
import skimage.transform as transform
from sklearn.model_selection import train_test_split
import os
import glob
import h5py

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD
import keras.utils as np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from matplotlib import pyplot as plt

#%matplotlib inline

NUM_CLASSES = 43 # 共有43種要辨識的交通標誌
IMG_SIZE = 48    # 每張圖像最後都要整理成 48x48的大小
h5File = 'X.h5'

#圖像亮度直方圖均衡化
#圖像大小的修改
from pathlib2 import PurePath   # 處理不同作業系統file path的解析問題 (*nix vs windows)

# 圖像標高度均衡、置中及大小調整
def preprocess_img( img ):
    # 進行"直方圖均衡化"處理
    hsv = color.rgb2hsv( img )  # 對彩色分量rgb分別做均衡化，會產生奇異的點，圖像不和諧。一般採用的是用hsv空間進行亮度的均衡
    hsv [:, :, 2] = exposure.equalize_hist( hsv[:, :, 2] )
    img = color.hsv2rgb( hsv )  # 再把圖像從hsv轉回rgb

    # 進行圖像置中
    min_side = min( img.shape[:-1] )
    centre = img.shape[0] //2, img.shape[1]//2
    img = img[ centre[0] - min_side//2 : centre[0] + min_side//2 ,
               centre[1] - min_side//2 : centre[1] + min_side//2,
               : ]
    # 改變大小
    img = transform.resize( img, ( IMG_SIZE, IMG_SIZE ) )

    return img

# 取得圖像檔的分類標籤
def get_class( img_path ):
    return int( PurePath( img_path ).parts[-2] )

#圖像處理並轉換成numpy ndarray

try:
    with h5py.File( h5File ) as hf:
        X, Y = hf['imgs'][:], hf['labels'][:]
    print( 'Loaded images from X.h5' )
except( IOError, OSError, KeyError ):
    print( 'Error in read X.h5, Process all images...' )
    root_dir = 'GTSRB/Final_Training/Images'
    imgs = []
    labels = []

    all_img_paths = glob.glob( os.path.join( root_dir, '*/*.ppm' ) )# 我們有 Test與Traing兩個檔案夾的資料要處理
    np.random.shuffle( all_img_paths )
    for img_path in all_img_paths:
        try:
            img = preprocess_img( io.imread( img_path ) )
            label = get_class( img_path )
            imgs.append( img )
            labels.append( label )

            if len( imgs ) % 1000 == 0:
                print( 'Precessd {}/{}'.format( len(imgs), len( all_img_paths ) ) )
        except( IOError, OSError ):
            print( 'Missed, ', img_path )
            pass

    X = np.array( imgs, dtype='float32' )    # 將資料轉換成numpy的ndarray, 資料型別為float32
    Y = np.eye( NUM_CLASSES, dtype='uint8')[labels]  # 對labels的資料進行one-hot (使用numpy.eye的函式)

    # 將處理過圖像資料與標籤保持在檔案系統, 下次可以加速載入與處理
    with h5py.File( h5File, 'w' ) as hf:
        hf.create_dataset( 'imgs', data=X )
        hf.create_dataset( 'labels', data=Y )

#網絡模型
# 產生一個Keras序貫模型
def cnn_model():
    model = Sequential()

    model.add( Conv2D( 32, (3,3), padding='same', activation='relu', input_shape=( IMG_SIZE, IMG_SIZE, 3 ) ) )
    model.add( Conv2D( 32, (3,3), activation='relu' ) )
    model.add( MaxPooling2D( pool_size=(2,2) ) )
    model.add( Dropout(0.2) )

    model.add( Conv2D( 64, (3,3), padding='same', activation='relu' ))
    model.add( Conv2D( 64, (3,3), padding='same', activation='relu' ))
    model.add( MaxPooling2D( pool_size=(2,2) ) )
    model.add( Dropout(0.2) )

    model.add( Conv2D( 128, (3,3), padding='same', activation='relu' ) )
    model.add( Conv2D( 128, (3,3), padding='same', activation='relu' ) )
    model.add( MaxPooling2D( pool_size=(2,2) ) )
    model.add( Dropout(0.2) )

    model.add( Flatten() )
    model.add( Dense( 512, activation='relu' ) )
    model.add( Dense( NUM_CLASSES, activation='softmax' ))

    return  model

model = cnn_model()
model.summary()

# 讓我們先配置一個常用的組合來作為後續優化的基準點
lr = 0.01
sgd = SGD( lr=lr, decay=1e-6, momentum=0.9, nesterov=True )
model.compile( loss='categorical_crossentropy',
               optimizer=sgd,
               metrics=['accuracy'])

#訓練 (Training)
def lr_schedule( epoch ):
    return lr * ( 0.1 ** int( epoch/10 ) )

batch_size = 32
nb_epoch = 30

history = model.fit( X, Y,
                     batch_size=batch_size,
                     epochs=nb_epoch,
                     validation_split=0.2,
                     shuffle=True,
                     callbacks=[ LearningRateScheduler( lr_schedule ),
                                 ModelCheckpoint( 'model.h5', save_best_only=True ) ] )

# 透過趨勢圖來觀察訓練與驗證的走向 (特別去觀察是否有"過擬合(overfitting)"的現象)
def plot_train_history( history, train_metrics, val_metrics ):
    plt.plot( history.history.get( train_metrics ), '-o' )
    plt.plot( history.history.get( val_metrics ), '-o' )
    plt.ylabel( train_metrics )
    plt.xlabel( 'Epochs' )
    plt.legend( ['train', 'validation'])

plt.figure( figsize=(12,4) )
plt.subplot( 1,2, 1 )
plot_train_history( history, 'loss', 'val_loss' )

plt.subplot( 1, 2, 2 )
plot_train_history( history, 'acc', 'val_acc' )

plt.show()

## 載入測試資料
import pandas as pd
test = pd.read_csv( 'GTSRB/GT-final_test.csv', sep=';')

X_test = []
Y_test = []
# 迭代處理每一筆要測試的圖像檔
for file_name, class_id in zip( list( test['Filename'] ), list( test['ClassId']) ):
    img_path = os.path.join( 'GTSRB/Final_Test/Images/', file_name )
    X_test.append( preprocess_img( io.imread( img_path ) ) )
    Y_test.append( class_id )

# 轉換成numpy ndarray
X_test = np.array( X_test )
Y_test = np.array( Y_test )

print( 'X_test.shape:', X_test.shape )
print( 'Y_test.shape:', Y_test.shape )

# 預測與比對
y_pred = model.predict_classes( X_test )
acc = np.sum( y_pred==Y_test )/np.size( y_pred )
print( 'Test accuracy = {} '.format( acc ) )

