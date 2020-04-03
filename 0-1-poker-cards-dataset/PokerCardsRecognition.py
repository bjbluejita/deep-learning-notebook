'''
Created on 2018年12月18日

@author: Administrator
'''
# 匯入相關所需的模組
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from pathlib import PurePath
import cv2

import keras
from keras.preprocessing.image import ImageDataGenerator
# 引入Tensorboard
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical


# 專案的根目錄路徑
ROOT_DIR  = os.getcwd()

# 置放資料的路徑
DATA_PATH = os.path.join( ROOT_DIR, "data" )

# 置放原始圖像檔案的路徑
ORIGIN_IMG_PATH = os.path.join( DATA_PATH, "origin_imgs" )

# 置放要用來訓練用圖像檔案的路徑
TRAIN_IMG_PATH = os.path.join( DATA_PATH, "train" )

# 訓練用的圖像大小與色階
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHANNEL = 1
WEIGHTS_FILE = 'weights.best.hdf5'

# 取得原始圖像的檔案路徑
all_ima_paths = glob.glob( os.path.join( ORIGIN_IMG_PATH, "*.PNG" ) )

'''
# 進行圖像色階轉換及大小的修改
for img_path in all_ima_paths:
    filename = ( PurePath( img_path ).stem )
    #print( filename )
    new_filename = filename + ".png"  # 更換附檔名 
    card_img_grey = cv2.imread( img_path, 0 )  # 使用OpenCV以灰階讀入
    card_img = cv2.resize( card_img_grey, ( IMG_HEIGHT, IMG_WIDTH ) )  # 轉換大小  
    cv2.imwrite( os.path.join( TRAIN_IMG_PATH, new_filename ), card_img )


plt.figure(figsize=(8,8))  # 設定每個圖像顯示的大小
# 產生一個3x2網格的組合圖像
for i in range(0,6):
    img_file = 'h0' + str(i+1) + '.PNG'
    img = cv2.imread( os.path.join( ORIGIN_IMG_PATH, img_file ) )
    img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
    
    plt.subplot( 330+1+i ) # (331) -> 第一個子圖像, (332) -> 第二個子圖像
    plt.title( img_file )
    plt.axis( 'off' )
    plt.imshow( img )
# 展現出圖像
plt.show()

plt.figure(figsize=(8,8))  # 設定每個圖像顯示的大小

# 產生一個3x2網格的組合圖像
for i in range(0, 6):
    img_file = 'h0' + str( i+1 ) + '.png'
    img = cv2.imread( os.path.join( TRAIN_IMG_PATH, img_file ), 0 )
    
    plt.subplot( 330 + 1 + i )
    plt.title( img_file )
    plt.axis( 'off' )
    plt.imshow( img, cmap=plt.get_cmap('gray') )
    
# 展現出圖像
plt.show()
'''

#train
num_classes = 52
img_rows, img_cols, img_channels = IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL # 圖像是 49像素 x 33像素 (灰色階: 1)
input_shape = ( img_rows, img_cols, img_channels )

# 載入標籤資料檔
cards_data = pd.read_excel( os.path.join( DATA_PATH, 'cards_data.xlsx') )
# 取得"card_label"的欄位資料
cards_label = cards_data['card_label']
# 產生相關的查找的字典物件
idx_to_label = { k:v for k, v in cards_label.iteritems() }
label_to_idx = { v:k for k, v in cards_label.iteritems() }

# 取得所有圖像的標籤值
y = np.array( cards_label.index.values )
# 進行標籤的one-hot編碼
y_train = to_categorical(y, num_classes)
y_test = y_train.copy()

# 將每個圖像從檔案中讀取進來
imgs = []
all_img_paths = glob.glob( os.path.join( TRAIN_IMG_PATH, '*.png'))

#進行圖像每個像素值的型別轉換與歸一化處理
for img_path in all_img_paths:
    img = cv2.imread( img_path, 0 )
    img = img.astype( 'float32' ) / 256
    imgs.append( img )
    
# 取得要進行訓練用的灰階圖像
X = np.array( imgs )

# 將圖像數據集的維度進行改變 
# 改變前: [樣本數, 圖像寬, 圖像高] -> 改變後: [樣本數, 圖像寬, 圖像高, 圖像頻道數]
X_train = X.reshape( X.shape[0], 28, 28, 1 )
X_test = X_train.copy()

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


# 產生一個Keras序貫模型
def cnn_model():
    model = Sequential()
    
    model.add( Conv2D( 32, (3,3), padding='same', activation='relu', input_shape=input_shape ) )
    model.add( Conv2D( 32, (3,3), activation='relu' ) )
    #model.add( MaxPool2D( pool_size=(2,2) ))
    model.add( MaxPooling2D( pool_size=(2,2) ))
    
    model.add( Dropout(0.2) )
    
    model.add( Conv2D( 64, (3,3), padding='same', activation='relu' ))
    model.add( Conv2D(64, (3,3), padding='same', activation='relu'  ))
    #model.add( MaxPool2D(pool_size=(2,2)))
    model.add( MaxPooling2D(pool_size=(2,2)))
    
    model.add( Dropout(0.2) )
    
    model.add( Conv2D(128, (3,3), padding='same', activation='relu' ))
    model.add( Conv2D(128, (3,3), padding='same', activation='relu' ))
    #model.add( MaxPool2D( pool_size=(2,2) ))
    model.add( MaxPooling2D( pool_size=(2,2) ))
    model.add( Dropout(0.2) )
    
    model.add( Flatten() )
    model.add( Dense(512, activation='relu' ) )
    model.add( Dropout(0.2) )
    model.add( Dense( num_classes, activation='softmax') )
    
    return model

model = cnn_model() # 初始化一個模型
model.summary()     # 秀出模型架構

# 讓我們先配置一個常用的組合來作為後續優化的基準點
lr = 0.001
sgd = SGD( lr=lr, decay=1e-6, momentum=0.9, nesterov=True )

model.compile( optimizer=sgd, 
               loss='categorical_crossentropy', 
               metrics=['accuracy'] )

if os.path.isfile( WEIGHTS_FILE ):
    #load weight file
    print( 'load weight file' )
    model.load_weights( WEIGHTS_FILE )
else:
    #create weight file
    print( 'create weight file' )
    model.save_weights( WEIGHTS_FILE, overwrite=True )

#因為我們只有52張圖像, 因此利用現有的圖像來生成新的訓練圖像，這將是一個很好的方式來增加訓練數據集的大小。
#讓我們直接使用keras的內置功能來完成圖像增強 (Data Augmentation)。
datagen_train = ImageDataGenerator( rotation_range=3. )
datagen_train.fit( X_train )

#訓練 (Training)
batch_size = 64
steps_per_epoch = 2000
training_epochs = 700

checkpointer = ModelCheckpoint( filepath=WEIGHTS_FILE, verbose=1, period=1  )

# 引入Tensorboard
tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                         histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                         batch_size=batch_size,     # 用多大量的数据计算直方图
                         write_graph=True,  # 是否存储网络结构图
                         write_grads=True, # 是否可视化梯度直方图
                         write_images=True,# 是否可视化参数
                         embeddings_freq=0, 
                         embeddings_layer_names=None, 
                         embeddings_metadata=None)

'''
# 透過data generator來產生訓練資料, 由於資料是可持續產生, 我們可以透過設定'steps_per_epoch'的數量來讓模型可以有更多的訓練批次
history = model.fit_generator( generator=datagen_train.flow( X_train, y_train, batch_size=batch_size ), 
                               steps_per_epoch=steps_per_epoch, 
                               epochs=training_epochs,
                               callbacks=[checkpointer, tbCallBack ] )


#透過趨勢圖來觀察訓練與驗證的走向 (特別去觀察是否有"過擬合(overfitting)"的現象)
import matplotlib.pyplot as plt

def plot_train_history( history, train_metrics ):
    plt.plot( history.history.get( train_metrics ) )
    plt.ylabel( train_metrics )
    plt.xlabel( 'Epochs' )
    plt.legend()
    
plt.figure( figsize=(12,4) )
plt.subplot( 1, 2, 1 )
plot_train_history( history, 'loss' )

plt.subplot( 1, 2, 2 )
plot_train_history( history, 'acc' )
plt.show()
'''
'''
score = model.evaluate( X_train, y_train, verbose=1 )
print( "Test loss: ", score[0] )
print( "Test accuracy:", score[1] )

# 打散圖像集的順序
randomize = np.arange( len( X_test ) )
np.random.shuffle( randomize )
X_test_randomize = X_test[ randomize ]
y_test_randomize = y_test[ randomize ]

# 計算打散後的圖像集驗證
score = model.evaluate( X_test_randomize, y_test_randomize, verbose=1  )
print( "Test loss: ", score[0] )
print( "Test accuracy:", score[1] )
'''

#
# 取得原始圖像的檔案路徑
all_ima_paths = glob.glob( os.path.join( ORIGIN_IMG_PATH, "*.PNG" ) )
plt.figure( figsize=(6,6) )
i = 1
for img_path in all_ima_paths:
    filename = ( PurePath( img_path ).stem )
    #print( filename )
    new_filename = filename + ".png"  # 更換附檔名 
    org_img = cv2.imread( img_path )
    card_img_grey = cv2.imread( img_path, 0 )  # 使用OpenCV以灰階讀入
    card_img = cv2.resize( card_img_grey, ( IMG_HEIGHT, IMG_WIDTH  ) )  # 轉換大小 
    card_img = card_img.astype( 'float32' ) / 256
    card_img = card_img.reshape( 1, IMG_HEIGHT, IMG_WIDTH, 1 ) 
    
    plt.subplot(4, 13, i )
    predict = model.predict( card_img, verbose=1 )
    predict_class = idx_to_label.get( np.argmax(predict) )
    plt.title( "[{}]".format(predict_class)   )
    plt.imshow( org_img )
    plt.axis('off')
    
    i += 1
plt.show()
