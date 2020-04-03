'''
Created on 2019年1月17日

@author: Administrator
'''
# 這個Jupyter Notebook的環境
import platform
import tensorflow
import keras
print("Platform: {}".format(platform.platform()))
print("Tensorflow version: {}".format(tensorflow.__version__))
print("Keras version: {}".format(keras.__version__))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython.display import Image
import os, shutil

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras.layers import Dropout
from keras.utils import plot_model
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

WEIGHTS_FILE = 'weights.best.hdf5'

# 專案的根目錄路徑
ROOT_DIR = os.getcwd()

# 置放coco圖像資料與標註資料的目錄
DATA_PATH = os.path.join(ROOT_DIR, "data")

# 原始數據集的路徑
original_dataset_dir = os.path.join(DATA_PATH, "train")
# 存儲小數據集的目錄
base_dir = os.path.join(DATA_PATH, "cats_and_dogs_small")
# 我們的訓練資料的目錄
train_dir = os.path.join(base_dir, 'train')
# 我們的驗證資料的目錄
validation_dir = os.path.join(base_dir, 'validation')
# 我們的測試資料的目錄
test_dir = os.path.join(base_dir, 'test')
# 貓的圖片的訓練資料目錄
train_cats_dir = os.path.join(train_dir, 'cats')
# 狗的圖片的訓練資料目錄
train_dogs_dir = os.path.join(train_dir, 'dogs')
# 貓的圖片的驗證資料目錄
validation_cats_dir = os.path.join(validation_dir, 'cats')
# 狗的圖片的驗證資料目錄
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# 貓的圖片的測試資料目錄
test_cats_dir = os.path.join(test_dir, 'cats')
# 狗的圖片的測試資料目錄
test_dogs_dir = os.path.join(test_dir, 'dogs')

# 所有的圖像將重新被進行歸一化處理 Rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True )
test_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

batch_size = 64
steps_per_epoch = 2000
training_epochs = 30

# 直接從檔案目錄讀取圖像檔資料
train_generator = train_datagen.flow_from_directory(
                                  #這是圖像資料的目錄
                                  directory=train_dir, 
                                  # 所有的圖像大小會被轉換成150x150
                                  target_size=(150, 150), 
                                  # 每次產生20圖像的批次資料
                                  batch_size = batch_size,
                                  # 由於這是一個二元分類問題, y的lable值也會被轉換成二元的標籤
                                  class_mode='binary')
# 直接從檔案目錄讀取圖像檔資料
validation_generator = test_datagen.flow_from_directory(
                                  directory=validation_dir, 
                                  target_size=(150,150), 
                                  class_mode ='binary',
                                  batch_size=batch_size )

model = models.Sequential()
model.add( layers.Conv2D( 32, (3,3), activation='relu', input_shape=(150, 150,3) ))
model.add( layers.MaxPooling2D( (2,2)) )
model.add( Dropout(0.2) )
model.add( layers.Conv2D( 64, (3,3), activation='relu' ))
model.add( layers.MaxPooling2D( (2,2) ))
model.add( Dropout(0.2) )
model.add( layers.Conv2D( 128, (3,3), activation='relu' ))
model.add( layers.MaxPooling2D( (2,2) ))
model.add( Dropout(0.2) )
model.add( layers.Conv2D( 128, (3,3), activation='relu' ))
model.add( layers.MaxPooling2D( (2,2) ))
model.add( Dropout(0.2) )
model.add( layers.Flatten() )
model.add( Dropout(0.2) )
model.add( layers.Dense( 512, activation='relu' ))
model.add( layers.Dense( 1, activation='sigmoid' ))

model.compile( optimizer=optimizers.RMSprop( lr=1e-4 ), 
               loss='binary_crossentropy', 
               metrics=['acc'] )

#checkpoint 文件保存
if os.path.isfile( WEIGHTS_FILE ):
    #load weight file
    print( 'load weight file' )
    model.load_weights( WEIGHTS_FILE )
else:
    #create weight file
    print( 'create weight file' )
    model.save_weights( WEIGHTS_FILE, overwrite=True )
    

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
#checkpoint 文件保存
checkpointer = ModelCheckpoint( filepath=WEIGHTS_FILE, verbose=1, period=1  )

model.fit_generator( generator=train_generator, 
                     steps_per_epoch=steps_per_epoch, 
                     epochs=training_epochs, 
                     verbose=1, 
                     callbacks = [tbCallBack, checkpointer], 
                     validation_data=validation_generator, 
                     validation_steps=50 )