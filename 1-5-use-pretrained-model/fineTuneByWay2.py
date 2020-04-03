'''
Created on 2019年1月21日
https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/1.5-use-pretrained-model-2.ipynb#%E7%A7%BB%E8%8A%B1
@author: Administrator
'''
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
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint

print("numpy version: {}".format( np.__version__ ))

WEIGHTS_FILE = 'weights.best.hdf5'

# 專案的根目錄路徑
ROOT_DIR =  'F:\\workspace\\Tensorflow\\src\\deep-learning-with-keras-notebooks\\1-4-small-datasets-image-augmentation'

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


from keras.applications import VGG16
conv_base = VGG16(include_top=False, # 在這裡告訴 keras我們只需要卷積基底的權重模型資訊
                  weights='imagenet', 
                  input_shape=( 150, 150, 3) )
conv_base.summary( )

#卷積基底:提取特徴 + 串接新的密集分類層:重新訓練
datagen = ImageDataGenerator(rescale=1./255) # 產生一個"圖像資料產生器"物件
batch_size = 20 # 設定每次產生的圖像的數據批量

#數據擴充(data augmentation): 擴展conv_base模型，並進行端(end)對端(end)的訓練。
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add( conv_base )    # 把預訓練的卷積基底疊上去
model.add( layers.Flatten() )  # 打平
model.add( layers.Dense( 256, activation='relu') )
model.add( layers.Dropout(0.2) )
model.add( layers.Dense( 1, activation='sigmoid') )
print( '-----------new model--------------')
model.summary(line_length=100 )

# 看一下“凍結前”有多少可以被訓練的權重
print( 'This is the num of trainable weights before'
        ' frezzing the conv base', len( model.trainable_weights ) )
conv_base.trainable = False
print( 'This is the num of trainable weights after'
        ' frezzing the conv base', len( model.trainable_weights ) )

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# 請注意: 驗證用的資料不要進行資料的增強
test_datagen = ImageDataGenerator( rescale=1./255 )

train_generator = train_datagen.flow_from_directory(
        # 圖像資料的目錄
        train_dir,
        # 設定圖像的高(height)與寬(width)
        target_size=(150, 150),
        batch_size=20,
        # 因為我們的目標資料集只有兩類(cat & dog)
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

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


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

#checkpoint 文件保存
if os.path.isfile( WEIGHTS_FILE ):
    #load weight file
    print( 'load weight file' )
    model.load_weights( WEIGHTS_FILE )
else:
    #create weight file
    print( 'create weight file' )
    model.save_weights( WEIGHTS_FILE, overwrite=True )
    


history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      callbacks = [tbCallBack, checkpointer],
      verbose=1)
