'''
Created on 2019年1月21日
https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/1.5-use-pretrained-model-2.ipynb#%E7%A7%BB%E8%8A%B1
@author: Administrator
'''
import platform
import tensorflow
import keras
from astropy.utils.argparse import directory
from astropy import samp
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
print("numpy version: {}".format( np.__version__ ))

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

# 提取圖像特徵
def extract_features( directory, sample_count ):# 影像的目錄, 要處理的圖像數
    features = np.zeros( shape=( sample_count, 4, 4, 512 ))  # 根據VGG16(卷積基底)的最後一層的輪出張量規格
    labels = np.zeros( shape=(sample_count ) )
    
    # 產生一個"圖像資料產生器"實例(資料是在檔案目錄中), 每呼叫它一次, 它會吐出特定批次數的圖像資料
    generator = datagen.flow_from_directory( directory=directory, 
                                             target_size=(150, 150), 
                                             class_mode='binary', 
                                             batch_size=batch_size )
    # 讓我們把訓練資料集所有的圖像都跑過一次
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict( inputs_batch ) # 透過“卷積基底”來淬取圖像特徵
        features[ i*batch_size : (i+1)*batch_size ] = features_batch
        labels[ i*batch_size : (i+1)*batch_size ] = labels_batch
        i +=1 
        
        if i*batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    
    print( 'extract_features complete!' )
    return features, labels

train_features, train_labels = extract_features( train_dir, 2000 )  # 訓練資料的圖像特徵淬取
validation_features, validation_labels = extract_features(validation_dir, 1000) # 驗證資料的圖像特徵淬取
test_features, test_labels = extract_features(test_dir, 1000) # 測試資料的圖像特徵淬取

#壓扁(flatten)成（樣本數, 8192）
train_features  = np.reshape( train_features, (2000, 4*4*512) )
validation_features = np.reshape( validation_features, (1000, (4*4*512)) )
test_features = np.reshape( test_features, (1000, (4*4*512) ) )

#定義我們密集連接(densely-connected)的分類器（注意使用dropout來進行正規化）
from keras import models
from keras import layers
from keras import optimizers

# 產生一個新的密集連接層來做為分類器
model = models.Sequential()
model.add( layers.Dense( 256, activation='relu', input_dim=4*4*512 ))
model.add( layers.Dropout(0.5) )
model.add( layers.Dense( 1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5), 
              loss='binary_crossentropy', 
              metrics=['acc'] )

history = model.fit( train_features, train_labels,
                     epochs=30,
                     batch_size=20,
                     validation_data=( validation_features, validation_labels ))
