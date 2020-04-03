'''
Created on 2019年1月1日
https://www.jianshu.com/p/d23b5994db64
@author: Administrator
'''
from keras.preprocessing.image import ImageDataGenerator
#from keras.datasets import mnist
#from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

num_classes = 10


# 專案的根目錄路徑
ROOT_DIR  = os.getcwd()
DATA_PATH = os.path.join( ROOT_DIR, "data" )
TRAIN_IMG_PATH = os.path.join( DATA_PATH, "train" )
img_path = os.path.join( TRAIN_IMG_PATH, 'c03.png' )
img = cv2.imread( img_path, 0 )
#img = img.astype( 'float32' ) / 256
img_train= img.reshape( 1, 28, 28, 1 )

imagegen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
maskgen = ImageDataGenerator(
     rescale = 1./255,
     rotation_range=20,
     width_shift_range=0.2,
     height_shift_range=0.2,
     horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
imagegen.fit( img_train )
image_iter = imagegen.flow( img_train, y=None, batch_size=8 )

ibatch = 0
for index, x_batch  in enumerate( image_iter ):
    if ( index % 24 ) == 0 :
        ibatch = 0
        plt.show()
        continue
    plt.subplot( 3, 8, ibatch+1 )
    plt.imshow( x_batch.reshape( 28, 28) )
    ibatch += 1


