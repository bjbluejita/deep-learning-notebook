'''
Created on 2019年1月14日

@author: Administrator
'''
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# load data
(X_train, y_train), ( X_test, y_test ) = mnist.load_data( )

plt.figure( figsize=(8,8) )

for i in range( 0, 6 ):
    plt.subplot( 330+1+i )
    plt.title( y_train[i] )
    plt.axis( 'off' )
    plt.imshow( X_train[i], cmap=plt.get_cmap('gray'))
plt.show()

# 將圖像數據集的維度進行改變 
# 改變前: [樣本數, 圖像寬, 圖像高] -> 改變後: [樣本數, 圖像寬, 圖像高, 圖像頻道數]
X_train = X_train.reshape( X_train.shape[0], 28, 28, 1 )
X_test = X_test.reshape( X_test.shape[0], 28, 28, 1 )

# 將像素值由"整數(0~255)"換成"浮點數(0.0~255.0)"
X_train = X_train.astype( 'float32' )
X_test = X_test.astype( 'float32' )

# 定義"圖像數據增強產生器(ImageDataGenerator)"的參數
#隨機旋轉 
datagen = ImageDataGenerator( rotation_range=90 )
#隨機偏移 
shift = 0.2
datagen = ImageDataGenerator( width_shift_range=shift, height_shift_range=shift )
#隨機推移錯切 
shear_range=1.25 # 推移錯切的強度
datagen = ImageDataGenerator(shear_range=shear_range)
#隨機鏡像翻轉 
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)


# 透過訓練數據集來訓練(fit)圖像數據增強產生器(ImageDataGenerator)的實例
datagen.fit( X_train )

# 設定要"圖像數據增強產生器(ImageDataGenerator)"產生的圖像批次值(batch size)
# "圖像數據增強產生器(ImageDataGenerator)"會根據設定回傳指定批次量的新生成圖像數據
for X_batch, y_batch in datagen.flow( X_train, y_train, batch_size=9 ):
    plt.figure( figsize=(8,8) )
    # 產生一個3x3網格的組合圖像
    for i in range(0, 9):
        plt.subplot( 331+i )
        plt.title( y_batch[i] )
        plt.axis( 'off' )
        plt.imshow( X_batch[i].reshape( 28, 28 ), cmap=plt.get_cmap('gray') )
    plt.show()