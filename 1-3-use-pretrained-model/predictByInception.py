'''
Created on 2019年1月16日

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

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions

# 載入權重
model_inception_v3 = InceptionV3(weights='imagenet')

# 載入圖像檔
img_file = 'C:/Users/Administrator/Pictures/20.jpg'
# InceptionV3的模型的輸入是299x299
img = load_img(img_file, target_size=(299, 299)) 

# 將圖像資料轉為numpy array
image = image.img_to_array(img) # RGB
print("image.shape:", image.shape)

# 調整張量的維度
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
print("image.shape:", image.shape)

# 準備模型所需要的圖像前處理
image = preprocess_input(image)

# 預測所有產出類別的機率
y_pred = model_inception_v3.predict(image)

# 將機率轉換為類別標籤
label = decode_predictions(y_pred)

# 檢索最可能的結果，例如最高的概率
label = label[0][0]

# 打印預測結果
print('%s (%.2f%%)' % (label[1], label[2]*100))