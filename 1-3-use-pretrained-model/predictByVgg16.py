'''
Created on 2019年1月16日

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

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


model_vgg16 = VGG16()
# 載入圖像檔
img_file = 'C:/Users/Administrator/Pictures/14.jpg'
image = load_img(img_file, target_size=(224, 224)) # 因為VGG16的模型的輸入是224x224

# 將圖像資料轉為numpy array
image = img_to_array(image) # RGB
print("image.shape:", image.shape)

# 調整張量的維度
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
print("image.shape:", image.shape)

# 準備VGG模型的圖像
image = preprocess_input(image)

# 預測所有產出類別的機率
y_pred = model_vgg16.predict(image)

# 將機率轉換為類別標籤
label = decode_predictions(y_pred)

# 檢索最可能的結果，例如最高的概率
label = label[0][0]

# 打印預測結果
print('%s (%.2f%%)' % (label[1], label[2]*100))