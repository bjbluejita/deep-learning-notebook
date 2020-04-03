'''
@Project: 7-0-opencv-face-detection
@Package 
@author: ly
@date Date: 2019年05月05日 11:02
@Description: 
@URL: https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/7.0-opencv-face-detection.ipynb
@version: V1.0
'''
import os
import sys
import random
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join( ROOT_DIR, 'data' )
TEST_IMGS_PATH = os.path.join( DATA_DIR, 'images' )
MODEL_PATH = os.path.join( ROOT_DIR, 'model' )
CV2_MODEL_PATH = os.path.join( MODEL_PATH, 'cv2' )
# Haar權重檔
HAAR_WEIGHT_FILE = os.path.join( CV2_MODEL_PATH, 'haarcascade_frontalface_alt2.xml' )

# 透過cv2.CascadeClassifier來產生實例
faceDetector = cv2.CascadeClassifier( HAAR_WEIGHT_FILE )

# 用來測試OpenCV自帶的Haar的人臉偵測的圖像
test_image = os.path.join( TEST_IMGS_PATH, 'part2.jpg' )

# 使用OpenCV讀入測試圖像

# 注意: OpenCV讀進來的圖像所產生的Numpy Ndaary格式是BGR (B:Blue, G: Green, R: Red)
bgr_imagee = cv2.imread( test_image )

# 將BGR圖像轉成灰階
gray_image = cv2.cvtColor( bgr_imagee, cv2.COLOR_BGR2GRAY )
# 秀出原圖像
plt.imshow( bgr_imagee[ :, :, ::-1 ] )
plt.show()

#進行偵測圖像中的人臉
# detectMultiScale方法它可以檢測出圖片中所有的人臉，並將人臉用向量保存各個人臉的坐標，大小（用矩形表示）
faces = faceDetector.detectMultiScale( gray_image, 1.3, 5 )
# 偵測出來的結果的資料結構 (x, y, w, h) -> 左上角的(x,y)座標, 以及矩型的寬高(w, h)
print( 'Face detected: ', len( faces ) )
print( 'Result data shape:', faces.shape )
print( '1st data:', faces[0] )

#把結果在原圖像中展現
for( x, y, w, h ) in faces:
    # 透過OpenCV來把邊界框畫出來
    # rectangle
    # 參數:
    #     要畫矩形的圖像
    #     左上角座標 tuple (x, y)
    #     右下角座標 tuple (x, y)
    #     邊框顏色 tuple (r,g,b)
    #     邊框寬度 int
    cv2.rectangle( bgr_imagee, (x, y), ( x+w, y+h ), (0, 255,0), 2 )

plt.figure( figsize=( 20, 10) )
plt.imshow( bgr_imagee[ :, :, ::-1 ] )
plt.show()