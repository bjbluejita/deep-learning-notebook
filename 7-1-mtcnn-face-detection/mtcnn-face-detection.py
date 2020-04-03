'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年05月05日 14:10
@Description: 
@URL: https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/7.1-mtcnn-face-detection.ipynb
@version: V1.0
'''
import warnings
warnings.filterwarnings( 'ignore' )

# Utilities相關函式庫
import os
import sys
import random
from tqdm import tqdm
from scipy import misc

import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

import detect_face

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join( ROOT_DIR, 'data' )
TEST_IMGS_PATH = os.path.join( DATA_DIR, 'images' )
TEST_VIDEOS_PATH = os.path.join( DATA_DIR, 'videos' )

minsize = 20  # 最小的臉部的大小
threshold = [ 0.6, 0.7, 0.7 ]
factor = 0.709

gpu_memeory_fraction = 1.0

# 構建MTCNN網絡架構與模型
print( 'Createing networks and loading parameters' )

# 由於這個mtcnn是使用tensorflow構建而成, 所以需要使用tensorflow執行
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction=gpu_memeory_fraction )
    sess = tf.Session( config=tf.ConfigProto( gpu_options=gpu_options,
                                              log_device_placement=False ))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn( sess, None )

test_image = os.path.join( TEST_IMGS_PATH, 'part2.jpg' )
# 使用OpenCV讀入測試圖像

# 注意: OpenCV讀進來的圖像所產生的Numpy Ndaary格式是BGR (B:Blue, G: Green, R: Red)
#      跟使用PIL或skimage的格式RGB (R: Red, G: Green, B:Blue)在色階(channel)的順序上有所不同
bgr_image = cv2.imread( test_image )
rgb_image = bgr_image[ :, :, ::-1 ]

# 偵測人臉
bounding_boxes, _ = detect_face.detect_face(rgb_image, minsize, pnet, rnet, onet, threshold, factor)

# 複製原圖像
draw = bgr_image.copy()

# 被偵測到的臉部總數
faces_detected = len(bounding_boxes)

print('Total faces detected ：{}'.format(faces_detected))

# 保留裁剪下來的人臉圖像
crop_faces=[]

# 每一個 bounding_box包括了（x1,y1,x2,y2,confidence score)：
# 　　左上角座標 (x1,y1)
#     右下角座標 (x2,y2)
#     信心分數 confidence score

# 迭代每一個偵測出來的邊界框
for face_position in bounding_boxes:
    # 把資料由float轉成int
    face_position=face_position.astype(int)

    # 取出左上角座標 (x1,y1)與右下角座標 (x2,y2)
    # 由於有可能預測出來的臉在圖像的圖邊而導致座標值為負值
    # 因此進行的負值的偵測與修正
    x1 = face_position[0] if face_position[0] > 0 else 0
    y1 = face_position[1] if face_position[1] > 0 else 0
    x2 = face_position[2] if face_position[2] > 0 else 0
    y2 = face_position[3] if face_position[3] > 0 else 0

    # 在原圖像上畫上這些邊界框
    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 進行臉部圖像裁剪
    crop=bgr_image[y1:y2,x1:x2,]

    # 把臉部大小進行大小的修改便拋出給其它模組進行辨識(face recognition)
    # crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )
    # crop_faces.append(crop)
    # plt.imshow(crop)
    # plt.show()

# 設定展示的大小
plt.figure(figsize=(20,10))

# 展示偵測出來的結果
plt.imshow(draw[:,:,::-1]) # 轉換成RGB來給matplotlib展示
plt.show()