'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年05月14日 14:55
@Description: 
@URL: https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/7.3-face-embedding-and-classifier.ipynb
@version: V1.0
'''
# 屏蔽Jupyter的warning訊息
import warnings
warnings.filterwarnings('ignore')

# Utilities相關函式庫
import sys
import os
from tqdm import tqdm
import math
# 多維向量處理相關函式庫
import numpy as np
# 圖像處理相關函式庫
import cv2
# 深度學習相關函式庫
import tensorflow as tf

import facenet
import detect_face

import pickle

from sklearn.svm import  SVC
from sklearn.svm import LinearSVC

# 專案的根目錄路徑
ROOT_DIR = os.getcwd()

# 訓練/驗證用的資料目錄
DATA_PATH = os.path.join(ROOT_DIR, "data")

# 模型的資料目錄
MODEL_PATH = os.path.join(ROOT_DIR, "model")

# FaceNet的模型
FACENET_MODEL_PATH = os.path.join(MODEL_PATH, "facenet","20170512-110547","20170512-110547.pb")

# Classifier的模型
SVM_MODEL_PATH = os.path.join(MODEL_PATH, "svm", "lfw_svm_classifier.pkl")

# 訓練/驗證用的圖像資料目錄
IMG_IN_PATH = os.path.join(DATA_PATH, "lfw")

# 訓練/驗證用的圖像資料目錄
IMG_OUT_PATH = os.path.join(DATA_PATH, "lfw_mtcnnpy_160")

#轉換每張人臉的圖像成為Facenet的人臉特徵向量(128 bytes)表示
# 使用Tensorflow的Facenet模型
with tf.Graph().as_default():
    with tf.Session() as sess:
        datadir = IMG_OUT_PATH# 經過偵測、對齊 & 裁剪後的人臉圖像目錄
        dataset = facenet.get_dataset( datadir )
        # 原始: 取得每個人臉圖像的路徑與標籤
        paths, labels, labels_dict = facenet.get_image_paths_and_labels( dataset )
        print( 'Origin: Number of classes : %d' % len( labels_dict ) )
        print( 'Origin: Number of images : %d' % len( paths ) )

        # 由於lfw的人臉圖像集中有很多的人臉類別只有1張的圖像, 對於訓練來說樣本太少
        # 因此我們只挑選圖像樣本張數大於5張的人臉類別

        # 過濾: 取得每個人臉圖像的路徑與標籤 (>=5)
        paths, labels, labels_dict = facenet.get_image_paths_and_labels(dataset, enable_filter=True, filter_size=5)
        print('Filtered: Number of classes: %d' % len(labels_dict))
        print('Filtered: Number of images: %d' % len(paths))

        #載入Facenet模型
        print( 'Loading feature extraction model' )
        modeldir = FACENET_MODEL_PATH
        facenet.load_model( modeldir )

        images_placeholder = tf.get_default_graph().get_tensor_by_name( 'input:0' )
        embeddings = tf.get_default_graph().get_tensor_by_name( 'embeddings:0' )
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name( 'phase_train:0' )
        embedding_size = embeddings.get_shape()[1]

        # 計算人臉特徵向量 (128 bytes)
        print( 'Calculating feature for images' )
        batch_size = 100 # 批次量
        image_size = 160 # 要做為Facenet的圖像輸入的大小

        nrof_images = len(paths) # 總共要處理的人臉圖像
        # 計算總共要跑的批次數
        nrof_batches_per_epoch = int( math.ceil( 1.0 * nrof_images / batch_size ) )
        # 構建一個變數來保存"人臉特徵向量"
        emb_array = np.zeros( ( nrof_images, embedding_size ))

        for i in tqdm( range( nrof_batches_per_epoch ) ):
            start_index = i * batch_size
            end_index = min( (i+1)*batch_size, nrof_images )
            paths_batch = paths[ start_index : end_index ]
            images = facenet.load_data( paths_batch, False, False, image_size )
            feed_dict ={ images_placeholder: images, phase_train_placeholder: False }
            emb_array[ start_index : end_index ] = sess.run( embeddings, feed_dict=feed_dict )
