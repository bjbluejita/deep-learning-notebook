'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年03月06日 17:20
@Description: 
@URL: https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/3.1-yolov2-object-detection.ipynb
@version: V1.0
'''
import warnings
warnings.filterwarnings( 'ignore' )

import os
import random
import numpy as np
import platform
import tensorflow
import keras
from keras import backend as K
from keras.models import  load_model
from keras.utils import  plot_model

from PIL import Image, ImageDraw, ImageFont
import colorsys
import imghdr
from yad2k.models.keras_yolo import yolo_eval, yolo_head
import cv2
from tqdm import tqdm
import time
import matplotlib
import matplotlib.pyplot as plt
import IPython.display as display

print( 'Platform: {}'.format( platform.platform() ))
print( 'Tensorflow version {}'.format( tensorflow.__version__ ))
print( 'Keras version {}'.format( keras.__version__ ))

#設定相關基相設定與參數
ROOT_DIR = os.getcwd()

# 模型相關資料的目錄
MODEL_PATH = os.path.join( ROOT_DIR, 'model_data' )

# 圖像類別定義文件路徑，預設為"coco_classes.txt"
CLASSES_FILE_PATH = os.path.join( MODEL_PATH, 'coco_classes.txt' )

# 設定模型權重檔案
MODEL_FILENAME = 'yolov2_coco_608X608.h5'
MODEL_FILE_PATH = os.path.join( MODEL_PATH, MODEL_FILENAME )

# 設定錨點文件檔案
ANCHORS_FILENAME = 'yolov2_coco_608X608_anchors.txt'
ANCHORS_FILE_PATH = os.path.join( MODEL_PATH, ANCHORS_FILENAME )

# 模型輸入的圖像大小與顏色頻道數
IMAGE_HEIGHT = 608
IMAGE_WEIGHT = 608
IMAGE_CHANNELS = 3

# 設定邊界框過濾的閥值(confidence score)
SCORE_THRESHOLD = 0.3

# IOU(Intersection over Union)的閥值
IOU_THRESHOLD = 0.5

# 驗證用的圖像目錄，預設為"images/"
TEST_PATH = os.path.join( ROOT_DIR, 'images')

## 處理結果的圖像目錄，預設為"images/out"
OUTPUT_PATH = os.path.join( ROOT_DIR, 'images_out' )
# 檢查"OUTPUT_PATH"的目錄是否存在
if not os.path.exists( OUTPUT_PATH ):
    print( 'Create output directory {}'.format( OUTPUT_PATH ) )
    os.mkdir( OUTPUT_PATH )

# 取得Tensorflow的session物件
sess = K.get_session()

# 取得物件的類別名稱(要看模型是用那一種資料集來進行訓練)
with open( CLASSES_FILE_PATH ) as f:
    class_names = f.readlines()
# 把這些圖像類別放到一個列表中
class_names = [ c.strip() for c in class_names ]

# 取得圖像的預設錨點 (x1,y1,x2,y2,x3,y3,x4,y4,x5,y5)
with open( ANCHORS_FILE_PATH ) as f:
    anchors = f.readline()
    anchors = [ float(x) for x in anchors.split(',') ]
    anchors = np.array( anchors ).reshape( -1, 2 ) # 錨點是 (x, y) 一組
#載入網絡結構模型
yolo_model = load_model( MODEL_FILE_PATH )
# 打印模型結構
yolo_model.summary()

# 產生網絡拓撲圖
plot_model( yolo_model, to_file='yolov2_model.png' )
#秀出網絡拓撲圖
display.Image( 'yolov2_model.png'  )

num_classes = len( class_names )
num_anchors = len( anchors )

# 取出YOLOv2模型的最後一層 -> "conv2d_23"的輸出為 (19, 19, 425)
model_output_channels = yolo_model.layers[-1].output_shape[-1]
# 最後的features map的維度是19x19x425 (為什麼是425)
# 主要是因為YOLOv2把整個圖像分成19x19個小區塊(cell)
# 而每個小區塊(cell)要predict"num_anchors(5)"個邊界框(bounding boxes)
# 因此(19, 19, 425)中的前2個維度就是定義每一個小區塊(cell), 而"425"則代表著:
#     - x, y, width, height (定義邊界框的座標) -> 4個浮點數
#     - 邊界框裡包含物體的信心分數(confidence score) -> 1個浮點數
#     - 每一種類別的機率(num_classes) -> 80個浮點數 (MS Coco的資料集有80種圖像類別)

# 驗證 model, anchors, 與 classes 是相符合的
if model_output_channels != ( num_anchors * ( num_classes + 5 ) ):
    print( 'Mismatch between model and given anchor class size.' )

print( 'model[{}], anchor[{}], and classes[{}] loaded'.format( MODEL_FILENAME, num_anchors, num_classes ))
## 檢查模型的Input圖像大小
model_image_size = yolo_model.layers[0].input_shape[1:3] # (h, w, channel)
is_fixed_size = model_image_size != (None, None)  # 如果 h, w都是"None"代表任何圖像的size都可以
print( 'Flag: is_fixed_size[{}]'.format( is_fixed_size ))

print( 'Model input image size:', model_image_size )

# 為不同的bounding boxes產生不同的顏色
hsv_tuples = [ ( x / len( class_names ), 1., 1.)  for x in  range( len(class_names) ) ]

colors = list( map( lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples ))
colors = list(map(lambda x: ( int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

random.seed( 10101 ) # 設定random的seed讓整個運行中的顏色有一致性。
random.shuffle( colors ) # 打散顏色
random.seed( None ) # 回複到預設值

# 將最後的圖層特徵轉換為邊界框的相對參數。
yolo_outputs = yolo_head( yolo_model.output, anchors, len( class_names ) )

input_image_shape = K.placeholder( shape=(2, ) )
# 使用YOLO模型評估給定的輸入批次和相關的閥值, 返回過濾後的邊界框。
boxes, scores, classes = yolo_eval( yolo_outputs,
                                    input_image_shape,
                                    score_threshold=SCORE_THRESHOLD,
                                    iou_threshold=IOU_THRESHOLD )

#圖像的物體偵測
# 指定要用來進行物體偵測的圖像
TEST_IMG = '7b47020b4965b327e4c1fd19a47ba2b1.jpg'
TEST_IMG_PATH = os.path.join( TEST_PATH, TEST_IMG )
# 將原始的圖像秀出來
plt.figure( figsize=(15,8) )
img = plt.imread( TEST_IMG_PATH )
plt.imshow( img )
plt.show()

#圖像資料的張量形狀(tensor shape)為(608, 608, 3)也就是圖像RGB三色頻且大小為608x608
#圖像的張量需要進行歸一化的處理(除以255)以及資料型別為浮點數32
# 載入圖像
image = Image.open( TEST_IMG_PATH )
print( 'Before image resize,', np.array( image, dtype='float32').shape )
# 修改輸入圖像大小來符合模型的要求
resized_image = image.resize( tuple( reversed( model_image_size ) ), Image.BICUBIC  )
image_data = np.array(  resized_image, dtype='float32' ) # (img_height, img_width, img_channels)
print( 'After image resized:', image_data.shape )

# 進行圖像歸一處理
image_data /= 255.

# 增加"批次"的維度
print( 'Before expand dims:', image_data.shape )
image_data = np.expand_dims( image_data, 0 ) # 增加 batch dimension
print( 'After expand dims:', image_data.shape )


# 取得YOLOv2模型偵測後的結果
out_boxes, out_scores, out_classes = sess.run( [ boxes, scores, classes ],
                                               feed_dict={
                                                   yolo_model.input: image_data,
                                                   input_image_shape: [ image.size[1], image.size[0] ],
                                                   K.learning_phase(): 0
                                               })
print( 'out_boxes.shape:', out_boxes.shape, out_boxes.dtype )
print( 'out_scores.shape:', out_scores.shape, out_scores.dtype )
print( 'out_classes.shape:', out_classes.shape, out_classes.dtype )

# 打印找到個bounding boxes
print( 'Found {} boxes for {}'.format( len(out_boxes), TEST_IMG ))

# 打印每個找到的物體類別、信心分數與邊界框的左上角與右下角座標
for i, c in reversed( list( enumerate( out_classes ) ) ):
    predicted_class = class_names[c]
    box = out_boxes[i]
    score = out_scores[i]

    top, left, bottom, right = box
    top = max( 0, np.floor( top + 0.5 ).astype( 'int32' ) )
    left = max( 0, np.floor( left + 0.5 ).astype( 'int32' ) )
    bottom = min( image.size[1], np.floor( bottom+0.5 ).astype( 'int32' ))
    right = min( image.size[0], np.floor( right+0.5 ).astype( 'int32' ))

    label = '{} {:.2f}[ ({},{}), ({},{})]'.format( predicted_class, score, top, left, bottom, right )
    print( label )

#圖像偵測結果
font = ImageFont.truetype( font='font/FiraMono-Medium.otf',
                           size=np.floor( 3e-2 * image.size[1]+0.5).astype('int32') )
# 設定邊界框的厚度
thickness = ( image.size[0] + image.size[1] ) // 300

# 迭代每個找到的物體類別
for i, c in reversed( list( enumerate( out_classes ) )):
    predicted_class = class_names[c]
    box = out_boxes[i]
    score = out_scores[i]

    label = '{} {:.2f}'.format( predicted_class, score )
    draw = ImageDraw.Draw( image ) # 使用Draw物件來在圖像上畫圖
    label_size = draw.textsize( label, font ) # 將"物體類別"與"信心分數"以文字的方式展現

    # "邊界框"的座標
    top, left, bottom, right = box
    top = max( 0, np.floor( top+0.5 ).astype('int32') )
    left = max( 0, np.floor( left+0.5 ).astype('int32') )
    bottom = min( image.size[1], np.floor( bottom+0.5 ).astype('int32') )
    right = min( image.size[0], np.floor( right+0.5 ).astype('int32') )

    # 打印圖像類別, 邊界框的左上角及右下角的座標
    print( label, (left, top), (right, bottom) )

    if top - label_size[1] >= 0:
        text_origin = np.array( [left, top-label_size[1] ] )
    else:
        text_origin = np.array( [left, top+1] )

    for i in range( thickness ):
        # 畫"邊界框"
        draw.rectangle(
            [ left+i, top+i, right-i, bottom-i ],
            outline=colors[c] )

    # 畫一個四方型來做為文字標籤的背景
    draw.rectangle( [ tuple(text_origin), tuple(text_origin+label_size)],
                    fill=colors[c] )
    # 把文字標籤的資訊放到圖像上
    draw.text( text_origin, label, fill=(0,0,0), font=font )

# 將最後的圖像秀出來
plt.figure(figsize=(15,8)) # 設定展示圖像的大小
plt.imshow(image) # 展示
plt.show()




