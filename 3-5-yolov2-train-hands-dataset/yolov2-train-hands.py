'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年03月21日 11:17
@Description: 
@URL: https://github.com/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/3.5-yolov2-train-hands-dataset.ipynb
@version: V1.0
'''

import os
import sys
sys.path.append( os.getcwd() )
import random
from tqdm import tqdm

import numpy as np

# 圖像處理相關函式庫
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import colorsys
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# 序列/反序列化相關函式庫
import pickle

# 深度學習相關函式庫
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import keras.backend as K
import tensorflow as tf

# 專案相關函式庫
from preprocessing import parse_annotation, BatchGenerator
from utils import WeightReader, decode_netout, draw_boxes, normalize
from utils import draw_bgr_image_boxes, draw_rgb_image_boxes,draw_pil_image_boxes

# 專案的根目錄路徑
ROOT_DIR = os.getcwd()

# 訓練/驗證用的資料目錄
DATA_PATH = os.path.join( ROOT_DIR, 'data' )
# 資料集目錄
DATA_SET_PATH = os.path.join( DATA_PATH, 'hands' )
TRAIN_DATA_PATH = os.path.join( DATA_SET_PATH, 'train' )

TRAIN_IMGS_PATH = os.path.join( TRAIN_DATA_PATH, 'pos' )
TRAIN_ANNOTATION_PATH = os.path.join( TRAIN_DATA_PATH, 'posGt' )

#但在這次的訓練中, 我們只要判斷是左/右手就行了
# 圖像類別的Label-encoding
map_categories = { 0:'left_hand', 1:'right_hand' }

# 取得所有圖像的圖像類別列表
labels=list(map_categories.values())

#設定YOLOv2模型的設定與參數
LABELS = labels
IMAGE_H, IMAGE_W = 416, 416
GRID_H, GRID_W = 13, 13
BOX            = 5
CLASS          = len( LABELS )
CLASS_WEIGHTS  = np.ones( CLASS, dtype='float32' )
OBJ_THRESHOLD  = 0.5
NMS_THRESHOLD  = 0.45
ANCHORS        = [ 0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828 ]

NO_OBJECT_SCALE = 1.0
OBJECT_SCALE    = 5.0
COORD_SCALE     = 1.0
CLASS_SCALE     = 1.0

BATCH_SIZE = 16
WARM_UP_BATCHS = 0
TRUE_BOX_BUFFER = 50

wt_path = os.path.join( ROOT_DIR, 'yolov2.weights' )

train_image_folder = TRAIN_IMGS_PATH
train_annot_folder = TRAIN_ANNOTATION_PATH
valid_image_folder = TRAIN_IMGS_PATH
valid_annot_folder = TRAIN_ANNOTATION_PATH

def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)

input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

# Layer 1
x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 2
x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 3
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 4
x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 5
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 6
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 7
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 8
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 9
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 10
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
x = BatchNormalization(name='norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 11
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
x = BatchNormalization(name='norm_11')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 12
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
x = BatchNormalization(name='norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 13
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)

skip_connection = x

x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 14
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 15
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
x = BatchNormalization(name='norm_15')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 16
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
x = BatchNormalization(name='norm_16')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 17
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
x = BatchNormalization(name='norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 18
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
x = BatchNormalization(name='norm_18')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 19
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
x = BatchNormalization(name='norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 20
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 21
skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)

x = concatenate([skip_connection, x])

# Layer 22
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 23
x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

# small hack to allow true_boxes to be registered when Keras build the model
# for more information: https://github.com/fchollet/keras/issues/2790
output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model([input_image, true_boxes], output)

model.summary()

#載入預訓練的模型權重
weight_reader = WeightReader( wt_path )
weight_reader.reset()
nb_conv = 23

for i in range( 1, nb_conv+1 ):
    conv_layer = model.get_layer( 'conv_' + str(i) )

    # 在conv_1~conv_22的卷積組合裡都包含了"conv + norm"二層, 只有conv_23是獨立一層
    if i < nb_conv:
        print( 'handle norm_ ' + str(i) + ' start' )
        norm_layer = model.get_layer( 'norm_' + str(i) ) # 取得BatchNormalization層

        size = np.prod( norm_layer.get_weights()[0].shape ) ## 取得BatchNormalization層的參數量
        print( 'shape:', norm_layer.get_weights()[0].shape )

        beta = weight_reader.read_bytes( size )
        gamma = weight_reader.read_bytes( size )
        mean = weight_reader.read_bytes( size )
        var = weight_reader.read_bytes( size )
        weights = norm_layer.set_weights( [ gamma, beta, mean, var ] )
        print( 'handle norm_' + str(i) + ' completed' )

    if len( conv_layer.get_weights() ) > 1:
        print( 'handle conv_' + str(i) + ' start' )
        print( 'len:', len( conv_layer.get_weights() ) )

        bias = weight_reader.read_bytes( np.prod( conv_layer.get_weights()[1].shape ))
        kernel = weight_reader.read_bytes( np.prod( conv_layer.get_weights()[0].shape ))
        kernel = kernel.reshape( list( reversed( conv_layer.get_weights()[0].shape ) ) )
        kernel = kernel.transpose( [ 2, 3, 1, 0 ] )
        conv_layer.set_weights( [ kernel, bias ] )
        print( 'handle conv_' + str(i) + ' completed' )
    else:
        print( 'handle conv_' + str(i) + ' only kernel start' )
        kernel = weight_reader.read_bytes( np.prod( conv_layer.get_weights()[0].shape ) )
        kernel = kernel.reshape( list( reversed( conv_layer.get_weights()[0].shape )))
        kernel = kernel.transpose( [2, 3, 1, 0] )
        print( 'kernel shape: ', kernel.shape )
        conv_layer.set_weights( [kernel] )
        print( 'handle conv_' + str(i) + ' completed' )

#設定要微調(fine-tune)的模型層級權重
layer = model.layers[ -4 ] # 找出最後一層的卷積層
weights = layer.get_weights()

new_kernel = np.random.normal( size=weights[0].shape ) / ( GRID_H*GRID_W)
new_bias   = np.random.normal( size=weights[1].shape ) / ( GRID_H*GRID_W)
layer.set_weights( [ new_kernel, new_bias ] ) # 重初始化權重

#損失函數:
def custom_loss( y_true, y_pred ):
    mask_shape = tf.shape( y_true )[:4]

    cell_x = tf.to_float( tf.reshape( tf.tile( tf.range(GRID_W), [GRID_H]), ( 1, GRID_H, GRID_W, 1, 1) ))
    cell_y = tf.transpose( cell_x, ( 0, 2, 1, 3 ,4 ) )

    cell_grid = tf.tile( tf.concat( [ cell_x, cell_y ], -1 ), [BATCH_SIZE, 1, 1, 5, 1] )

    coord_mask = tf.zeros( mask_shape )
    conf_mask  = tf.zeros( mask_shape )
    class_mask = tf.zeros( mask_shape )

    seen = tf.Variable( 0. )
    total_recall = tf.Variable( 0. )

    '''
    Adjust prediction
    '''
    ### adjust x and y
    pred_box_xy = tf.sigmoid( y_pred[ ..., :2 ] ) + cell_grid

    ### adjust w and h
    pred_box_wh = tf.exp( y_pred[ ..., 2:4 ]) * np.reshape( ANCHORS, [1,1,1,BOX,2] )

    ### adjust confidence
    pred_box_conf = tf.sigmoid( y_pred[ ...,4] )

    ### adjust class probabilities
    pred_box_class = y_pred[ ..., 5: ]

    '''
    Adjust ground truth
    '''
    ### adjust x and y
    true_box_xy = y_true[ ..., 0:2 ]

    ### adjust w and h
    true_box_wh = y_true[ ..., 2:4 ]

    ### adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins = true_box_xy - true_wh_half
    true_maxes = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.
    pred_mins = pred_box_xy - pred_wh_half
    pred_maxes = pred_box_xy + pred_wh_half

    intersect_minis = tf.maximum( pred_mins, true_mins )
    intersect_maxes = tf.minimum( pred_maxes, true_maxes )
    intersect_wh = tf.maximum( intersect_maxes-intersect_minis, 0.)
    intersect_araeas = intersect_wh[ ..., 0 ] * intersect_wh[ ..., 1 ]

    true_areas = true_box_wh[ ...,0 ] * true_box_wh[ ..., 1 ]
    pred_areas = pred_box_wh[ ..., 0 ] * pred_box_wh[ ..., 1 ]

    union_areas = pred_areas + true_areas - intersect_araeas
    iou_socres = tf.truediv( intersect_araeas, union_areas )

    true_box_conf = iou_socres * y_true[ ..., 4 ]

    ### adjust class probabilities
    true_box_class = tf.argmax( y_true[ ..., 5:], -1 )

    '''
    Determine the masks
    '''
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims( y_true[ ..., 4 ], axis=-1 ) * COORD_SCALE

    ### confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[ ..., 0:2 ]
    true_wh = true_boxes[ ..., 2:4 ]

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_xy = tf.expand_dims( pred_box_xy, 4 )
    pred_wh = tf.expand_dims( pred_box_wh, 4 )

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_minis = tf.maximum( pred_mins, true_maxes )
    intersect_maxes = tf.minimum( pred_maxes, true_mins )
    intersect_wh = tf.maximum( intersect_maxes - intersect_minis, 0. )
    intersect_araeas = intersect_wh[ ..., 0 ] * intersect_wh[ ..., 1 ]

    true_areas = true_wh[ ..., 0 ] * true_wh[ ..., 1 ]
    pred_areas = pred_wh[ ..., 0 ] * pred_wh[ ..., 1 ]

    union_areas = pred_areas + true_areas - intersect_araeas
    iou_scores = tf.truediv( intersect_araeas, union_areas )

    best_ious = tf.reduce_max( iou_scores, axis=4 )
    conf_mask = conf_mask + tf.to_float( best_ious < 0.6 ) * ( 1 -y_true[ ..., 4] ) * NO_OBJECT_SCALE

    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[ ..., 4 ] * OBJECT_SCALE

    ### class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[ ..., 4 ] * tf.gather( CLASS_WEIGHTS, true_box_class ) * CLASS_SCALE

    '''
    Warm-up training
    '''
    no_boxes_mask = tf.to_float( coord_mask < COORD_SCALE / 2. )
    seen = tf.assign_add( seen, 1 )

    true_box_xy, true_box_wh, coord_mask = tf.cond( tf.less( seen, WARM_UP_BATCHS ),
                                                    lambda :[ true_box_xy + ( 0.5 + cell_grid ) * no_boxes_mask,
                                                              true_box_wh + tf.ones_like( true_box_wh ) * np.reshape( ANCHORS, [ 1, 1,1, BOX, 2 ] ) *
                                                              no_boxes_mask,
                                                              tf.ones_like( coord_mask ) ],
                                                    lambda :[true_box_xy,
                                                             true_box_wh,
                                                             coord_mask]
                                                    )

    '''
    Finalize the loss
    '''
    nb_coord_box = tf.reduce_sum( tf.to_float( coord_mask > 0.0 ) )
    nb_conf_box  = tf.reduce_sum( tf.to_float( conf_mask > 0.0 ) )
    nb_class_box = tf.reduce_sum( tf.to_float( class_mask > 0.0 ) )

    loss_xy = tf.reduce_sum( tf.square( true_box_xy - pred_box_xy ) * coord_mask ) / ( nb_coord_box + 1e-6 ) / 2.
    loss_wh = tf.reduce_sum( tf.square( true_box_wh - pred_box_wh)  * coord_mask ) / ( nb_coord_box + 1e-6 ) / 2.
    loss_conf = tf.reduce_sum( tf.square( true_box_conf - pred_box_conf ) * conf_mask ) / ( nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits( labels= true_box_class, logits=pred_box_class )
    loss_class = tf.reduce_sum( loss_class * class_mask ) / ( nb_class_box + 1e-6 )

    loss = loss_xy + loss_wh + loss_conf + loss_class

    nb_true_box = tf.reduce_sum( y_true[ ..., 4 ] )
    nb_pred_box = tf.reduce_sum( tf.to_float(  true_box_conf > 0.5 ) * tf.to_float( pred_box_conf >0.3 ) )

    '''
    Debugging code
    '''
    current_recall = nb_pred_box / ( nb_true_box +  1e-6 )
    total_recall = tf.assign_add( total_recall, current_recall )

    loss = tf.Print( loss, [ tf.zeros((1)) ], message='Dummy line', summarize=1000 )
    loss = tf.Print( loss, [loss_xy], message='Loss XY', summarize=1000 )
    loss = tf.Print( loss, [loss_wh], message='Loss WH', summarize=1000 )
    loss = tf.Print( loss, [loss_conf], message='Loss conf', summarize=1000 )
    loss = tf.Print( loss, [loss_class], message='Loss Class', summarize=1000 )
    loss = tf.Print( loss, [loss], message='Total loss', summarize=1000 )
    loss = tf.Print( loss, [current_recall], message='Current Recall', summarize=1000 )
    loss = tf.Print( loss, [total_recall/seen], message='Average Recall', summarize=1000 )

    return loss

#用來產生Keras訓練模型的BatchGenerator的設定
generator_config = {
    'IMAGE_H' : IMAGE_H,
    'IMAGE_W' : IMAGE_W,
    'GRID_H'  : GRID_H,
    'GRID_W'  : GRID_W,
    'BOX'     : BOX,
    'LABELS'  : LABELS,
    'CLASS'   : len( LABELS ),
    'ANCHORS' : ANCHORS,
    'BATCH_SIZE' : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : 50
}
#由於hands資料集的標註檔並不是採用PASCAL VOC格式而是自行定義的格式
from tqdm import tqdm
from PIL import Image
def parse_hands_annotation( ann_dir, img_dir, labels=[] ):
    '''
    解析圖像標註檔

    根據手部標註檔存放的目錄路徑迭代地解析每一個標註檔，
    將每個圖像的檔名(filename)、圖像的寬(width)、高(height)、圖像的類別(name)以
    及物體的邊界框的坐標(xmin,ymin,xmax,ymax)擷取出來。以下是圖像標註檔的範例:

    % bbGt version=3
    leftHand_driver 87 295 57 67 0 0 0 0 0 0 0
    rightHand_driver 223 283 62 64 0 0 0 0 0 0 0
    leftHand_passenger 483 356 91 71 0 0 0 0 0 0 0
    rightHand_passenger 548 328 86 70 0 0 0 0 0 0 0

    擷取目標: [hands_class x y w h ...]
    hands_class: leftHand_driver/leftHand_passenger: left_hand, rightHand_driver/rightHand_passenger: right_hand,

    參數:
        ann_dir: 圖像標註檔存放的目錄路徑
        img_dir: 圖像檔存放的目錄路徑
        labels: 圖像資料集的物體類別列表

    回傳:
        all_imgs: 一個列表物件, 每一個物件都包括了要訓練用的重要資訊。例如:
                    {
                        'filename': '/tmp/img/img001.jpg',
                        'width': 128,
                        'height': 128,
                        'object':[
                            {'name':'person',xmin:0, ymin:0, xmax:28, ymax:28},
                            {'name':'person',xmin:45, ymin:45, xmax:60, ymax:60}
                        ]
                    }
        seen_labels: 一個字典物件(k:圖像類別, v:出現的次數)用來檢視每一個圖像類別出現的次數
    '''
    print( 'start parsing annotation' )

    # 產生一個標註圖資料標註的mapping
    hands_label_map = { 'leftHand_driver':'left_hand', 'leftHand_passenger': 'left_hand',
                        'rightHand_driver':'right_hand', 'rightHand_passenger':'right_hand' }
    all_imgs = []
    seen_labels = {}

    # 迭代每個標註檔
    for ann in tqdm( sorted( os.listdir( ann_dir ) ) ):
        img = { 'object' : [] }
        # 處理圖檔檔案路徑
        img_filename = ann[ 0:len(ann)-3 ] + 'png'

        # 圖檔檔案路徑
        img[ 'filename' ] = os.path.join( img_dir, img_filename )

        im = Image.open( img['filename'] )
        img_width, img_height = im.size

        # 圖檔大小
        img['width'] = img_width
        img['height'] = img_height

        line = 0 # 行數
        with open( os.path.join( ann_dir, ann ), 'r' ) as fann:
            # 一行一行讀進來處理
            for cnt, line in enumerate( fann ):
                # 忽略第一行的資料
                if cnt == 0 : continue

                # 建立物件來保留bbox
                obj = {}

                tokens = line.split()
                label = hands_label_map[ tokens[0] ]
                bbox_x = int( tokens[1] )
                bbox_y = int( tokens[2] )
                bbox_w = int( tokens[3] )
                bbox_h = int( tokens[4] )

                obj['name'] = label

                if obj['name'] in seen_labels:
                    seen_labels[ obj['name'] ] += 1
                else:
                    seen_labels[ obj['name'] ] = 1

                obj[ 'xmin' ] = bbox_x
                obj[ 'ymin' ] = bbox_y
                obj[ 'xmax' ] = bbox_x + bbox_w
                obj[ 'ymax' ] = bbox_y + bbox_h

                #檢看是是否有物體的標籤是沒有在傳入的物體類別(labels)中
                if len( labels ) > 0 and obj['name'] not in labels:
                    continue
                else:
                    img['object'] += [obj]
            if len( ['object'] ) > 0:
                all_imgs += [img]

    print( 'Parsing annotation completed!' )
    print( 'Total: {} images preprocessed.'.format( len( all_imgs ) ) )
    return all_imgs, seen_labels



# 進行圖像標註檔的解析 (在Racoon資料集的標註採用的是PASCAL VOC的XML格式)
train_imgs, seen_train_labels = parse_hands_annotation( train_annot_folder, train_image_folder, labels=LABELS )
# 建立一個訓練用的資料產生器
train_batch = BatchGenerator( train_imgs, generator_config, norm=normalize )

# 進行圖像標註檔的解析 (在Racoon資料集的標註採用的是PASCAL VOC的XML格式)
valid_imgs, seen_valid_labels = parse_hands_annotation( valid_annot_folder, valid_image_folder, labels=LABELS )
# 建立一個驗證用的資料產生器
valid_batch = BatchGenerator( valid_imgs, generator_config, norm=normalize, jitter=False )

#設置一些回調函式並開始訓練
early_stop = EarlyStopping( monitor='val_loss',
                            min_delta=0.001,
                            patience=30,  #如果超過30次的循環在loss的收歛上沒有改善就停止
                            mode='min',
                            verbose=1 )

# 每次的訓練循都去比較模型的loss是否有改善, 有就把模型的權重儲存下來
checkpoint = ModelCheckpoint( 'weight_kangaroo.h5',
                              monitor='val_loss',
                              verbose=1,
                              save_best_only=True,
                              mode='min',
                              period=1 )

optimizer = Adam( lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0 )
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
#optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile( loss=custom_loss, optimizer=optimizer )

history = model.fit_generator( generator=train_batch,
                               steps_per_epoch=len( train_batch ),
                               epochs=200,
                               verbose=0,
                               validation_data=valid_batch,
                               validation_steps=len( valid_batch ),
                               callbacks=[ early_stop, checkpoint ],
                               max_queue_size=3
                               )

#圖像的物體偵測
# 載入訓練好的模型權重
model.load_weights( 'weight_kangaroo.h5' )

# 產生一個Dummy的標籤輸入

# 在訓練階段放的是真實的邊界框與圖像類別訊息
# 但在預測階段還是需要有一個Dummy的輸入, 因為定義在網絡的結構中有兩個輸入：
#   1.圖像的輸人
#   2.圖像邊界框/錨點/信心分數的輸入
dummy_array = np.zeros( ( 1, 1, 1, TRUE_BOX_BUFFER, 4) )

# 選一張圖像
img_filepath = train_imgs[ np.random.randint( len( train_imgs))][ 'filename']

image = cv2.imread( img_filepath )

plt.figure( figsize=(10, 10) )

# 進行圖像輸入的前處理
input_image = cv2.resize( image, ( IMAGE_W, IMAGE_H ) )
input_image = input_image / 255.
input_image = np.expand_dims( input_image, 0 ) # 增加 batch dimension

# 進行圖像偵測
netout = model.predict( [ input_image, dummy_array ] )

# 解析網絡的輸出來取得最後偵測出來的邊界框(bounding boxes)列表
boxes = decode_netout( netout[0],
                       obj_threshold=OBJ_THRESHOLD,
                       nms_threshold=NMS_THRESHOLD,
                       anchors=ANCHORS,
                       nb_class=CLASS )

# "draw_bgr_image_boxes"
# 一個簡單把邊界框與預測結果打印到原始圖像(BGR)上的工具函式
# 參數: image 是image的numpy ndarray [h, w, channels(BGR)]
#       boxes 是偵測的結果
#       labels 是模型訓練的圖像類別列表
# 回傳： image 是image的numpy ndarray [h, w, channels(RGB)]
image = draw_bgr_image_boxes( image, boxes, labels=LABELS )

#把最後的結果秀出來
plt.imshow( image )
plt.show()