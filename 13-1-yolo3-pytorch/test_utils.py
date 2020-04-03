#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_utils.py
@Time    :   2020/03/08 10:58:52
@Author  :   LY 
@Version :   1.0
@URL     :
@License :   (C)Copyright 2017-2020
@Desc    :   utils 单元测试
'''
# here put the import lib
import unittest
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models import Upsample, EmptyLayer, YOLOLayer, Darknet
from utils.datasets import ImageFolder, ListDataset, pad_to_square

class test_utils( unittest.TestCase ):

    def test_ImageFolder( self ):
        img_folder = "13-1-yolo-pytorch/samples"
        dataloader = DataLoader(
            ImageFolder( img_folder, img_size=416 ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        for batch_i, ( img_path, input_imgs ) in enumerate( dataloader ):
            print( img_path, '->', input_imgs[ 0, 0, 100, 50: ])

    def test_ListDataset( self ):
        img_paht = 'E:/ML_data/trainvalno5k.txt'
        dataset = ListDataset(  img_paht, augment=True, multiscale=True )
        print( len( dataset ))
        dataloader_list = DataLoader( 
            dataset,
            batch_size=6,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=dataset.collate_fn
            )
        # for batch_i, ( img_path, imgs, targets ) in enumerate( dataloader_list ):
        #     print(img_path )

    def test_dataloader_getitem( self ):
        unloader = transforms.ToPILImage()

        self.normalized_labels = True
        # img_path =   'e:/ML_data/images/train2014/COCO_train2014_000000000092.jpg'
        # label_path = 'e:/ML_data/labels/train2014/COCO_train2014_000000000092.txt'

        img_path =   'E:/trafficlight_detect/light_img/IMG_20181124_125821.jpg'
        label_path = 'E:/trafficlight_detect/labels/IMG_20181124_125821.txt'

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()( Image.open( img_path ).convert( 'RGB') )
        
        fig, ax = plt.subplots( 1 )
        image_orig = unloader(img )
        ax.imshow( image_orig  )
        #ax.add_patch( patches.Rectangle( xy=( 225, 131), width=75, height=50, linewidth=1 ))
        plt.show()
        # Handle images with less than three channels
        if len( img.shape ) != 3:
            img = img.unsqueeze( 0 )
            img = img.expand( ( 3, img.shape[1:] ) )

        _, h, w = img.shape
        h_factor, w_factor = ( h, w ) if self.normalized_labels else ( 1, 1 )
        img, pad = pad_to_square( img, 0 )
        _, padded_h, padded_w = img.shape
        
        image_padded = unloader(img )
        fig, ax = plt.subplots( 1 )
        ax.imshow( image_padded  )
        plt.show()

        boxes = torch.from_numpy( np.loadtxt( label_path ).reshape( -1, 5 ))
        x1 = w_factor * ( boxes[ :, 1 ] - boxes[ :, 3 ] / 2 )
        y1 = h_factor * ( boxes[ :, 2 ] - boxes[ :, 4 ] / 2 )
        x2 = w_factor * ( boxes[ :, 1 ] + boxes[ :, 3 ] / 2 )
        y2 = h_factor * ( boxes[ :, 2 ] + boxes[ :, 4 ] / 2 )

        fig, ax = plt.subplots( 1 )
        ax.imshow( image_orig  )
        #ax.add_patch( patches.Rectangle( xy=( 225, 131), width=75, height=50, linewidth=1 ))
        for box_i, ( xx1, yy1, xx2, yy2 ) in enumerate( zip( x1, y1, x2, y2 ) ):
            ax.add_patch( patches.Rectangle( xy=( xx1.item(), yy1.item() ), width=( xx2 - xx1).item(), height=( yy2 - yy1).item(), linewidth=1 ))
        plt.show()

        # Adjust for added padding
        x1 += pad[ 0 ]
        y1 += pad[ 2 ]
        x2 += pad[ 1 ]
        y2 += pad[ 3 ]

        fig, ax = plt.subplots( 1 )
        ax.imshow( image_padded  )
        #ax.add_patch( patches.Rectangle( xy=( 225, 131), width=75, height=50, linewidth=1 ))
        for box_i, ( xx1, yy1, xx2, yy2 ) in enumerate( zip( x1, y1, x2, y2 ) ):
            ax.add_patch( patches.Rectangle( xy=( xx1.item(), yy1.item() ), width=( xx2 - xx1).item(), height=( yy2 - yy1).item(), linewidth=1 ))
        plt.show()

        boxes[ :, 1 ] = ( ( x1 + x2 ) / 2 ) / padded_w
        boxes[ :, 2 ] = ( ( y1 + y2 ) / 2 ) / padded_h
        boxes[ :, 3 ] = boxes[ :, 3 ] * w_factor / padded_w
        boxes[ :, 4 ] = boxes[ :, 4 ] * h_factor / padded_h
        fig, ax = plt.subplots( 1 )
        ax.imshow( image_padded  )
        #ax.add_patch( patches.Rectangle( xy=( 225, 131), width=75, height=50, linewidth=1 ))
        for box_i in range( len( boxes ) ):
            x = boxes[ box_i, 1 ] * padded_w
            y = boxes[ box_i, 2 ] * padded_h
            width = boxes[ box_i, 3 ] * padded_w
            height = boxes[ box_i, 3 ] * padded_h
            ax.add_patch( patches.Circle( xy=( x.item(), y.item()), radius=10, linewidth=1 ))
        plt.show()

        

if __name__ == '__main__':
    unittest.main()

