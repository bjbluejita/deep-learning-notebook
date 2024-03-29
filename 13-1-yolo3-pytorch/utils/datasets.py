#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2020/03/01 11:41:38
@Author  :   LY 
@Version :   1.0
@URL     :   https://github.com/liuyuemaicha/PyTorch-YOLOv3/blob/master/utils/datasets.py
@License :   (C)Copyright 2017-2020
@Desc    :   None
'''
# here put the import lib
import glob
import random
import os
import sys
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch
import torch.nn.functional as F

from .augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

def pad_to_square( img, pad_value ):
    '''
    将一个img填充成正方形img
    '''
    c, h, w = img.shape
    dim_diff = np.abs( h - w )
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ( 0, 0, pad1, pad2 ) if h <= w else ( pad1, pad2, 0, 0 )
    img = F.pad( img, pad, 'constant', value=pad_value )

    return img, pad


def resize( image, size ):
    '''
    输入进行下/上采样
    '''
    image = F.interpolate( image.unsqueeze( 0 ), size=size, mode='nearest' ).squeeze( 0 )
    return image

def random_resize( images, min_size=288, max_size=448 ):
    '''
    从一组image挑选一张，并按其维度对其他image进行下/上采样（resize）
    '''
    new_size = range.sample( list( range( min_size, max_size + 1, 32 )), 1 )[0]
    images = F.interpolate( images, size=new_size, mode='nearest' )
    return images

class ImageFolder( Dataset ):
    def __init__( self, folder_path, img_size=416 ):
        self.files = sorted( glob.glob( "%s/*.*" % folder_path ) )
        self.img_size = img_size

    def __getitem__( self, index ):
        img_path = self.files[ index % len( self.files ) ]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()( Image.open( img_path ) )
        # Pad to square resolution
        img, _ = pad_to_square( img, 0 )
        # Resize
        img = resize( img, self.img_size )

        return img_path, img

    def __len__( self ):
        return len( self.files )


class ListDataset( Dataset ):
    def __init__( self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True ):
        with open( list_path, 'r' ) as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace( 'images', 'labels' ).replace( '.png', '.txt' ).replace( '.jpg', '.txt' )
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objcets = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__( self, index ):
        # ---------
        #  Image
        # ---------        
        img_path = self.img_files[ index % len( self.img_files ) ].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()( Image.open( img_path ).convert( 'RGB') )

        # Handle images with less than three channels
        if len( img.shape ) != 3:
            img = img.unsqueeze( 0 )
            img = img.expand( ( 3, img.shape[1:] ) )

        _, h, w = img.shape
        h_factor, w_factor = ( h, w ) if self.normalized_labels else ( 1, 1 )
        # Pad to square resolution
        img, pad = pad_to_square( img, 0 )
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        label_path = self.label_files[ index % len( self.img_files ) ].rstrip()

        targets = None
        if os.path.exists( label_path ):
            boxes = torch.from_numpy( np.loadtxt( label_path ).reshape( -1, 5 ))
            # Extract coordinates for unpadded + unscaled image
            # label: (center_x, center_y, width, height)
            x1 = w_factor * ( boxes[ :, 1 ] - boxes[ :, 3 ] / 2 )
            y1 = h_factor * ( boxes[ :, 2 ] - boxes[ :, 4 ] / 2 )
            x2 = w_factor * ( boxes[ :, 1 ] + boxes[ :, 3 ] / 2 )
            y2 = h_factor * ( boxes[ :, 2 ] + boxes[ :, 4 ] / 2 )
            # Adjust for added padding
            x1 += pad[ 0 ]
            y1 += pad[ 2 ]
            x2 += pad[ 1 ]
            y2 += pad[ 3 ]

            # 计算box的中心点(anchor)和box宽和长
            boxes[ :, 1 ] = ( ( x1 + x2 ) / 2 ) / padded_w
            boxes[ :, 2 ] = ( ( y1 + y2 ) / 2 ) / padded_h
            boxes[ :, 3 ] = boxes[ :, 3 ] * w_factor / padded_w
            boxes[ :, 4 ] = boxes[ :, 4 ] * h_factor / padded_h

            targets = torch.zeros( ( len( boxes ), 6 ) )
            targets[ :, 1: ] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip( img, targets )

        return img_path, img, targets

    def collate_fn( self, batch ):
        paths, imgs, targets = list( zip( *batch ) )
        # Remove empty placeholder targets
        targets = [ boxes for boxes in targets if boxes is not None ]
        # Add sample index to targets
        for i, boxes in enumerate( targets ):
            boxes[ :, 0 ] = i
        try:
            targets = torch.cat( targets, 0 )
        except RuntimeError:
            print( batch, paths, targets )
        # Select new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice( range( self.min_size, self.max_size + 1, 32 ))
        # Resize imgage to input shape
        imgs = torch.stack( [ resize( img, self.img_size ) for img in imgs ] )
        self.batch_count += 1
        return paths, imgs, targets

    def __len__( self ):
        return len( self.img_files ) 


if __name__ == '__main__':
    a = torch.randn( ( 3, 37, 42 ) )
    img_a, pad1 = pad_to_square( a, pad_value=1 )
    print( img_a.shape )

    a = torch.randn( ( 3, 37, 42 ) )
    img_a = resize( a, size=[ 30, 24 ]  )
    print( img_a.shape )