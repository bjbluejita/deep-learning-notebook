#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   unittest_model.py
@Time    :   2020/03/05 17:28:01
@Author  :   LY 
@Version :   1.0
@URL     :
@License :   (C)Copyright 2017-2020
@Desc    :   model.py 单元测试程序
'''
# here put the import lib
import unittest
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from models import Upsample, EmptyLayer, YOLOLayer, Darknet
from utils.parse_config import parse_model_config

class test_Darknet_model( unittest.TestCase ):

    def test_Upsample( self ):
        upsample = Upsample( scale_factor = 2, mode='nearest' )
        print( upsample )

    def test_EmptyLayer( self ):
        emptyLayer = EmptyLayer(  )
        print( emptyLayer )

    def test_YOLOLayer( self ):
        anchor_idxs = [ 6,7,8  ]
        anchors = [ 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326 ]
        anchors = [ ( anchors[ i ], anchors[ i + 1 ] ) for i in range( 0, len( anchors ), 2 ) ]
        anchors = [ anchors[ i ] for i in anchor_idxs ]

        yoloLayer = YOLOLayer( anchors=anchors, num_classes=80 )
        x = torch.randn( ( 4, 255, 47, 47 ) )
        yoloLayer.forward( x, img_dim=yoloLayer.img_dim )
        print( yoloLayer )

    def test_parse_model_config( self ):
        module_defs = parse_model_config(  './13-1-yolo-pytorch/config/yolov3.cfg' )
        print( module_defs )

    def test_a( self ):
        anchors=[ ( 1.0, 2.0 ), ( 1.0, 3.0 ), ( 4.5, 6.1 ), (5.1, 6.0 ) ]
        aGen = ( [ a_w, a_h  ] for a_w, a_h  in anchors )
        for a in aGen:
            print( '***', a )

    def test_bool_not( sefl ):
        a = True
        print( not a )

    def test_Darknet( self ):
        darkNet = Darknet( './13-1-yolo-pytorch/config/yolov3.cfg' )
        print( darkNet )

if __name__ == '__main__':
    unittest.main()
