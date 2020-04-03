#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   augmentations.py
@Time    :   2020/03/01 11:44:07
@Author  :   LY 
@Version :   1.0
@URL     :   https://github.com/liuyuemaicha/PyTorch-YOLOv3/blob/master/utils/augmentations.py
@License :   (C)Copyright 2017-2020
@Desc    :   None
'''
# here put the import lib
import torch
import torch.nn.functional as F
import numpy as np


# 按照给定维度翻转张量 水平翻转
def horisontal_flip( images, targets=None ):
    # print images
    images = torch.flip( images, dims=[2] )
    #print( images )
    return images, targets

if __name__ == '__main__':
    a = torch.randn( ( 2, 5, 4 ) )
    flip_a, _ = horisontal_flip( a )
    print( flip_a.shape )
