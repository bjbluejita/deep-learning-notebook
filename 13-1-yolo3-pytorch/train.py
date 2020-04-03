#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2020/02/24 13:24:42
@Author  :   LY 
@Version :   1.0
@URL     :   https://github.com/liuyuemaicha/PyTorch-YOLOv3/blob/master/train.py
             https://zhuanlan.zhihu.com/p/69278495
             https://zhuanlan.zhihu.com/p/32525231
             https://zhuanlan.zhihu.com/p/35325884
             https://zhuanlan.zhihu.com/p/36899263
@License :   (C)Copyright 2017-2020
@Desc    :   None
'''
# here put the import lib
from __future__ import division

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from models import Darknet
from utils.utils import weights_init_normal, load_classes
from utils.datasets import ListDataset
from utils.parse_config import parse_data_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=1, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="13-1-yolo-pytorch/config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="13-1-yolo-pytorch/config/trafficlight.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    # device = torch.device( 'cpu' )
    os.makedirs( 'output', exist_ok=True )
    os.makedirs( 'checkpoints', exist_ok=True )

    # Get data configuration
    data_config = parse_data_config( opt.data_config )
    train_path = data_config[ 'train' ]
    valid_path = data_config[ 'valid' ]
    class_names = load_classes( data_config[ 'names' ] )

    # Initiate model
    model = Darknet( opt.model_def ).to( device )
    model.apply( weights_init_normal )

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith( '.pth' ):
            model.load_state_dict( torch.load( opt.pretrained_weights ) )
        else:
            model.load_darknet_weights( opt.pretrained_weights )

    # Get dataloader
    dataset = ListDataset(  train_path, augment=True, multiscale=opt.multiscale_training )
    dataloader = torch.utils.data.DataLoader( 
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    optimizer = torch.optim.Adam( model.parameters() )

    metrics = [
        'grid_size',
        'loss',
        'x',
        'y',
        'w',
        'h',
        'conf',
        'cls',
        'cls_acc',
        'recall50',
        'recall75',
        'precision',
        'conf_obj',
        'conf_noobj'
    ]

    for epoch in range( opt.epochs ):
        model.train()
        start_time = time.time()
        for batch_i, ( img_path, imgs, targets ) in enumerate( dataloader ):
            batches_done = len( dataloader ) * epoch + batch_i
            imgs = Variable( imgs.to( device ) )
            targets = Variable( targets.to( device ), requires_grad=False )
           
            loss, outputs = model( imgs, targets )
            torch.cuda.empty_cache()
            loss.backward()

            if batches_done % opt.gradient_accumulations == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            print( 'Epoch[{}/{}] step[{}/{}] loss[{}]'.format( epoch, opt.epochs, batch_i, len( dataloader ), loss ) )

        if epoch % opt.checkpoint_interval == 0:
            print( 'Save model to ', opt.pretrained_weights )
            torch.save( model.state_dict(), '13-1-yolo-pytorch/model/yolov3.pth' )
        