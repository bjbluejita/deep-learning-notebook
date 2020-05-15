#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/05/15 11:17:38
@Author  :   LY 
@Version :   1.0
@URL     :   https://zhuanlan.zhihu.com/p/112655246
             https://github.com/ymcui/Chinese-BERT-wwm
@License :   (C)Copyright 2017-2020
@Desc    :   基于pytorch的bert实践
'''
# here put the import lib
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import *
import os

path = 'data/'
bert_path = 'E:/ML_data/bert'
tokenizer = BertTokenizer( vocab_file= os.path.join( bert_path, 'vocab.txt' ) )

# 预处理数据集
print( '预处理数据集' )
input_ids = []
input_types = []

