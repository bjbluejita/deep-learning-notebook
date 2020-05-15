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
input_ids = []  # input char ids
input_types = []   # segment ids
input_masks = []    # attention mask
label = []
pad_size = 32      # 也称为 max_len (前期统计分析，文本长度最大值为38，取32即覆盖99%)

with open( os.path.join( bert_path, 'train_simple.txt' ), encoding='utf-8' ) as f:
    for i, l in tqdm( enumerate( f ) ):
        x1, y = l.strip().split( '\t' )
        x1 = tokenizer.tokenize( x1 )
        tokens = [ '[CLS]' ] + x1 + [ '[SEP]' ]

        # 得到input_id, seg_id, att_mask
        ids = tokenizer.convert_tokens_to_ids( tokens )
        types = [0] * ( len( ids ) )
        masks = [1] * len( ids )
        # 短则补齐，长则切断
        if len( ids ) < pad_size:
            types = types + [1] * ( pad_size - len( ids ) )
            masks = masks + [0] * ( pad_size - len( ids ) )
            ids = ids + [0] * ( pad_size - len( ids ) )
        else:
            types = types[ :pad_size ]
            masks = masks[ :pad_size ]
            ids = ids[ :pad_size ]
        input_ids.append( ids )
        input_types.append( types )
        input_masks.append( masks )

        assert len( ids ) == len( masks ) == len( types ) == pad_size
        label.append( int( y ))

# 切分训练集和测试集
# 随机打乱索引
random_order = list( range( len( input_ids ) ) )
np.random.seed( 2020 )
np.random.shuffle( random_order )

# 4:1 划分训练集和测试集
split_ratio = 0.8
input_ids_train = np.array( [ input_ids[i] for i in random_order[ : int( len( input_ids ) * split_ratio ) ] ] )
input_types_train = np.array( [ input_types[i] for i in random_order[ : int( len( input_ids ) * split_ratio ) ] ])
input_masks_train = np.array( [ input_masks[i] for i in random_order[ : int( len( input_ids ) * split_ratio ) ] ] )
y_train = np.array( [ label[ i ] for i in random_order[ : int( len( input_ids ) * split_ratio ) ] ]  )
print(input_ids_train.shape, input_types_train.shape, input_masks_train.shape, y_train.shape)

input_ids_test = np.array( [ input_ids[i] for i in random_order[ int( len( input_ids ) * split_ratio ): ] ] )
input_types_test = np.array( [ input_types[i] for i in random_order[ int( len( input_ids ) * split_ratio ): ] ])
input_masks_test = np.array( [ input_masks[i] for i in random_order[ int( len( input_ids ) * split_ratio ): ] ] )
y_test = np.array( [ label[ i ] for i in random_order[ int( len( input_ids ) * split_ratio ): ] ] )
print(input_ids_test.shape, input_types_test.shape, input_masks_test.shape, y_test.shape)

# 加载到高效的DataLoader
BATCH_SIZE = 16
train_data = TensorDataset( torch.LongTensor( input_ids_train ),
                            torch.LongTensor( input_types_train ),
                            torch.LongTensor( input_masks_train ),
                            torch.LongTensor( y_train ) )
train_simpler = RandomSampler( train_data )     
train_loader = DataLoader( train_data, sampler=train_simpler, batch_size=BATCH_SIZE )

test_data = TensorDataset( torch.LongTensor( input_ids_test),
                           torch.LongTensor( input_types_test ),
                           torch.LongTensor( input_masks_test),
                           torch.LongTensor( y_test ) )
test_sampler = SequentialSampler( test_data )
test_loader = DataLoader( test_data, sampler=test_sampler, batch_size=BATCH_SIZE )

# 定义bert模型
class Model( nn.Module ):
    def __init__( self ):
        super( Model, self ).__init__()
        self.bert = BertModel.from_pretrained( bert_path )
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear( 768, 10 )

    def forward( self, x ):
        context = x[0] # 输入的句子   (ids, seq_len, mask)
        types = x[1]
        mask = x[2]

        _, pooled = self.bert( context, token_type_ids=types,
                               attention_mask=mask,
                               output_all_encoded_layers=False )
        out = self.fc( pooled )
        return out

# 实例化bert模型
DEVICE = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
model = Model().to( device=DEVICE )
# print( model )

# 定义优化器
param_optimizer = list( model.named_parameters() ) # 模型参数名字列表
no_decay = [ 'bias', 'LayerNorm.bias', 'LayerNorm.weight' ]
optimizer_grouped_parameters = [
    { 'params': [ p for n, p in param_optimizer if not any ( nd in n for nd in no_decay )], 'weight_decay': 0.01 },
    { 'params': [ p for n, p in param_optimizer if any( nd in n for nd in no_decay)], 'weight_decay': 0.0 }
]
# optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
NUM_EPOCHS = 3
optimizer = BertAdam( optimizer_grouped_parameters,
                      lr=2e-5,
                      warmup=0.05,
                      t_total=len( train_loader ) * NUM_EPOCHS )

# 定义训练函数和测试函数
def train( model, device, train_loader, optimizer, epoch ):