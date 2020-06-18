#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pytext_test.py
@Time    :   2020/04/26 18:48:25
@Author  :   LY 
@Version :   1.0
@URL     :   https://blog.csdn.net/nlpuser/article/details/88067167
@License :   (C)Copyright 2017-2020
@Desc    :   TorchText 测试程序
'''
# here put the import lib
from torchtext import data
from torchtext.vocab import Vectors
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import GloVe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os

# tokenize = lambda x: x.split()
# TEXT = data.Field( sequential=True, tokenize=tokenize, lower=True, fix_length=200 )
TEXT = data.Field(sequential=True)
LABEL = data.Field( sequential=False, use_vocab=False )

# 使用torchtext内置的Dataset构建数据集
train_path = 'E:/ML_data/torchText/train_one_label.txt'
valid_path = 'E:/ML_data/torchText/valid_one_label.txt'
test_path = 'E:/ML_data/torchText/test.txt'

train_data = pd.read_csv( train_path )
valid_data = pd.read_csv( valid_path )
test_data = pd.read_csv( test_path )

# get_dataset构造并返回Dataset所需的examples和fields
def get_dataset( csv_data, text_field, label_fiedld, test=False ):
    fields = [ ('id', None ), ( 'comment_text', text_field ), ( 'toxic', label_fiedld ) ]
    examples = []

    if test:
        for text in tqdm( csv_data[ 'comment_text' ] ):
            examples.append( data.Example.fromlist( [ None, text, None ], fields ) )
    else:
        for text, label in tqdm( zip( csv_data[ 'comment_text' ], csv_data[ 'toxic' ] )):
            examples.append( data.Example.fromlist( [ None, text, label ], fields ) )

    return examples, fields

# 得到构建Dataset所需的examples和fields
train_examples, train_fields = get_dataset( train_data, TEXT, LABEL )
valid_examples, valid_fields = get_dataset( valid_data, TEXT, LABEL )
test_examples, test_fields = get_dataset( test_data, TEXT, None, test=True )

# 构建Dataset数据集
train = data.Dataset( train_examples, train_fields )
valid = data.Dataset(valid_examples, valid_fields)
test = data.Dataset(test_examples, test_fields)

for batch in train:
    print( batch.toxic )


# 自定义Dataset类
class MyDataset( data.Dataset ):
    def __init__( self, path, text_field, label_fiedld, test=False, aug=False, **kwargs ):
        fields = [ ( 'id', None), ( 'comment_text', text_field ), ( 'toxic', label_fiedld ) ]

        examples = []
        csv_data = pd.read_csv( path )
        print( 'read data from {}'.format( path ) )

        if test:
            for text in tqdm( csv_data[ 'comment_text' ] ):
                examples.append( data.Example.fromlist( [ None, text, None ], fields ) )
        else:
            for text, label in tqdm( zip( csv_data[ 'comment_text' ], csv_data[ 'toxic' ] ) ):
                if aug:
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout( text )
                    else:
                        text = self.shuffle( text )
                examples.append( data.Example.fromlist( [ None, text, label ], fields ) )

        # super( MyDataset, self ).__init__( examples, fields, **kwargs )
        super( MyDataset, self ).__init__( examples, fields )

    def shuffle( self, text ):
        text = np.random.permutation( text.strip().split() )
        return ' '.join( text )

    def dropout( self, text, p=0.5 ):
        text = text.strip().split()
        len_ = len( text )
        indexs = np.random.chioce( len_, int( len_ * p ) )
        for i in indexs:
            text[i] = ''
        return ' '.join( text )


def data_iter( train_path, valid_path, test_path, TEXT, LABEL ):

    train_dataset = MyDataset( train_path, TEXT, LABEL )
    valid_dataset = MyDataset( valid_path, TEXT, LABEL )
    test_dataset = MyDataset( test_path, text_field=TEXT, label_fiedld=None, test=True, aug=1 )

    # 若只针对训练集构造迭代器
    # train_iter = data.BucketIterator( dataset=train_dataset, batch_size=8, shuffle=True, sort_within_batch=False, repeat=False )

    # 同时对训练集和验证集进行迭代器的构建
    train_iter, valid_iter = BucketIterator.splits( datasets=( train_dataset, valid_dataset ),
                                                    batch_sizes=( 8, 8 ),
                                                    sort_key=lambda x: len( x.comment_text ),
                                                    sort_within_batch=False,
                                                    repeat=False )
    test_iter = Iterator( dataset=test_dataset, batch_size=8, sort=False, sort_within_batch=False, repeat=False )

    TEXT.build_vocab( train_dataset, vectors=GloVe( name='6B', dim=300 ) )
    weight_matrix = TEXT.vocab.vectors

    return train_iter, valid_iter, test_iter, weight_matrix

class LSTM( nn.Module ):
    def __init__( self, weight_matrix ):
        super( LSTM, self ).__init__()
        self.word_embedings = nn.Embedding( len( TEXT.vocab ), 300 )
        # embedding.weight.data.copy_( weight_matrix )
        self.word_embedings.weight.data.copy_( weight_matrix )
        self.lstm = nn.LSTM( input_size=300, hidden_size=128, num_layers=1 )
        self.decoder = nn.Linear( 128, 2 )

    def forward( self, sentence ):
        embeds = self.word_embedings( sentence )
        lstm_out = self.lstm( embeds )[0] # lstm_out:200x8x128
        # 取最后一个时间步
        final = lstm_out[ -1 ]  # 8*128
        y = self.decoder( final )
        return y


def main():
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

    train_iter, valid_iter, test_iter, weight_matrix = data_iter( train_path, valid_path, test_path, TEXT, LABEL)
    model = LSTM(weight_matrix)
    model.train()

    optimizer = optim.Adam( filter( lambda p: p.requires_grad, model.parameters() ), lr=0.01 )
    loss_funtion = F.cross_entropy
    
    for epoch, batch in enumerate( train_iter ):
        optimizer.zero_grad()
        predicted = model( batch.comment_text )

        loss = loss_funtion( predicted, batch.toxic )
        loss.backward()
        optimizer.step()
        print( loss.item() )

if __name__ == '__main__':
    main()
    print( '******finished******' )