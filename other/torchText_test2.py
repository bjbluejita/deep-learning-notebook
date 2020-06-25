#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   torchText_test2.py
@Time    :   2020/06/04 15:50:46
@Author  :   LY 
@Version :   1.0
@URL     :
@License :   (C)Copyright 2017-2020
@Desc    :   None
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
import spacy
import thulac
import re
from tqdm import tqdm

spacy_en = spacy.load('en')

thu1 = thulac.thulac( seg_only=True )  #默认模式

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_cn(text):
    return [tok for tok in thu1.cut(text, text=True )]

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_cn, init_token = BOS_WORD, 
                    eos_token=EOS_WORD, pad_token=BLANK_WORD)
train_fields = [ ('src', SRC), ('trg', TGT ) ]                    

train_path = 'E:/ML_data/translate/cmn.txt'
train_examples = []
with open( train_path, 'r', encoding ='utf-8' )as f:
    for line in tqdm( f.readlines() ):
        en = re.split( r"\.|\!|\?", line.split( 'CC-BY 2.0' )[0].strip() )[0]
        cn = re.split( r"\.|\!|\?", line.split( 'CC-BY 2.0' )[0].strip() )[1].replace( '\t', '' )
        # print( [ en, cn ] )
        # print( thu1.cut( cn, text=True ))
        train_examples.append( data.Example.fromlist( [ en, cn ], train_fields ) )


# 构建Dataset数据集
train_dataset, val_dataset = data.Dataset( train_examples, train_fields ).split( split_ratio=0.8 )
# train, val = data.Dataset( train_examples, train_fields ).split( split_ratio=0.8 )

MIN_FREQ = 2
SRC.build_vocab(train_dataset.src, min_freq=MIN_FREQ)
TGT.build_vocab(train_dataset.trg, min_freq=MIN_FREQ) 

train_iter = data.Iterator( train_dataset, batch_size=5, train=True )
#train_iter = data.BucketIterator( train_dataset, batch_size=5, train=True )
for i, batch in enumerate( train_iter ):
    print( 'src->', batch.src )
    print( 'trg->', batch.trg )

print( 'finished')

