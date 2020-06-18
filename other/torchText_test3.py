#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   torchText_test3.py
@Time    :   2020/06/11 15:58:33
@Author  :   LY 
@Version :   1.0
@URL     :   https://blog.csdn.net/JWoswin/article/details/92821752
@License :   (C)Copyright 2017-2020
@Desc    :   None
'''
# here put the import lib

import pandas as pd
data = pd.read_csv('E:\\ML_data\\torchText\\train.tsv', sep='\t')
test = pd.read_csv('E:\\ML_data\\torchText\\test.tsv', sep='\t')

from sklearn.model_selection import train_test_split
train, val = train_test_split( data, test_size=0.2 )

train.to_csv("E:\\ML_data\\torchText\\train.csv", index=False)
val.to_csv("E:\\ML_data\\torchText\\val.csv", index=False)

# 定义Field
import spacy
import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors
from torch.nn import init

DEVICE = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
spacy_en = spacy.load( 'en' )

# create a tokenizer function
def tokenizer( text ):
    return [ tok.text for tok in spacy_en.tokenizer( text ) ]

"""
field在默认的情况下都期望一个输入是一组单词的序列，并且将单词映射成整数。
这个映射被称为vocab。如果一个field已经被数字化了并且不需要被序列化，
可以将参数设置为use_vocab=False以及sequential=False。
"""
LABEL = data.Field( sequential=False, use_vocab=False )
TEXT = data.Field( sequential=True, use_vocab=tokenizer, lower=True )

# 定义Dataset
# 对于csv/tsv类型的文件，TabularDataset很容易进行处理，故我们选它来生成Dataset
"""
我们不需要 'PhraseId' 和 'SentenceId'这两列, 所以我们给他们的field传递 None
如果你的数据有列名，如我们这里的'Phrase','Sentiment',...
设置skip_header=True,不然它会把列名也当一个数据处理
"""
train, val = data.TabularDataset.splits( 
    path='E:\\ML_data\\torchText\\',  train='train.csv',validation='val.csv', format='csv', skip_header=True,
    fields= [ ( 'PhraseId', None ), ( 'SentenceId', None ), ( 'Phrase', TEXT ), ( 'Sentiment', LABEL ) ]
 )
test = data.TabularDataset( 'E:\\ML_data\\torchText\\test.tsv', format='tsv', skip_header=True,
                              fields= [ ( 'PhraseId', None ), ( 'SentenceId', None ), ( 'Phrase', TEXT ) ] 
                              )
print( train[5] )
print( train[5].__dict__.keys() )
print( train[5].Phrase, train[5].Sentiment )

# 建立vocab
# Torchtext可以将词转化为数字，但是它需要被告知需要被处理的全部范围的词。
TEXT.build_vocab( train, vectors='glove.6B.100d' )
TEXT.vocab.vectors.unk_init = init.xavier_uniform

print( TEXT.vocab.stoi[ 'enough' ] )
print( TEXT.vocab.itos[ 1501 ] )
print(TEXT.vocab.vectors.shape)
word_vect = TEXT.vocab.vectors[ TEXT.vocab.stoi['enough']]
print( word_vect.shape )
print( word_vect )

# 构造迭代器
'''
和Dataset一样，torchtext有大量内置的迭代器，我们这里选择的是BucketIterator，官网对它的介绍如下：

Defines an iterator that batches examples of similar lengths together.
Minimizes amount of padding needed while producing freshly shuffled batches for each new epoch.
'''
train_iter = data.BucketIterator( train, batch_size=20, sort_key=lambda  x : len( x.Phrase ),
                                  shuffle=True, device=DEVICE )
test_iter = data.BucketIterator( test, batch_size=20, sort_key=lambda  x : len( x.Phrase ),
                                  shuffle=True, device=DEVICE )

for batch_idx, batch in enumerate( train_iter ):
    data = batch.Phrase
    print( 'data shape:', data.shape )
    data =data.permute( 1, 0 )
    print( 'after permute, data shape:', data.shape )
    target = batch.Sentiment
    print( 'target shape:', target.shape )
    target = torch.sparse.torch.eye( 5 ).index_select( dim=0, index=target.cpu().data )
    print( 'after sparse target shape:', target.shape )
    target = target.to( DEVICE )
    