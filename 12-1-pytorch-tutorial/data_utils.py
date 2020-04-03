# !/usr/bin/env python
#  -*- encoding: utf-8 -*-
'''
@File    :   data_utils.py
@Time    :   2020/01/14 10:34:14
@Author  :   LY 
@Version :   1.0
@URL     :   https://github.com/yunjey/pytorch-tutorial/blob/56bba936197f46968aa2cc04c8096055de7f710c/tutorials/02-intermediate/language_model/data_utils.py# L5
@License :   (C)Copyright 2017-2020
@Desc    :   None
'''
#  here put the import lib
import torch

class Dictionary( object ):
    def __init__( self ):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word( self, word ):
        if word not in self.word2idx:
            self.word2idx[ word ] = self.idx
            self.idx2word[ self.idx ] = word
            self.idx += 1

    def __len__( self ):
        return len( self.word2idx )


class Corpus( object ):
    def __init__( self ):
        self.dictonary = Dictionary()

    def get_data( self, path, batch_size=20 ):
        # Add word to the dictionary
        with open( path, 'r' ) as f:
            tokens = 0
            for line in f:
                words = line.split() + [ '<eos>' ]
                tokens += len( words )
                for word in words:
                    self.dictonary.add_word( word )

        # Tokenize the file content
        ids = torch.LongTensor( tokens )
        token = 0
        with open( path, 'r' ) as f:
            for line in f:
                words = line.split() + [ '<eos>' ]
                for word in words:
                    ids[ token ] = self.dictonary.word2idx[word]
                    token += 1

        num_batches = ids.size(0) // batch_size
        ids = ids[ :num_batches * batch_size ]
        return ids.view( batch_size, -1 )

