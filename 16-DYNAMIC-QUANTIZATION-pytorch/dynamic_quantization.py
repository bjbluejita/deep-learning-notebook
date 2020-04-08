#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dynamic_quantization.py
@Time    :   2020/04/08 12:04:47
@Author  :   LY 
@Version :   1.0
@URL     :
@License :   (C)Copyright 2017-2020
@Desc    :   None
'''
# here put the import lib
import os
from io import open
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model
class LSTMModel( nn.Module ):
    '''Container module with an encoder, a recurrent module, and a decoder.'''

    def __init__( self, ntoken, ninp, nhid, nlayers, dropout=0.5 ):
        super( LSTMModel, self ).__init__()
        self.drop = nn.Dropout( dropout )
        self.encoder = nn.Embedding( ntoken, ninp )
        self.rnn = nn.LSTM( ninp, nhid, nlayers, dropout=dropout )
        self.decoder = nn.Linear( nhid, ntoken )

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights( self ):
        initrange = 0.1
        self.encoder.weight.data.uniform( -initrange, initrange )
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform( -initrange, initrange )

    def forward( self, input, hidden ):
        emb = self.drop( self.encoder( input ) )
        output, hidden = self.rnn( emb, hidden )
        output = self.drop( output )
        decoded = self.decoder( output )
        return decoded, hidden

    def init_hidden( self, bsz ):
        weight = next( self.parameters() )
        return( weight.new_zeros( self.nlayers, bsz, self.nhid ),
                weight.new_zeros( self.nlayers, bsz, self.init_hid ) )

class Dictionary( object ):
    def __init__( self ):
        self.word2idx = {}
        self.idx2word = []

    def add_word( self, word ):
        if word not in self.word2idx:
            self.idx2word.append( word )
            self.word2idx[ word ] = len( self.idx2word ) -1 
        return self.word2idx[ word ]

    def __len__( self ):
        return len( self.idx2word )

class Corpus( object ):
    def __init__( self, path ):
        self.dictionary = Dictionary()
        self.train = self.tokenize( os.path.join( path, 'wiki.train.tokens') )
        self.valid = self.tokenize( os.path.join( path, 'wiki.valid.tokens') )
        self.test  = self.tokenize( os.path.join( path, 'wiki.test.tokens') )

    def tokenize( self, path ):
        """Tokenizes a text file."""
        assert os.path.exists( path )
        # Add words to the dictionary
        with open( path, 'r', encoding='utf8' ) as f:
            idss = []
            for line in f:
                words = line.split() + [ '<eos>' ]
                ids = []
                for word in words:
                    self.dictionary.add_word( word )

        # Tokenize file content
        with open( path, 'r', encoding='utf8' ) as f:
            idss = []
            for line in f:
                words = line.split() + [ '<eos>' ]
                ids = []
                for word in words:
                    ids.append( self.dictionary.word2idx[ word ] )
                idss.append( torch.tensor( ids ).type( torch.int64 ) )
            ids = torch.cat( idss )
                    
        return ids

model_data_filepath = 'E:/ML_data/wikitext-2'
corpus = Corpus( model_data_filepath )


