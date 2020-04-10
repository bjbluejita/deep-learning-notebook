#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dynamic_quantization.py
@Time    :   2020/04/08 12:04:47
@Author  :   LY 
@Version :   1.0
@URL     :   https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html
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
import torch.quantization

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
        self.encoder.weight.data.uniform_( -initrange, initrange )
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_( -initrange, initrange )

    def forward( self, input, hidden ):
        emb = self.drop( self.encoder( input ) )
        output, hidden = self.rnn( emb, hidden )
        output = self.drop( output )
        decoded = self.decoder( output )
        return decoded, hidden

    def init_hidden( self, bsz ):
        weight = next( self.parameters() )
        return( weight.new_zeros( self.nlayers, bsz, self.nhid ),
                weight.new_zeros( self.nlayers, bsz, self.nhid ) )
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

ntokens = len( corpus.dictionary )
print( 'ntokens lenght=', ntokens )

model = LSTMModel( ntoken=ntokens,
                   ninp=512,
                   nhid=256,
                   nlayers=5, )

model.load_state_dict( 
    torch.load( os.path.join( model_data_filepath, 'word_language_model_quantize.pth' ),
                map_location=torch.device( 'cpu' ) )
)

model.eval()
print( model )

input_ = torch.randint( ntokens, ( 1, 1 ), dtype=torch.long )
hidden = model.init_hidden( 1 )
temperature = 1.0
num_words = 1000

with open( os.path.join( model_data_filepath, 'out.txt'), 'w' ) as outf:
    with torch.no_grad():
        for i in range( num_words ):
            output, hidden = model( input_, hidden )
            word_weights = output.squeeze().div( temperature ).exp().cpu()
            word_idx = torch.multinomial( word_weights, 1 )[ 0 ]  # 抽样。
            input_.fill_( word_idx )

            word = corpus.dictionary.idx2word[ word_idx ]

            outf.write( str( word.encode( 'utf-8' )) + ( '\n' if i % 20 == 19 else ' ' ) )
            if i % 100 == 0:
                print( '| Generated {}/{} words'.format( i, 1000 ) )

with open( os.path.join( model_data_filepath, 'out.txt'), 'r' ) as outf:
    all_output = outf.read()
    print( all_output )

# to demonstrate dynamic quantization
bptt = 25
criterion = nn.CrossEntropyLoss()
eval_batch_size = 1

# create test data set
def batchify( data, bsz ):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size( 0 ) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow( 0, 0, nbatch * bsz )
    # Evenly divide the data across the bsz batches.
    return data.view( bsz, -1 ).t().contiguous()

test_data = batchify( corpus.test, eval_batch_size )

# Evaluation functions
def get_batch( source, i ):
    seq_len = min( bptt, len( source ) - 1 - i )
    data = source[ i:i+seq_len ]
    target = source[ i+1:i+1+seq_len ].view( -1 )
    return data, target

def repackage_hidden( h ):
    # wraps hidden states in new Tensors, to detach them from their history.
    if isinstance( h, torch.Tensor ):
        return h.detach
    else:
        return tuple( repackage_hidden( v ) for v in h )

def evaluate( model_, data_source ):
    # Turn on evaluation mode which disables dropout.
    model_.eval()
    total_loss = 0
    hidden = model_.init_hidden( eval_batch_size )
    with torch.no_grad():
        for i in range( 0, data_source.size(0) - 1, bptt ):
            data, target = get_batch( data_source, i )
            output, hidden = model_( data, hidden )
            hidden = repackage_hidden( hidden )
            output_flat = output.view( -1, ntokens )
            total_loss += len( data ) * criterion( output_flat, targets ).item()

    return total_loss / ( len( data_source ) - 1 )

# torch.quantization.quantize_dynamic on the model:
# 1 specify the nn.LSTM and nn.Linear modules in our model to be quantized
# 2 specify the weights to be converted to int8 values
print( torch.backends.quantized.supported_engines )
quantized_model = torch.quantization.quantize_dynamic(
    model, { nn.LSTM, nn.Linear }, dtype=torch.qint8
)
print( quantized_model )
