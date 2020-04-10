#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/04/10 11:46:04
@Author  :   LY 
@Version :   1.0
@URL     :   https://github.com/pytorch/examples/blob/master/word_language_model/main.py
@License :   (C)Copyright 2017-2020
@Desc    :   None
'''
# here put the import lib
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='E:/ML_data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')

args = parser.parse_args()
print( args )

# Set the random seed manually for reproducibility.
torch.manual_seed( args.seed )
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

###############################################################################
# Load data
###############################################################################
corpus = data.Corpus( args.data )

def batchify( data, bsz ):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow( 0, 0, nbatch * bsz )
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to( device )

eval_batch_size = 10
train_data = batchify( corpus.train, args.batch_size )
val_data = batchify( corpus.valid, eval_batch_size )
test_data = batchify( corpus.test, eval_batch_size )

###############################################################################
# Build the model
###############################################################################
ntokens = len( corpus.dictionary )
if args.model == 'Transformer':
    model = model.TransformerModel( ntoken=ntokens, ninp=args.emsize, nhead=args.nhead, nhid=args.nhid, nlayers=args.nlayers, dropout=args.dropout ).to( device )
else:
    model = model.RNNModel( args.model, ntoken=ntokens, ninp=args.emsize, nhid=args.nhid, nlayers=args.nlayers, dropout=args.dropout ).to( device )

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################
def repackage_hidden( h ):
    '''Wraps hidden states in new Tensors, to detach them from their history.'''
    if isinstance( h, torch.Tensor ):
        return h.detach()
    else:
        return tuple( repackage_hidden( v ) for v in h )

def get_batch( source, i ):
    seq_len = min( args.bptt, len( source ) - 1 - i )
    data = source[ i:i+seq_len ]
    target = source[ i+1:i+1+seq_len ].view( -1 )
    return data, target