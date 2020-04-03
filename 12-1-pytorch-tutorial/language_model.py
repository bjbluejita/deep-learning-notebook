# !/usr/bin/env python
#  -*- encoding: utf-8 -*-
'''
@File    :   language_model.py
@Time    :   2020/01/14 10:58:38
@Author  :   LY 
@Version :   1.0
@URL     :   https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py# L30-L50
@License :   (C)Copyright 2017-2020
@Desc    :   None
'''
#  here put the import lib
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Corpus

# device configuration
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

# Hyper-parameters
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 1000
batch_size = 20
seq_length = 30
learning_rate = 0.002

# Load 'Penn Treebank' dataset
corpus = Corpus()
ids = corpus.get_data( './data/train.txt', batch_size )
vocab_size = len( corpus.dictonary )
num_batches = ids.size(1) // seq_length

# RNN based language model
class RNNLM( nn.Module ):
    def __init__( self, vocab_size, embed_size, hidden_size, num_layers ):
        super( RNNLM, self ).__init__()
        self.embed = nn.Embedding( vocab_size, embed_size )
        self.lstm = nn.LSTM( embed_size, hidden_size, num_layers, batch_first=True )
        self.linear = nn.Linear( hidden_size, vocab_size )

    def forward( self, x, h ):
        #  Embed word ids to vectors
        x = self.embed( x )

        #  forward propagate LSTM
        out, ( h, c ) = self.lstm( x, h )

        #  Reshape output to( batch_size*sequence_length, hidden_size )
        out = out.reshape( out.size(0)*out.size(1), out.size(2) )

        #  Decode hidden state of all time steps
        out = self.linear( out )
        return out, ( h, c )

model = RNNLM( vocab_size, embed_size, hidden_size, num_layers ).to( device )    

#  Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate )

#  Truncated backpropagation
def detach( states ):
    return [ state.detach() for state in states ]

#  Train the model
for epoch in range( num_epochs ):
    #  Set initial hidden and cell states
    states = ( torch.zeros( num_layers, batch_size, hidden_size ).to( device ),
               torch.zeros( num_layers, batch_size, hidden_size ).to( device )
             )    
    for i in range( 0, ids.size(1) - seq_length, seq_length ):
        #  Get mini-batch inputs and targets
        inputs = ids[ :, i:i+seq_length ].to( device )
        targets = ids[ :, (i+1):(i+1)+seq_length ].to( device )

        #  Forward pass
        states = detach( states )
        outputs, states = model( inputs, states )
        loss = criterion( outputs, targets.reshape( -1 ) )

        #  Backward and optimize
        model.zero_grad()
        loss.backward()
        clip_grad_norm_( model.parameters(), 0.5 )
        optimizer.step()

        step = ( i+1 ) // seq_length
        if step % 100 == 0:
            print( 'Epoch[{}/{}], Step[{}/{}], Loss{:.4f}, Perplexity:{:5.2f}'
                   .format( epoch+1, num_epochs, step, num_batches, loss.item(), np.exp( loss.item() ) ) )

# Test the model
with torch.no_grad():
    with open( 'sample.txt', 'w' ) as f:
        #  Set inital hidden ane cell states
        state = ( torch.zeros( num_layers, 1, hidden_size ).to( device ),
                  torch.zeros( num_layers, 1, hidden_size ).to( device ) )

        #  Select one word id randomly
        prob = torch.ones( vocab_size )
        input = torch.multinomial( prob, num_samples=1 ).unsqueeze(1).to( device )

        for i in range( num_samples ):
            #  Forward propagate RNN
            output, state = model( input, state )

            #  Sample a word id
            prob = output.exp()
            word_id = torch.multinomial( prob, num_samples=1 ).item()

            #  Fill input with sampled word id for the next time step
            input.fill_( word_id )

            #  file write
            word = corpus.dictonary.idx2word[ word_id ]
            word = '\n' if word == '<eos>' else word + ' '
            f.write( word )

            if ( i + 1 ) % 100 == 0:
                print( 'Sample [{}/{}] word and save to {}'.format( i + 1, num_samples, 'sample.txt' ) )

