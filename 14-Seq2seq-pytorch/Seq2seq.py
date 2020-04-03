#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Seq2seq.py
@Time    :   2020/03/29 16:35:57
@Author  :   LY 
@Version :   1.0
@URL     :   https://zhuanlan.zhihu.com/p/52302561
             https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
@License :   (C)Copyright 2017-2020
@Desc    :   None
'''
# here put the import lib
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import os
import re
import random
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# plt.switch_backend('agg')


device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

# sentenceFilePath = 'E:/ML_data/Seq2seq-pytorch/eng-fra.txt'
sentenceFilePath = 'E:/ML_data/Seq2seq-pytorch/chn-eng.txt'
encoderFilePath = 'F:/workspace/Tensorflow/src/deep-learning-with-keras-notebooks/encoder.pth'
decoderFilePath = 'F:/workspace/Tensorflow/src/deep-learning-with-keras-notebooks/decoder.pth'

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__( self, name ):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.index2word = { 0: "SOS", 1: "EOS" }
        self.n_words = 2

    def addStentence( self, sentence ):
        for word in sentence.split( ' ' ):
            self.addWord( word )

    def addWord( self, word ):
        if word not in self.word2index:
            self.word2index[ word ] = self.n_words
            self.word2count[ word ] = 1
            self.index2word[ self.n_words ] = word
            self.n_words += 1
        else:
            self.word2count[ word ] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii( s ):
    return "".join( 
        c for c in unicodedata.normalize( 'NFD', s )
            if unicodedata.category( c ) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString( s ):
    s = unicodeToAscii( s.lower().strip() )
    s = re.sub( r"([.!?])", r"\1", s )
    s = re.sub( r"[^a-zA-Z.!?]+", r" ", s )
    return s

# To read the data file we will split the file into lines, 
# and then split lines into pairs. The files are all English → Other Language, 
# so if we want to translate from Other Language → English I added the reverse 
# flag to reverse the pairs.
def readLangs( corpusFilePath, lang1, lang2, reverse=False ):
    print( 'Reading lines...' )

    # Read the file and split into lines
    lines = open( corpusFilePath, encoding='utf-8' ).read().strip().split( '\n' )

    # Split every line into pairs and normalize
    pairs = [ [ normalizeString( s ) for s in l.split( '\t' ) ] for l in lines ]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [ list( reversed( p )) for p in pairs ]
        input_lang = Lang( lang2 )
        output_lang = Lang( lang1 )
    else:
        input_lang = Lang( lang1 )
        output_lang = Lang( lang2 )
    return input_lang, output_lang, pairs

# 过滤长度超过MAX_LENGTH和开头不是eng_prefixes的句子
MAX_LENGTH = 10

eng_prefixes = (
    'i am', 'i m',
    'he is', 'he s',
    'she is', 'she s',
    'you are', 'you re',
    'we are', 'we re',
    'they are', 'they re'
)

def filterPair( p ):
    return len( p[0].split( ' ' )) < MAX_LENGTH and \
        len( p[1].split( ' ' )) < MAX_LENGTH and \
        p[1].startswith( eng_prefixes )

def filterPairs( paris ):
    return [ pari for pari in paris if filterPair( pari ) ]

# Read text file and split into lines, split lines into pairs
# Normalize text, filter by length and content
# Make word lists from sentences in pairs
def prepareData( lang1, lang2, reverse=False ):
    input_lang, output_lang, pairs = readLangs( sentenceFilePath, lang1, lang2, reverse )
    print( 'Read %s sentence pairs' % len( pairs ) )
    pairs = filterPairs( pairs )
    print( 'Trimmed to %s sentence pairs' % len( pairs ) )

    for pair in pairs:
        input_lang.addStentence( pair[0] )
        output_lang.addStentence( pair[1] )
    print( 'Counted words:')
    print( input_lang.name, ':', input_lang.n_words )
    print( output_lang.name, ':', output_lang.n_words )
    return input_lang, output_lang, pairs

# The Seq2Seq Model
class EncoderRNN( nn.Module ):
    def __init__( self, input_size, hidden_size ):
        super( EncoderRNN, self ).__init__()
        self. hidden_size = hidden_size
        
        # 单词向量化
        self.embedding = nn.Embedding( input_size, hidden_size )

        # 定义一个gru单元，input和hidden的维度定义成相同，都用hidden_size
        # 第一个hidden_size是定义了input的维度，就是词向量的维度
        # 第二个定义了input_hidden和output_hidden的维度，它们俩的维度是一样的
        self.gru = nn.GRU( hidden_size, hidden_size )

    def forward( self, input, hidden ):
        embedded = self.embedding( input ).view( 1, 1, -1 )  # 整个tensor维变成[1,1,input_size]的向量
        output = embedded
        output, hidden = self.gru( output, hidden )
        return output, hidden

    def initHidden( self ):
        return torch.zeros( 1, 1, self.hidden_size, device=device )

class DecoderRNN( nn.Module ):
    def __init__( self, hidden_size, output_size ):
        super( DecoderRNN, self ).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding( output_size, hidden_size )
        self.gru = nn.GRU( hidden_size, hidden_size )
        self.out = nn.Linear( hidden_size, output_size )
        self.softmax = nn.LogSoftmax( dim=1 )

    def forward( self, input, hidden, encoder_outputs ):
        output = self.embedding( input ).view( 1, 1, -1 )
        output = F.relu( output )
        output, hidden = self.gru( output, hidden )
        output = self.softmax( self.out( output[0] ) )
        return output, hidden, None

    def initHidden( self ):
        return torch.zeros( 1, 1, self.hidden_size, device=device )

class AttnDecoderRNN( nn.Module ):
    def __init__( self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH ):
        super( AttnDecoderRNN, self ).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding( self.output_size, self.hidden_size )
        self.attn = nn.Linear( self.hidden_size * 2, self.max_length )
        self.attn_combine = nn.Linear( self.hidden_size * 2, self.hidden_size )
        self.dropout = nn.Dropout( self.dropout_p )
        self.gru = nn.GRU( self.hidden_size, self.hidden_size )
        self.out = nn.Linear( self.hidden_size, self.output_size )

    def forward( self, input, hidden, encoder_outputs ):
        embedded = self.embedding( input ).view( 1, 1, -1 )
        embedded = self.dropout( embedded )

        attn_weights = F.softmax( 
            self.attn( torch.cat( ( embedded[0], hidden[0] ), 1 )), dim=1
        )
        attn_applied = torch.bmm( attn_weights.unsqueeze(0),
                                  encoder_outputs.unsqueeze(0) )
        
        output = torch.cat( ( embedded[0], attn_applied[0] ), 1 )
        output = self.attn_combine( output ).unsqueeze( 0 )

        output = F.relu( output )
        output, hidden = self.gru( output, hidden )

        output = F.log_softmax( self.out( output[0] ), dim=1 )

        return output, hidden, attn_weights

    def initHidden( self ):
        return torch.zeros( 1, 1, self.hidden_size, device=device )

# To train, for each pair we will need an input tensor (indexes of 
# the words in the input sentence) and target tensor (indexes of the 
# words in the target sentence). While creating these vectors we 
# will append the EOS token to both sequences.
def indexesFromSentence( lang, sentence ):
    return [ lang.word2index[ word ] for word in sentence.split( ' ' ) ]

def tensorFromSentence( lang, sentence ):
    indexes = indexesFromSentence( lang, sentence )
    indexes.append( EOS_token )
    return torch.tensor( indexes, dtype=torch.long, device=device ).view( -1, 1 )

def tensorFromPair( pair ):
    input_tensor = tensorFromSentence( input_lang, pair[0] )
    target_tensor = tensorFromSentence( output_lang, pair[1] )
    return ( input_tensor, target_tensor )

# -----Training the Model-------
teacher_forcing_ratio = 0.5

def train( input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH ):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size( 0 )
    target_length = target_tensor.size( 0 )

    encoder_outputs = torch.zeros( max_length, encoder.hidden_size, device=device )

    loss = 0
    
    for ei in range( input_length ):
        encoder_output, encoder_hidden = encoder( 
            input_tensor[ ei ], encoder_hidden
        )
        encoder_outputs[ ei ] = encoder_output[ 0, 0 ]

    decoder_input = torch.tensor( [ [ SOS_token ] ], device=device )

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range( target_length ):
            decoder_output, decoder_hidden, decoder_attention = decoder( 
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion( decoder_output, target_tensor[di] )
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range( target_length ):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk( 1 )
            decoder_input = topi.squeeze().detach()   # detach from history as input

            loss += criterion( decoder_output, target_tensor[di] )
            if decoder_input.item() == EOS_token:
                break
    
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# helper function to print time elapsed and estimated time remaining 
# given the current time and progress 

def asMinutes( s ):
    m = math.floor( s / 60 )
    s -= m * 60
    return '%dm %ds' % ( m, s )

def timeSince( since, percent ):
    now = time.time()
    s = now - since
    es = s / ( percent )
    rs = es - s
    return '%s (- %s)' % ( asMinutes( s ), asMinutes( rs ) )

# whole training process:
# 1.Start a timer
# 2.Initialize optimizers and criterion
# 3.Create set of training pairs
# 4.Start empty losses array for plotting
def trainIters( encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01 ):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    fig, ax = plt.subplots()

    encoder_optimizer = optim.SGD( encoder.parameters(), lr=learning_rate )
    decoder_optimizer = optim.SGD( decoder.parameters(), lr=learning_rate )
    training_pairs = [ tensorFromPair( random.choice( pairs )) 
                    for i in range( n_iters ) ]

    criterion = nn.NLLLoss()

    for iter in range( 1, n_iters + 1 ):
        training_pair = training_pairs[ iter - 1 ]
        input_tensor = training_pair[ 0 ]
        target_tensor = training_pair[ 1 ]

        loss = train( input_tensor, target_tensor, encoder,
                       decoder, encoder_optimizer, decoder_optimizer, criterion )

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append( plot_loss_avg )
            plot_loss_total = 0

            showPlot( plt, ax, plot_losses )

        if iter % 2000 == 0:
            print( 'Saving model...' )
            torch.save( encoder.state_dict(), encoderFilePath )
            torch.save( decoder.state_dict(), decoderFilePath )

def showPlot( plt, ax, points ):
    ax.cla()
    ax.scatter( range( len( points )), points, label='loss' )
    ax.legend()
    plt.pause( 0.1 )

# Evaluation
def evaluate( encoder, input_lang, decoder, output_lang, sentence, max_length=MAX_LENGTH ):
    with torch.no_grad():
        input_tensor = tensorFromSentence( input_lang, sentence )
        input_length = input_tensor.size( 0 )
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros( max_length, encoder.hidden_size, device=device )

        for ei in range( input_length ):
            encoder_output, encoder_hidden = encoder( input_tensor[ei],
                                                      encoder_hidden )
            encoder_outputs[ei] += encoder_output[ 0, 0 ]

        decoder_input = torch.tensor( [[SOS_token]], device=device )
        decoder_hidden = encoder_hidden

        decoder_words = []
        decoder_attentions = torch.zeros( max_length, max_length )

        for di in range( max_length ):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            if decoder_attention is not None:
                decoder_attentions[di] = decoder_attention.data

            topv, topi = decoder_output.topk( 1 )
            if topi.item() == EOS_token:
                decoder_words.append( '<EOS>')
                break
            else:
                decoder_words.append( output_lang.index2word[ topi.item() ] )
            
            decoder_input = topi.squeeze().detach()

        return decoder_words, decoder_attentions[ :di + 1 ]

def evaluateRandomly( encoder, input_lang,  decoder, output_lang, pairs, n=10 ):
    for i in range( n ):
        pair = random.choice( pairs )
        print( '>', pair[0], end='' )
        print( '=', pair[1] )
        output_words, attentions = evaluate( encoder, input_lang, decoder, output_lang, pair[0] )
        output_sentence = ' '.join( output_words )
        print( '<', output_sentence )
        print( ' ' )


if __name__ == '__main__':
    hidden_size = 256
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    encoder1 = EncoderRNN( input_lang.n_words, hidden_size ).to( device )
    # decoder1 = DecoderRNN( hidden_size, output_lang.n_words ).to( device )
    decoder1 = AttnDecoderRNN( hidden_size, output_lang.n_words, dropout_p=0.1 ).to( device )    
    
    if os.path.exists( encoderFilePath ) and os.path.exists( decoderFilePath ):
        print( 'Load model...' )
        encoder1.load_state_dict( torch.load( encoderFilePath ) )
        decoder1.load_state_dict( torch.load( decoderFilePath ) )
    

    trainIters( encoder1, decoder1, 750000, print_every=5, plot_every=10 )

    evaluateRandomly( encoder1, input_lang, decoder1, output_lang, pairs )