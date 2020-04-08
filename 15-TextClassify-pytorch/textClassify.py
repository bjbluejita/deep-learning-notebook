#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   textClassify.py
@Time    :   2020/04/04 11:48:18
@Author  :   LY 
@Version :   1.0
@URL     :   https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
@License :   (C)Copyright 2017-2020
@Desc    :   None
'''
# here put the import lib
# ----Load data with ngrams-----
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torchtext
from torchtext.datasets import text_classification
from torchtext.datasets.text_classification import _create_data_from_iterator, _csv_iterator, TextClassificationDataset
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.vocab import build_vocab_from_iterator

NGRAMS = 2

def _setup_datasets( root='.data', ngrams=1, vocab=None, include_unk=False):
    # dataset_tar = 
    # 
    # (URLS[dataset_name], root=root)
    extracted_files = os.listdir(root)

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = os.path.join( root, fname )
        if fname.endswith('test.csv'):
            test_csv_path = os.path.join( root, fname )

    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_csv_path))
        vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
    logging.info('Creating testing data')
    test_data, test_labels = _create_data_from_iterator(
        vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(vocab, train_data, train_labels),
            TextClassificationDataset(vocab, test_data, test_labels))

dataDir = 'E:/ML_data/data/ag_news_csv'
if not os.path.isdir( dataDir ):
    os.makedirs( dataDir )
print( 'begin download data...' )
train_dataset, test_dataset = _setup_datasets(
    root=dataDir, ngrams=NGRAMS, vocab=None
)
BATCH_SIZE = 16
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

class TextSentiment( nn.Module ):
    def __init__( self, vocab_size, embed_dim, num_class ):
        super().__init__()
        self.embedding = nn.EmbeddingBag( vocab_size, embed_dim, sparse=True )
        self.fc = nn.Linear( embed_dim, num_class )
        self.init_weights()

    def init_weights( self ):
        initrange = 0.5
        self.embedding.weight.data.uniform_( -initrange, initrange )
        self.fc.weight.data.uniform_( -initrange, initrange )
        self.fc.bias.data.zero_()

    def forward( self, text, offset ):
        embedded = self.embedding( text, offset )
        return self.fc( embedded )

VACAB_SIZE = len( train_dataset.get_vocab() )
EMBED_DIM = 32
NUM_CLASS = len( train_dataset.get_labels() )
model = TextSentiment( VACAB_SIZE, EMBED_DIM, NUM_CLASS ).to( device )

# function generate_batch() is used to generate data batches and 
# offsets. The function is passed to collate_fn in torch.utils.data.DataLoader. 
# The input to collate_fn is a list of tensors with the size of 
# batch_size, and the collate_fn function packs them into a mini-batch
def generate_batch( batch ):
    label = torch.tensor( [ entry[0]  for entry in batch ] )
    text = [ entry[1] for entry in batch ]
    offsets = [ 0 ] + [ len( entry ) for entry in text ]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor( offsets[:-1] ).cumsum( dim=0 )
    text = torch.cat( text )
    return text, offsets, label

# use DataLoader here to load AG_NEWS datasets and send it to the model for training/validation.
def train_func( sub_train_ ):
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader( sub_train_, 
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        collate_fn=generate_batch )
    for i, ( text, offsets, cls ) in enumerate( data ):
        optimizer.zero_grad()
        text, offsets, cls = text.to( device ), offsets.to( device ), cls.to( device )
        output = model( text, offsets )
        loss = criterion( output, cls )
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += ( output.argmax(1) == cls ).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len( sub_train_ ), train_acc / len( sub_train_ )

def test( data_ ):
    loss = 0
    acc = 0
    data = DataLoader( data_,
                        batch_size=BATCH_SIZE,
                        collate_fn=generate_batch )
    for text, offsets, cls in data:
        text, offsets, cls = text.to( device ), offsets.to( device ), cls.to( device )
        with torch.no_grad():
            output = model( text, offsets )
            loss = criterion( output, cls )
            loss += loss.item()
            acc += ( output.argmax(1) == cls ).sum().item()

    return loss / len( data_ ), acc / len( data_ )

# Split the dataset and run the model
N_EPOCHS = 5
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to( device )
optimizer = torch.optim.SGD( model.parameters(), lr=4.0 )
scheduler = torch.optim.lr_scheduler.StepLR( optimizer, 1, gamma=0.9 )

train_len = int( len( train_dataset ) * 0.95 )
sub_train_, sub_valid_ = random_split( train_dataset, [ train_len, len( train_dataset ) - train_len ] )

for epoch in range( N_EPOCHS ):
    start_time = time.time()
    train_loss, train_acc = train_func( sub_train_ )
    valid_loss, valid_acc = test( sub_valid_ )

    secs = int( time.time() - start_time )
    mins = secs / 60
    secs = secs % 60

    print( 'Epoch: %d' % ( epoch + 1 ), ' | time: %dM%dS' % ( mins, secs ), end='' )
    #print(f'\tLoss:{train_loss:.4f}(train)\t|\tacc:{train_acc * 100:1.f}%(train)')
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')