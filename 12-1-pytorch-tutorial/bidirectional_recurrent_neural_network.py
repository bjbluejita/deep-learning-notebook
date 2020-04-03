# !/usr/bin/env python
#  -*- encoding: utf-8 -*-
'''
@File    :   bidirectional_recurrent_neural_network.py
@Time    :   2020/01/13 15:50:27
@Author  :   LY 
@Version :   1.0
@URL     :   https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py# L39-L58
@License :   (C)Copyright 2017-2020
@Desc    :    引入一种需要同时考虑下一个时间步和上一个时间步的信息来做当前时间步决定的结构:
             有2层隐藏层，其中每个隐藏层都连接到输出和输入。这2个隐藏层可以微分，且都有自
             循环连接，不过一个是朝着下一个时间步连接的，另一个是朝着上一个时间步连接的
'''
#  here put the import lib
import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms

# Device configuration
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.003

# MNIST dataset
train_dataset = torchvision.datasets.MNIST( root='./data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True )
test_dataset = torchvision.datasets.MNIST( root='./data',
                                           train=False,
                                           transform=transforms.ToTensor() )

# Data loader
train_loader = torch.utils.data.DataLoader( dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True )
test_loader = torch.utils.data.DataLoader( dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False )

# Bidirectional recurrent neural network( many-to-one )
class BiRNN( nn.Module ):
    def __init__( self, input_size, hidden_size, num_layers, num_classes ):
        super( BiRNN, self ).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM( input_size, hidden_size, num_layers, batch_first=True, bidirectional=True )
        self.fc = nn.Linear( hidden_size*2, num_classes ) #  2 for bidirection

    def forward( self, x ):
        # Set initial state
        h0 = torch.zeros( self.num_layers*2, x.size(0), self.hidden_size ).to( device )
        c0 = torch.zeros( self.num_layers*2, x.size(0), self.hidden_size ).to( device )

        # Forward propagate LSTM
        out, _ = self.lstm( x, ( h0, c0 ) ) #  out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc( out[ :, -1, : ] )
        return out

model = BiRNN( input_size, hidden_size, num_layers, num_classes ).to( device )

# Loss and optimize
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate )

# Train the model
total_step = len( train_loader )
for epoch in range( num_epochs ):
    for i, ( images, labels ) in enumerate( train_loader ):
        images = images.reshape( -1, sequence_length, input_size ).to( device )
        labels = labels.to( device )

        # Forward pass
        outputs = model( images )
        loss = criterion( outputs, labels )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if ( i+1 ) % 100 == 0:
            print( 'Epoch[{}/{}], Step[{}/{}], loss:'
                   .format( epoch+1, num_epochs, i+1, total_step, loss.item() ))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape( -1, sequence_length, input_size ).to( device )
        labels = labels.to( device )
        outputs = model( images )
        _, predicted = torch.max( outputs.data, 1 )
        total += labels.size(0)
        correct += ( predicted == labels ).sum().item()

    print( 'Test accuracy {}%'.format( 100*correct/total ) )                   