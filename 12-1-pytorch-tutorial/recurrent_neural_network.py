# !/usr/bin/env python
#  -*- encoding: utf-8 -*-
'''
@File    :   recurrent_neural_network.py
@Time    :   2020/01/13 14:04:37
@Author  :   LY 
@Version :   1.0
@Description :
@URL     :https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py# L39-L58
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

#  here put the import lib
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configureation
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST( root='./data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True  )
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

# Recurrent  neural network( many-to-one )
class RNN( nn.Module ):
    def __init__( self, input_size, hidden_size, num_layers, num_classes ):
        super( RNN, self ).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM( input_size, hidden_size, num_layers, batch_first=True )  
        self.fc = nn.Linear( hidden_size, num_classes )

    def forward( self, x ):
        # Set initial hidden and cell states
        h0 = torch.zeros( self.num_layers, x.size(0), self.hidden_size ).to( device )
        c0 = torch.zeros( self.num_layers, x.size(0), self.hidden_size ).to( device )

        # Forward propagate LSTM
        out, _ = self.lstm( x, ( h0, c0 ) )  # out: tensor of shape( batch_size, seq_length,hidden_size )                                                                                        

        # Decode the hidden state of the last time step
        out = self.fc( out[ :, -1, : ] )
        return out

model = RNN( input_size, hidden_size, num_layers, num_classes ).to( device )        

# Loss and optimizer
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
            print( 'Epoch[{}/{}], Step[{}/{}], Loss{:4f}'
                    .format( epoch+1, num_epochs, i, total_step, loss.item() ))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for iamges, labels in ( test_loader ):
        images = images.reshape( -1, sequence_length, input_size ).to( device )
        labels = labels.to( device )
        outputs = model( images )
        _, prediced = torch.max( outputs.data, 1 )
        total += labels.size(0)
        correct += ( prediced == labels ).sum().item()

    print( 'Test Accuract Score:{}%'.format( 100*correct/total ) )