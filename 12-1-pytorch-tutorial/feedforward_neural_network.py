# !/usr/bin/env python
#  -*- encoding: utf-8 -*-
'''
@File    :   feedforward_neural_network.py
@Time    :   2020/01/09 15:55:32
@Author  :   LY 
@Version :   1.0
@Description :
@URL     :https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py# L37-L49
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

#  here put the import lib
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# device configuration
# device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
device = 'cpu'

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learing_rate = 0.001

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

# Fully connected neural network with on hidden layer
class NeuralNet( nn.Module ):
    def __init__( self, input_size, hidden_size, num_classes ):
        super( NeuralNet, self ).__init__()
        self.fc1 = nn.Linear( input_size, hidden_size )
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear( hidden_size, num_classes )

    def forward( self, x ):
        out = self.fc1( x )
        out = self.relu( out )
        out = self.fc2( out )
        return out


model = NeuralNet( input_size, hidden_size, num_classes ).to( device )

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( model.parameters(), lr=learing_rate )

# Train the model
total_step = len( train_loader )
for epoch in range( num_epochs ):
    for i, ( images, labels ) in enumerate( train_loader ):
        # Move tensor to configured device
        images = images.reshape( -1, 28*28 ).to( device )
        labels = labels.to( device )

        # Forward pass
        outputs = model( images )
        loss = criterion( outputs, labels )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if  (i+1) % 100 == 0:
            print( 'Epoch[{}/{}], Step[{}/{}], Loss:{:4f}'
                   .format( epoch+1, num_epochs, i+1, total_step, loss.item() ))


# Test the model
with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in   test_loader :
        images = images.reshape( -1, 28*28 ).to( device )
        labels = labels.to( device )
        outputs = model( images )
        _, predicted = torch.max( outputs.data, 1 )
        total += labels.size(0)
        correct += ( predicted == labels ).sum().item()

    print( 'Test accuracy:{}%'.format( 100 * correct / total ))
