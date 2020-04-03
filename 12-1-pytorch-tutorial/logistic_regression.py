# !/usr/bin/env python
#  -*- encoding: utf-8 -*-
'''
@File    :   logistic_regression.py
@Time    :   2020/01/09 14:29:36
@Author  :   LY 
@Version :   1.0
@Description :
@URL     :
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

#  here put the import lib
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Hyper-parameters
input_size = 784
num_classes = 10
num_epoches = 6
batch_size = 100
learing_rate = 0.001

# MNIST database
train_dataset = torchvision.datasets.MNIST( root='./data',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True 
                                             )
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

# Logistic regression model
model = nn.Linear( input_size, num_classes )

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD( model.parameters(), lr=learing_rate )

# Train the model
total_step = len( train_loader )
for epoch in range( num_epoches ):
    for i, ( images, labels ) in enumerate( train_loader ):
        #  Reshape images to (batch_size, input_size)
        images = images.reshape( -1, 28*28 )

        # Forward pass
        outputs = model( images )
        loss = criterion( outputs, labels ) # 前向传播求出预测的值

        # Backward and optimize
        optimizer.zero_grad()  # 将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
        loss.backward()
        optimizer.step()

        if ( i+1 ) % 100 == 0:
            print( 'Epoch[{}/{}], Step[{}/{}], Loss:{:4f}'
                    .format( epoch+1, num_epoches, i+1, total_step, loss.item() ))


# Test the model
# In test phase, we don't need to compute gradients
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape( -1, 28*28 )
        outputs = model( images )
        _, predicted = torch.max( outputs.data, 1 )
        total += labels.size( 0 )
        correct += ( predicted == labels ).sum()

    print( 'Accuracy {}%'.format( 100 * correct / total ) )

