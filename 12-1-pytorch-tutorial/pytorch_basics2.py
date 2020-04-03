'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2020年01月07日 14:37
@Description: 
@URL: https://www.bilibili.com/video/av79803229?p=15
@version: V1.0
'''
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt
import numpy as np

train_set = FashionMNIST( root='./data/FashionMNIST',
                              train=True,
                              download=True,
                              transform=transforms.Compose( [
                                  transforms.ToTensor()
                              ]))
train_loader = torch.utils.data.DataLoader( train_set, batch_size=64 )

# sample = next( iter( train_set ))
sample = next( iter( train_loader ))
image, label = sample
print( 'label=', label )
print( 'shape=', image.numpy().shape )
plt.imshow( image[3].squeeze(), cmap='gray' )
plt.show()

grid = torchvision.utils.make_grid( image, nrow=8 )
plt.figure( figsize=( 15, 15 ) )
plt.imshow( np.transpose( grid, (1, 2, 0 ) ) )
plt.show()
