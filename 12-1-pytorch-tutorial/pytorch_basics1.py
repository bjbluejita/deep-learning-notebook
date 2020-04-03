'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2020年01月06日 16:10
@Description: 
@URL: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
@version: V1.0
'''
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

#  ================================================================== # 
#                          Table of Contents                          # 
#  ================================================================== # 

#  1. Basic autograd example 1               (Line 25 to 39)
#  2. Basic autograd example 2               (Line 46 to 83)
#  3. Loading data from numpy                (Line 90 to 97)
#  4. Input pipline                          (Line 104 to 129)
#  5. Input pipline for custom dataset       (Line 136 to 156)
#  6. Pretrained model                       (Line 163 to 176)
#  7. Save and load model                    (Line 183 to 189)

#  ================================================================== # 
#                      1. Basic autograd example 1                    # 
#  ================================================================== # 
# Create tensors
x = torch.tensor( 1., requires_grad=True )
w = torch.tensor( 2., requires_grad=True )
b = torch.tensor( 3., requires_grad=True )

# Build a computional graph
y = w * x + b     # y = 2 * x + 3

# Compute gradients
y.backward()

# Print out the gradients
print( x.grad )
print( w.grad )
print( b.grad )

#  ================================================================== # 
#                     2. Basic autograd example 2                     # 
#  ================================================================== # 
# Create tensor of shape( 10, 3 ) and ( 10, 2 )
x = torch.randn( 10, 3 )
y = torch.randn( 10, 2 )

# Build a full connected layer
linear = nn.Linear( 3, 2 )
print( 'w: ', linear.weight )
print( 'b: ', linear.bias )

# Build loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD( linear.parameters(), lr=0.01 )

# Forward pass.
pred = linear( x )

# Compute loss
loss = criterion( pred, y )
print( 'loss:', loss.item() )

# Backforward pass:
loss.backward()

# Print out the gradients
print( 'dL/dw:', linear.weight.grad )
print( 'dL/db:', linear.bias.grad )

#  1-step gradient descent.
optimizer.step()

# Print out the loss after 1 step gradient descent
pred = linear( x )
loss = criterion( pred, y )
print( 'weight after 1 step optimization:', linear.weight )
print( 'bias after 1 step optimization:', linear.bias )
print( 'loss after 1 step optimization: ', loss.item() )

#  ================================================================== # 
#                      3. Loading data from numpy                     # 
#  ================================================================== # 
# Create a numpy array.
x = np.array( [ [1,2], [3,4] ] )

# Convert the numpy array to a torch tensor
y = torch.from_numpy( x )

# Convert the torch tensor to a numpy array
z = y.numpy()
print( x ,y, z )

#  ================================================================== # 
#                          4. Input pipline                           # 
#  ================================================================== # 
'''
# Download and construct CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10( root='./',
                                              train=True,
                                              transform=transforms.ToTensor(),
                                              download=True
                                              )
# Fetch one data pair( read data from disk )
image, label = train_dataset[0]
print( image.size() )
print( 'label=', label )

#  Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader( dataset=train_dataset,
                                            batch_size=64,
                                            shuffle=True )

# When iteration starts, queue and thread start to load data from file
data_iter = iter( train_loader )

# Mini-batch images and labels
images, labels = data_iter.next()
print( 'Mini-batch:', len( labels.numpy()) )

# Actual usage of the data loader is as below
# for iamges, labels in train_loader:
#     print( 'loop batch:', len( labels.numpy() ) )
'''
t = torch.tensor([
    [ 1, 1, 1, 1 ],
    [ 2, 2, 2, 2 ],
    [ 3, 3, 3, 3 ]
   ], dtype=torch.float32 )
print( t[0], 'sum=', t[0].sum() )
print( t.sum( dim=1) )

t = torch.tensor( [
    [ 1, 0, 0, 2 ],
    [ 0, 3, 2, 0 ],
    [ 4, 0, 0, 5 ]
], dtype=torch.float32 )
print( t.max() )
print( t.flatten() )
print( t.argmax() )
print( t.max( dim=1 ) )
print( 't.argmax', t.argmax( dim=1 ) )

#  ================================================================== # 
#                 5. Input pipline for custom dataset                 # 
#  ================================================================== # 
class CustomDataset( torch.utils.data.Dataset ):
    def __init__(self):
        #  TODO
        #  1. Initialize file paths or a list of file names.
        pass

    def __getitem__(self, item):
        #  TODO
        #  1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        #  2. Preprocess the data (e.g. torchvision.Transform).
        #  3. Return a data pair (e.g. image and label).
        pass

    def __len__(self):
        #  You should change 0 to the total size of your dataset.
        return 0
cusome_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader( dataset=cusome_dataset,
                                           batch_size=64,
                                           shuffle=False )

#  ================================================================== # 
#                         6. Pretrained model                         # 
#  ================================================================== # 
# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18( pretrained=True )

# if you want to fineturn only the top layer of the model, set as below
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear( resnet.fc.in_features, 100 )  # 100 is an example

# forward pass.
images = torch.randn( 64, 3,  224, 224 )
ouputs = resnet( images )
print( ouputs  )


#  ================================================================== # 
#                       7. Save and load the model                    # 
#  ================================================================== # 
# Save and load the entire model
torch.save( resnet, 'model.ckpt' )
model = torch.load( 'model.ckpt' )

# save and load only the parmaters 
torch.save( resnet.state_dict(), 'params.ckpt' )
resnet.load_state_dict( torch.load( 'params.ckpt' ) )


