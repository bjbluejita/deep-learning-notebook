
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, copy
from torch.autograd import  Variable
import matplotlib.pyplot as plt
import seaborn
import spacy
import os
# For data loading.
from torchtext import data, datasets

A = [ 1, 2, 3, 4 ]
B = [ 'b1', 'b2', 'b3' ]

for a, b in zip( A, B ):
    print( a, '->', b ) 

def clones( module, N ):
    '''Produce N identical layers.'''
    return nn.ModuleList( [ copy.deepcopy( module ) for _ in range( N ) ] )

d_model = 512
h = 8
d_k = d_model // h

query = key = value = torch.Tensor( np.random.randint( 1, 25, size=( 1, 25, d_model ) ) )
# key = torch.Tensor( np.random.randint( 1, 25, size=( 1, 25, 512 ) ) )
# value = torch.Tensor( np.random.randint( 1, 25, size=( 1, 25, 512 ) ) )

linears = clones( nn.Linear( d_model, d_model ), 4 )

'''
for l, x in zip( linears, ( query, key, value) ):
    nbatches = x.size(0)
    print( x.shape, '->', l(x).view( nbatches, -1, h, d_k ).transpose( 1, 2 ).shape )
'''
nbatches = query.size( 0 )
query, key, value = [ l(x).view( nbatches, -1, h, d_k ).transpose( 1, 2 ) 
                       for l, x in zip( linears, ( query, key, value )) ]
print( query.shape, key.shape, value.shape )
print( 'transpose:', key.transpose( -2, -1 ).shape )
print( '--attentino:')
scores = torch.matmul( query, key.transpose( -2, -1 )) / math.sqrt( d_k )
print( 'matmul:', scores.shape )
p_attn = F.softmax( scores, dim=-1 )
print( 'softmax:', p_attn.shape )



print( 'done')