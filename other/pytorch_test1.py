'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2020年01月06日 11:11
@Description: 
@URL: 
@version: V1.0
'''
import torch
import torch.nn as nn
from torch.autograd import  Variable
import numpy  as np

print( torch.__version__ )
print( torch.cuda.is_available() )
print( torch.version.cuda )

t = torch.tensor( [ 1, 2, 3 ] )
print( t )
t = t.cuda()
print( t )

a=torch.tensor([[4, 5, 2, 3],
                [3, 6, 7, 8]])
# 将 mask必须是一个 ByteTensor 而且shape必须和 a一样 并且元素只能是 0或者1,是将 
# mask中为1的 元素所在的索引，在a中相同的的索引处替换为 value 
print( a.masked_fill(mask = torch.ByteTensor([1,0, 0, 1 ]), value=torch.tensor(-1e9)))

a = torch.tensor( np.random.randint(0, 10, size=( 3, 8, 10, 10 )) )
aMask = torch.ByteTensor( np.random.choice( [ True, False ], size=( 3, 1, 1, 10 )))
print( 'a=', a )
print( 'mask=', aMask )
aMasked = a.masked_fill( mask=aMask, value=torch.tensor(999) )
print( 'mask a =',  aMasked.shape, '  shape' )
print( 'mask a =',  aMasked )

word_idx = { 'hello':0, 'world':1 }
embeds = nn.Embedding( 2, 5 )
helloTensor = torch.LongTensor( [ word_idx[ 'hello' ]] )
hello_idx = Variable( helloTensor )
worldTensor = torch.LongTensor( [ word_idx[ 'hello' ]] )
hello_idx = Variable( helloTensor )
hello_emb = embeds( hello_idx )
print( hello_emb )


print( '-----finished------')


