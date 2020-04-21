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
print( torch.__version__ )
print( torch.cuda.is_available() )
print( torch.version.cuda )

t = torch.tensor( [ 1, 2, 3 ] )
print( t )
t = t.cuda()
print( t )

a=torch.tensor([4,5,2,3])
print( a.masked_fill(mask = torch.ByteTensor([1,0,0,1]), value=torch.tensor(-1e9)))
