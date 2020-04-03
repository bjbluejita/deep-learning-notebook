'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2020年01月02日 17:21
@Description: 
@URL: https://wiki.tiker.net/PyCuda/Examples/PlotRandomData
@version: V1.0
'''
import pycuda.autoinit
import pycuda.curandom as curandom

size = 5000
a = curandom.rand( (size, ) ).get()

from matplotlib.pyplot import *
subplot( 211 )
plot( a )
grid( True )
ylabel( 'plot - gpu' )

subplot( 212 )
hist( a, 100 )
grid( True )
ylabel( 'Histogram - gpu' )

show()