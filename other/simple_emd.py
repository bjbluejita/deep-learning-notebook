'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月10日 12:45
@Description: 
@URL: https://github.com/jaidevd/pyhht/blob/dev/docs/examples/simple_emd.py
@version: V1.0
'''
import numpy as np
from numpy import pi, sin, linspace, sqrt
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs
import matplotlib
import matplotlib.pyplot as plt

t = linspace(0, 1, 1000)
modes = sin(2 * pi * 5 * t) + sin(2 * pi * 10 * t) + sin(2 * pi * 3 * t)
x = modes + t
plt.plot( t, modes )
plt.plot( t, x )
plt.show()
decomposer = EMD(x)
imfs = decomposer.decompose()

plot_imfs(x, imfs, t )

sumValues = np.zeros_like( imfs[0] )
for i in range( imfs.shape[0] ):
    sumValues += imfs[i]

plt.plot( t, sumValues )
plt.show()
