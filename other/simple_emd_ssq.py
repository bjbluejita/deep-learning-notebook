'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月11日 10:43
@Description: 
@URL: 
@version: V1.0
'''
import  numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs

column_names = ['term','date','red1','red2','red3','red4','red5','red6','blue',
                'appear1','appear2','appear3','appear4','appear5','appear6',
                'prize', 'Total','first_num', 'first_amount','second_num', 'second_amount',
                'third_num', 'third_amount', 'fourth_num','fourth_amount', 'fifth_num', 'fifth_amount',
                'sixth_num', 'sixth_amount']
data = pd.read_csv( 'F:/workspace/Tensorflow/src/ssq/ssq.txt', sep=' ', header=None, names=column_names )
print( data.info() )
plt.figure( figsize=( 15, 5 ))
x = range( 0, len( data['red1']), 1 )
plt.plot( x, data['red1'] )
plt.show()

decomposer = EMD( data['red1'] )
imfs = decomposer.decompose()
print( imfs.shape )
plot_imfs( data['red2'].values, imfs, None )