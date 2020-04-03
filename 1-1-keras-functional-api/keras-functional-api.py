'''
Created on 2019年1月14日

@author: Administrator
'''
import platform
import tensorflow
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input
from keras.utils import plot_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython.display import Image

print( 'Platform: {}'.format( platform.platform() ))
print( 'Tensorflow version:{}'.format( tensorflow.__version__ ) )
print( 'Kears version: {}'.format( keras.__version__ ))

#model = Sequential( [ Dense(2, input_shape=(1,)), Dense(1)] )
mnist_input = Input( shape=(784,), name='input' )
hidden1 = Dense( 512, activation='relu', name='hidden1')( mnist_input )
hidden2 = Dense( 216, activation='relu', name='hidden2')( hidden1 )
hidden3 = Dense( 128, activation='relu', name='hidden3')( hidden2 )
output = Dense( 10, activation='softmax', name='output')( hidden3 )

model = Model( inputs=mnist_input, outputs=output )
# 打印網絡結構
model.summary()
# 產生網絡拓撲圖
plot_model( model, to_file='multilayer_perceptron_graph.png' )
# 秀出網絡拓撲圖
Image( 'multilayer_perceptron_graph.png' )