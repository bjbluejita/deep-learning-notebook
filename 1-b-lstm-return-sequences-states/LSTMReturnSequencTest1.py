'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年02月22日 10:19
@Description: 
@URL: https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/1.b-lstm-return-sequences-states.ipynb
@version: V1.0
'''

from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
import numpy as np

# 定義模型架構
input_x = Input( shape=(3,1) )
lstm_1 = LSTM(1, return_sequences=True, return_state=True )( input_x )
model = Model( inputs=input_x, outputs=lstm_1 )
# LSTM的模型需要的輸入張量格式為:
# (batch_size，timesteps，input_features)
data = np.array( [ 0.1, 0.2, 0.3 ] ).reshape( 1, 3, 1 )

print( model.predict( data ) )