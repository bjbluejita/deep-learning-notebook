'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年01月22日 13:44
@Description: 
@URL: https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/1.6-visualizing-what-convnets-learn.ipynb
@version: V1.0
'''
import warnings
warnings.filterwarnings('ignore')

import platform
import tensorflow
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython.display import Image
print("Platform: {}".format(platform.platform()))
print("Tensorflow version: {}".format(tensorflow.__version__))
print("Keras version: {}".format(keras.__version__))
print("numpy version: {}".format( np.__version__ ) )

MODEL_FILE = 'cats_and_dogs_small_2.h5'

from keras.models import load_model
model = load_model( MODEL_FILE )


