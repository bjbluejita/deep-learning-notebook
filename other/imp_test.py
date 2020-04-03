'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月05日 14:53
@Description: 
@URL: 
@version: V1.0
'''
import  imp

pathname = "C:\\Anaconda3\\envs\\tensorflow2.0\\lib\\site-packages\\tensorflow\\python\\_pywrap_tensorflow_internal.pyd"
description = ('.pyd', 'rb', 3)
with open(  pathname, 'rb' ) as fp:
    _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)