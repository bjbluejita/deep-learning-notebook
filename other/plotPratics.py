#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   plotPratics.py
@Time    :   2020/04/01 11:15:22
@Author  :   LY 
@Version :   1.0
@URL     :
@License :   (C)Copyright 2017-2020
@Desc    :   plt 画动态图
'''
# here put the import lib
import matplotlib.pyplot as plt
import random

fig, ax = plt.subplots()
y1 = []
for i in range( 50 ):
    y1.append( i + random.random() * 5 )
    ax.cla()
    #ax.bar( y1, label='test', height=y1, width=0.3 )
    ax.scatter( range(len(y1)), y1, label='test' )
    ax.legend()
    plt.pause( 0.1 )
