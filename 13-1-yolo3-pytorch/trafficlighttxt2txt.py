#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trafficlighttxt2txt.py
@Time    :   2020/03/22 09:25:29
@Author  :   LY 
@Version :   1.0
@URL     :
@License :   (C)Copyright 2017-2020
@Desc    :   traffic light label txt file convert to other txt
'''
# here put the import lib
import numpy as np
import pandas as pd
import os
from PIL import Image
from PIL import ImageFile


def checkImgFile( imgPath, labelPdf ):
    for item in os.listdir( imgPath ):
        if item not in data[ 'fileName' ]:
            print( 'Not found :', item )

if __name__ == '__main__':
    origLabelFilePath = 'D:/ChineseTrafficSignDetection/GroundTruth/GroundTruth.txt'
    origImagesFilePath = 'D:/ChineseTrafficSignDetection/images'
    targetLabelPath = 'D:/ChineseTrafficSignDetection/labels'

    column_names = ['fileName', 'x1', 'y1', 'x2', 'y2', 'type']
    data = pd.read_csv( origLabelFilePath, sep=';', header=None, names=column_names )
    trafficlightTypes = data['type'].unique()
    print( trafficlightTypes )

    checkImgFile( origImagesFilePath, data )

    for idx in range( len( data ) ):
        imgPath = data['fileName'].iloc[idx]
        try:
            img = Image.open( os.path.join( origImagesFilePath, imgPath ) )
        except FileNotFoundError:
            continue
        img_width = img.size[0]
        img_height = img.size[1]
        # print( labelFileName )
        x1 = data[ 'x1' ].iloc[ idx ]
        y1 = data[ 'y1' ].iloc[ idx ]
        x2 = data[ 'x2' ].iloc[ idx ]
        y2 = data[ 'y2' ].iloc[ idx ]
        trafficlightType = data[ 'type' ].iloc[ idx ]

        x1_scale = float( ( x1 + ( x2 - x1 ) / 2 ) / img_width )
        y1_scale = float( ( y1 + ( y2 - y1 ) / 2 ) / img_height )
        x2_scale = float( ( x2 - x1 ) / img_width )
        y2_scale = float( ( y2 - y1 ) / img_height )

        typeIdex = np.where( trafficlightTypes == trafficlightType )[0][0]
        labelPath = imgPath.replace( 'png', 'txt' )
        fp = open( os.path.join( targetLabelPath, labelPath ), 'w' )
        outputStr = '{} {:6f} {:6f} {:6f} {:6f}'.format( typeIdex, x1_scale, y1_scale, x2_scale, y2_scale )
        fp.write( outputStr )
        fp.close()

        
