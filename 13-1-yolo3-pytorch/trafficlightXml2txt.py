#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trafficlightXml2txt.py
@Time    :   2020/03/19 23:06:30
@Author  :   LY 
@Version :   1.0
@URL     :
@License :   (C)Copyright 2017-2020
@Desc    :   traffic light label xml file convert to txt
'''
# here put the import lib
import os
import shutil
from xml.dom.minidom import parse

def getOneXmlFile( xmlPath ):
    domTree = parse( xmlPath )
    # 文档根元素
    rootNode = domTree.documentElement
    print( rootNode.getElementsByTagName( 'folder' )[0].childNodes[0].data )

    fileName = rootNode.getElementsByTagName( 'filename' )[0].childNodes[0].data
    print( 'fileName=', fileName )

    sizeNodes = rootNode.getElementsByTagName( 'size' )
    img_width = int( sizeNodes[0].getElementsByTagName( 'width' )[0].childNodes[0].data )
    img_height = int( sizeNodes[0].getElementsByTagName( 'height' )[0].childNodes[0].data )
    print( 'img width=', img_width , ' height=', img_height )

    bndBoxes = []
    targetObjectes = rootNode.getElementsByTagName( 'object' )
    for targetObject in targetObjectes:
        bndbox = targetObject.getElementsByTagName( 'bndbox' )
        x1 = int( bndbox[0].getElementsByTagName( 'xmin' )[0].childNodes[0].data )        
        y1 = int( bndbox[0].getElementsByTagName( 'ymin' )[0].childNodes[0].data )        
        x2 = int( bndbox[0].getElementsByTagName( 'xmax' )[0].childNodes[0].data )        
        y2 = int( bndbox[0].getElementsByTagName( 'ymax' )[0].childNodes[0].data )

        x1_scale = ( x1 + ( x2 - x1 ) / 2 ) / img_width
        y1_scale = ( y1 + ( y2 - y1 ) / 2 ) / img_height
        x2_scale = ( x2 - x1 ) / img_width
        y2_scale = ( y2 - y1 ) / img_height
       
        bndBoxes.append( ( x1_scale, y1_scale, x2_scale, y2_scale ))
        print( ( x1_scale, y1_scale, x2_scale, y2_scale ) ) 

    return fileName, bndBoxes

if __name__ == '__main__':
    rootDir = 'E:/trafficlight_detect/light_label'
    targetDir = 'E:/trafficlight_detect/labels'

    files = os.listdir( rootDir )
    for file in files:
        fileName, bndBoxes = getOneXmlFile( os.path.join( rootDir, file ) )
        fileName = fileName.replace( 'jpg', 'txt' )
        fp = open( os.path.join( targetDir, fileName), 'w' )
        for bbox in bndBoxes:
            outputStr = '1 {:.6f} {:.6f} {:.6f} {:.6f}'.format( bbox[0], bbox[1], bbox[2], bbox[3])
            fp.write( outputStr )
            fp.write( '\n' )
        fp.close()
