#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   logger.py
@Time    :   2020/02/29 12:10:04
@Author  :   LY 
@Version :   1.0
@URL     :   https://github.com/liuyuemaicha/PyTorch-YOLOv3/blob/master/utils/logger.py
@License :   (C)Copyright 2017-2020
@Desc    :   None
'''
# here put the import lib
import tensorflow as tf

class Logger( object ):
    def __init__( self, log_dir ):
        '''Create a summary writer logging to log_dir.'''
        self.writer = tf.summary.FileWriter( log_dir )

    def scalar_summary( self, tag, value, step ):
        '''Log a scaler variable'''
        summary = tf.Summary( value=[ tf.Summary.Value( tag=tag, simple_value=value ) ] )
        self.writer.add_summary( summary, step )

    def list_of_scalars_summary( self, tag_value_pairs, step ):
        '''Log scalar variables'''
        summary = tf.Summary( value=[ tf.Summary.value( tag=tag, simple_value=value ) for tag, value in tag_value_pairs ] )
        self.writer.add_summary( summary, step )