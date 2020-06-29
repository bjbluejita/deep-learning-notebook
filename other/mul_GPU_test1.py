#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mul_GPU_test1.py
@Time    :   2020/06/27 15:26:07
@Author  :   LY 
@Version :   1.0
@URL     :   https://www.cnblogs.com/marsggbo/p/11534141.html
@License :   (C)Copyright 2017-2020
@Desc    :   None
'''
# here put the import lib
import torch
import torch.nn as nn
# import ipdb; ipdb.set_trace()

class DataParalleModel( nn.Module ):
    def __init__( self ):
        super().__init__()
        self.block1 = nn.Linear( 10, 20 )

    def forward( self, x ):
        x = self.block1( x )
        return x

def data_parallel( module, input, device_ids, output_device=None ):
    if not device_ids:
        return module( input )
    
    if output_device is None:
        output_device = device_ids[ 0 ]

    replicas = nn.parallel.replicate( module, device_ids )
    print( f"replicas:{replicas}" )

    inputs = nn.parallel.scatter( input, device_ids )
    print( f'inputs:{inputs}' )

    for i in range( len( inputs )):
        print( f"input {i}:{ inputs[i].shape}" )

    replicas = replicas[ :len( inputs ) ]
    outputs = nn.parallel.parallel_apply( replicas, inputs )
    print( f'outputs: {type( outputs )}' )
    for i in range( len( outputs ) ):
        print( f"output {i}:{ outputs[i].shape}" )

    result = nn.parallel.gather( outputs, output_device )

    return result

model = DataParalleModel()
x = torch.rand( 16, 10 )
result = data_parallel( model.cuda(), x.cuda(), [0] )
print( f"result: { type(result) }" )