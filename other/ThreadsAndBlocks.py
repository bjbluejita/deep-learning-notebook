'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2020年01月03日 11:11
@Description: 
@URL: https://wiki.tiker.net/PyCuda/Examples/ThreadsAndBlocks
@version: V1.0
'''
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule( """
    #include <stdio.h>
    __global__ void say_hi()
    {
         printf( "I am %dth thread in threadIdx.x:%d threadIdx.y:%d blockIdx.x:%d blockIdx.y:%d blockDim.x:%d blockDim.y:%d\\n",
                 (threadIdx.x + threadIdx.y*blockDim.x + ( blockIdx.x*blockDim.x*blockDim.y) + ( blockIdx.y*blockDim.x*blockDim.y )),
                 threadIdx.x, threadIdx.y,
                 blockIdx.x, blockIdx.y,
                 blockDim.x, blockDim.y );
     }
""")

func = mod.get_function( "say_hi" )
func( grid=(2,2,1), block=(4,4,1) )