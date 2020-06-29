import numpy as np

def softmax( X ):
    """
    softmax 函数实现
    X : 一个二维矩阵, m * n,其中m表示向量个数，n表示向量维度
    返回：
    softmax计算结果
    """
    assert( len( X.shape )== 2 )
    row_max = np.max( X ).reshape( -1, 1 )
    X -= row_max
    X_exp = np.exp( X )
    S = X_exp / np.sum( X_exp, keepdims=True )

    return S

a = [[ 1, 2, 3], [ -1, -2, -3 ] ]

a = np.array( a )

print( softmax( a ) )
