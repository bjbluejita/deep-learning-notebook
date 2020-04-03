'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月31日 17:44
@Description: 
@URL: https://blog.csdn.net/zdcs/article/details/53389371
@version: V1.0
'''
import inspect

def printLineAndFileFunc():
    calledFrameRecord = inspect.stack()[1]
    print( inspect.stack() )
    frame = calledFrameRecord[0]
    info = inspect.getframeinfo( frame )
    print( 'file:%s function:%s line:%d' %( info.filename, info.function, info.lineno ) )

def main():
    printLineAndFileFunc()

if __name__ == '__main__':
    main()