'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月30日 16:33
@Description: 
@URL: https://www.jianshu.com/p/94ce4c96f987
@version: V1.0
'''

class A( object ):

    def foo(self, x ):
        print( 'executing foo( %s, %s)' % ( self, x ) )

    @classmethod
    def class_foo(cls, x ):
        print( 'executing class_foo( %s, %s)' % ( cls, x ) )

    @staticmethod
    def static_foo( x ):
        print( 'executing static_foo( %s ) ' % x )


a = A()
A.class_foo( 1 )
a.foo( 2 )
A.static_foo( 3 )