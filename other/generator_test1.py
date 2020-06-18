#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   generator_test1.py
@Time    :   2020/06/09 11:33:20
@Author  :   LY 
@Version :   1.0
@URL     :   https://www.cnblogs.com/xybaby/p/6322376.html
@License :   (C)Copyright 2017-2020
@Desc    :   python yield generator 详解
'''
# here put the import lib
def gen_generator():
    yield 1

def gen_value():
    return 1

def gen_example1():
    print( 'before any yield')
    yield "first yields"
    print( 'between yield')
    yield "second yields"
    print( 'no yield any more')

def Fib():
    a, b = 1, 1
    while( True ):
        yield a
        a, b = b, a+b


def test1():
    ret = gen_generator()
    print( ret, type( ret ) )
    ret = gen_value()
    print( ret, type( ret ) )

def test2():
    gen = gen_example1()
    t = gen.__next__()
    print( t )
    t = gen.__next__()
    print( t )
    t = gen.__next__()
    print( t )

def test3():
    # for语句能自动捕获StopIteration异常
    gen = gen_example1()
    for i in range( 1, 3 ):
        for t in gen:
            print( t )

def test4():
    # 在函数中使用Yield，然后调用该函数是生成generator的一种方式。
    # 另一种常见的方式是使用generator expression，For example：
    gen = ( x*x for x in range( 5) )
    print( gen )
    for el in gen:
        print( el )

def test5():
    RANGE_NUM = 100
    for i in [ x*x for x in range( RANGE_NUM ) ]:
        print( i )

    for i in ( x*x for x in range( RANGE_NUM )):
        print( i )

def test6():
    fib = Fib()
    for i in range( 100 ):
        print( fib.__next__() )


if __name__ == "__main__":
    test1()
    test3()
    test4()
    test5()
    test6()