'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年06月18日 16:23
@Description: 
@URL: https://docs.python.org/zh-cn/3.6/library/unittest.html
@version: V1.0
'''
import unittest

class TestStringMethods( unittest.TestCase ):

    def test_upper(self ):
        self.assertEqual( 'foo'.upper(), 'FOO' )

    def test_isupper(self):
        self.assertTrue( 'FOO'.isupper() )
        self.assertFalse( 'Foo'.isupper() )

    def test_split(self ):
        s = 'hello world'
        self.assertEqual( s.split(), [ 'hello', 'world'] )
        with self.assertRaises( TypeError ):
             s.split( 2 )


if __name__ == '__main__':
    unittest.main()