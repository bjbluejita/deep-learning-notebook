'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年05月28日 16:20
@Description: 
@URL: https://blog.csdn.net/huilan_same/article/details/52944782
@version: V1.0
'''
import unittest
import mathfunc as mf

class TestMathfunc( unittest.TestCase ):

    def test_add( self ):
        self.assertEqual( 3, mf.add( 1, 2 ) )
        self.assertNotEqual(5, mf.add( 2,3 ))

    def test_minus( self ):
        self.assertEqual( 0, mf.minus(1, 1 ))

    def test_multi(self):
        """Test method multi(a, b)"""
        self.assertEqual(6, mf.multi(2, 3))

    def test_divide(self):
        """Test method divide(a, b)"""
        self.assertEqual(2, mf.divide(6, 3))
        self.assertEqual(2.5, mf.divide(5, 2))

if __name__ == '__main__':
    unittest.main()