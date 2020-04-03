'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年06月24日 14:47
@Description: 
@URL: 
@version: V1.0
'''
import unittest
from  read_utils import TextConverter, batch_generator
from model import CharRNN
import tensorflow as tf
import codecs

class testReadUtils( unittest.TestCase ):

    def test_vocab_size(self):
        testConverter = TextConverter( text=[ "We","are","accounted","poor","citizens,","the","patricians","goodare","accounted","poor","citizens,","the","patricians","good" ],
                                       max_vocab=10 )
        print( testConverter.vocab_size )
        print( testConverter.int_to_word(4) )
        print( testConverter.text_to_arr( ['the']))
        print( testConverter.arr_to_text( [ 3, 4]))

    def test_save_file(self):
        testConverter = TextConverter( text=[ "We","are","accounted","poor","citizens,","the","patricians","goodare","accounted","poor","citizens,","the","patricians","good" ],
                                       max_vocab=10 )
        testConverter.save_to_file( 'test.pcl')

    def test_CharRNN(self):
        t = CharRNN( num_classes=26 )
        print( t )
        print( t.optimizer )

    def test_batch_generator(self):
        with codecs.open( 'data/shakespeare.txt', encoding='utf-8' ) as f:
            text = f.read()
        converter = TextConverter( text, 35000  )
        arr = converter.text_to_arr( text )
        g = batch_generator( arr, 32, 50 )
        count = 0
        for x, y in g:
            count += 1
            print( count )

if __name__ == "__main__":
    unittest.main()