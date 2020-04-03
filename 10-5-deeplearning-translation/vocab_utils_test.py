'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月18日 14:40
@Description: Tests for vocab_utils.
@URL: https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_16/nmt/utils/vocab_utils_test.py
@version: V1.0
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import tensorflow as tf

print( '__file__={0:<35} | __name__={1:<20}|__package__:{2:<20}'.format( __file__, __name__, str( __package__) ))

from nmt.utils import vocab_utils

class testCheckVocab( tf.test.TestCase ):

    def testCheckVocab( self ):
        # Create a vocab file
        vocab_dir = os.path.join( tf.test.get_temp_dir(), 'vocab_dir' )
        os.makedirs( vocab_dir )
        vocab_file = os.path.join( vocab_dir, 'vocab_file' )
        vocab = [ "alpha", "beta", "charli", 'delta' ]
        with codecs.getreader( 'utf-8' )( tf.gfile.GFile( vocab_file, 'wb' ) ) as f:
            for word in vocab:
                f.write( '%s\n' % word )

        # Call vocab_utils
        out_dir = os.path.join( tf.test.get_temp_dir(), 'out_dir' )
        os.makedirs( out_dir )
        vocab_size, new_vocab_file = vocab_utils.check_vocab(
            vocab_file, out_dir
        )

        # Assert: we expect the code to add  <unk>, <s>, </s> and
        # create a new vocab file
        self.assertEqual( len( vocab ) + 3, vocab_size )
        self.assertEqual( os.path.join( out_dir, 'vocab_file' ), new_vocab_file )
        new_vocab = []
        with codecs.getreader( 'utf-8' )( tf.gfile.GFile( new_vocab_file, 'rb' ) ) as f:
            for line in f:
                new_vocab.append( line.strip() )
        self.assertEqual(
            [ vocab_utils.UNK, vocab_utils.SOS, vocab_utils.EOS ] + vocab, new_vocab
        )


if __name__ == '__main__':
    tf.test.main()