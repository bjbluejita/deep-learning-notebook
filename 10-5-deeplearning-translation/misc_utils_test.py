'''
Tests for vocab_utils.
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月18日 11:10
@Description: 
@URL: https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_16/nmt/utils/misc_utils_test.py
@version: V1.0
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import time

from nmt.utils import misc_utils

class MiscUtilsTest( tf.test.TestCase ):

    def testFormatBpeText(self ):
        bpe_line =(
            b"En@@ ough to make already reluc@@ tant men hesitate to take screening"
            b" tests ."
        )
        expected_result = (
            b"Enough to make already reluctant men hesitate to take screening tests"
            b" ."
        )
        self.assertEqual( expected_result,
                          misc_utils.format_bpe_text( bpe_line.split( b" " )))

    def testFormatText(self):
        text_line = (
            b"Enough to make already reluctant men hesitate to"
            b" take screening tests all."
        )
        expected_result = (
            b"Enough to make already reluctant men hesitate to take screening tests"
            b" all."
        )
        self.assertEqual( expected_result,
                          misc_utils.format_text( text_line.split( b" ") ) )

    def testGet_config_proto(self):
        start_time = time.time()
        misc_utils.print_time( misc_utils.get_config_proto(), start_time )

    def testCheck_tensorflow_version(self):
        misc_utils.check_tensorflow_version()

    def testLoad_hparams(self):
        hparms_file = './nmt/standard_hparams/iwslt15.json'
        hprams = misc_utils.load_hparams( './nmt/standard_hparams/' )
        print( hprams )

    '''
    def testmaybe_parase_standard_hparams(self):
        hparms_file = './nmt/standard_hparams/iwslt15.json'
        hprams = misc_utils.load_hparams( './nmt/standard_hparams/' )
        hprams = misc_utils.maybe_parase_standard_hparams( hprams, hparms_file )
        self.assertEqual( hprams.init_weight, 0.2  )
    '''

    def test_save_hparams(self):
        hprams = misc_utils.load_hparams( './nmt/standard_hparams/' )
        misc_utils.save_hparams( 'C:\\Users\\Administrator\\', hprams  )

    def testPrint_out(self ):
        misc_utils.print_out( 'hhhh' )
        misc_utils.print_out( 'hhhh' )

if __name__ == '__main__':
    tf.test.main()
