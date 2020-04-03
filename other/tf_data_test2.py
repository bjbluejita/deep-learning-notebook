'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月22日 15:11
@Description: 
@URL: 
@version: V1.0
'''
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from tensorflow.python.data import Dataset
from tensorflow.python.data.experimental import  group_by_window

tf.logging.set_verbosity( tf.logging.INFO )

def print_Dataset( dataset ):
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        while True:
            try:
                print( sess.run( one_element ) )
            except tf.errors.OutOfRangeError:
                print( 'end' )
                break

def print_Dataset_stateful( dataset ):
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    one_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run( tf.tables_initializer() )
        sess.run( init_op )
        while True:
            try:
                print( sess.run( one_element ) )
            except tf.errors.OutOfRangeError:
                print( 'end' )
                break
batch_size= 2
src_max_len = 3
num_buckets = 5

tgt_vocab_table = src_vocab_table = lookup_ops.index_table_from_tensor(  ## 这里构造了个查找表 ##
    tf.constant( [ 'a', 'b', 'c', 'eos', 'sos' ] )
)
src_dataset = Dataset.from_tensor_slices(
    tf.constant( [ 'c c a', 'c a', 'd', 'f e a g' ] )
)
tgt_dataset = Dataset.from_tensor_slices(
    tf.constant( [ 'a b', 'b c', '',  'c c' ] )
)
src_eos_id = tf.cast(
    src_vocab_table.lookup( tf.constant( 'eos' ) ),
    tf.int32
)
tgt_sos_id = tf.cast(
    tgt_vocab_table.lookup( tf.constant( 'sos' ) ),
    tf.int32
)
tgt_eos_id = tf.cast(
    tgt_vocab_table.lookup( tf.constant( 'eos' ) ),
    tf.int32
)
src_tgt_dataset = Dataset.zip( ( src_dataset, tgt_dataset ) )
print( 'begin')
print_Dataset( src_tgt_dataset )

src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (
        tf.string_split( [src] ).values, tf.string_split( [tgt] ).values
    )
)
print( 'string_split')
print_Dataset( src_tgt_dataset )

src_tgt_dataset = src_tgt_dataset.filter(
    lambda src, tgt: tf.logical_and( tf.size(src) >0, tf.size( tgt) > 0 )
)
print( 'Filter zero length input sequences')
print_Dataset( src_tgt_dataset )

src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: ( src[ :src_max_len ], tgt )
)
print( 'src_max_len')
print_Dataset( src_tgt_dataset )

src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (
        tf.cast( src_vocab_table.lookup( src ), tf.int32 ),
        tf.cast( tgt_vocab_table.lookup( tgt ), tf.int32 )
    )
)
print( 'Convert the word strings to ids')
print_Dataset_stateful( src_tgt_dataset )

src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt:( src,
                      tf.concat( ( [tgt_sos_id], tgt ), 0 ),
                      tf.concat( ( tgt, [ tgt_eos_id ] ), 0)
                      )
)
print( ' Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>')
print_Dataset_stateful( src_tgt_dataset )

src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt_in, tgt_out: (
        src, tgt_in, tgt_out, tf.size( src ), tf.size( tgt_in )
    )
)
print( 'Add in the word counts')
print_Dataset_stateful( src_tgt_dataset )



def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=( tf.TensorShape( [None] ), # src
                        tf.TensorShape( [None] ), #tgt_input
                        tf.TensorShape( [None] ), # tgt_output
                        tf.TensorShape( [] ),     # src_len
                        tf.TensorShape( [] )      # tgt_len
                        ),
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=( src_eos_id,   # src
                         tgt_sos_id,   # tgt_input
                         tgt_eos_id,   #tgt_outpout
                         0,            # src_len -- unused
                         0             # tgt_len -- unused
                         )
    )
def key_func( unused_1, unused_2, unused_3, src_len, tgt_len ):
    # Calculate bucket_width by maximum source sequence length.
    # Pairs with length [0, bucket_width) go to bucket 0, length
    # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
    # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
    #tf.logging.info( 'unused_1'  )
    if src_max_len:
        bucket_width = ( src_max_len + num_buckets - 1 ) // num_buckets
    else:
        bucket_width = 10
    # Bucket sentence pairs by the length of their source sentence and target
    # sentence.
    bucket_id = tf.maximum( src_len // bucket_width, tgt_len // bucket_width )
    return tf.to_int64( tf.minimum( num_buckets, bucket_id ) )

def reduce_func( unused_key, windowed_data ):
    return batching_func( windowed_data )


batched_dataset = src_tgt_dataset.apply( group_by_window(
    key_func=key_func, reduce_func=reduce_func, window_size=batch_size
) )

print( 'group_by_window')
print_Dataset_stateful( batched_dataset )
