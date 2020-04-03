'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年02月27日 10:08
@Description: 
@URL: https://zhuanlan.zhihu.com/p/28979653
      https://www.jianshu.com/p/e3b825bc3950?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation
      https://x-algo.cn/index.php/2016/04/10/323/
@version: V1.0
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import  print_function

import collections
import math
import random
import jieba
import numpy as  np
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Download the data.
# Read the data into a list of strings.
def read_data():
    """
    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
    """
    #读取停用词
    stop_words = []
    with open( 'stop_words.txt', 'r', encoding='utf8' ) as f:
        line = f.readline()
        while line:
            stop_words.append( line[:-1] )
            line = f.readline()
    stop_words = set( stop_words )
    print( '停用词读取完毕，共{n}个词'.format( n=len(stop_words) ) )

    # 读取文本，预处理，分词，得到词典
    raw_word_list = []
    with open( 'doupocangqiong.txt', 'r', encoding='utf8' ) as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace( '\n', '' )
            while ' ' in line:
                line = line.replace( ' ', '' )
            if len( line ) > 0 : # 如果句子非空
                raw_words = list( jieba.cut( line, cut_all=False ) )
                raw_word_list.extend( raw_words )
            line = f.readline()

    return raw_word_list

words = read_data()
print( 'Data size', len(words) )

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000
def build_dataset( word ):
    count = [ ['UNK', -1 ] ]
    count.extend( collections.Counter( words ).most_common( vocabulary_size - 1) )
    print( 'count:', len( count ) )
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len( dictionary )
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append( index )

    count[0][1] = unk_count
    reverse_dictionary = dict( zip( dictionary.values(), dictionary.keys() ) )
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset( words )
#删除words节省内存
del words
print( 'Most common words (+UNK):', count[:5] )
print( 'Sample data', data[:10], [ reverse_dictionary[i] for i in data[:10]] )

data_index = 0
# Step 3: Function to generate a training batch for the skip-gram model.
'''
假如我们有一个句子“The dog barked at the mailman”,选取“dog”作为input word,
skip_window参数代表着我们从当前input word的一侧（左边或右边）选取词的数量,如
果我们设置skip\_window=2，那么我们最终获得窗口中的词（包括input word在内）就
是['The', 'dog'，'barked', 'at']
num_skips参数 它代表着我们从整个窗口中选取多少个不同的词作为我们的output word，
当skip_window=2，num_skips=2时，我们将会得到两组 (input word, output word) 形
式的训练数据，即 ('dog', 'barked')，('dog', 'the')
'''
def generate_batch( batch_size, num_skips, skip_window ):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray( shape=( batch_size ), dtype=np.int32 )
    labels = np.ndarray( shape=( batch_size, 1 ), dtype=np.int32 )
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque( maxlen=span )
    for _ in range( span ):
        buffer.append( data[data_index] )
        data_index = ( data_index + 1 ) % len(data)
    for i in range( batch_size // num_skips ):
        target = skip_window # target label at the center of the buffer
        target_to_avoid = [ skip_window ]
        for j in range( num_skips ):
            while target in target_to_avoid:
                target = random.randint( 0, span-1 )
            target_to_avoid.append( target )
            batch[ i * num_skips + j ] = buffer[ skip_window ]
            labels[ i * num_skips + j, 0 ] = buffer[ target ]
        buffer.append( data[data_index] )
        data_index = ( data_index + 1 ) % len( data )

    return batch, labels
batch, labels = generate_batch( batch_size=8, num_skips=2, skip_window=1 )
for i in range(8):
    print( batch[i], reverse_dictionary[ batch[i]], '->', labels[i,0], reverse_dictionary[labels[i,0]])

# Step 4: Build and train a skip-gram model.
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
valid_size = 9 #切记这个数字要和len(valid_word)对应，要不然会报错哦
valid_window = 100
num_sampled = 64  # Number of negative examples to sample.

#验证集
valid_word = ['萧炎','灵魂','火焰','脸颊','药老','天阶',"云岚宗","乌坦城","惊诧"]
valid_examples = [ dictionary[word] for word in valid_word ]
graph = tf.Graph()
with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder( tf.int32, shape=[batch_size] )
    train_lables = tf.placeholder( tf.int32, shape=[batch_size, 1] )
    valid_dataset = tf.constant( valid_examples, dtype=tf.int32 )

    # Ops and variables pinned to the CPU because of missing GPU implementation
    embeddings = tf.Variable(
        tf.random_uniform( [ vocabulary_size, embedding_size ], -1.0, 1.0 ) )
    embed = tf.nn.embedding_lookup( embeddings, train_inputs )

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable( tf.truncated_normal( [vocabulary_size, embedding_size],
                                                    stddev=1.0 / math.sqrt( embedding_size )))
    nce_biases = tf.Variable( tf.zeros( [vocabulary_size,]), dtype=tf.float32 )

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss( weights=nce_weights, biases=nce_biases, inputs=embed, labels=train_lables,
                        num_sampled=num_sampled, num_classes=vocabulary_size )
    )
    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer( 1.0 ).minimize( loss )

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt( tf.reduce_sum( tf.square( embeddings ), 1, keepdims=True ) )
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup( normalized_embeddings, valid_dataset )
    similarity = tf.matmul( valid_embeddings, normalized_embeddings, transpose_b=True )
    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 200
with tf.Session( graph=graph ) as session:
    # We must initialize all variables before we use them.
    init.run()
    print( 'Initialized' )

    average_loss = 0
    for step in range( num_steps ):
        batch_inputs, batch_labels = generate_batch( batch_size=batch_size, num_skips=num_skips, skip_window=skip_window )
        feed_dict =  {  train_inputs: batch_inputs, train_lables: batch_labels }
        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run( [ optimizer, loss ], feed_dict=feed_dict )
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print( 'Average loss at step {} : {}'.format( step, average_loss ) )

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range( valid_size ):
                valid_word = reverse_dictionary[ valid_examples[i] ]
                top_k = 8 # number of nearest neighbors
                nearest = ( -sim[ i,: ] ).argsort()[ :top_k ]
                log_str =  'Nearest to %s:' % valid_word
                for k in range( top_k ):
                    closed_word = reverse_dictionary[ nearest[k] ]
                    log_str = '%s %s' % (  log_str, closed_word )
                print( log_str )

    final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.
print( 'Visualize the embeddings' )
def plot_with_labels( low_dim_embs, labels, filenam='tsne.png', font=None ):
    assert low_dim_embs.shape[0] >= len( labels ), 'More labels then embeddings'
    plt.figure( figsize=(18, 18)) # in inches
    for i, label in enumerate( labels ):
        x, y = low_dim_embs[ i, : ]
        plt.scatter( x, y )
        plt.annotate( label,
                      xy=(x, y),
                      xytext=(5, 3),
                      textcoords='offset points',
                      ha='right',
                      va='bottom' )
        plt.savefig( filenam, dpi=600 )

try:
    from sklearn.manifold import TSNE
    from matplotlib.font_manager import FontProperties

    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact'
    )
    plot_only = 500
    low_dim_embs = tsne.fit_transform( final_embeddings[:plot_only, :] )
    labels = [ reverse_dictionary[i] for i in range( plot_only ) ]
    plot_with_labels( low_dim_embs, labels, 'tsne.png' )
except ImportError as ex:
    print( 'Please install skearn matplotlib and scipy to show embeddings' )
    print( ex )