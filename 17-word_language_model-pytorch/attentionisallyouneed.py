# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Attention is All You Need
# 源自：
# > http://nlp.seas.harvard.edu/2018/04/03/attention.html#encoder
# 
# > https://zhuanlan.zhihu.com/p/48731949
#
# > https://zhuanlan.zhihu.com/p/137578323

# %%
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, copy
from torch.autograd import  Variable
import matplotlib.pyplot as plt
import seaborn
import spacy
import os
import re
from tqdm import tqdm

import thulac

# For data loading.
from torchtext import data, datasets
seaborn.set_context( context='talk' )
# get_ipython().magic('matplotlib inline')

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

# modelFilePath = '/content/gdrive/My Drive/Colab Notebooks/17-word_language_model-pytorch/transformer.pt'
# train_path = '/content/cmn.txt'
modelFilePath = 'transformer.pt'
# train_path = 'E:/ML_data/translate/cmn_sample.txt'
train_path = 'E:/ML_data/translate/cmn.txt'

# %% [markdown]
# The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed- forward network.<br>
# ![Transformer] (http://nlp.seas.harvard.edu/images/the-annotated-transformer_14_0.png "Transformer")

# %%
class EncoderDecoder( nn.Module ):
    '''
    A standard Encoder-Decoder architecture. Base for this          and many other models.
    '''
    def __init__( self, encoder, decoder, src_embed, tgt_embed, generator ):
        super( EncoderDecoder, self ).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward( self, src, tgt, src_mask, tgt_mask ):
        '''Take in and process masked src and target sequences.'''
        return self.decode( self.encode( src, src_mask ), src_mask, tgt, tgt_mask )

    def encode( self, src, src_mask ):
        return self.encoder( self.src_embed( src ), src_mask )
    
    def decode( self, memory, src_mask, tgt, tgt_mask ):
        return self.decoder( self.tgt_embed( tgt ), memory, src_mask, tgt_mask )


# %%
class Generator( nn.Module ):
    '''Define standard linear + softmax generation step.'''
    def __init__( self, d_model, vocab ):
        super( Generator, self ).__init__()
        self.proj = nn.Linear( d_model, vocab )

    def forward( self, x ):
        return F.log_softmax( self.proj( x ), dim=-1 )

# %% [markdown]
# # Encoder
# The encoder is composed of a stack of $N=6$ identical layers.

# %%
def clones( module, N ):
    '''Produce N identical layers.'''
    return nn.ModuleList( [ copy.deepcopy( module ) for _ in range( N ) ] )


# %%
class Encoder( nn.Module ):
    "Core encoder is a stack of N layers"
    def __init__( self, layer, N ):
        super( Encoder, self ).__init__()
        self.layers = clones( layer, N )
        self.norm = LayerNorm( layer.size )

    def forward( self, x, mask ):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer( x, mask )
        return self.norm( x )

# %% [markdown]
# We employ a residual connection (cite) around each of the two sub-layers, followed by layer normalization (cite).
# 
# EncoderLayer: 每层都有两个子层组成。第一个子层实现了“多头”的 Self-attention，第二个子层则是一个简单的Position-wise的全连接前馈网络。

# %%
class EncoderLayer( nn.Module ):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__( self, size, self_attn, feed_forward, dropout ):
        super( EncoderLayer, self ).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones( SublayerConnection( size, dropout=dropout ), 2 )
        self.size = size

    def forward( self, x, mask ):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0]( x, lambda x: self.self_attn( x, x, x, mask ) )
        return self.sublayer[1]( x, self.feed_forward )


# %%
class LayerNorm( nn.Module ):
    "Construct a layernorm module (See citation for details)."
    def __init__( self, features, eps=1e-6 ):
        super( LayerNorm, self ).__init__()
        self.a_2 = nn.Parameter( torch.ones( features ) )
        self.b_2 = nn.Parameter( torch.zeros( features ) )
        self.eps = eps

    def forward( self, x ):
        mean = x.mean( -1, keepdim=True )
        std = x.std( -1, keepdim=True )
        return self.a_2 * ( x - mean ) / ( std + self.eps ) + self.b_2


# %%
class SublayerConnection( nn.Module ):
    '''
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    '''
    def __init__( self, size, dropout ):
        super( SublayerConnection, self ).__init__()
        self.norm = LayerNorm( size )
        self.dropout = nn.Dropout( dropout )

    def forward( self, x, sublayer ):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout( sublayer( self.norm( x ) ) )

# %% [markdown]
# # Decoder
# 
# Decoder也是由N=6个相同层组成。

# %%
class Decoder( nn.Module ):
    "Generic N layer decoder with masking."
    def __init__( self, layer, N ):
        super( Decoder, self ).__init__()
        self.layers = clones( layer, N )
        self.norm = LayerNorm( layer.size )

    def forward( self, x, memory, src_mask, tgt_mask ):
        for layer in self.layers:
            x = layer( x, memory, src_mask, tgt_mask )
        return self.norm( x ) 

# %% [markdown]
# (DecoderLayer)

# %%
class DecoderLayer( nn.Module ):
    "Decoder is made of self_attn, src_attn, and feed forward (defined below)"
    def __init__( self, size, self_attn, src_attn, feed_forward, dropout ):
        super( DecoderLayer, self ).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones( SublayerConnection( size, dropout ), 3 )

    def forward( self, x, memory, src_mask, tgt_mask ):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0]( x, lambda x: self.self_attn( x, x, x, tgt_mask ) )
        x = self.sublayer[1]( x, lambda x: self.src_attn( x, m, m, src_mask ) )
        return self.sublayer[2]( x, self.feed_forward )

# %% [markdown]
# 我们还修改了解码器中的Self-attetion子层以防止当前位置attend到后续位置。这种Masked的Attention是考虑到输出Embedding会偏移一个位置，确保了生成位置i的预测时，仅依赖于i的位置处已知输出，相当于把后面不该看的信心屏蔽掉

# %%
def subsequent_mask( size ):
    "Mask out subsequent positions."
    attn_shape = ( 1, size, size )
    subsequent_mask = np.triu( np.ones( attn_shape ), k=1 ).astype( 'uint8' )
    return torch.from_numpy( subsequent_mask ) == 0

plt.figure( figsize=( 5, 5 ) )
plt.imshow( subsequent_mask( 19 )[0] )

# %% [markdown]
# ## Attention
# An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
# We call our particular attention “Scaled Dot-Product Attention”. The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$. We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.
# <img src="https://pic2.zhimg.com/80/v2-c5dcddf20d8b2d7ce0130fac2071317d_720w.jpg" />
# 

# %%
def attention( query, key, value, mask=None, dropout=None ):
    "Compute 'Scaled Dot Product Attention"
    d_k = query.size( -1 )
    scores = torch.matmul( query, key.transpose( -2, -1 )) / math.sqrt( d_k )
    if mask is not None:
        scores = scores.masked_fill( mask == 0.0, -1e9 )
    p_attn = F.softmax( scores, dim=-1 )
    if dropout is not None:
        p_attn = dropout( p_attn )
    return torch.matmul( p_attn, value ), p_attn

# %% [markdown]
# MultiHead:<br>
# 
# <img  src="http://nlp.seas.harvard.edu/images/the-annotated-transformer_38_0.png" width=25% height=25% />
# 
# Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.
# 
# $\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ...,
# \mathrm{head_h})W^O    \\
#     \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
# $
# 
# Where the projections are parameter matrices $W^Q_i \in
# \mathbb{R}^{d_{\text{model}} \times d_k}$, WKi∈Rdmodel×dk, WVi∈Rdmodel×dv and WO∈Rhdv×dmodel. In this work we employ h=8 parallel attention layers, or heads. For each of these we use dk=dv=dmodel/h=64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.
# 
# “多头”机制能让模型考虑到不同位置的Attention，另外“多头”Attention可以在不同的子空间表示不一样的关联关系，使用单个Head的Attention一般达不到这种效果。

# %%
class MultiHeaderAttention( nn.Module ):
    def __init__( self, h, d_model, dropout=0.1 ):
        "Take in model size and number of heads."
        super( MultiHeaderAttention, self ).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones( nn.Linear( d_model, d_model ), 4 )
        self.attn = None
        self.dropout = nn.Dropout( p=dropout )

    def forward( self, query, key, value, mask=None ):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze( 1 )
        nbatches = query.size( 0 )
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [ l(x).view( nbatches, -1, self.h, self.d_k ).transpose( 1, 2 )
              for l, x in zip( self.linears, ( query, key, value )) ]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention( query, key, value, mask=mask, dropout=self.dropout )
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose( 1, 2 ).contiguous().view( nbatches, -1, self.h * self.d_k )
        return self.linears[-1]( x )

# %% [markdown]
# > attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

# %%
class PositionwiseFeedForward( nn.Module ):
    "Implements FFN equation."
    def __init__( self, d_model, d_ff, dropout=0.1 ):
        super( PositionwiseFeedForward, self ).__init__()
        self.w_1 = nn.Linear( d_model, d_ff )
        self.w_2 = nn.Linear( d_ff, d_model )
        self.dropout = nn.Dropout( dropout )

    def forward( self, x ):
        return self.w_2( self.dropout( F.relu( self.w_1( x ) )) )

# %% [markdown]
# # Emedding 和 softmax
# 
# <img src="https://pic2.zhimg.com/80/v2-ca9c861576e2aac1ee7211d4e0bc6281_720w.jpg" />

# %%
class Embeddings( nn.Module ):
    def __init__( self, d_model, vocab ):
        super( Embeddings, self ).__init__()
        self.lut = nn.Embedding( vocab, d_model )
        self.d_model = d_model

    def forward( self, x ):
        return self.lut( x ) * math.sqrt( self.d_model )

# %% [markdown]
# # 位置编码

# %%
class PositionalEncoding( nn.Module ):
    "Implement the PE function."
    def __init__( self, d_model, dropout, max_len=5000 ):
        super( PositionalEncoding, self ).__init__()
        self.dropout = nn.Dropout( p=dropout )

        # Compute the positional encodings once in log space.
        pe = torch.zeros( max_len, d_model )
        position = torch.arange( 0, max_len ).unsqueeze( 1 ).float()
        div_term = torch.exp( torch.arange( 0., d_model, 2) * -(math.log(10000.0) / d_model ) )
        pe[ :, 0::2 ] = torch.sin( position * div_term )
        pe[ :, 1::2 ] = torch.cos( position * div_term )
        pe = pe.unsqueeze( 0 )
        self.register_buffer( 'pe', pe )

    def forward( self, x ):
        x = x + Variable( self.pe[ :, :x.size(1) ], requires_grad=False )
        return self.dropout( x )


# %% [markdown]
# 
# Here we define a function that takes in hyperparameters and produces a full model.

# %%
def make_model( src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1 ):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeaderAttention( h, d_model )
    ff = PositionwiseFeedForward( d_model, d_ff, dropout )
    position = PositionalEncoding( d_model, dropout )

    model = EncoderDecoder( 
        Encoder( EncoderLayer( d_model, c(attn), c(ff), dropout ), N ),
        Decoder( DecoderLayer( d_model, c(attn), c(attn), c(ff), dropout ), N ),
        nn.Sequential( Embeddings( d_model, src_vocab ), c(position) ),
        nn.Sequential( Embeddings( d_model, tgt_vocab ), c(position) ),
        Generator( d_model, tgt_vocab )
    )

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_( p )
    return model


# %%
tmp_model = make_model(10, 10, 2)
# print( tmp_model )

# %% [markdown]
# 
# ## Batches and Masking

# %%
class Batch( ):
    "Object for holding a batch of data with mask during training."
    def __init__( self, src, trg=None, pad=0 ):
        self.src = src.to( device )
        self.src_mask = ( self.src != pad ).unsqueeze( -2 )
        if trg is not None:
            trg = trg.to( device )
            self.trg = trg[ :, :-1 ]
            self.trg_y = trg[ :, 1: ]
            self.trg_mask = self.make_std_mask( self.trg, pad  )
            self.ntokens = ( self.trg_y != pad ).data.sum()

    @staticmethod       
    def make_std_mask( tgt, pad ):
        "Create a mask to hide padding and future words."
        tgt_mask = ( tgt != pad ).unsqueeze( -2 )
        tgt_mask = tgt_mask & Variable( 
            subsequent_mask( tgt.size( -1 ) ).type_as( tgt_mask.data )
        )
        return tgt_mask

# %% [markdown]
# 
# ## Training Loop
# 

# %%
def run_epoch( data_iter, model, loss_compute ):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    try:
        for i, batch in enumerate( data_iter ):
            out = model.forward( batch.src, batch.trg, batch.src_mask, batch.trg_mask )
            loss = loss_compute( out, batch.trg_y, batch.ntokens )
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % 100 == 1:
                elapsed = time.time() - start
                print( 'Epoch Step: %4d Loss:%5.4f Tokens per sec  %4.2f' % ( i, loss / batch.ntokens.item(), tokens / elapsed ))
                start = time.time()
                tokens = 0

                torch.save( model.state_dict(), modelFilePath )
    except ValueError:
        print( 'run_epoch except: len of batch.src:', len( batch.src ) )
    
    return ( total_loss / total_tokens.float() ).item()

# %% [markdown]
# and Batching

# %%
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn( new, count, sofar ):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max( max_src_in_batch, len( new.src ) )
    max_tgt_in_batch = max( max_tgt_in_batch, len( new.trg ) + 2 )
    src_elemnets = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch

    return max( src_elemnets, tgt_elements )

# %% [markdown]
# 
# 
# We used the Adam optimizer (cite) with β1=0.9, β2=0.98 and ϵ=10−9.
# 
# >> Note: This part is very important. Need to train with this setup of the model.

# %%
class NoamOpt():
    "Optim wrapper that implements rate."
    def __init__( self, model_size, factor, warmup, optimizer ):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step( self ):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p[ 'lr' ] = rate
        self._rate = rate
        self.optimizer.step()

    def rate( self, step=None ):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * ( self.model_size ** ( -0.5 ) * 
                   min( step ** ( -0.5 ), step * self.warmup ** ( -1.5)))


def get_std_opt( model ):
    return NoamOpt( model.src_embed[0].d_model, 2, 4000,
                  torch.optim.Adam( model.parameters(), lr=0, betas=( 0.9, 0.98 ), eps=1e-9 ))


# %%
# Three settings of the lrate hyperparameters.
opts = [ NoamOpt( 512, 1, 4000, None ),
         NoamOpt( 1024, 1, 8000, None ),
         NoamOpt( 256, 1, 4000, None )]
plt.plot( np.arange( 1, 20000 ), [ [ opt.rate( i ) for opt in opts ] for i in range( 1, 20000 ) ])
plt.legend( [ '512:4000', '1024:8000', '256:4000'])

# %% [markdown]
# 
# ## Label Smoothing
#  令人迷惑，但确实能改善accuracy and BLEU 成绩.

# %%
class LabelSmoothing( nn.Module ):
    "Implement label smoothing."
    def __init__( self, size, padding_idx, smoothing=0.0 ):
        super( LabelSmoothing, self ).__init__()
        self.criterion = nn.KLDivLoss( size_average=False )
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward( self, x, target ):
        assert x.size(1) == self.size
        true_dist = x.data.clone()  # 先深复制过来
        true_dist.fill_( self.smoothing / ( self.size - 2 ) )  # otherwise的公式
        # 变成one-hot编码，1表示按列填充，target.data.unsqueeze(1)表示索引, confidence表示填充的数字
        true_dist.scatter_( 1, target.data.unsqueeze(1), self.confidence )  
        true_dist[ :, self.padding_idx ] = 0
        mask = torch.nonzero( target.data == self.padding_idx )
        if mask.dim() > 0:
            true_dist.index_fill( 0, mask.squeeze(), 0.0 )
        self.true_dist = true_dist
        return self.criterion( x, Variable( true_dist, requires_grad=False ) )


# %% [markdown]
# # A First Example

# %%
# Synthetic Data
def data_gen( V, batch, nbatches ):
    "Generate random data for a src-tgt copy task."
    for i in range( nbatches ):
        data = torch.from_numpy( np.random.randint( 1, V, size=( batch, 10 ) ) )
        data[ :, 0 ] = 1
        src = Variable( data, requires_grad=False ).long()
        tgt = Variable( data, requires_grad=False ).long()
        yield Batch( src, tgt, 0 )

# Loss Computation
class SimpleLossCompute():
    "A simple loss compute and train function."
    def __init__( self, generator, criterion, opt=None ):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__( self, x, y, norm ):
        x = self.generator( x )
        loss = self.criterion( x.contiguous().view( -1, x.size(-1) ),
                               y.contiguous().view( -1 ) ) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return ( loss * norm ).item()

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

'''
# Greedy Decoding
V = 11
criterion = LabelSmoothing( size=V, padding_idx=0, smoothing=0.0 )
model = make_model( V, V, N=2 )
model_opt = NoamOpt( model.src_embed[0].d_model, 1, 400,
                     torch.optim.Adam( model.parameters(), lr=1, betas=( 0.9, 0.98 ), eps=1e-9 ) )

for epoch in range( 10 ):
    model.train()
    run_epoch( data_gen( V, 30, 20 ), model, 
                SimpleLossCompute( model.generator, criterion, model_opt ))
    model.eval()
    print( 'eval:', run_epoch(data_gen( V, 30, 5 ), model, 
                    SimpleLossCompute( model.generator, criterion, None ) ) )



model.eval()
src = Variable(torch.LongTensor([[1,8,1,4,5,6,7,4,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))


'''
class MyIterator( data.Iterator ):
    def create_batches( self ):
        if self.train:
            def pool( d, random_shuffler ):
                for p in data.batch( d, self.batch_size * 100 ):
                    p_batch = data.batch( sorted( p, key=self.sort_key ),
                                           self.batch_size, self.batch_size_fn )
                    for b in random_shuffler( list( p_batch ) ):
                        yield b
            self.batches = pool( self.data(), self.random_shuffler )
        else:
            self.batches = []
            for b in data.batch( self.data(), self.batch_size, self.batch_size_fn ):
                self.batches.append( sorted( b, key=self.sort_key ) )


def rebatch( pad_idx, batch ):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose( 0, 1 ), batch.trg.transpose( 0, 1 )
    return Batch( src, trg, pad_idx ) 

class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__( self, generator, criterion, devices, opt=None, chunk_size=5 ):
        self.generator = generator
        self.criterion = nn.parallel.replicate( criterion, devices=devices )
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__( self, out, targets, normalize ):
        total = 0.0
        generator = nn.parallel.replicate( self.generator, devices=self.devices )
        out_scatter = nn.parallel.scatter( out, target_gpus=self.devices )
        out_grad = [ [] for _ in out_scatter ]
        targets = nn.parallel.scatter( targets, target_gpus=self.devices )

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range( 0, out_scatter[0].size(1), chunk_size ):
            # Predict distributions
            out_column = [ [ Variable( o[ :, i:i+chunk_size ].data, requires_grad=self.opt is not None )]
                           for o in out_scatter ]
            gen = nn.parallel.parallel_apply( generator, out_column )

            # Compute loss
            y = [ ( g.contiguous().view( -1, g.size( -1 )), 
                   t[ :, i:i+chunk_size ].contiguous().view( -1 ))
                     for g, t in zip( gen, targets ) ]
            loss = nn.parallel.parallel_apply( self.criterion, y )

            # Sum and normalize loss
            l = nn.parallel.gather( loss, target_device=self.devices[0] )
            l = l.sum() / normalize
            total += l.item()

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate( loss ):
                    out_grad[ j ].append( out_column[ j ][ 0 ].grad.data.clone() ) 

        # Backprop loss to output of transformer
        if self.opt is not None:
            out_grad = [ Variable( torch.cat( og, dim=1 )) for og in out_grad ]
            o1 = out
            o2 = nn.parallel.gather( out_grad,
                                     target_device=self.devices[0] )
            o1.backward( gradient=o2 )
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return total * normalize.float()

def main_de_en():
    
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits( exts=( '.de', '.en' ), fields=( SRC, TGT ),
                                               filter_pred=lambda x: len( vars(x)[ 'src']) <= MAX_LEN and
                                                  len( vars(x)[ 'trg' ]) <= MAX_LEN )
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ) 

    pad_idx = TGT.vocab.stoi[ '<blank>' ]
    model = make_model( len(SRC.vocab ), len(TGT.vocab ), N=6 )
    model.to( device )
    criterion = LabelSmoothing( size=len( TGT.vocab ), padding_idx=pad_idx, smoothing=0.1 )
    criterion.to( device )

    BATCH_SIZE = 20
    train_iter = MyIterator( train, batch_size=BATCH_SIZE, device=device, repeat=False, 
                             sort_key=lambda x: ( len( x.src ), len( x.trg ) ), 
                             batch_size_fn=batch_size_fn, train=True )
    valid_iter = MyIterator( val, batch_size=BATCH_SIZE, device=device, repeat=False,
                              sort_key=lambda x: ( len( x.src ), len( x.trg ) ),
                              batch_size_fn=batch_size_fn, train=False )      

    # model_par = nn.DataParallel( model )

    model_opt = NoamOpt( model.src_embed[0].d_model, 1, 2000,
                         torch.optim.Adam( model.parameters(), lr=0, betas=( 0.9, 0.98 ), eps=1e-9 ) )

    if os.path.exists( modelFilePath ):
        print( 'Load model...' )
        model.load_state_dict( torch.load( modelFilePath ) )

    for epoch in range( 10 ):
        model.train()
        run_epoch( ( rebatch( pad_idx, b ) for b in train_iter ),
                    model,
                    SimpleLossCompute( model.generator, criterion,
                                       opt=model_opt ))
        model.eval()
        loss = run_epoch( ( rebatch( pad_idx, b ) for b in valid_iter ),
                           model, 
                           SimpleLossCompute( model.generator, criterion,
                                              opt=None ) )
        
        print( loss )

def main_cn_en():
    devices = [0]

    spacy_en = spacy.load('en')

    thu1 = thulac.thulac( seg_only=True )  #默认模式

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def tokenize_cn(text):
        return [tok for tok in thu1.cut(text, text=True )]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_cn, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    train_fields = [ ('src', SRC), ('trg', TGT ) ]       

    train_examples = []
    with open( train_path, 'r', encoding ='utf-8' )as f:
        for line in tqdm( f.readlines() ):
            en = re.split( r"\.|\!|\?", line.split( 'CC-BY 2.0' )[0].strip() )[0]
            cn = re.split( r"\.|\!|\?", line.split( 'CC-BY 2.0' )[0].strip() )[1].replace( '\t', '' )
            # print( [ en, cn ] )
            # print( thu1.cut( cn, text=True ))
            train_examples.append( data.Example.fromlist( [ en, cn ], train_fields ) )

    # 构建Dataset数据集
    train = data.Dataset( train_examples, train_fields )

    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ) 
    # print( 'SRC vocab stoi:', SRC.vocab.stoi )
    # print( 'TGT vocab stoi:', TGT.vocab.stoi )

    pad_idx = TGT.vocab.stoi[ '<blank>' ]
    model = make_model( len(SRC.vocab ), len(TGT.vocab ), N=6 )
    model.to( device )
    criterion = LabelSmoothing( size=len( TGT.vocab ), padding_idx=pad_idx, smoothing=0.1 )
    criterion.to( device )

    BATCH_SIZE = 25
    '''
    train_iter = MyIterator( train, batch_size=BATCH_SIZE, device=device, repeat=False, 
                             sort_key=lambda x: ( len( x.src ), len( x.trg ) ), 
                             batch_size_fn=batch_size_fn, train=True )
    '''
    train_iter = data.Iterator( train, batch_size=BATCH_SIZE, train=True )
  
    # model_par = nn.DataParallel( model )

    model_opt = NoamOpt( model.src_embed[0].d_model, 1, 2000,
                         torch.optim.Adam( model.parameters(), lr=0, betas=( 0.9, 0.98 ), eps=1e-9 ) )

    if os.path.exists( modelFilePath ):
        print( 'Load model...' )
        model.load_state_dict( torch.load( modelFilePath ) )    

    epoches = 300
    for epoch in range( epoches ):
        print( 'epoch train: {}/{}'.format( epoch, epoches ))
        model.train()
        loss = run_epoch( ( rebatch( pad_idx, b ) for b in train_iter ),
                    model,
                    MultiGPULossCompute( model.generator, criterion,
                                         devices=devices, opt=model_opt ))              
        print( loss )

def testEnToCn():
    spacy_en = spacy.load('en')

    thu1 = thulac.thulac( seg_only=True )  #默认模式

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def tokenize_cn(text):
        return [tok for tok in thu1.cut(text, text=True )]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_cn, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    train_fields = [ ('src', SRC), ('trg', TGT ) ]       

    train_examples = []
    with open( train_path, 'r', encoding ='utf-8' )as f:
        for line in tqdm( f.readlines() ):
            en = re.split( r"\.|\!|\?", line.split( 'CC-BY 2.0' )[0].strip() )[0]
            cn = re.split( r"\.|\!|\?", line.split( 'CC-BY 2.0' )[0].strip() )[1].replace( '\t', '' )
            # print( [ en, cn ] )
            # print( thu1.cut( cn, text=True ))
            train_examples.append( data.Example.fromlist( [ en, cn ], train_fields ) )

    # 构建Dataset数据集
    train = data.Dataset( train_examples, train_fields )

    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ) 

    pad_idx = TGT.vocab.stoi[ '<blank>' ]
    
    modelEvl = make_model( len(SRC.vocab ), len(TGT.vocab ), N=6 )
    modelEvl.load_state_dict( torch.load( modelFilePath ) )
    modelEvl.eval()

    testStr = "I'm very hot."
    testTokens = tokenize_en( testStr )
    testTokens = torch.Tensor( [[ SRC.vocab.stoi[ token ] for token in testTokens  ]] ).long()
    testTokens_mask = ( testTokens != SRC.vocab.stoi[ BLANK_WORD ]).unsqueeze( -2 )

    memory = modelEvl.encode( testTokens, testTokens_mask )

    ys = torch.ones(1, 1).fill_( TGT.vocab.stoi[ BOS_WORD ] ).type_as( testTokens.data)
    for i in range( 20 ):
        out = modelEvl.decode(memory, testTokens_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1)).type_as( testTokens.data )))
        prob = modelEvl.generator(out[:, -1])
        _, next_word = torch.max( prob, dim=1 )
        next_word = next_word.item()
        ys = torch.cat( [ ys,
                         torch.ones( 1, 1 ).type_as( testTokens.data ).fill_( next_word )], dim=1 )
    
    for i in range( 1, ys.size(1) ):
        sym = TGT.vocab.itos[ ys[ 0, i ] ]
        if sym == "</s>": break
        print( sym, end='' )
    print()

    return ys

if __name__ == '__main__':
    # main_cn_en()
    print( testEnToCn() )
    print( '---finished---' )

