'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年06月06日 14:42
@Description: 
@URL: https://www.bilibili.com/video/av53772494/?p=8
@version: V1.0
'''
import  tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets( './data/mnist', one_hot=False )

real_img = tf.placeholder( tf.float32, [ None, 784 ], name='real_image' )
noise_img = tf.placeholder( tf.float32, [ None, 100 ], name='noise_image' )

def generator( noise_img, hidden_units, out_dim, reuse=False, alpha=0.01 ):
    with tf.variable_scope( 'generator', reuse=reuse ):
        hidden1 = tf.layers.dense( noise_img, hidden_units )
        hidden1 = tf.nn.relu( hidden1 )
        hidden1 = tf.layers.dropout( hidden1, rate=0.2 )

        #logits & outputs
        logits = tf.layers.dense( hidden1, out_dim )
        outputs = tf.tanh( logits )

        return logits, outputs

def discriminator( img, hidden_units, reuser=False, alpha=0.01 ):
    with tf.variable_scope( 'discriminator', reuse=reuser ):
        hidden1 = tf.layers.dense( img, hidden_units )
        hidden1 = tf.maximum( alpha * hidden1, hidden1 )

        #logits & outputs
        logits = tf.layers.dense( hidden1, 1 )
        outputs = tf.sigmoid( logits )

        return logits, outputs

def plot_images( samples ):
    samples = ( samples + 1 ) / 2
    fig, axes = plt.subplots( nrows=1, ncols=25, sharex=True, sharey=True, figsize=(50, 2) )
    for img, ax in zip( samples, axes ):
        ax.imshow( img.reshape( 28, 28 ), cmap='Greys_r' )
        ax.get_xaxis().set_visible( False )
        ax.get_yaxis().set_visible( False )
    fig.tight_layout( pad=0 )


img_size = 784
noise_size = 100
hidden_units = 128
alpha = 0.01
learning_rate = 0.001
smooth = 0.1

g_logits, g_outputs = generator( noise_img, hidden_units=hidden_units, out_dim=img_size )
#discriminator
d_logits_real, d_outputs_real = discriminator( real_img, hidden_units )
d_logits_fake, d_outputs_fake = discriminator( g_outputs, hidden_units, reuser=True )

#discriminator loss
d_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=d_logits_real,
                                                                       labels=tf.ones_like( d_logits_real )) * (1 - smooth) )
d_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=d_logits_fake,
                                                                       labels=tf.zeros_like( d_logits_fake )) )
d_loss = tf.add( d_loss_real, d_loss_fake )

#generator loss
g_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=d_logits_fake,
                                                                  labels=tf.ones_like( d_logits_fake)) * ( 1 - smooth ))

train_vars = tf.trainable_variables()
g_vars = [ var for var in train_vars if var.name.startswith( 'generator' ) ]
d_vars = [ var for var in train_vars if var.name.startswith( 'discriminator' ) ]

d_train_op = tf.train.AdamOptimizer( learning_rate ).minimize( d_loss, var_list=d_vars )
g_trian_op = tf.train.AdamOptimizer( learning_rate ).minimize( g_loss, var_list=g_vars )

batch_size = 64
epochs = 300
n_sample = 25
samples = []
losses = []
saver = tf.train.Saver( var_list=g_vars )

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    for e in range( epochs ):
        for batch_i in range( int( mnist.train.num_examples / batch_size ) ):
            batch = mnist.train.next_batch( batch_size )

            batch_images = batch[0].reshape( batch_size, 784 )
            batch_images = batch_images * 2 - 1
            #generator输入噪声
            batch_noise = np.random.uniform( -1, 1, size=( batch_size, noise_size ) )

            #Run optimizer
            _ = sess.run( d_train_op, feed_dict={ real_img:batch_images, noise_img:batch_noise } )
            _ = sess.run( g_trian_op, feed_dict={ noise_img:batch_noise } )

        if e % 30 == 0:
            sample_noise = np.random.uniform( -1, 1, size=( n_sample, noise_size ) )
            _, samples = sess.run( generator( noise_img, hidden_units, img_size, reuse=True),
                                   feed_dict={ noise_img: sample_noise } )
            plot_images( samples )

        #每轮结束，结束loss
        train_loss_d = sess.run( d_loss,
                                 feed_dict={ real_img: batch_images,
                                             noise_img:batch_noise} )
        #real image loss
        train_loss_d_real = sess.run( d_loss_real,
                                      feed_dict={ real_img: batch_images,
                                                  noise_img:batch_noise} )
        train_loss_d_fake = sess.run( d_loss_fake,
                                      feed_dict={ real_img:batch_images,
                                                  noise_img:batch_noise} )
        #generator loss
        train_loss_g = sess.run( g_loss,
                                 feed_dict={ noise_img:batch_noise } )

        print( 'Epoch {}/{}'.format( e+1, epochs ),
               'Discriminator Loss:{:.4f}(Real: {:4f} + Fake:{:4f} )'.format( train_loss_d, train_loss_d_real, train_loss_d_fake),
               'Generator Loss:{:4f}'.format( train_loss_g ) )
        losses.append( ( train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g ) )
        saver.save( sess, './checkpoint/generator.ckpt' )


