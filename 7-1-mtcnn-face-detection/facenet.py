'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年05月09日 11:19
@Description: 
@URL: https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_6/src/facenet.py
@version: V1.0
'''
from __future__ import absolute_import
from __future__ import division

import os
from subprocess import Popen, PIPE
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from scipy import  misc
from sklearn.model_selection import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile

def triplet_loss( anchor, positive, negative, alpha ):
    '''
    DESC: Calculate the triplet loss according to the FaceNet paper
    :param anchor: the embeddings for the anchor images.
    :param positive: the embeddings for the positive images.
    :param negative: the embeddings for the negative images.
    :param alpha:
    :return: the triplet loss according to the FaceNet paper as a float tensor.
    '''
    with tf.variable_scope( 'triplet_loss' ):
        pos_dist = tf.reduce_sum( tf.square( tf.subtract( anchor, positive )) ,1 )
        neg_dist = tf.reduce_sum( tf.square( tf.subtract( anchor, negative )),1  )

        basic_loss = tf.add( tf.subtract( pos_dist, neg_dist ), alpha )
        loss =  tf.reduce_mean( tf.maximum( basic_loss, 0.0 ), 0 )

    return loss

def devcov_loss( xs ):
    '''
    DESC: Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf
          Reducing Overfitting In Deep Networks by Decorrelating Representation
    :param xs:
    :return:
    '''
    x = tf.reshape( xs, [ int( xs.get_shape()[0] ), -1 ] )
    m = tf.reduce_mean( x, 0, True )
    z = tf.expand_dims( x-m, 2 )
    corr = tf.reduce_mean( tf.matmul( z, tf.transpose( z, perm=[0,2,1] ) ), 0 )
    corr_frob_sqr = tf.reduce_sum( tf.square( corr ) )
    corr_diag_sqr = tf.reduce_sum( tf.square( tf.diag_part( corr ) ) )
    loss = 0.5 * ( corr_frob_sqr - corr_diag_sqr )
    return loss

def center_loss( features, label, alfa, nrof_classes ):
    '''
    DESC: Center loss based on the pape http://ydwen.github.io/papers/WenECCV16.pdf
    :param features:
    :param label:
    :param alfa:
    :param nrof_classes:
    :return:
    '''
    #nrof_features 就是feature size,即神经网络计算的人脸的维度
    nrof_features = features.get_shape()[1]

    #centers为变量，它是各个类别对应的类别中心
    centers = tf.get_variable( 'centers', [ nrof_classes, nrof_features ], dtype=tf.float32,
                               initializer=tf.constant_initializer(0), trainable=False )
    label = tf.reshape( label, [-1] )

    #根据label，取出features中每一个样本对应的类别中心
    #centers_batch的形状应该和features一致，为[ batch_size, feature_size]
    centers_batch = tf.gather( centers, label )
    #计算类别中心和各个样本特征的差距diff
    #计算diff时用到的alfa是应该超参数，它可以控制中心位置的更新幅度
    diff = ( 1 - alfa ) * ( centers_batch - features )
    #用diff来更新中心
    centers = tf.scatter_sub( centers, label, diff )
    #计算loss
    loss = tf.reduce_mean( tf.square( features - centers_batch ))
    #返回loss和更新后的中心（center）
    return  loss, centers

def _add_loss_summaries( total_loss ):
    '''
    DESC: Add summaries for losses, Generates moving average for all losses and associated summaries for
          visualizing the performance of the network.
    :param total_loss:  Total loss from loss().
    :return: op for generating moving averages of losses.
    '''
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage( 0.9, name='avg' )
    losses = tf.get_collection( 'losses' )
    loss_averages_op = loss_averages.apply( losses + [total_loss] )

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [ total_loss ]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar( l.op.name + '(rae_', l )
        tf.summary.scalar( l.op.name, loss_averages.average(l) )

    return loss_averages_op

def train( total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True ):
    '''

    :param total_loss:
    :param global_step:
    :param optimizer:
    :param learning_rate:
    :param moving_average_decay:
    :param update_gradient_vars:
    :param log_histograms:
    :return:
    '''
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries( total_loss )

    # Compute gradients.
    with tf.control_dependencies( [loss_averages_op] ):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer( learning_rate )
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer( learning_rate, rho=0.9, epsilon=1e-6 )
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer( learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1 )
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer( learning_rate, decay=0.9, momentum=0.9, epsilon=1.0 )
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer( learning_rate, 0.9, use_nesterov=True )
        else:
            raise  ValueError( 'Invalid optimization algorithm' )

        grads = opt.compute_gradients( total_loss, update_gradient_vars )

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients( grads, global_step=global_step )

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram( var.op.name, var )

    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram( var.op.name + '/gradients', grad )

    # Track the moving averages of all trainable variables.
    variabel_averages = tf.train.ExponentialMovingAverage( moving_average_decay, global_step )
    variabel_averages_op = variabel_averages.apply( tf.trainable_variables() )

    with tf.control_dependencies( [ apply_gradient_op, variabel_averages_op ]):
        train_op = tf.no_op( name='train' )

    return train_op




def get_learning_rate_from_file( filename, epoch ):
    #TODO
    return 0.0

class ImageClass():
    '''
    Stores the paths to images for a given class
    '''
    def __init__(self, name, image_paths ):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ',' + str( len( self.image_paths ) ) + ' images'

    def __len__(self):
        return len( self.image_paths )

def get_dataset( paths ):
    '''

    :param paths:
    :return:
    '''
    dataset = []
    for path in paths.split( ';' ):
        path_exp = os.path.expanduser( path )
        classes = os.listdir( path_exp )
        classes.sort()
        nrof_classes = len( classes )
        for i in range( nrof_classes ):
            class_name = classes[i]
            facedir = os.path.join( path_exp, class_name )
            if os.path.isdir( facedir ):
                images = os.listdir( facedir )
                image_paths = [ os.path.join( facedir, img ) for img in images ]
                dataset.append( ImageClass( class_name, image_paths) )

    return dataset

def get_image_paths_and_labels( dataset ):
    image_paths_flat = []
    labels_flat = []
    for i in range( len( dataset ) ):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len( dataset[i].image_paths )
    return image_paths_flat, labels_flat

def calculate_roc( thresholds, embedding1, embeddings2, actual_issame, nrof_folds=10  ):
    assert ( embedding1.shape[0] == embeddings2.shape[0] )
    assert ( embedding1.shape[1] == embeddings2.shape[1] )
    nrof_pairs = min( len( actual_issame ), embedding1.shape[0] )
    nrof_thresholds = len( thresholds )
    k_fold = KFold( n_splits=nrof_folds, shuffle=False )

    tprs = np.zeros(( nrof_folds, nrof_thresholds ) )
    fprs = np.zeros(( nrof_folds, nrof_thresholds ) )
    accuracy = np.zeros(( nrof_folds ) )
    diff = np.subtract(( embedding1, embeddings2 ) )
    dist = np.sum( np.square( diff ), 1 )
    indices = np.aran( nrof_pairs )

    for fold_idx, ( train_set, test_set ) in enumerate( k_fold.split( indices ) ):

        #Find the threshold that gives FAR = far_target
        acc_train = np.zeros( ( nrof_thresholds ) )
        for threshold_idx, threshold in enumerate( thresholds ):
            _, _, acc_train[ threshold_idx ] = calculate_accuracy( threshold, dist[ train_set ], actual_issame[ train_set ] )
        best_threshold_index = np.argmax( acc_train )
        for threshold_idx, threshold in enumerate( thresholds ):
            tprs[ fold_idx, threshold_idx ], fprs[ fold_idx, threshold_idx ], _ = calculate_accuracy( threshold, dist[test_set], actual_issame[test_set] )
        _, _, accuracy[ fold_idx ] = calculate_accuracy( thresholds[best_threshold_index], dist[test_set], actual_issame[test_set] )

        tpr = np.mean( tprs, 0 )
        fpr = np.mean( fprs, 0 )

    return tpr, fpr, accuracy

def calculate_accuracy( threshold, dist, actual_issame ):
    predict_issame = np.less( dist, threshold )
    tp = np.sum( np.logical_and( predict_issame, actual_issame ) )
    fp = np.sum( np.logical_and( predict_issame, np.logical_not( actual_issame ) ) )
    tn = np.sum( np.logical_and( np.logical_not( predict_issame), np.logical_not( actual_issame ) ) )
    fn = np.sum( np.logical_and( np.logical_not( predict_issame ), actual_issame ))

    tpr = 0 if ( tp+fn == 0 ) else float( tp ) / float( tp+fn )
    fpr = 0 if ( fp+tn == 0 ) else float( fp ) / float( fp+tn )
    acc = float( tp+tn ) / dist.size
    return tpr, fpr, acc

def calculate_val( thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10 ):
    assert( embeddings1.shape[0] == embeddings2.shape[0] )
    assert( embeddings1.shape[1] == embeddings2.shape[1] )
    nrof_pairs = min( len( actual_issame ), embeddings1.shape[0] )
    nrof_thresholds = len( thresholds )
    k_fold = KFold( n_splits=nfro_folds, shuffle=False )

    val = np.zeros( nrof_folds )
    far = np.zeros( nrof_folds )

    diff = np.subtract( embeddings1, embeddings2 )
    dist = np.sum( np.square( diff ), 1 )
    indices = np.arange( nrof_pairs )

    for fold_idx, ( train_set, test_set ) in enumerate( k_fold.split( indices ) ):
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros( nrof_thresholds )
        for threshold_idx, threshold in enumerate( thresholds ):
            _, far_train[ threshold_idx ] = calculate_val_far( threshold, dist[train_set], actual_issame[train_set] )
        if np.max( far_train ) >= far_target:
            f = interpolate.interp1d( far_train, thresholds, kind='slinear' )
            threshold = f( far_target )
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far( threshold, dist[test_set], actual_issame[test_set] )

    val_mean = np.mean( val )
    far_mean = np.mean( far )
    val_std = np.std( val )
    return  val_mean, val_std, far_mean

def calculate_val_far( threshold, dist, actual_issame ):
    predict_issame = np.less( dist, threshold )
    true_accept = np.sum( np.logical_and( predict_issame, actual_issame ) )
    false_accept = np.sum( np.logical_and( predict_issame, np.logical_not( actual_issame) ) )
    n_same = np.sum( actual_issame )
    n_diff = np.sum( np.logical_not( actual_issame ) )
    val = float( true_accept ) / float( n_same )
    far = float( false_accept ) / float( n_diff )
    return val, far

def store_revision_info(src_path, output_dir, arg_string):

    # Get git hash
    gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_hash = stdout.strip()

    # Get local changes
    gitproc = Popen(['git', 'diff', 'HEAD'], stdout = PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_diff = stdout.strip()

    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)

def list_variables(filename):
    reader = training.NewCheckpointReader(filename)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    return names