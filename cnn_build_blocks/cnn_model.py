#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The CNN build blocks model. If you'd like to change your model, you need to edit 
this files

Author: André Pacheco
Email: pacheco.comp@gmail.com

If you find any bug, please email-me
"""

from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np


def print_summary_model (tensor, first=False, last=False, verbose=True):
    if (verbose):
        if (first):
            print("\n------- CNN summary model -------\n")
            
        print ("Layer: {}".format(tensor.name))
        print ("Shape: {}\n------------------".format(tensor.shape))
        
        if (last):
            print("\n------- End of CNN summary model -------\n")


def build_model (dict_input, params, is_train, verbose=True):
    
    images = dict_input['images']
    
    if ('scalar_feat' in dict_input):
        scalar_feat = dict_input['scalar_feat']
    else:
        scalar_feat = None
    
    with tf.variable_scope('block_1'):
        # Convolution and activation
        out = tf.layers.conv2d(
                inputs=images, 
                filters=64, 
                kernel_size=7, 
                strides=1, 
                padding='same', 
                activation=tf.nn.relu)
        
        print_summary_model (out, first=True, verbose=verbose)
        
        # pooling/subsampling
        out = tf.layers.max_pooling2d(inputs=out, pool_size=2, strides=2)
        print_summary_model (out, verbose=verbose)
    
    with tf.variable_scope('block_2'):
        # Convolution and activation
        out = tf.layers.conv2d(
                inputs=images, 
                filters=32, 
                kernel_size=4, 
                strides=1, 
                padding='same', 
                activation=tf.nn.relu)
        
        print_summary_model (out, verbose=verbose)
        
        # pooling/subsampling
        out = tf.layers.max_pooling2d(inputs=out, pool_size=2, strides=2)
        print_summary_model (out, verbose=verbose)
        
    with tf.variable_scope('block_3'):
        # Convolution and activation
        out = tf.layers.conv2d(
                inputs=images, 
                filters=16, 
                kernel_size=3, 
                strides=1, 
                padding='same', 
                activation=tf.nn.relu)
        
        print_summary_model (out, verbose=verbose)
        
        # pooling/subsampling
        out = tf.layers.max_pooling2d(inputs=out, pool_size=2, strides=2)
        print_summary_model (out, verbose=verbose)
    
    
    with tf.variable_scope('block_fc1'):
        shape_out = out.get_shape().as_list()
        out = tf.reshape(out, [-1, shape_out[1] * shape_out[2] * shape_out[3]]) 
        
        # Feature aggregation
        if (scalar_feat is not None):
#            print (out.shape)
#            print (scalar_feat.shape)
            out = tf.concat([out, scalar_feat], 1)
#            print (out.shape)
        
        out = tf.layers.dense(inputs=out, units=512, activation=tf.nn.relu)
        print_summary_model (out, verbose=verbose)
        
    with tf.variable_scope('block_fc2'):
        logits = tf.layers.dense(inputs=out, units=params['num_labels'], activation=tf.nn.relu)
        print_summary_model (logits, last=True, verbose=verbose)
        
    return logits
    

def model (dict_input, params, is_train, verbose=True):    
    
    labels = dict_input['labels']
    
    with tf.variable_scope('model', reuse = not is_train):
        logits = build_model (dict_input, params, is_train, verbose=verbose)
        predictions = tf.argmax(logits, 1)
        
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    acc = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
    
    # Define training step that minimizes the loss with the Adam optimizer
    if (is_train):
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        global_step = tf.train.get_or_create_global_step()
        if (params['batch_norm']):
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)    
        
    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', acc)
    tf.summary.image('train_image', dict_input['images'])


    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params['num_labels']):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(dict_input['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = dict_input
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = acc
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if (is_train):
        model_spec['train_op'] = train_op

    return model_spec



## Resetando o grafo para rodar aqui no spyder
#tf.reset_default_graph()
#
#images = tf.Variable(tf.ones([2,64,64,3]))
#feats = tf.Variable(tf.ones([2,7]))
#dict_input = {'images': images, 'labels': np.array([1,0]), 'scalar_feat': feats}
#params = {'classes': 5, 'learning_rate': 0.001, 'batch_norm': True, 'num_labels': 2}
#
#model_spec_train = model(dict_input, params, True)
#
#model_spec_val = model(dict_input, params, False, verbose=False)
#
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print (sess.run(model_spec_train['accuracy']))





'''

out_conv1 = tf.layers.conv2d(
        inputs=images, 
        filters=16, 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        activation=tf.nn.relu)

out_pool1 = tf.layers.max_pooling2d(inputs=out_conv1, pool_size=2, strides=2)

out_pool1_shape = out_pool1.shape

print (out_pool1_shape)

# First, reshape the output into [batch_size, flat_size]
pool1_flat = tf.reshape(out_pool1, [-1, 32 * 32 * 16])

dense1 = tf.layers.dense(inputs=pool1_flat, units=20, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=dense1,units=6, activation=tf.nn.softmax)




'''





'''
        # Batch normalization
        if (params['batch_norm']):
            if ('bn_momentum' in params):
                momentum = params['momentum']
            else:
                momentum = 0.99
                        
            out = tf.layers.batch_normalization(out, momentum=momentum, training=is_train)
            
        # Dropout
        if (params['dropout']):
            if ('dp_rate' in params):
                dp_rate = params['dropout']
            else:
                dp_rate = 0.5
                        
            out = tf.layers.dropout(out, rate=dp_rate, training=is_train)

''' 