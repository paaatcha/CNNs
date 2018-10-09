#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
A CNN model to tackle the Signs dataset using tf.keras

Author: André Pacheco
Email: pacheco.comp@gmail.com

If you find any bug, please email-me
"""

from __future__ import print_function
from __future__ import division
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

import sys
#sys.path.append('/home/patcha/Dropbox/Doutorado/Codigos/Python/utils')
sys.path.append('/home/labcin/CODIGOS/utils/')
sys.path.append('/home/labcin/CODIGOS/CNNs/cnn_build_blocks/')

from utils_img import get_path_and_labels_from_folders, get_dataset_tf
from utils import plot_conf_matrix
from cnn_model import model
from cnn_train import train_and_evaluate



################################# PARAMETERS ##################################
NUM_LABLES = 9
INPUT_SHAPE = (64,64,3)
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
MODEL_DIR = '/home/labcin/CODIGOS/CNNs/skin-cancer/pad/Modelo'

###############################################################################

path_train = '/home/labcin/AndrePacheco/Datasets/PAD/pad_menor_splited/TRAIN'
path_val = '/home/labcin/AndrePacheco/Datasets/PAD/pad_menor_splited/VAL'
path_test = '/home/labcin/AndrePacheco/Datasets/PAD/pad_menor_splited/TEST'

params_load = {
        'img_size': (INPUT_SHAPE[0],INPUT_SHAPE[1]),
        'channels': INPUT_SHAPE[2],
        'shuffle': True,
        'repeat': True,
        'augmentation': True,
        'sat': True,
        'bright': True,
        'flip': True,
        'threads': 4,
        'batch_size': BATCH_SIZE
        }




# Loading the train dataset and creating the dataser TF iterator
paths_train, labels_train, scalar_feat_train, dict_labels_train = get_path_and_labels_from_folders (path_train, one_hot=False)
train_data, train_dataset = get_dataset_tf (paths_train, labels_train, True, params_load)

params_load['shuffle'] = False
params_load['batch_size'] = 7

# Loading the validation dataset and creating the dataser TF iterator
paths_val, labels_val, scalar_feat_val, dict_labels_val = get_path_and_labels_from_folders (path_val, one_hot=False)
val_data, val_dataset = get_dataset_tf (paths_val, labels_val, False, params_load)

# Loading the test dataset and creating the dataser TF iterator
paths_test, labels_test, scalar_feat_test, dict_labels_test = get_path_and_labels_from_folders (path_test, one_hot=False)
test_data, test_dataset = get_dataset_tf (paths_test, labels_test, False, params_load)   
  

#with tf.Session() as sess:
#    sess.run(train_data['iterator_init_op'])
##    print (sess.run(train_data['scalar_feat']))
#    print (sess.run(train_data['scalar_feat']).shape)

###############################################################################

#print (len(paths_train))

params_model = {        
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'batch_norm': False,
        'num_labels': NUM_LABLES,
        'num_epochs': EPOCHS,
        'train_size': len(paths_train),
        'eval_size': len(paths_val),
        'save_summary_steps': 5        
        }


# Creating the model    
train_model_spec = model(train_data, params_model, True)
val_model_spec = model(val_data, params_model, False, verbose=False)



## Train the model
#print ("Starting training for {} epoch(s)".format(params_load['num_epochs']))
train_and_evaluate(train_model_spec, val_model_spec, MODEL_DIR, params_model)
