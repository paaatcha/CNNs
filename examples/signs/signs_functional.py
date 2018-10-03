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

import sys
#sys.path.append('/home/patcha/Dropbox/Doutorado/Codigos/Python/utils')
sys.path.append('/home/labcin/CODIGOS/utils/')
from utils_img import get_path_and_labels_from_folders, get_dataset_tf



########## CONSTANTS #############
NUM_CLASS = 6
INPUT_SHAPE = (64,64,3)
BATCH_SIZE = 32
EPOCHS = 20
FILEPATH = 'best_model.hdf5'
##################################

def training (model, train_dataset, test_dataset, n_samples_train, n_samples_test):

    # Choosing the loss function, optimizer and metrics
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])    
    
    # Defining a checkpoint to save the best model
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=FILEPATH, verbose = 1, save_best_only=True)    
    
    # Including tensorboard
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=5, batch_size=BATCH_SIZE, write_graph=False, write_images=True)
    
    # Including early stop    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
        
    model.fit(
            train_dataset,
            steps_per_epoch=n_samples_train // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=test_dataset,
            validation_steps=n_samples_test // BATCH_SIZE,
            callbacks=[checkpointer, tensorboard, early_stop])   
    
    return model
    
    
def evaluate_model (filepath, model, dataset, n_samples):
    
    model.load_weights(filepath)
    
    # Evaluate the model on test set
    score = model.evaluate(dataset, steps=n_samples // BATCH_SIZE)
    
    # Print test accuracy
    print('\n', 'Test accuracy:', score[1])    
    
def predict_model (filepath, model, dataset, n_samples):
    
    model.load_weights(filepath)
    
    # Evaluate the model on test set
    pred = model.predict(dataset, steps=n_samples // BATCH_SIZE)
    
    return pred     
    
def keras_model_functional (summary=True):
    img_input = tf.keras.layers.Input(shape=(INPUT_SHAPE), name="img_input")
    
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, padding='same', activation='relu')(img_input)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=4, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
#    model.add(tf.keras.layers.Dropout(0.5))
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
#    model.add(tf.keras.layers.Dropout(0.5))
    
    pred = tf.keras.layers.Dense(NUM_CLASS, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=img_input, outputs=pred)
    
    if (summary):    
        model.summary()
    
    return model


def keras_model (summary=True):
    model = tf.keras.Sequential()

    # The same model of fm_1.py
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=7, padding='same', activation='relu', input_shape=INPUT_SHAPE)) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#    model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#    model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#    model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
#    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CLASS, activation='softmax'))
    
    if (summary):    
        model.summary()
    
    return model


##################################################################################################
params = {
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


paths_train, labels_train, dict_labels_train = get_path_and_labels_from_folders ('/home/labcin/AndrePacheco/Datasets/SIGNS/TRAIN')
train_data, train_dataset = get_dataset_tf (paths_train, labels_train, True, params)

params['shuffle'] = False

paths_test, labels_test, dict_labels_test = get_path_and_labels_from_folders ('/home/labcin/AndrePacheco/Datasets/SIGNS/TEST')
test_data, test_dataset = get_dataset_tf (paths_test, labels_test, False, params)    

model = keras_model_functional()

model = training (model, train_dataset, test_dataset, len(paths_train), len(paths_test))

evaluate_model(FILEPATH, model, test_dataset, len(paths_test))
