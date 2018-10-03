#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
A CNN model to tackle the Fashion MNIST using tf.dataset and tf.keras

Author: André Pacheco
Email: pacheco.comp@gmail.com

If you find any bug, please email-me
"""

import tensorflow as tf   

########## CONSTANTS #############
NUM_CLASS = 10
INPUT_SHAPE = (28,28,1)
BATCH_SIZE = 64
EPOCHS = 2
FILEPATH = 'best_model.hdf5'
##################################

def training (imgs_train, labs_train, imgs_test, labs_test):
    
    # Getting the samples for each set
    n_samples_train = imgs_train.shape[0]
    n_samples_val = imgs_test.shape[0]
    
    # Getting the tf.data.dataset for each set
    train_dataset = get_dataset (imgs_train, labs_train, True, BATCH_SIZE)
    val_dataset = get_dataset (imgs_val, labs_val, False, BATCH_SIZE)
    
    model = keras_model()

    # Choosing the loss function, optimizer and metrics
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])    
    
    # Defining a checkpoint to save the best model
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=FILEPATH, verbose = 1, save_best_only=True)    
    
    # Including tensorboard
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=5, batch_size=BATCH_SIZE, write_graph=False, write_images=True)
    
    # Including early stop    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
        
    model.fit(
            train_dataset,
            steps_per_epoch=n_samples_train // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_dataset,
            validation_steps=n_samples_val // BATCH_SIZE,
            callbacks=[checkpointer, tensorboard, early_stop])   
    
    return model
    
    
def evaluate_model (filepath, model, imgs, labs):
    
    model.load_weights(filepath)
    test_dataset = get_dataset (imgs, labs, False, BATCH_SIZE)
    n_samples_test = imgs.shape[0]
    # Evaluate the model on test set
    score = model.evaluate(test_dataset, steps=n_samples_test // BATCH_SIZE)
    
    # Print test accuracy
    print('\n', 'Test accuracy:', score[1])    
    


def keras_model (summary=True):
    model = tf.keras.Sequential()

    # The same model of fm_1.py
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=INPUT_SHAPE)) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CLASS, activation='softmax'))
    
    if (summary):    
        model.summary()
    
    return model



def process_input_data (img, lab):
    x = tf.reshape(tf.cast(img, tf.float32), INPUT_SHAPE)
    y = tf.one_hot(tf.cast(lab, tf.uint8), NUM_CLASS)
    return x,y


def get_dataset (imgs, labs, is_training, batch_size=32):
    
    n_samples = imgs.shape[0]
    print n_samples
    
    # Preprocessing the images and labels
    pre_proc = lambda x, y: process_input_data (x, y)    
    
    # Creating the dataset
    dataset = tf.data.Dataset.from_tensor_slices( (imgs, labs) )
    
    if (is_training):
        dataset = dataset.shuffle(n_samples)   
    
    # Preprocessing the data
    dataset = dataset.map(pre_proc, num_parallel_calls=4)    
    
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1) 
    
    return dataset


###############################################################################################################

# Loading data
(imgs_train, labs_train), (imgs_test, labs_test) = tf.keras.datasets.fashion_mnist.load_data()  

imgs_train = imgs_train / 255.0
imgs_test = imgs_test / 255.0  

# Getting a validation set
(imgs_train, imgs_val) = imgs_train[5000:], imgs_train[:5000] 
(labs_train, labs_val) = labs_train[5000:], labs_train[:5000]

# Training
model = training (imgs_train, labs_train, imgs_test, labs_test)

# Testing
evaluate_model (FILEPATH, model, imgs_test, labs_test)



 
    