#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Andre Pacheco
Email: pacheco.comp@gmail.com

"""
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from utils import *


def cnn_model_1 (shapeIn, nClass):
    model = Sequential()
    
    model.add(Conv2D(4, kernel_size=(11,11), activation='relu', input_shape=shapeIn))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    
    model.add(Conv2D(8, kernel_size=(8,8), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
#    model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
#    model.add(Dropout(0.25))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    
#    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
#    model.add(Dropout(0.25))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
    

    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClass, activation='softmax'))
    
    model.compile(loss=keras.losses.binary_crossentropy, 
                  optimizer=keras.optimizers.Adadelta(),                   
                  metrics=['accuracy'])
    
    model.summary()
    
    return model      

def cnnVGG(shapeIn, nClass):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=shapeIn))
    model.add(Conv2D(4, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(4, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(8, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(8, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(8, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(8, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(784, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(392, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClass, activation='softmax'))

    model.summary()
    
#    model.compile(loss=keras.losses.categorical_crossentropy, 
#                  optimizer=keras.optimizers.Adadelta(),                   
#                  metrics=['accuracy']) 

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=keras.losses.mean_squared_error, metrics=['accuracy'])

    return model


def cnnModel (shapeIn, nClass):
    model = Sequential()
    
    model.add(Conv2D(12, kernel_size=(11,11), activation='relu', input_shape=shapeIn))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    
    model.add(Conv2D(12, kernel_size=(8,8), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    
    model.add(Conv2D(12, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    
    model.add(Conv2D(12, kernel_size=(4,4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    
    model.add(Conv2D(12, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    

    
    model.add(Flatten())
    model.add(Dense(576, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(288, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(nClass, activation='softmax'))
    
    model.compile(loss='mean_squared_error', 
                  optimizer=keras.optimizers.Adadelta(),                   
                  metrics=['accuracy'])
    
    model.summary()
    
    return model    

###############################################################################
###############################################################################
path = "/home/patcha/Datasets/Cancer"
rows, cols = 128, 128
shapeIn = (rows,cols,3)
nClass = 2
batchSize = 50
epochs = 50
train = True



################################################################################
# Loading the dataset
print 'Loading the dataset...'
inputData, outputData = loadDatasetImgs (path, norm=True, resize=(rows,cols))

# Spliting the dataset into train, validation and test data
inTrain, outTrain, inVal, outVal, inTest, outTest = splitData (inputData, outputData, inputData.shape[0])

del inputData
del outputData

## Data augmentation
#datagen = ImageDataGenerator (horizontal_flip=True,
#                              vertical_flip=True,
#                              featurewise_center=True,
#                              featurewise_std_normalization=True,
#                              shear_range=0.2,
#                              zoom_range=0.2,
#                              rotation_range=5)
#
#datagen.fit(inTrain)
#
#print 'Data augmentation process...'
#i = 0
#nmax = 300
#for xbatch,ybatch in datagen.flow(inTrain, outTrain, batch_size=10):#, save_to_dir='/home/patcha/Aquila/Aug/', save_prefix='aug', save_format='png'):
#    inTrain = np.concatenate((inTrain,xbatch))
#    outTrain = np.concatenate((outTrain,ybatch))
#    i+=1    
#    if i==nmax:
#        break



if train:
    
    # Setting the model
    model = cnn_model_1 (shapeIn, nClass)
    #
    #
    ##callbacks = [
    ##    EarlyStopping(monitor='val_loss', patience=3, verbose=0),
    ##]
    #
    history = model.fit(inTrain,outTrain,
              batch_size=batchSize,
              epochs=epochs,
              verbose=1,
              validation_data=(inVal, outVal))
              #callbacks=callbacks)
    
    # PLoting the history
    plotHistory (history)
    
    # Saving the model
    model.save('modelCancer.h5')
    model.save_weights('modelCancerWeights.h5')
    
    ###############################################################################
    # Evaluating the model
    score = model.evaluate(inTest, outTest, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    outPredict = model.predict(inTest)
    cm = confusion_matrix(outTest.argmax(axis=1),outPredict.argmax(axis=1))
    print classification_report(outTest.argmax(axis=1),outPredict.argmax(axis=1), target_names=['Healthy','Unhealthy'])
    plotConfMatrix (cm, ['Healthy','Unhealthy'], normalize=True)
