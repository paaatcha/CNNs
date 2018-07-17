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
import sys
sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/Python/utils/')
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
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(32, (3,3), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(32, (3,3), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(32, (3,3), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClass, activation='softmax'))

    model.summary()
    
    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=keras.optimizers.Adadelta(),                   
                  metrics=['accuracy']) 

#    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(optimizer=sgd, loss=keras.losses.mean_squared_error, metrics=['accuracy'])

    return model


def cnn_model2 (shapeIn, nClass):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(11,11), activation='relu', input_shape=shapeIn))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, kernel_size=(8,8), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, kernel_size=(4,4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
    
#    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
    

    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(nClass, activation='softmax'))
    
    model.compile(loss=keras.losses.binary_crossentropy, 
                  optimizer=keras.optimizers.Adadelta(),                   
                  metrics=['accuracy'])
    
    model.summary()
    
    return model    


def getGenerators (path, rows, cols, batchSize):
    train_data = ImageDataGenerator ()
    val_data = ImageDataGenerator ()
    test_data = ImageDataGenerator ()
    
    train_ge = train_data.flow_from_directory(path + '/TRAIN',
			target_size=(rows, cols),
			batch_size=batchSize)
    
    val_ge = val_data.flow_from_directory(path + '/VAL',
			target_size=(rows, cols),
			batch_size=batchSize)
    
    test_ge = test_data.flow_from_directory(path + '/TEST',
			target_size=(rows, cols),
			batch_size=batchSize)
    
    return train_ge, val_ge, test_ge
###############################################################################
###############################################################################
#path = '/home/labcin/AndrePacheco/ISIC/ISIC_Classification/HSV'
#path = '/home/labcin/AndrePacheco/ISIC/ISIC_SPLITED/RGB/'
path = '/home/labcin/AndrePacheco/ISIC/ISIC_ALL/RGB_bin/'
#folders = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
folders = ['Non_NV', 'NV']
rows, cols = 224, 224
shapeIn = (rows,cols,3)
nClass = len(folders)
batchSize = 32
epochs = 25
train = True

################################################################################
# Loading the dataset
#print 'Loading the dataset...'


#inTrain, outTrain = loadImgsFromFolders (path + 'TRAIN', folders, norm=True, resize=(rows,cols), verbose=True, doubleFolder=True)
#inVal, outVal = loadImgsFromFolders (path + 'VAL/', folders, norm=True, resize=(rows,cols), verbose=True)
#inTest, outTest = loadImgsFromFolders (path + 'TEST/', folders, norm=True, resize=(rows,cols), verbose=True, doubleFolder=True)

#inputData, outputData = loadImgsFromFolders (path, folders, norm=True, resize=(rows,cols), verbose=True)

# Spliting the dataset into train, validation and test data
#inTrain, outTrain, inVal, outVal, inTest, outTest = splitData (inputData, outputData, inputData.shape[0])

#del inputData
#del outputData

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


train_ge, val_ge, test_ge = getGenerators(path, rows, cols, batchSize)


if train:
    
    # Setting the model
    model = cnn_model_2 (shapeIn, nClass)
    #
    #
    ##callbacks = [
    ##    EarlyStopping(monitor='val_loss', patience=3, verbose=0),
    ##]
    #
#    history = model.fit(inTrain,outTrain,
#              batch_size=batchSize,
#              epochs=epochs,
#              verbose=1,
#              validation_data=(inVal, outVal))
#              #callbacks=callbacks)
    
    
    history = model.fit_generator(train_ge, 
                                  steps_per_epoch=train_ge.samples // batchSize,
                                  epochs=epochs, 
                                  validation_data=val_ge, 
                                  validation_steps= val_ge.samples // batchSize)    
  
    
    
    # PLoting the history
    plotHistory (history)
    
    # Saving the model
    model.save('modelCancer.h5')
    model.save_weights('modelCancerWeights.h5')
    
    ###############################################################################
    # Evaluating the model
    #score = model.evaluate(inTest, outTest, verbose=0)
    score = model.evaluate_generator(test_ge, test_ge.samples // batchSize)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    outPredict = model.predict_generator(test_ge, test_ge.samples // batchSize)
    
    print outPredict
    print test_ge.classes
    
    cm = confusion_matrix(test_ge.classes,outPredict.argmax(axis=1))
    print classification_report(test_ge.classes,outPredict.argmax(axis=1), target_names=folders)
    plotConfMatrix (cm, folders, normalize=True)
