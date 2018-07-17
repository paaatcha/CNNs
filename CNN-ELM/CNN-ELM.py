# -*- coding: utf-8 -*-
"""
Author: André Pacheco
Email: pacheco.comp@gmail.com

This code implements a CNN using an ELM as fully connected layer.
It's on tensting and may not work properly for all datasets. If you wanna use it,
do on your on risk.

If you find any bug, please email-me
"""

import tensorflow as tf
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.utils import shuffle
import sys
import numpy as np
#sys.path.append('/home/patcha/Dropbox/Doutorado/Codigos/Python/utils')
sys.path.append('/home/labcin/Dropbox/Codigos/Python/utils/')
sys.path.append('/home/labcin/Dropbox/Codigos/Python/CNN/CNN-Blocks/')
from utilsCNN import formatInputData, formatOutputData
from CNNBlocks import convLayer2D, maxPooling, flatLayer, fcLayer, elm, oselm, rbm, trainRbm, trainCNN, getPerformance, oselmRun, elmRun#, evalPerformance 


def getMnist (imgShape, nClass, verbose=False):
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    xTrain = formatInputData (xTrain, imgShape[0], imgShape[1], imgShape[2])
    xTest = formatInputData (xTest, imgShape[0], imgShape[1], imgShape[2])
    yTrain = formatOutputData (yTrain, nClass)
    yTest = formatOutputData (yTest, nClass)
    
    if verbose:
        print('\nMNIST:')
        print('xTrain shape:', xTrain.shape)
        print('yTrain shape:', yTrain.shape)
        print(xTrain.shape[0], 'train samples')
        print(xTest.shape[0], 'test samples')
        
    return xTrain, yTrain, xTest, yTest
        

def getCifar10 (imgShape, nClass, verbose=False):
    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

    xTrain = formatInputData (xTrain, imgShape[0], imgShape[1], imgShape[2])
    xTest = formatInputData (xTest, imgShape[0], imgShape[1], imgShape[2])
    yTrain = formatOutputData (yTrain, nClass)
    yTest = formatOutputData (yTest, nClass)
    
    if verbose:
        print('\nCIFAR 10:')
        print('xTrain shape:', xTrain.shape)
        print('yTrain shape:', yTrain.shape)        
        print(xTrain.shape[0], 'train samples')
        print(xTest.shape[0], 'test samples')
        
        
    return xTrain, yTrain, xTest, yTest
    
    
def cnnModel (X, dropout=1.0, verbose=False):    
    conv11 = convLayer2D(X, kernel=[3,3,3], nFM=8, padd='SAME')
    conv12 = convLayer2D(conv11, kernel=[3,3,8], nFM=8, padd='SAME')    
    pool1 = maxPooling(conv12)        
    pool1_dropout = tf.nn.dropout(pool1, keep_prob=dropout)    
    
    conv21 = convLayer2D(pool1_dropout, kernel=[3,3,8], nFM=16, padd='SAME')    
    conv22 = convLayer2D(conv21, kernel=[3,3,16], nFM=16, padd='SAME')    
    pool2 = maxPooling(conv22)
    pool2_dropout = tf.nn.dropout(pool2, keep_prob=dropout)
    
#    conv31 = convLayer2D(pool2_dropout, kernel=[3,3,16], nFM=16, padd='SAME')    
#    conv32 = convLayer2D(conv31, kernel=[3,3,16], nFM=16, padd='SAME')    
#    pool3 = maxPooling(conv32)
#    pool3_dropout = tf.nn.dropout(pool3, keep_prob=dropout)

    
    flatFM = flatLayer(pool2_dropout)
    
    if verbose:
        print ('----------- CNN summary -------------')
        print (conv12.get_shape())
        print (pool1.get_shape())
        print (conv22.get_shape())
        print (pool2.get_shape())
#        print (conv32.get_shape())
#        print (pool3.get_shape())
        print (flatFM.get_shape())
        print ('-------------------------------------')


    return flatFM
    
def fcModelBp (inputFC, Y, neurons, nClass, dropout):
    # FC with BP
    fc1 = fcLayer(inputFC, neurons, 'relu')    
    fc1_dropout = tf.nn.dropout(fc1, keep_prob=dropout)
    ypred = fcLayer(fc1_dropout, nClass, 'softmax')        
    
    return ypred

def fcModelElm (inputFC, Y, neurons, act='sig', W=None, b=None, beta_prev=0.0):
    beta, W, b = elm (inputFC, Y, neurons, act=act, W=W, b=b, beta_prev=beta_prev)
    return beta, W, b
    

def fcModelOsElm (X, Y, neurons, nClass, batchSize, isInit, reg=0.5, act='sig', W=None, b=None, betaPrev=None, Pprev=None):
    params = oselm(X, Y, neurons, nClass, batchSize, isInit, reg, act, W, b, betaPrev, Pprev)    
    return params

    
###############################################################################
############################# START SCRIPT ####################################
###############################################################################

########################### PARAMETERS ########################################
inputShapeCifar = (32, 32, 3)
batchSize = 50
nClass = 10
epochs = 5
lr = 0.001
hidRBM = 512
neuronsFC = 512
fcType = 'ELM'
########################### PARAMETERS ########################################


######################## Getting the dataset ##################################
xTrain, yTrain, xTest, yTest = getCifar10 (inputShapeCifar, nClass, True)
###############################################################################

################ Place holders for the training phase #########################
X = tf.placeholder(tf.float32,(None,32,32,3), name='X')
Y = tf.placeholder(tf.float32,(None), name='Y')
dropoutFC = tf.placeholder(tf.float32, name='dropoutFC')
dropoutCNN = tf.placeholder(tf.float32, name='dropoutCNN')
momentum = tf.placeholder(tf.float32, name='momentum')
isInit = tf.placeholder(tf.bool,(None), name='isInit')
Pprev = tf.placeholder(tf.float32, name='Pprev')
betaPrev = tf.placeholder(tf.float32, name='betaPrev')
oselmParamsPrev = [np.zeros([neuronsFC, nClass], dtype=np.float32), np.zeros([neuronsFC, neuronsFC], dtype=np.float32)]

###############################################################################

############################# The model #######################################
flatFM = cnnModel(X, verbose=True, dropout=dropoutCNN)
#featMap = tf.placeholder(tf.float32, flatFM.get_shape())


#layersRBM = rbm (flatFM, hidRBM, batchSize)
#recError = tf.losses.mean_squared_error(layersRBM[2], layersRBM[3])

# with ELM without RBM
#beta, W, b = fcModelElm (flatFM, Y, neuronsFC, beta_prev=betaPrev)
#netOut = elmRun (flatFM, [beta, W, b])

# With ELM with RBM
beta, W, b = fcModelElm (flatFM, Y, neuronsFC, act='sig', beta_prev=betaPrev)
netOut = elmRun (flatFM, [beta, W, b])


# With OS-ELM without RBM
#oselmParams = fcModelOsElm (flatFM, Y, neuronsFC, nClass, batchSize, isInit, betaPrev=betaPrev, Pprev=Pprev)                                    
#netOut = oselmRun (flatFM, oselmParams)

# With OS-ELM with RBM
# TODO

# with BP with RBM
#cnnPred = fcModelBp (layersRBM[1], Y, neuronsFC, nClass, dropoutFC)

# with BP Without RBM
#netOut = fcModelBp (flatFM, Y, neuronsFC, nClass, dropoutFC)



############################# Training phase ##################################

trainOp, lossOp = trainCNN (netOut, Y, opt='adam', loss='log', lr=lr)
accuracy = getPerformance (netOut, Y)
# Rbm training
#atts = trainRbm (layersRBM, lr=0.001, wc=0.001, mom=momentum, batchSize=batchSize)





if fcType == 'BP':
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        sess.run(tf.global_variables_initializer())
        nSam = xTrain.shape[0]
        
        for i in xrange(epochs):
            xTrain, yTrain = shuffle(xTrain, yTrain)        
            print '####### Epoch {}/{} #######'.format(i, epochs)
            
            for offset in xrange(0,nSam,batchSize):
                end = offset + batchSize
                xBatch, yBatch = xTrain[offset:end], yTrain[offset:end]  
                sess.run(trainOp, feed_dict={X: xBatch, Y: yBatch, dropoutFC: 1.0, dropoutCNN: 1.0})     
                
                if offset % 1000 == 0:
                    trainAcc, loss = sess.run([accuracy, lossOp], feed_dict={X: xBatch, Y: yBatch, dropoutFC: 1.0, dropoutCNN: 1.0})
                    print 'Batch: {}/{} | training acc: {} | loss: {} | recError: {}'.format(end,nSam,trainAcc,loss,0)                
                    
    #            if offset % 10000 == 0 and offset != 0:
    #                break
            
########################## Evaluating the test data ###########################
            nSamTest, accFinal, n = xTest.shape[0], 0, 0
            for offsetTest in xrange(0,nSamTest,batchSize):
                end = offsetTest + batchSize
                xBatch, yBatch = xTest[offsetTest:end], yTest[offsetTest:end]                
                pred, acc = sess.run([netOut, accuracy], feed_dict={X: xBatch, Y: yBatch, dropoutFC: 1.0, dropoutCNN: 1.0})
                accFinal += acc
                n+=1
            print 'Epoch {} - Test acc {}'.format(i,accFinal/n)
######################### Evaluating the test data ############################
            
        writer.close()
        
elif fcType=='OS-ELM':
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        sess.run(tf.global_variables_initializer())
        nSam = xTrain.shape[0]
        
        for i in xrange(epochs):
            xTrain, yTrain = shuffle(xTrain, yTrain)        
            print '####### Epoch {}/{} #######'.format(i, epochs)
            
            for offset in xrange(0,nSam,batchSize):
                end = offset + batchSize
                xBatch, yBatch = xTrain[offset:end], yTrain[offset:end]  
                
                if offset == 0:
                    # Init phase
                    _, oselmParamsPrev = sess.run([trainOp,oselmParams], feed_dict={X: xBatch, Y: yBatch, dropoutCNN: 1.0, betaPrev: oselmParamsPrev[0], Pprev: oselmParamsPrev[1], isInit: True})            
#                    print '\n\n INIT PHASE \n\n'   
                    print oselmParamsPrev[-1]
#                    print '\n\n', oselmParamsPrev[-1].shape, '\n\n'
                else:
                    # Training phase
                    _, oselmParamsPrev = sess.run([trainOp,oselmParams], feed_dict={X: xBatch, Y: yBatch, dropoutCNN: 1.0, betaPrev: oselmParamsPrev[0], Pprev: oselmParamsPrev[1], isInit: True})            
#                    print '\n\n TRAINING PHASE \n\n'
                                
                
                if offset % 1000 == 0:
                    trainAcc, loss = sess.run([accuracy, lossOp], feed_dict={X: xBatch, Y: yBatch, dropoutCNN: 1.0, betaPrev: oselmParamsPrev[0], Pprev: oselmParamsPrev[1], isInit: True})
                    print 'Batch: {}/{} | training acc: {} | loss: {} | recError: {}'.format(end,nSam,trainAcc,loss,0)                
                    print oselmParamsPrev[0]
                    print oselmParamsPrev[1]
#                    exit()
                    
    #            if offset % 10000 == 0 and offset != 0:
    #                break
            
########################## Evaluating the test data ###########################
            nSamTest, accFinal, n = xTest.shape[0], 0, 0
            for offsetTest in xrange(0,nSamTest,batchSize):
                end = offsetTest + batchSize
                xBatch, yBatch = xTest[offsetTest:end], yTest[offsetTest:end]                
                pred, acc = sess.run([netOut, accuracy], feed_dict={X: xBatch, Y: yBatch, dropoutCNN: 1.0, betaPrev: oselmParamsPrev[0], Pprev: oselmParamsPrev[1], isInit: False})
                accFinal += acc
                n+=1                
            print 'Epoch {} - Test acc {}'.format(i,accFinal/n)
######################### Evaluating the test data ############################
            
        writer.close()    
        
elif fcType=='ELM':
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        sess.run(tf.global_variables_initializer())
        nSam = xTrain.shape[0]
        rbmError = 0
        betaValues = np.zeros([neuronsFC, nClass], dtype=np.float32)
        betaBatch = np.zeros([neuronsFC, nClass], dtype=np.float32)
        for i in xrange(epochs):
            xTrain, yTrain = shuffle(xTrain, yTrain)        
            
            
            print '####### Epoch {}/{} #######'.format(i, epochs)
            
            for offset in xrange(0,nSam,batchSize):
                end = offset + batchSize
                xBatch, yBatch = xTrain[offset:end], yTrain[offset:end]                

                if offset < 5*batchSize:                
#                    _, betaValues,_,rbmError = sess.run([trainOp,beta, atts, recError], feed_dict={X: xBatch, Y: yBatch, dropoutCNN: 1.0, momentum: 0.5, betaPrev: betaValues})      
                    _, betaValues = sess.run([trainOp,beta], feed_dict={X: xBatch, Y: yBatch, dropoutCNN: 1.0, betaPrev: betaValues})      
                else:
#                    _, betaValues,_,rbmError = sess.run([trainOp,beta, atts, recError], feed_dict={X: xBatch, Y: yBatch, dropoutCNN: 1.0, momentum: 0.9, betaPrev: betaValues})
                    _, betaValues = sess.run([trainOp,beta], feed_dict={X: xBatch, Y: yBatch, dropoutCNN: 1.0, betaPrev: betaValues})                    
                
#                print (params[0]), '\n\n\n'                
#                betaValues = betaBatch#(betaBatch+betaValues)/2
                
                
                if offset % 1000 == 0:
                    trainAcc, loss = sess.run([accuracy, lossOp], feed_dict={X: xBatch, Y: yBatch, dropoutCNN: 1.0, beta: betaValues})
                    print 'Batch: {}/{} | training acc: {} | loss: {} | recError: {}'.format(end,nSam,trainAcc,loss,rbmError)                
#                    print betaValues.mean()
                    
    #            if offset % 10000 == 0 and offset != 0:
    #                break
            
########################## Evaluating the test data ###########################
            nSamTest, accFinal, n = xTest.shape[0], 0, 0
            for offsetTest in xrange(0,nSamTest,batchSize):
                end = offsetTest + batchSize
                xBatch, yBatch = xTest[offsetTest:end], yTest[offsetTest:end]                
                pred, acc = sess.run([netOut, accuracy], feed_dict={X: xBatch, Y: yBatch, dropoutCNN: 1.0, beta: betaValues})
                accFinal += acc
                n+=1                
            print 'Epoch {} - Test acc {}'.format(i,accFinal/n)
######################### Evaluating the test data ############################
            
        writer.close()            

















