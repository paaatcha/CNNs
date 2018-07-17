#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: André Pacheco
Email: pacheco.comp@gmail.com

This code implements the build blocks for a convolutional neural network using
the TensorFlow as backend. Moreover, it also has some code related to the CNN-ELM and CNN-RBM-ELM
(It's on testing, if you wanna use it, do on your on risk)

"""

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


# X: the inputdata (image)
# kernel: the shape of the kernel, ex: [3,3,1]
# nFM: the number of features map
# stride: The stride of the sliding window for each dimension of input
# padd: the padding (SAME or VALID)
# activation: the activation function (relu, logit)
def convLayer2D (x, kernel, nFM=1, stride = [1,1,1,1], padd='SAME', activation='relu', dropout=None, wi=None, bi=None):
    with tf.name_scope('Conv'):
        if wi is None:        
            w = tf.Variable(tf.truncated_normal([kernel[0],kernel[1],kernel[2],nFM],
                                            stddev=0.1, dtype=tf.float32), name='Weights')
        else:
            w = tf.Variable(wi, dtype=tf.float32, name='Weights')                                                        
            
        if bi is None:
            b = tf.Variable(tf.constant(0.1, shape=[nFM]), name='Biases')
        else:
            b = tf.Variable(bi, dtype=tf.float32, name='Biases')
        
        cv2d = tf.nn.conv2d(x,w,strides=stride, padding=padd, name='Conv2D')
        out = tf.nn.bias_add(cv2d,b)
        
        if activation=='relu':
            actOut = tf.nn.relu(out, name='relu')
        elif activation=='sigmoid':
            actOut = tf.nn.sigmoid(out,name='sigmoid')
        else:
            print ('There is no {} activation'.format(activation))
            raise AttributeError
        
        if dropout is not None:
            actOut = tf.nn.dropout(actOut, keep_prob=dropout)
        
        return actOut
    
def maxPooling (x, kernel=[1,2,2,1], stride=[1,2,2,1], padd='SAME'):
    with tf.name_scope('pool'):
        return tf.nn.max_pool(x, ksize=kernel, strides=stride, padding=padd)

def flatLayer (x):
    with tf.name_scope('flat'):
        nFM = int(np.prod(x.get_shape()[1:]))
        return tf.reshape(x, [-1, nFM])
    
def fcLayer (x, neurons, activation='relu', dropout=None, wi=None, bi=None):
    with tf.name_scope('FC'):
        shape = int(x.get_shape()[1])
        if wi is None:
            w = tf.Variable(tf.truncated_normal([shape, neurons],
                                            dtype=tf.float32, stddev=0.1),
                                                name='Weights')
        else:
            w = tf.Variable(w, dtype=tf.float32, name='Weights')                                                
            
        if bi is None:
            b = tf.Variable(tf.constant(1.0, shape=[neurons], dtype=tf.float32),
                                         name='Biases')            
        else:
            b = tf.Variable(bi, dtype=tf.float32, name='Biases')                                             
        
        hidfc = tf.nn.bias_add(tf.matmul(x, w), b, name='hidfc')
        if activation=='relu':
            out = tf.nn.relu(hidfc, name='relu')
        elif activation=='softmax':
            out = tf.nn.softmax(hidfc, name='softmax')
        elif activation=='sigmoid':
            out = tf.nn.sigmoid(hidfc, name='sigmoid')
        else:
            print ('There is no {} activation'.format(activation))
            
        if dropout is not None:
            out = tf.nn.dropout(out, keep_prob=dropout)
            
        return out
        
def trainCNN (Ypred, Y, opt='adam', loss='log', lr=0.001, others=None):
    
    if loss == 'log':
        lossOp = tf.losses.log_loss(Y,Ypred)
    elif loss == 'mse':                   
        lossOp = tf.losses.mean_squared_error(Y,Ypred)
    else:
        print 'The loss function {} is not available yet'.format(loss)
        raise ValueError
        
    if opt == 'adam':        
        optimizer = tf.train.AdamOptimizer(lr)
    elif opt == 'adadelta':        
        optimizer = tf.train.AdadeltaOptimizer(lr)
    else:
        print 'The opt function {} is not available yet'.format(opt)
        raise ValueError
        
    return optimizer.minimize(lossOp), lossOp   
    
def getPerformance (Ypred, Y):    
    correctPred = tf.equal(tf.argmax(Ypred,1), tf.argmax(Y,1))
    acc = tf.reduce_mean(tf.cast(correctPred, tf.float32))    
    return acc
    
        
def elm (X, Y, neurons, reg=0.1, act='sig', W=None, b=None, beta_prev=0.0):
    nFeat = int(X.get_shape()[1])  
    #Xc = tf.concat([X, tf.constant(1.0, shape=(batchSize,1))],1)
    
    with tf.name_scope('ELM'):
#        Wnp = np.random.uniform(-1,1,[nFeat,neurons])
#        Wnp,_ = np.linalg.qr(Wnp.T)
#        Wnp = Wnp.T
#        W = tf.Variable(Wnp, dtype=tf.float32, name='Weights', trainable=False)   
#        W = tf.convert_to_tensor(Wnp, tf.float32, name='Weights', trainable=False)        
        if W is None and b is None:            
            W = tf.Variable(tf.random_uniform([nFeat,neurons],-1,1,tf.float32), name='Weights', trainable=True)           
            b = tf.Variable(tf.random_uniform([neurons],-1,1,tf.float32), name='Biases', trainable=True)
        elif W is None or b is None:
            print 'Check the ELM initialization'
            raise ValueError
         

        # Computing the matrix H
        if act == 'sig':
            H = tf.sigmoid(tf.nn.bias_add(tf.matmul(X,W), b, name='H'))
        elif act == 'relu':
            H = tf.nn.relu(tf.nn.bias_add(tf.matmul(X,W), b, name='H'))
        elif act == 'tanh':
            H = tf.nn.tanh(tf.nn.bias_add(tf.matmul(X,W), b, name='H'))
           
        # Computing the weights
        I = tf.eye(neurons,dtype=tf.float32)       
        HInv = tf.matmul(tf.matrix_inverse(tf.matmul(H,H,True) - (reg*I)),H,False,True) 
        beta = tf.matmul(HInv,Y,name='Beta')   

        lr = 0.5                  
        beta = tf.div(tf.add(beta,lr*beta_prev),1+lr)
        
#        pred = tf.nn.softmax(tf.matmul(H,beta))
      
    return beta, W, b
    
def elmRun (X, params, act='sig'):
    beta, W, b = params[0], params[1], params[2]
    
    # Computing the matrix H
    if act == 'sig':
        H = tf.sigmoid(tf.nn.bias_add(tf.matmul(X,W), b, name='H'))
    elif act == 'relu':
        H = tf.nn.relu(tf.nn.bias_add(tf.matmul(X,W), b, name='H'))
    elif act == 'tanh':
        H = tf.nn.tanh(tf.nn.bias_add(tf.matmul(X,W), b, name='H'))  
    
    pred = tf.nn.softmax(tf.matmul(H,beta,name='Pred'))
    
    return pred    
      
def oselm (X, Y, neurons, nClass, batchSize, isInit, reg=0.5, act='sig', W=None, b=None, betaPrev=None, Pprev=None):
    nFeat = int(X.get_shape()[1])   
    
    with tf.name_scope('OS-ELM'):
        if W is None and b is None:
            W = tf.Variable(tf.random_uniform([nFeat,neurons],-1,1,tf.float32), name='Weights', trainable=False)   
            b = tf.Variable(tf.random_uniform([neurons],-1,1,tf.float32), name='Biases', trainable=False)      
#            b = tf.Variable(tf.zeros([neurons], dtype=tf.float32), name='Biases', trainable=False)
        elif W is None or b is None:
            print 'Check the ELM initialization'
            raise ValueError            
        
#        betaFinal = tf.Variable(tf.zeros((neurons,nClass)), name='BetaFinal')
#        Pfinal = tf.Variable(tf.zeros((neurons,neurons)), name='Pfinal')

        # Computing the matrix H
        if act == 'sig':
            H = tf.sigmoid(tf.nn.bias_add(tf.matmul(X,W), b, name='H'))
        elif act == 'relu':
            H = tf.nn.relu(tf.nn.bias_add(tf.matmul(X,W), b, name='H'))
        elif act == 'tanh':
            H = tf.nn.tanh(tf.nn.bias_add(tf.matmul(X,W), b, name='H'))        
        
        I = tf.eye(batchSize,dtype=tf.float32, name='I')                            
        Ireg = tf.eye(neurons,dtype=tf.float32, name='I')  
        
        # Op for init phase
        Pinit = tf.matrix_inverse(tf.matmul(H,H,True)-(reg*Ireg), name='Pinit')        
        
        # Ops for training phase
        PtrainAux1 = tf.matrix_inverse(tf.add(I,tf.matmul(tf.matmul(H,Pprev),H,False,True)), name='PtrainAux1')
        PtrainAux2 = tf.matmul(tf.matmul(Pprev,H,False,True),PtrainAux1, name='PtrainAux2')
        Ptrain = tf.subtract(Pprev,tf.matmul(tf.matmul(PtrainAux2,H),Pprev), name='Ptrain')  
        
        # Decision for the phase
        P = tf.cond(isInit, lambda: Pinit, lambda: Ptrain, name='P')                
        
        # Ops for init phase
        betaInit = tf.matmul(tf.matmul(Pinit,H,False,True),Y)
        
        # Ops for training phase
        betaTrainAux = tf.subtract(Y, tf.matmul(H,betaPrev), name='betaTrainAux')
        betaTrain = tf.add(betaPrev, tf.matmul(tf.matmul(Ptrain,H,False,True),betaTrainAux), name='betaTrain')
        
        # Decision for the phase
        beta = tf.cond(isInit, lambda: betaInit, lambda: betaTrain, name='Beta')
        
    return [beta,P,W,b,tf.matmul(H,H,True)]    

def oselmRun (X, params, act='sig'):
    beta, W, b = params[0], params[2], params[3]
    
    # Computing the matrix H
    if act == 'sig':
        H = tf.sigmoid(tf.nn.bias_add(tf.matmul(X,W), b, name='H'))
    elif act == 'relu':
        H = tf.nn.relu(tf.nn.bias_add(tf.matmul(X,W), b, name='H'))
    elif act == 'tanh':
        H = tf.nn.tanh(tf.nn.bias_add(tf.matmul(X,W), b, name='H'))  
    
    pred = tf.nn.softmax(tf.matmul(H,beta,name='Pred'))
    return pred

    
def rbm (X, numHid, batchSize=50): 
    numVis = int(X.get_shape()[1]) 
    
    vis0 = X
    
    with tf.name_scope('RBM'):
        visBias = tf.Variable(tf.zeros(shape=[numVis], dtype=tf.float32), name='visBias', trainable=False)
        hidBias = tf.Variable(tf.zeros(shape=[numHid], dtype=tf.float32), name='hidBias', trainable=False)
        weights = tf.Variable (tf.random_normal(shape=[numVis,numHid], mean=0.0, stddev=0.1, dtype=tf.float32), name='weights', trainable=False)       
        
        # Init as ELM
#        hidBias = tf.Variable(tf.random_uniform([numHid],-1,1,tf.float32), name='hidBias', trainable=False)       
        # Init as ELM
#        weights = tf.Variable (tf.random_uniform([numVis,numHid],-1,1,tf.float32), name='weights', trainable=False)      
        
        hid0 = tf.sigmoid(tf.matmul(vis0,weights) + hidBias)    
        vis1 = tf.matmul(hid0, weights, False, True) + visBias 
        hid1 = tf.sigmoid(tf.matmul(vis1,weights) + hidBias)                         

    return [hid0, hid1, vis0, vis1, visBias, hidBias, weights]             
    
def trainRbm (layers, lr=0.001, wc=0.0002, mom=0.7, batchSize=50):
    hid0, hid1, vis0, vis1 = layers[0], layers[1], layers[2], layers[3]
    visBias, hidBias, weights = layers[4], layers[5], layers[6]    
        
    with tf.name_scope('TrainRBM'):
        deltaWeights = tf.Variable(tf.zeros_like(weights, dtype=tf.float32), name='deltaWeights', trainable=False)       
        deltaVisBias = tf.Variable(tf.zeros_like(visBias, dtype=tf.float32), name='deltaVisBias', trainable=False)
        deltaHidBias = tf.Variable(tf.zeros_like (hidBias, dtype=tf.float32), name='deltaHidBias', trainable=False)     
   

    # Computing the value to update rules
    dw = tf.matmul(vis0,hid0,True) - tf.matmul(vis1,hid1,True)                
    dv = tf.reduce_sum(vis0, axis=0) - tf.reduce_sum(vis1, axis=0)
    #dv = dv[np.newaxis,:] # Just ajusting the numpy format
    dh = tf.reduce_sum(hid0, axis=0) - tf.reduce_sum(hid1, axis=0)
    
    attDeltaWeights = deltaWeights.assign((mom*deltaWeights) + (lr*dw/batchSize) - (wc*weights))
    attDeltaVisBias = deltaVisBias.assign((mom*deltaVisBias) + (lr*dv/batchSize))
    attDeltaHidBias = deltaHidBias.assign((mom*deltaHidBias) + (lr*dh/batchSize))
    
    # Updating the weights and bias
    attWeights = weights.assign_add(attDeltaWeights)
    attVisBias = visBias.assign_add(attDeltaVisBias)
    attHidBias = hidBias.assign_add(attDeltaHidBias)
    
    return [attWeights, attVisBias, attHidBias]    
    
#    sess.run([attDeltaWeights,attDeltaVisBias,attDeltaHidBias])                                     
#    sess.run([attWeights, attVisBias, attHidBias])       
    

          

        
      
#def evaluate (Y, Ypred):
#    correctPred = tf.equal(tf.argmax(Y,1), tf.argmax(Ypred,1))
#    acc = tf.reduce_mean(tf.cast(correctPred, tf.float32))    
#    loss = tf.losses.mean_squared_error(Y,Ypred)
#    
#    return acc, loss
#      
#def train (X,Y,model,loss='mse',optim='adam',lr=0.001):
#    
#    if loss=='mse':
#        lossOp = tf.losses.mean_squared_error(Y,model)
#        
#    if optim=='adam':
#        optimizerOp = tf.train.AdadeltaOptimizer(lr)
#        
#    trainOp = optimizerOp.minimize(lossOp)
#    return trainOp
    
        
        
##mninstIN = np.ones(shape=[100,28,28,1], dtype=np.float32)
##mninstOUT = np.random.randint(0,2,size=(100,10))
#lr = 0.001
#epochs = 1
#batchSize = 50
#
#X = tf.placeholder(tf.float32,(None,28,28,1), name='X')
#Y = tf.placeholder(tf.float32,(None), name='Y')
#
#conv1 = convLayer2D(X, kernel=[5,5,1], nFM=5, padd='VALID')
#pool1 = maxPooling(conv1)
#flatFM = flatLayer(pool1)
#
#
##fc1 = fcLayer(flatFM, 100, 'relu')
##cnnPred = fcLayer(fc1, 10, 'softmax')
#
#cnnPred = elm (flatFM, Y, 500, batchSize)
#
#
##
##with tf.Session() as sess:
##    sess.run(tf.global_variables_initializer())
##    x = sess.run(cnnPred, feed_dict={X: mninstIN, Y: mninstOUT})
##    print x.shape
#
#
##crossEntropy = tf.nn.softmax_cross_entropy_with_logits(cnnPred,Y)
##MSE = tf.reduce_mean(tf.square(Y-cnnPred), name='mse')
#lossOp = tf.losses.mean_squared_error(Y,cnnPred)
#optimizer = tf.train.AdamOptimizer(lr)
#trainOp = optimizer.minimize(lossOp)
#
#correctPred = tf.equal(tf.argmax(cnnPred,1), tf.argmax(Y,1))
#accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
#
#
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    nSam = xTrain.shape[0]
#    
#    for i in range(epochs):
#        xTrain, yTrain = shuffle(xTrain, yTrain)
#        
##        if i % 10 == 0:
##            trainAcc = sess.run(accuracy, feed_dict={X: mninstIN, Y: mninstOUT})
##            print 'step {}, training acc {}'.format(i,trainAcc)
#            
#        for offset in range(0,nSam,batchSize):
#            end = offset + batchSize
#            xBatch, yBatch = xTrain[offset:end], yTrain[offset:end]           
#            sess.run(trainOp, feed_dict={X: xBatch, Y: yBatch})
#            trainAcc = sess.run(accuracy, feed_dict={X: xBatch, Y: yBatch})
#            
#            if offset % 10 == 0:
#                print 'Batch {}/{}, training acc {}'.format(offset,nSam,trainAcc)    
#
#                
#            if offset % 100 == 0 and offset != 0:
#                break
#            
#    acc, testAcc = sess.run([cnnPred, accuracy], feed_dict={X: xTest, Y: yTest})
#    print 'Test acc {}'.format(testAcc)


    
    
    
    
    
    
    
    
    
    
