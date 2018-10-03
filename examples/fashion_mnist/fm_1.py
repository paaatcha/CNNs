#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
A CNN model to tackle the Fashion MNIST using tf.keras. It's just a first example inspired by 
https://colab.research.google.com/github/margaretmz/deep-learning/blob/master/fashion_mnist_keras.ipynb#scrollTo=UD1tecxUZZhE

Author: André Pacheco
Email: pacheco.comp@gmail.com

If you find any bug, please email-me
"""

import tensorflow as tf

# Load the fashion-mnist pre-shuffled train data and test data
(imgs_train, labs_train), (imgs_test, labs_test) = tf.keras.datasets.fashion_mnist.load_data()

#print ("###### Data summary ######")
#print ("Image shape: ", imgs_train.shape[1:3])
#print ("Training samples: ", imgs_train.shape[0])
#print ("Test samples: ", imgs_test.shape[0])
#print ("##########################")
       
# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9


# Normalizeing and spliting the dataset
imgs_train = imgs_train / 255.0
imgs_test = imgs_test / 255.0

# Creating the val set from the train set
(imgs_train, imgs_val) = imgs_train[5000:], imgs_train[:5000] 
(labs_train, labs_val) = labs_train[5000:], labs_train[:5000]

# Putting the images in TF format, that is, (batch_size, w, h, channels)
imgs_train = imgs_train.reshape(imgs_train.shape[0], 28, 28, 1)
imgs_val = imgs_val.reshape(imgs_val.shape[0], 28, 28, 1)
imgs_test = imgs_test.reshape(imgs_test.shape[0], 28, 28, 1)

# Creating the one hot encode labels
labs_train = tf.keras.utils.to_categorical(labs_train, 10)
labs_val = tf.keras.utils.to_categorical(labs_val, 10)
labs_test = tf.keras.utils.to_categorical(labs_test, 10)


print ("###### Data summary ######")
print ("Train - Input:", imgs_train.shape, "Outuput: ", labs_train.shape)
print ("Test - Input:", imgs_test.shape, "Outuput: ", labs_test.shape)
print ("Val - Input:", imgs_val.shape, "Outuput: ", labs_val.shape)
print ("##########################")




# Creating the model using tf.keras

model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Take a look at the model summary
model.summary()

# Choosing the loss function, optimizer and metrics
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# Defining a checkpoint to save the best model
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.hdf5', verbose = 1, save_best_only=True)

# Including tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=5, batch_size=64, write_graph=False, write_images=True)

model.fit(imgs_train,
         labs_train,
         batch_size=64,
         epochs=10,
         validation_data=(imgs_val, labs_val),
         callbacks=[checkpointer, tensorboard])


# Load the weights with the best validation accuracy
model.load_weights('model.weights.best.hdf5')

# Evaluate the model on test set
score = model.evaluate(imgs_test, labs_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])






