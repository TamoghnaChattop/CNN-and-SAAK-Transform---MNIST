# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 19:04:40 2018

@author: tchat
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from keras.utils import np_utils
from keras.datasets import mnist
from pylab import *
import numpy as np
import tensorflow
import time
from keras.utils.data_utils import get_file
from time import localtime, strftime
from keras.callbacks import TensorBoard
np.random.seed(123)

# Loading the training, validation and testing data
batch_size = 256
num_classes = 10
epochs = 200

# Load training and eval data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Returns np.array
X_train = mnist.train.images 
y_train = np.asarray(mnist.train.labels, dtype=np.int32)
# Returns np.array
X_test = mnist.test.images 
y_test = np.asarray(mnist.test.labels, dtype=np.int32)
# Returns np.array
X_valid = mnist.validation.images 
y_valid = np.asarray(mnist.validation.labels, dtype=np.int32)

# define LeNet5 model
def base_model():    
    
    model = Sequential()
    model.add(Reshape((28,28,1), input_shape=(784,)))
    model.add(Conv2D(30, kernel_size=(5, 5), strides=(1, 1), 
                     activation='relu',padding='SAME', use_bias=True, kernel_initializer='RandomNormal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu', use_bias=True))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', use_bias=True))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu', use_bias=True))
    model.add(Dense(num_classes, activation='softmax'))
    # define optimizer
    #sgd = optimizers.SGD(lr=0.001, momentum=0.5)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer=adam, metrics=['accuracy'])
    return model

# build the model
model = base_model()
# print model information
print(model.summary())
print('Training the LeNet 5 Model')
# Fit the model
start = time.time()
ts = localtime()

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
                    epochs=epochs, batch_size=batch_size, verbose=2) 
end = time.time()
print ("\nModel took %0.2f seconds to train\n"%(end - start))
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=2)
model.save_weights("model_rn.h5", overwrite=True)
model.save('my_model_rn.h5', overwrite=True)
# save as JSON
json_string = model.to_json()

print('Test Error: %.4f%%' % (100-scores[1]*100))
print('Test Loss: %.4f%%' % scores[0])
print('Test Accuracy: %.4f%%' % (scores[1]*100))

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(12,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')

    axs[0].legend(['Training', 'Validation'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')

    axs[1].legend(['Training', 'Validation'], loc='best')
    plt.show()
    
plot_model_history(history)


def PlotHistory(train_value, test_value, value_is_loss_or_acc):
    f, ax = plt.subplots()
    ax.plot([None] + train_value, 'o-')
    ax.plot([None] + test_value, 'x-')
    
    # Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Train ' + value_is_loss_or_acc, 'Validation ' + value_is_loss_or_acc], loc = 'best') 
    ax.set_title('Training/Validation ' + value_is_loss_or_acc + ' per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(value_is_loss_or_acc)  
    
PlotHistory(history.history['loss'], history.history['val_loss'], 'Loss')
PlotHistory(history.history['acc'], history.history['val_acc'], 'Accuracy')    
	