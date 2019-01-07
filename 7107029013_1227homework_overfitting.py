#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:15:17 2018
@author: rock

'''four Optimization '''

"""
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten , Dropout
from keras.optimizers import Adam
import keras
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')[:10000]
X_test = X_test.astype('float32')[:10000]
y_train = y_train[:10000]
y_test = y_test[:10000]
# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28)/255. #正規化
X_test = X_test.reshape(-1, 1,28, 28)/255.   #正規化
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


print("Use CNN Method(relu):\n")

   
# build your CNN
model = Sequential()
'''
#processing step

input:1*28*28 
-->Convolution(output:25*26*26) filter=25
-->MaxPooling(25*13*13) #find max value(matrix 2*2)
-->Convolution(output:50*11*11)#使用前一個MaxPooling輸出結果繼續做Convolution，並重新設定filter數量(50)
-->MaxPooling(50*5*5)   #find max value(matrix 2*2)

'''
# Conv layer 1 output shape (25, 26, 26)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=25,#濾波器找特徵
    kernel_size=3,#濾波器大小3*3
    strides=1,#間格 1
    padding='same',# Padding method 不更改長寬大小
    data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (25, 13, 13)
model.add(MaxPooling2D(
    pool_size=2,#大小２＊２
    strides=1,#間格1
    padding='same',#Padding method 不更改長寬大小
    data_format='channels_first',
))


#Conv layer 2 output shape (50, 11, 11)
#Convolution2D(filter,pixel,strides, padding='same', data_format='channels_first')
model.add(Convolution2D(50, 3, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (50, 5, 5)
model.add(MaxPooling2D(2, 1, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (50 * 5 * 5) = (1250), output shape (1024)
model.add(Flatten()) #壓縮成一維
model.add(Dense(1024))
model.add(Activation('relu'))


# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))#分類

# Another way to define your optimizer
adam = Adam(lr=1e-4)
# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
print('降低資料量(10000筆)達成 overfitting')
# Another way to train the model
#（2）batch-size：1次迭代所使用的样本量；
# (3）epoch：1个epoch表示过了1遍训练集中的所有样本。
#https://www.zhihu.com/question/43673341
model.fit(X_train, y_train, epochs=5, batch_size=1000,verbose=1,validation_data=(X_test, y_test))

'''====================================================================================''' 

# build your CNN
model = Sequential()

# Conv layer 1 output shape (25, 26, 26)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=25,#濾波器找特徵
    kernel_size=3,#濾波器大小3*3
    strides=1,#間格 1
    padding='same',# Padding method 不更改長寬大小
    data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (25, 13, 13)
model.add(MaxPooling2D(
    pool_size=2,#大小２＊２
    strides=1,#間格1
    padding='same',#Padding method 不更改長寬大小
    data_format='channels_first',
))


#Conv layer 2 output shape (50, 11, 11)
#Convolution2D(filter,pixel,strides, padding='same', data_format='channels_first')
model.add(Convolution2D(50, 3, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (50, 5, 5)
model.add(MaxPooling2D(2, 1, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (50 * 5 * 5) = (1250), output shape (1024)
model.add(Dropout(0.25))
model.add(Flatten()) #壓縮成一維
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Activation('relu'))


# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))#分類

# Another way to define your optimizer
adam = Adam(lr=1e-4)
# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
print("使用 Dropout 達成 overfitting")

# Another way to train the model
#（2）batch-size：1次迭代所使用的样本量；
# (3）epoch：1个epoch表示过了1遍训练集中的所有样本。
#https://www.zhihu.com/question/43673341
model.fit(X_train, y_train, epochs=5, batch_size=1000,verbose=1,validation_data=(X_test, y_test))








































