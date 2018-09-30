#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model

from sklearn.model_selection import train_test_split

import utils

class Model():

	def __init__(self):
		self.learning_rate = 0.01
		self.batch_size = 64
		self.epochs = 10

	def train_test_split(self, all_images, data_list):
		x_train, x_test, y_train, y_test = train_test_split(all_images, data_list, test_size=0.33, random_state=42)
		x_train = np.asarray(x_train)
		x_test = np.asarray(x_test)
		y_train = np.asarray(y_train)
		y_test = np.asarray(y_test)

		shape = x_train.shape

		if K.image_data_format() == 'channels_first': # channels, rows, cols
		  x_train = x_train.reshape(x_train.shape[0], 1, shape[1], shape[2])
		  x_test = x_test.reshape(x_test.shape[0], 1, shape[1], shape[2])
		  input_shape = (1, shape[1], shape[2])
		else:
		  x_train = x_train.reshape(x_train.shape[0], shape[1], shape[2], 1) # rows, cols, channels
		  x_test = x_test.reshape(x_test.shape[0], shape[1], shape[2], 1)
		  input_shape = (shape[1], shape[2], 1)

		return x_train, x_test, y_train, y_test, input_shape

	def model(self, input_shape):
		model = Sequential()
		model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2), input_shape=input_shape))
		model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
		model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
		model.add(Conv2D(64, 3, 3, activation='elu'))
		model.add(Conv2D(64, 3, 3, activation='elu'))
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(100, activation='elu'))
		model.add(Dense(50, activation='elu'))
		model.add(Dense(10, activation='elu'))
		model.add(Dense(3))
		model.summary()

		return model

	def model_fit(self, model, x_train, y_train, x_test, y_test, n):																																																						
		model.compile(loss='mean_squared_error', 
			optimizer=Adam(lr=self.learning_rate), 
			metrics=['accuracy'])

		model.fit(x_train, y_train,
			verbose=1,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(x_test, y_test))
		score = model.evaluate(x_test, y_test, verbose=0)
		print('Test loss: ', score[0])
		print('Test accuracy: ', score[1])

		self.save_model(model)

	def save_model(self, model, name):
		model.save('saved_models/my_model2.h5')  # creates a HDF5 file 'my_model.h5'

def main():

	dataname = 'testedois'
	mypath = '/home/luiza/Documents/autocone_dataset/'

	p = utils.Preprocess(path=mypath, dataname=dataname)
	all_images, data_list, n = p.return_data()

	m = Model()
	x_train, x_test, y_train, y_test, input_shape = m.train_test_split(all_images, data_list.iloc[:, 1:4])
	print(input_shape, x_train.shape, y_train, y_train.shape)
	model = m.model(input_shape)
	m.model_fit(model, x_train, y_train, x_test, y_test, n)


main()