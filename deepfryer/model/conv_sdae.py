#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:13:26 2018

@author: mcamara
"""
import sys

import tensorflow as tf

from .noise import MaskingNoise, GaussianNoise, SaltPepper

from keras.layers import Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten
from keras.models import Sequential
from keras.utils import multi_gpu_model


def gpu_pretrain_convDAE(x_train, nb_layers, batch, epochs, ngpu, nrate=0.25, noise='mask', optimizer='rmsprop', loss='mean_squared_error'):
	'''GPU pretraining of an Deep Convolutional AutoEncoder.
	-x_train: training set.
	-nb_layers: list of three elements. Number of neurons in the three layers of the AE.
	-batch: Size of the batch
	-epochs: number of iterations
	-ngpu: number of gpu used for data parallelization.
	-nrate: portion of noise added.
	-noise: type of noise.
	-optimizer: optimizer for the neural network.
	-loss: loss function used.'''
	
	trained_encoders=[]
	input_dim = x_train.shape
	if noise == 'mask':
		x_noisy = MaskingNoise(x_train, nrate)
	elif noise == 'gaussian':
		x_noisy = GaussianNoise(x_train, nrate)
	elif noise == 'salt':
		x_noisy = SaltPepper(x_train, nrate)
	elif noise == 'clean':
		x_noisy = x_train
		nrate = None
	#encoder
	with tf.device('/cpu:0'):
		ae = Sequential()
		ae.add(MaxPooling1D(125, input_shape=(input_dim[1],2), padding='same'))
		ae.add(Conv1D(nb_layers[0], 3, activation='relu', padding='same'))
		ae.add(MaxPooling1D(2, padding='same'))
		ae.add(Conv1D(nb_layers[1], 3, activation='relu', padding='same'))
		ae.add(MaxPooling1D(2, padding='same'))
		ae.add(Conv1D(nb_layers[2], 3, activation='relu', padding='same'))
		ae.add(MaxPooling1D(2, padding='same'))
		#decoder
		ae.add(Conv1D(nb_layers[2], 3, activation='relu', padding='same'))
		ae.add(UpSampling1D(2))
		ae.add(Conv1D(nb_layers[1], 3, activation='relu', padding='same'))
		ae.add(UpSampling1D(2))
		ae.add(Conv1D(nb_layers[0], 3, activation='relu', padding ='same'))
		ae.add(UpSampling1D(2))
		ae.add(Conv1D(2, (int(input_dim[1]/125)), activation='sigmoid', padding='same'))
		ae.add(UpSampling1D(125))
	parallel_model = multi_gpu_model(ae, gpus=ngpu)
	parallel_model.compile(loss=loss, optimizer=optimizer)
	parallel_model.fit(x_noisy, x_train, batch_size=int(batch)*ngpu, epochs=int(epochs))
	for i in range(7):
		trained_encoders.append(ae.layers[i])
	return trained_encoders, nb_layers[-1]


def gpu_fine_tuning_convANN(trained_encoders, x_train, y_train, x_test, y_test, nb_classes, batch, epochs, ngpu, activation='softmax', optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics='categorical_accuracy'):
	'''Fine tunning of an Deep Convolutional AutoEncoder pretrained.
	-trained_encoders: list containing encoding layers as output of gpu_pretrain_convDAE function.
	-x_train: training set.
	-y_train: labels of training set.
	-x_test: testing set.
	-y_test: labels of testing set.
	-nb_classes: number of classes.
	-batch: Size of the batch
	-epochs: number of iterations
	-ngpu: number of gpu used for data parallelization.
	-nrate: portion of noise added.
	-noise: type of noise.
	-activation: activation function used on output layer.
	-optimizer: optimizer for the neural network.
	-loss: loss function used.'''
	sys.stdout.write('Fine-tuning on ANN')
	with tf.device('/cpu:0'):
		model = Sequential()
		for encoder in trained_encoders:
			model.add(encoder)
		model.add(Flatten())
		model.add(Dense(1000, activation='relu'))
		model.add(Dense(nb_classes, activation=activation))
	model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
	model.fit(x_train, y_train, batch_size=int(batch), epochs=int(epochs), validation_data=(x_test, y_test))#, callbacks=[callback])
	score = model.evaluate(x_test, y_test, verbose=0)
	sys.stdout.write('Test score ANN: '+str(score[0])+'\n')
	sys.stdout.write('Test accuracy ANN: '+str(score[1]+'\n'))
	return model, score[1]