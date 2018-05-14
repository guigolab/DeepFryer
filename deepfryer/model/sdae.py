# -*- coding: utf-8 -*-
import sys
import copy
import tensorflow as tf

from .noise import MaskingNoise, GaussianNoise, SaltPepper

from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
from keras.utils import multi_gpu_model
from keras import backend as K


def cpu_pretrain_SAE(x_train, nb_layers, batch, epochs, nrate=0.25, noise='mask', activation='tanh', optimizer='rmsprop', loss='mean_squared_error'):
	'''Implementation of pretraining of SDAE in CPU.
	-x_train: training set.
	-nb_layers: list of three elements. Number of neurons in the three layers of the AE.
	-batch: Size of the batch
	-epochs: number of iterations
	-nrate: portion of noise added.
	-noise: type of noise.
	-activation: activation function for the hidden layers.
	-optimizer: optimizer for the neural network.
	-loss: loss function used.'''
	trained_encoders = []
	input_dim = x_train.shape
	nb_layers.insert(0,input_dim[1])
	if noise == 'mask':
		x_noisy = MaskingNoise(x_train, nrate)
	elif noise == 'gaussian':
		x_noisy = GaussianNoise(x_train, nrate)
	elif noise == 'salt':
		x_noisy = SaltPepper(x_train, nrate)
	elif noise == 'clean':
		x_noisy = x_train
		nrate = None
	for n_in, n_out in zip(nb_layers[:-1], nb_layers[1:]):
		sys.stdout.write('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
		# Create AE and training
		ae = Sequential()
		ae.add(Dense(int(n_out),input_dim=int(n_in), activation = activation))
		ae.add(Dense(int(n_in), activation = activation))
		ae.compile(loss=loss, optimizer=optimizer)
		ae.fit(x_noisy, x_train, batch_size=int(batch), epochs=int(epochs))
		# Store trained weight
		trained_encoders.append(ae.layers[0])
		# Compute the transformation of training data
		encoded_output = K.function([ae.layers[0].input], [ae.layers[0].output]) 
		# Update training data
		x_train = encoded_output([x_train])[0]
		x_noisy = copy.deepcopy(x_train)
	return trained_encoders, nb_layers[-1]

def cpu_fine_tuning_ANN(trained_encoders, nb_layers, x_train, y_train, x_test, y_test, x_val, y_val, nb_classes,batch, epochs, activation='softmax', optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics='categorical_accuracy'):
	'''Fine tunning of an SDAE pretrained on CPU.
	-trained_encoders: list containing encoding layers as output of cpu_pretrain_convDAE function.
	-nb_layers: for the size of the Fully connected layer
	-x_train: training set.
	-y_train: labels of training set.
	-x_test: testing set.
	-y_test: labels of testing set.
	-x_val: validation set.
	-y_val: labels of validation set
	-nb_classes: number of classes.
	-batch: Size of the batch
	-epochs: number of iterations
	-nrate: portion of noise added.
	-noise: type of noise.
	-activation: activation function used on output layer.
	-optimizer: optimizer for the neural network.
	-loss: loss function used.'''
	sys.stdout.write('Fine-tuning on ANN')
	model = Sequential()
	for encoder in trained_encoders:
		model.add(encoder)
	#model.add(BatchNormalization(axis = 1, epsilon = 0.000001))
	model.add(Dense(int(nb_layers),input_dim=int(nb_layers), activation="relu"))
	model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
	model.fit(x_train, y_train, batch_size=int(batch), epochs=int(epochs), validation_data=(x_val, y_val))#, callbacks=[callback])
	score = model.evaluate(x_test, y_test, verbose=0)
	sys.stdout.write('Test score ANN: '+str(score[0]))
	sys.stdout.write('Test accuracy ANN: '+str(score[1]))
	return model, score[1]

def gpu_pretrain_SAE(x_train, nb_layers, batch, epochs, ngpu=4, nrate = 0.25, noise='mask', activation='tanh', optimizer='rmsprop', loss='mean_squared_error'):
	'''Implementation of pretraining of SDAE in GPU.
	-x_train: training set.
	-nb_layers: list of three elements. Number of neurons in the three layers of the AE.
	-batch: Size of the batch
	-epochs: number of iterations
	-ngpu: number of gpu used for data parallelization.
	-nrate: portion of noise added.
	-noise: type of noise.
	-activation: activation function for the hidden layers.
	-optimizer: optimizer for the neural network.
	-loss: loss function used.'''
	trained_encoders = []
	input_dim = x_train.shape
	nb_layers.insert(0,input_dim[1])
	if noise == 'mask':
		x_noisy = MaskingNoise(x_train, nrate)
	elif noise == 'gaussian':
		x_noisy = GaussianNoise(x_train, nrate)
	elif noise == 'salt':
		x_noisy = SaltPepper(x_train, nrate)
	elif noise == 'clean':
		x_noisy = copy.deepcopy(x_train)
		nrate = None
	for n_in, n_out in zip(nb_layers[:-1], nb_layers[1:]):
		with tf.device('/cpu:0'):
			sys.stdout.write('Pre-training the layer: Input {} -> Output {} \n'.format(n_in, n_out))
			# Create AE and training
			ae = Sequential()
			ae.add(Dense(int(n_out),input_dim=int(n_in), activation = activation))
			#ae.add(Dropout(0.2, None, 0))
			ae.add(Dense(int(n_in), activation = activation))
		parallel_model = multi_gpu_model(ae, gpus=ngpu)
		parallel_model.compile(loss=loss, optimizer=optimizer)
		parallel_model.fit(x_noisy, x_train, batch_size=int(batch)*ngpu, epochs=int(epochs))
		# Store trained weight
		trained_encoders.append(ae.layers[0])
		# Compute the transformation of training data
		encoded_output = K.function([ae.layers[0].input], [ae.layers[0].output]) 
		# Update training data
		x_train = encoded_output([x_train])[0]
		x_noisy = copy.deepcopy(x_train)
	return trained_encoders, nb_layers[-1]

def gpu_fine_tuning_ANN(trained_encoders,nb_layers, x_train, y_train, x_test, y_test, x_val, y_val, nb_classes,batch, epochs, ngpu = 4,  activation='softmax', optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics='categorical_accuracy'):
	'''Fine tunning of an SDAE pretrained on GPU.
	-trained_encoders: list containing encoding layers as output of cpu_pretrain_convDAE function.
	-nb_layers: for the size of the Fully connected layer
	-x_train: training set.
	-y_train: labels of training set.
	-x_test: testing set.
	-y_test: labels of testing set.
	-x_val: validation set.
	-y_val: labels of validation set
	-nb_classes: number of classes.
	-batch: Size of the batch
	-epochs: number of iterations
	-ngpu: number of gpu used for data parallelization.
	-nrate: portion of noise added.
	-noise: type of noise.
	-activation: activation function used on output layer.
	-optimizer: optimizer for the neural network.
	-loss: loss function used.'''
	sys.stdout.write('Fine-tuning on ANN \n')
	with tf.device('/cpu:0'):
		model = Sequential()
		for encoder in trained_encoders:
			model.add(encoder)
		#model.add(BatchNormalization(trainable=True))
		model.add(Dense(int(nb_layers),input_dim=int(nb_layers), activation="relu"))
		model.add(Dense(nb_classes,input_dim=int(nb_layers), activation=activation))
	parallel_model = multi_gpu_model(model, gpus=None)
	parallel_model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
	parallel_model.fit(x_train, y_train, batch_size=int(batch)*ngpu, epochs=int(epochs), validation_data=(x_val, y_val))
	score = parallel_model.evaluate(x_test, y_test, batch_size=int(batch)*ngpu, verbose=0)
	sys.stdout.write('Test score ANN: '+str(score[0])+'\n')
	sys.stdout.write('Test accuracy ANN: '+str(score[1])+'\n')
	return model, score[1]