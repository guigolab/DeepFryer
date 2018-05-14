# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

def set_prepare(x, y):
	'''Prepare sets for DL algorithm.
	-x: dataframe of values
	-y: dataframe of labels'''
	nb_classes = len(np.unique(y[0]))
	y[0] = y[0].astype('category').cat.codes
	y = np_utils.to_categorical(y[0], nb_classes)
	x_tmp = x.as_matrix()
	return x_tmp, y, nb_classes

def train_test_val(x_model, y_model, train_size=0.6 , test_size=0.2 , val_size=0.2 , strat = 0):
	'''Does stratified random split for train, test, and validation sets.
	-x_model: values for splitting.
	-y_model: labels for splitting.
	-train_size: proportion of data for training.
	-test_size: same for test.
	-val_size: same for validation.
	-strat: list of labels to stratify for.'''
	train_size = int((x_model.shape[0])*train_size)
	test_size = int((x_model.shape[0])*test_size)
	val_size = int((x_model.shape[0])*val_size)
	X_train, X_rest, y_train, y_rest = train_test_split(x_model, y_model, train_size=train_size, random_state = 0, stratify=y_model[strat])
	X_rest = X_rest.sort_index(axis=0)
	y_rest = y_rest.sort_index(axis=0)
	X_test, X_val, y_test, y_val = train_test_split(X_rest, y_rest, train_size=test_size, random_state = 0, stratify=y_rest[strat])
	X_train = X_train.sort_index(axis=0)
	X_test = X_test.sort_index(axis=0)
	X_val = X_val.sort_index(axis=0)
	y_train = y_train.sort_index(axis=0)
	y_test = y_test.sort_index(axis=0)
	y_val = y_val.sort_index(axis=0)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_val = X_val.astype('float32')
	return X_train, X_test, X_val, y_train, y_test, y_val