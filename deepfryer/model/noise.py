# -*- coding: utf-8 -*-
import numpy as np

def SaltPepper(X, rate=0.3):
	'''Adds salt and pepper noise to the data at a rate.
	-X: data to be corrupted.
	-rate: rate of noise'''
	drop = np.arange(X.shape[1])
	np.random.shuffle(drop)
	sep = int(len(drop)*rate)
	drop = drop[:sep]
	X[:, drop[:int(sep/2)]]=0
	X[:, drop[int(sep-sep/2):]]=1
	return X

def MaskingNoise(X, rate=0.5):
	'''Adds Masking noise to the data at a rate.
	-X: data to be corrupted.
	-rate: rate of noise'''
	mask = (np.random.uniform(0,1, X.shape)<rate).astype("float32")
	X = mask*X
	return X

def GaussianNoise(X, rate=0.5):
	'''Adds Gaussian noise to the data at a rate.
	-X: data to be corrupted.
	-rate: rate of noise'''
	X = X + rate * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
	X = np.clip(X, 0., 1.)
	return X
