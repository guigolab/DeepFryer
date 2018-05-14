# -*- coding: utf-8 -*-
import sys

import numpy as np
import pandas as pd

from scipy import stats as sst
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector


def calc_weights(model):
	'''Calculates the Linear contribution of weights extracted from a Deep Learning model built on Keras API.
	-Model: Keras model.'''
	model_layer = model.get_weights()
	W=1
	for i in range(len(model.get_weights())+1):
		if i%2 == 0:
			Wi = model_layer[-i]
			if i == 0:
				continue
			if i == 2:
				W = model_layer[-i]	
				continue
			W = np.matmul(Wi, W)
	sys.stderr.write('Weight matrix computed. \n')
	return W

def element_analysis(W, elements, class_list, top, alpha, path):
	'''Does the significance analysis on weights assuming normality in their distribution.
	-W: weight matrix
	-elements: list of features
	-class_list: list of classes.
	-top: select a top number of features
	-alpha: significance level
	-path: path to store the resulting csv of significant features.'''
	padjust=[]
	mat_pos = []
	stats = importr('stats')
	for cl in W.T:
		if top is None:
			top = W.shape[0]
		class_pos = np.argpartition(cl, -top)[-top:]
		mat_pos.append(class_pos)
		z_values = cl
		mean=z_values.mean()
		std=z_values.std()
		normal_distribution = sst.norm(loc=mean,scale=std)
		p_values = normal_distribution.cdf(z_values)
		p_adjust = stats.p_adjust(FloatVector(p_values), method = 'BH')
		p_adjust = np.array(p_adjust)
		padjust.append(p_adjust)
	df = pd.DataFrame(padjust)
	class_ele = []
	for cls, pos in zip(df.as_matrix(), mat_pos):
		sign = np.where((cls > 1-alpha))[0]
		ele_sign=[]
		for p in sign:
			if p in pos:
				ele_sign.append(p)
		if len(ele_sign) > 0:
			ele_sign = np.take(elements, ele_sign)
			class_ele.append(ele_sign)
		else:
			class_ele.append(np.array(ele_sign))
	df_class_ele = pd.DataFrame(class_ele).T.sort_index()
	df_class_ele.to_csv(path,sep=',', na_rep="", header = class_list, index=False)
	sys.stderr.write('Element analysis completed. \n')
	return df_class_ele

def perc_analysis(w, elements, class_list, perc, path):
	'''Select top percentile features depending on weights.
	-W: weight matrix
	-elements: list of features
	-class_list: classes.
	-perc: percentile threshold.
	-path: location of the resulting file. '''
	df_class = pd.DataFrame(columns=class_list)
	for cla in w.columns:
		cl = w.loc[:,cla]
		n = np.percentile(cl,perc)
		genes = np.array(((cl >= n)[(cl >=n) == True]).index)
		df_class.loc[:,cla] = genes
	df_class.to_csv(path,sep=',', na_rep="", header = class_list, index=False)
	sys.stderr.write('Element analysis completed. \n')
	return df_class

