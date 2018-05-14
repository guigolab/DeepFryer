# -*- coding: utf-8 -*-
import sys

import pandas as pd
import numpy as np

import copy

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import KernelPCA
from sklearn.kernel_ridge import KernelRidge

from .plot import hier_heatmap

def assign_val(dataset, valueset, var, name):
	''' It takes two datasets and map values from one to the other.
		-dataset: Pandas DataFrame to where the values are going to be added.
		-valueset: Pandas DataFrame to where the values are going to be taken.
		-var: String of the value taken in valueset
		-name: String. New name of the column in dataset. If the name is already in the Dframe it will overwrite values.'''
	if dataset.index[0] in valueset.index:
		dataset.loc[dataset.index, name] = valueset.loc[dataset.index, var]
	else:	
		dataset.loc[:,'SUBID'] = np.array([i.split('-')[0]+'-'+i.split('-')[1] for i in dataset.index])
		dataset.loc[:,name] = valueset.loc[dataset['SUBID'], var].values
		dataset = dataset.drop('SUBID', axis = 1)
	sys.stderr.write(str(var)+' values assigned.\n')
	return dataset

def linear_cov_reads_mat(dset, labels, meta, var, cov_matrix):
	'''Calculates linear covariates for a Dataset based on nucleotide resolution RNA-seq
		-dset: Pandas DataFrame with RNA-seq expression values.
		-labels: DataFrame used for assigning variables on meta.
		-meta: DataFrame of potential covariates.
		-cov_matrix: Dataframe were store R^2 values.'''
	y_model = copy.deepcopy(dset)
	x_model = copy.deepcopy(labels)
	cov = copy.deepcopy(meta)
	cov_list = cov.columns
	#x_model = x_model[x_model[var] == tissue]
	y_model = y_model[y_model.index.isin(x_model.index)]
	pca = PCA(n_components = 10, random_state = 0)
	pc = pca.fit_transform(y_model)
	x_=copy.deepcopy(x_model)
	for w in cov_list:
		#sys.stderr.write(tissue+" "+str(w)+"\n")
		x_model = copy.deepcopy(x_)
		covariate = pd.DataFrame(cov.loc[:,w])
		if w.startswith('MH') and (cov[w].dtype == 'float64'):
			covariate[w] = covariate.loc[:,w].astype('category').cat.codes
			x_model = assign_val(x_model, covariate,w, 0)
			x_model = pd.DataFrame(x_model.loc[:,0])
			x_model = pd.get_dummies(x_model)
			lm = LinearRegression()
			lm.fit(x_model, pc)
			r2 = lm.score(x_model, pc)
			cov_matrix.loc[w, tissue] = r2
		elif covariate[w].dtype == object:
			covariate[w] = covariate.loc[:,w].astype('category').cat.codes
			x_model = assign_val(x_model, covariate,w, 0)
			x_model = pd.DataFrame(x_model.loc[:,0])
			x_model = pd.get_dummies(x_model)
			lm = LinearRegression()
			lm.fit(x_model, pc)
			r2 = lm.score(x_model, pc)
			cov_matrix.loc[w, tissue] = r2
		elif covariate[w].dtype == 'int64' and w != 'AGE':
			covariate[w] = covariate.loc[:,w].astype('category').cat.codes
			x_model = assign_val(x_model, covariate,w, 0)
			x_model = pd.DataFrame(x_model.loc[:,0])
			x_model = pd.get_dummies(x_model)
			lm = LinearRegression()
			lm.fit(x_model, pc)
			r2 = lm.score(x_model, pc)
			cov_matrix.loc[w, tissue] = r2
		else:
			x_model = assign_val(x_model, covariate,w, 0)
			x_model = pd.DataFrame(x_model.loc[:,0])
			if x_model[0].max() != 0.0:
				x_model = x_model/x_model.max()
			lm = LinearRegression()
			lm.fit(x_model.values.reshape(-1,1), pc)
			r2 = lm.score(x_model.values.reshape(-1,1), pc)
			cov_matrix.loc[w, tissue] = r2
	return cov_matrix

def linear_covariate_mat(dset, labels, meta, tissue, cov_matrix):
	'''Calculates linear covariates for a Dataset based on gene expression data.
		-dset: Pandas DataFrame with RNA-seq expression values.
		-labels: DataFrame used for assigning variables on meta.
		-meta: DataFrame of potential covariates.
		-cov_matrix: Dataframe were store R^2 values.'''
	y_model = copy.deepcopy(dset)
	x_model = copy.deepcopy(labels)
	cov = copy.deepcopy(meta)
	cov_list = cov.columns
	x_model = x_model[x_model[0] == tissue]
	y_model = y_model[y_model.index.isin(x_model.index)]
	pca = PCA(n_components = 10, random_state = 0)
	pc = pca.fit_transform(y_model)
	x_=copy.deepcopy(x_model)
	for w in cov_list:
		sys.stderr.write(tissue +" "+w+"\n")
		x_model = copy.deepcopy(x_)
		covariate = pd.DataFrame(cov.loc[:,w])
		if w.startswith('MH') and (cov[w].dtype == 'float64'):
			covariate[w] = covariate.loc[:,w].astype('category').cat.codes
			x_model = assign_val(x_model, covariate,w, 0)
			x_model = pd.get_dummies(x_model)
			lm = LinearRegression()
			lm.fit(x_model, pc)
			r2 = lm.score(x_model, pc)
			cov_matrix.loc[w, tissue] = r2
		elif covariate[w].dtype == object:
			covariate[w] = covariate.loc[:,w].astype('category').cat.codes
			x_model = assign_val(x_model, covariate,w, 0)
			x_model = pd.get_dummies(x_model)
			lm = LinearRegression()
			lm.fit(x_model, pc)
			r2 = lm.score(x_model, pc)
			cov_matrix.loc[w, tissue] = r2
		elif covariate[w].dtype == 'int64' and w != 'AGE':
			covariate[w] = covariate.loc[:,w].astype('category').cat.codes
			x_model = assign_val(x_model, covariate,w, 0)
			x_model = pd.get_dummies(x_model)
			lm = LinearRegression()
			lm.fit(x_model, pc)
			r2 = lm.score(x_model, pc)
			cov_matrix.loc[w, tissue] = r2
		else:
			x_model = assign_val(x_model, covariate,w, 0)
			if x_model[0].max() != 0.0:
				x_model = x_model/x_model.max()
			lm = LinearRegression()
			lm.fit(x_model.values.reshape(-1,1), pc)
			r2 = lm.score(x_model.values.reshape(-1,1), pc)
			cov_matrix.loc[w, tissue] = r2
	return cov_matrix

def linear_nonlinear_covariate_mat(dset, labels, meta, tissue, cov_matrix):
	'''Calculates non-linear covariates per tissue for a Dataset based on nucleotide resolution RNA-seq
		-dset: Pandas DataFrame with RNA-seq expression values.
		-labels: DataFrame used for assigning variables on meta.
		-meta: DataFrame of potential covariates.
		-cov_matrix: Dataframe were store R^2 values.
		-tissue: class used.'''
	y_model = copy.deepcopy(dset)
	x_model = copy.deepcopy(labels)
	cov = copy.deepcopy(meta)
	cov_list = cov.columns
	x_model = x_model[x_model[0] == tissue]
	y_model = y_model[y_model.index.isin(x_model.index)]
	pca = PCA(n_components = 10,  random_state = 0)
	pc = pca.fit_transform(y_model)
	x_=copy.deepcopy(x_model)
	for w in cov_list:
		sys.stderr.write(tissue +" "+w+"\n")
		x_model = copy.deepcopy(x_)
		covariate = pd.DataFrame(cov.loc[:,w])
		if w.startswith('MH') and (cov[w].dtype == 'float64'):
			covariate[w] = covariate.loc[:,w].astype('category').cat.codes
			x_model = assign_val(x_model, covariate,w, 0)
			x_model = pd.get_dummies(x_model)
			lm = KernelRidge(alpha=1, kernel='rbf')
			lm.fit(x_model, pc)
			r2 = lm.score(x_model, pc)
			cov_matrix.loc[w, tissue] = r2
		elif covariate[w].dtype == object:
			covariate[w] = covariate.loc[:,w].astype('category').cat.codes
			x_model = assign_val(x_model, covariate,w, 0)
			x_model = pd.get_dummies(x_model)
			lm = KernelRidge(alpha=1, kernel='rbf')
			lm.fit(x_model, pc)
			r2 = lm.score(x_model, pc)
			cov_matrix.loc[w, tissue] = r2
		elif covariate[w].dtype == 'int64' and w != 'AGE':
			covariate[w] = covariate.loc[:,w].astype('category').cat.codes
			x_model = assign_val(x_model, covariate,w, 0)
			x_model = pd.get_dummies(x_model)
			lm = KernelRidge(alpha=1, kernel='rbf')
			lm.fit(x_model, pc)
			r2 = lm.score(x_model, pc)
			cov_matrix.loc[w, tissue] = r2
		else:
			x_model = assign_val(x_model, covariate,w, 0)
			if x_model[0].max() != 0.0:
				x_model = x_model/x_model.max()
			lm = KernelRidge(alpha=1, kernel='rbf')
			lm.fit(x_model.values.reshape(-1,1), pc)
			r2 = lm.score(x_model.values.reshape(-1,1), pc)
			cov_matrix.loc[w, tissue] = r2
	return cov_matrix

def nonlinear_covariate_mat(dset, labels, meta, tissue, cov_matrix):
	'''Calculates non-linear covariates for a Dataset based on nucleotide resolution RNA-seq
		-dset: Pandas DataFrame with RNA-seq expression values.
		-labels: DataFrame used for assigning variables on meta.
		-meta: DataFrame of potential covariates.
		-cov_matrix: Dataframe were store R^2 values.
		-tissue: class used'''
	y_model = copy.deepcopy(dset)
	x_model = copy.deepcopy(labels)
	cov = copy.deepcopy(meta)
	cov_list = cov.columns
	x_model = x_model[x_model[0] == tissue]
	y_model = y_model[y_model.index.isin(x_model.index)]
	pca = KernelPCA(n_components = None, kernel = 'rbf',  random_state = 0)
	pc = pca.fit_transform(y_model)
	x_=copy.deepcopy(x_model)
	for w in cov_list:
		sys.stderr.write(tissue +" "+w+"\n")
		x_model = copy.deepcopy(x_)
		covariate = pd.DataFrame(cov.loc[:,w])
		if w.startswith('MH') and (cov[w].dtype == 'float64'):
			covariate[w] = covariate.loc[:,w].astype('category').cat.codes
			x_model = assign_val(x_model, covariate,w, 0)
			x_model = pd.get_dummies(x_model)
			lm = KernelRidge(alpha=1, kernel='rbf')
			lm.fit(x_model, pc)
			r2 = lm.score(x_model, pc)
			cov_matrix.loc[w, tissue] = r2
		elif covariate[w].dtype == object:
			covariate[w] = covariate.loc[:,w].astype('category').cat.codes
			x_model = assign_val(x_model, covariate,w, 0)
			x_model = pd.get_dummies(x_model)
			lm = KernelRidge(alpha=1, kernel='rbf')
			lm.fit(x_model, pc)
			r2 = lm.score(x_model, pc)
			cov_matrix.loc[w, tissue] = r2
		elif covariate[w].dtype == 'int64' and w != 'AGE':
			covariate[w] = covariate.loc[:,w].astype('category').cat.codes
			x_model = assign_val(x_model, covariate,w, 0)
			x_model = pd.get_dummies(x_model)
			lm = KernelRidge(alpha=1, kernel='rbf')
			lm.fit(x_model, pc)
			r2 = lm.score(x_model, pc)
			cov_matrix.loc[w, tissue] = r2
		else:
			x_model = assign_val(x_model, covariate,w, 0)
			if x_model[0].max() != 0.0:
				x_model = x_model/x_model.max()
			lm = KernelRidge(alpha=1, kernel='rbf')
			lm.fit(x_model.values.reshape(-1,1), pc)
			r2 = lm.score(x_model.values.reshape(-1,1), pc)
			cov_matrix.loc[w, tissue] = r2
	return cov_matrix

def corr_mat(data, labels, title, met, path):
	'''Computes Correlation Matrix across your data.
	-data: Pandas DataFrame with the values to correlate.
	-labels: list of labels.
	-title: Title for the plot
	-met: Metric used
	-path: Directory to save plot.'''
	corr = pd.DataFrame(np.corrcoef(data), index = labels, columns = labels)
	hier_heatmap(corr, corr.columns, corr.index, title, met, path)
	return corr