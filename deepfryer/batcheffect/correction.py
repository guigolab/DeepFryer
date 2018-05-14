# -*- coding: utf-8 -*-
import copy
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


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

def correct_linear_cov(dset, labels, meta, cov_list):
	'''It takes dset values and corrects the effect of variables contained in cov_list using labels as the receiving dframe for values in metadata.
	-dset: DataFrame of independent variables.
	-labels: DataFrame of dependent variables.
	-meta: DataFrame of covariates.
	-cov_list: List of covariables.'''
	y_model = copy.deepcopy(dset)
	x_model = copy.deepcopy(labels)
	cov = copy.deepcopy(meta)
	for w in cov_list:
		for tiss in np.unique(labels[0]):
			x_class = x_model[x_model[0] == tiss]
			y_class = y_model[y_model.index.isin(x_class.index)]
			if w.startswith('MH') and (cov[w].dtype == 'float64'):
				cov[w] = cov.loc[:,w].astype('category').cat.codes
				x_class = assign_val(x_class, cov, w, 0)
				x_class = pd.get_dummies(x_class)
				lm = LinearRegression()
				lm.fit(x_class, y_class)
				r2 = lm.score(x_class, y_class)
				sys.stdout.write(tiss+" pre-correction R² for "+w+": "+str(r2)+'\n')
				res = y_class - np.matmul(x_class.astype('float32'), lm.coef_.T)
				lm = LinearRegression()
				lm.fit(x_class, res)
				r2 = lm.score(x_class, res)
				sys.stdout.write(tiss+" post-correction R² for "+w+": "+str(r2)+'\n')
				y_model.loc[res.index,:] = res.loc[res.index,:]
			elif cov[w].dtype == object:
				cov[w] = cov.loc[:,w].astype('category').cat.codes
				x_class = assign_val(x_class, cov, w, 0)
				x_class = pd.get_dummies(x_class)
				lm = LinearRegression()
				lm.fit(x_class, y_class)
				r2 = lm.score(x_class, y_class)
				sys.stdout.write(tiss+" pre-correction R² for "+w+": "+str(r2)+'\n')
				res = y_class - np.matmul(x_class.x_class.astype('float32'), lm.coef_.T)
				lm = LinearRegression()
				lm.fit(x_class, res)
				r2 = lm.score(x_class, res)
				sys.stdout.write(tiss+" post-correction R² for "+w+": "+str(r2)+'\n')
				y_model.loc[res.index,:] = res.loc[res.index,:]
			elif cov[w].dtype == 'int64' and w != 'AGE':
				cov[w] = cov.loc[:,w].astype('category').cat.codes
				x_class = assign_val(x_class, cov, w, 0)
				x_class = pd.get_dummies(x_class)
				lm = LinearRegression()
				lm.fit(x_class, y_class)
				r2 = lm.score(x_class, y_class)
				sys.stdout.write(tiss+" pre-correction R² for "+w+": "+str(r2)+'\n')
				res = y_class - np.matmul(x_class.astype('float32'), lm.coef_.T)
				lm = LinearRegression()
				lm.fit(x_class, res)
				r2 = lm.score(x_class, res)
				sys.stdout.write(tiss+" post-correction R² for "+w+": "+str(r2)+'\n')
				y_model.loc[res.index,:] = res.loc[res.index,:]
			else:
				x_class = assign_val(x_class, cov, w, 0)
				if x_class[0].max() != 0.0:
					x_class = x_class/x_class.max()
				lm = LinearRegression()
				lm.fit(x_class.values.reshape(-1,1), y_class)
				r2 = lm.score(x_class.values.reshape(-1,1), y_class)
				sys.stdout.write(tiss+" pre-correction R² for "+w+": "+str(r2)+'\n')
				res = y_class - np.matmul(x_class.astype('float32'), lm.coef_.reshape(1,-1))
				lm = LinearRegression()
				lm.fit(x_class.values.reshape(-1,1), res)
				r2 = lm.score(x_class.values.reshape(-1,1), res)
				sys.stdout.write(tiss+" post-correction R² for "+w+": "+str(r2)+'\n')
				y_model.loc[res.index,:] = res.loc[res.index,:]
	return y_model
