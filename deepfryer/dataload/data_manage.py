# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


import pyarrow.feather as feather

# types = np.array(['protein_coding','lincRNA','antisense','miRNA','sense_intronic'])

def build_types(path, types):
	'''Select which type of elements from a GTF you want for your analysis.
	-path: Path of the GTF.
	-types: list with the elements.'''
	element_types = pd.read_csv(path, sep='\t', header=None)
	element_types = element_types[element_types[1].isin(types)]
	element_types = element_types.set_index(keys=0, drop=True)
	return element_types

def preprocess_data(dset, mode='max'):
	'''Preprocessing of the data.
	-dset: DataFrame with values.
	-mode: String with the mode, max and lib available. Max takes the maximum for the sample and Lib takes the library size if your data is not normalized.'''
	dset = dset.loc[~(dset<=1).all(axis=1)]
	dset = dset.loc[~(dset.var(axis=1)<1)]
	dset = np.log2(dset+1)
	dset = dset.T
	dset.index = np.array([w.replace('.','-') for w in dset.index])
	dset = dset.sort_index(axis=0)
	if mode == 'max':
		maxcov = dset.max(axis=1)
		dset = dset.T
		dset = dset/maxcov
		dset = dset.T
	if mode == 'lib':
		dset = dset.T
		libsize = dset.sum(axis = 0)
		dset = (dset/libsize).T
	return dset


def ft_import_data(path, element_types, preprocess = False, mode='max'):
	'''Importer for feather format.
	-path: path of file
	-element_types: type of element that we selected previosly. If not, it will take everything
	-preprocess: Boolean. Select to preprocess the data or not.
	-mode: If preprocess true , the kind of normalization'''
	dset = feather.read_feather(path, nthreads = 12)
	if 'transcript_id' in dset.columns.values:
		dset = dset.set_index(keys='transcript_id', drop=True)
		dset = dset.drop(labels=['gene_id'], axis = 1)
	elif 'Name' in dset.columns.values:
		dset = dset.set_index(keys='Name', drop=True)
	if 'Description' in dset.columns.values:
		dset = dset.drop(labels=['Description'], axis = 1)
	if 'Unnamed: 0' in dset.columns.values:
		dset = dset.drop(labels=['Unnamed: 0'], axis = 1)
	if 'transcript_id(s)' in dset.columns.values:
		dset = dset.drop(labels=["transcript_id(s)"], axis = 1)
	if element_types is not None:
		dset = dset[dset.index.isin(element_types.index)]
	if preprocess == True:
		dset = preprocess_data(dset, mode)
	sys.stderr.write('Data imported from '+path+'\n')
	return dset

def pd_import_data(path, element_types, preprocess = False, mode ='max'):
	'''Importer for csv format.
	-path: path of file
	-element_types: type of element that we selected previosly. If not, it will take everything
	-preprocess: Boolean. Select to preprocess the data or not.
	-mode: If preprocess true , the kind of normalization'''
	dset = pd.read_csv(path, sep ='\t', header=0)
	if 'transcript_id' in dset.columns.values:
		dset = dset.set_index(keys='transcript_id', drop=True)
		dset = dset.drop(labels=['gene_id'], axis = 1)
	elif 'Name' in dset.columns.values:
		dset = dset.set_index(keys='Name', drop=True)
	if 'Description' in dset.columns.values:
		dset = dset.drop(labels=['Description'], axis = 1)
	if 'Unnamed: 0' in dset.columns.values:
		dset = dset.drop(labels=['Unnamed: 0'], axis = 1)
	if 'transcript_id(s)' in dset.columns.values:
		dset = dset.drop(labels=["transcript_id(s)"], axis = 1)
	if element_types is not None:
		dset = dset[dset.index.isin(element_types.index)]
	if preprocess == True:
		dset = preprocess_data(dset, mode)
	sys.stderr.write('Data imported from '+path+'\n')
	return dset

def folder_readpile(path, npset, samples, list_index):
	'''Importer for files stored in a folder in which we have more than one strand. It quantile normalizes the data to make it comparable between samples.
	-path: Location of the files.
	-npset: np.array to add the files in the path.
	-samples: list of samples selected. If not it will take everything in the file.
	-list_index: stores the samples id to recognize each row of the npset.'''
	for file in os.listdir(path):
		if samples is not None:
			if file.replace('_read.pile', "") in samples:
				tmp = pd.read_csv(path+file, sep='\t')
				if 'Unnamed: 3' in tmp.columns:  
					tmp= tmp.drop(['Unnamed: 3'], axis=1)
				if 'pos' in tmp.columns:  
					tmp= tmp.drop(['pos'], axis=1)
				qqnorm = QuantileTransformer(n_quantiles = 1000, output_distribution = 'uniform', random_state = 0)
				tmpnorm = qqnorm.fit_transform(tmp)
				tmp.loc[:,:] = tmpnorm
				tmp_np = np.reshape(tmp.as_matrix(), (1, tmp.shape[0] ,tmp.shape[1]))
			else:
				continue
		elif samples is None:
			tmp = pd.read_csv(path+file, sep='\t')
			if 'Unnamed: 3' in tmp.columns:  
				tmp= tmp.drop(['Unnamed: 3'], axis=1)
			if 'pos' in tmp.columns:  
				tmp= tmp.drop(['pos'], axis=1)
			qqnorm = QuantileTransformer(n_quantiles = 1000, output_distribution = 'uniform', random_state = 0)
			tmpnorm = qqnorm.fit_transform(tmp)
			tmp.loc[:,:] = tmpnorm
			tmp_np = np.reshape(tmp.as_matrix(), (1, tmp.shape[0] ,tmp.shape[1]))
		if npset is None:
			npset = tmp_np
			list_index.append(file.replace('_read.pile', ""))
		elif file.replace('_read.pile', "") in samples:
			npset = np.append(npset, tmp_np, axis=0)
			list_index.append(file.replace('_read.pile', ""))
		elif samples is None:
			npset = np.append(npset, tmp_np, axis=0)
			list_index.append(file.replace('_read.pile', ""))
	return npset, list_index

def folder_readpile_mat(path, dset):
	'''Load data contained in several files with more than one strand into a single strand and unique Matrix.
	-path: Location of the data.
	-dset: Dataset in which you want to store everything.'''
	for file in os.listdir(path):
		tmp = pd.read_csv(path+file, sep = '\t')
		if 'Unnamed: 3' in tmp.columns:  
			tmp= tmp.drop(['Unnamed: 3'], axis=1)
		if 'pos' in tmp.columns:  
			tmp= tmp.drop(['pos'], axis=1)
		sample = file.replace('_read.pile', "")
		gsize = tmp.shape[0]
		dset.loc[:,sample] = tmp.loc[:,'plus']
		minus = tmp.loc[:,'minus']
		minus.index = dset.index[gsize:]
		dset.loc[gsize:,sample] = minus
	return dset

def label_prepare(dset, path):
	'''Prepare the labels set according to dset.
	-dset: DataFrame with values and index used.
	-path: Location of labels dataset.'''
	labels = pd.read_csv(path, sep ='\t', header=0)
	samples = np.array(dset.index.values.tolist())
	samples = np.array([w.replace('.', '-') for w in samples])
	labels = labels[labels['SAMPID'].isin(samples)]
	labels = labels.loc[:,['SAMPID','SMTSD']]
	labels = labels.set_index(keys='SAMPID', drop=True)
	labels.columns = [0]
	sys.stderr.write('Labels imported from '+path+'\n')
	return labels

def load_dset(path):
	'''Load a dset compatible.
	-path: Location of the file'''
	dset = pd.read_csv(path, sep = '\t', header=0, index_col=0)
	sys.stderr.write('Data from '+path+' Loaded \n')
	if len(dset.columns.values) == 1:
		dset.columns = [0]
	return dset

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

