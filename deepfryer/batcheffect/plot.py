# -*- coding: utf-8 -*-

import copy
import numpy as np

import matplotlib.pyplot as plt
import pylab as pl

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

def hier_heatmap(data, labels, gene_list, title, met, path):
	''' Hierarchical Heatmap on data provided.
	-data: DataFrame with the values.
	-labels: list of tags for the samples
	-gene_list: list of features used.
	-title: title of the plot
	-met: metric used.
	-path: Path where the plot is saved'''
	import sys
	#Hierarchical
	sys.setrecursionlimit(1000000000)
	D = copy.deepcopy(data)
	D1 = ssd.pdist(D,metric ='euclidean')
	fig = pl.figure(figsize=(15,13))
	Y1 = sch.linkage(D1, method='single')
	Z1 = sch.dendrogram(Y1, orientation='left', no_plot = True)
	D2 = ssd.pdist(D.T,metric ='euclidean')
	Y2 = sch.linkage(D2, method='single')
	Z2 = sch.dendrogram(Y2, orientation='top', no_plot = True)
	axmatrix = fig.add_axes([0.03,0.23,0.83,0.70])
	idx1 = Z1['leaves']
	idx2 = Z2['leaves']
	D = D.iloc[idx1,:]
	D = D.iloc[:,idx2]
	cl_lab = np.take(labels, idx2)
	cl_gene = np.take(gene_list, idx1)
	im = axmatrix.matshow(D.astype('float64'), aspect='auto', origin='lower', cmap=pl.cm.YlGnBu)
	dr = np.array(idx2, dtype=int)
	dc = np.array(range(len(idx2)))
	dr.shape = (1,len(dr))
	dc.shape = dr.shape
	#ticks for matrix
	axmatrix.set_xticks(np.arange(D.shape[1]), minor=False)
	axmatrix.set_xticklabels(cl_lab, rotation=60, minor=False)
	axmatrix.xaxis.set_label_position('bottom')
	axmatrix.xaxis.tick_bottom()
	if len(cl_lab) > 40:
		pl.xticks(rotation=90, fontsize='small')
	else:
		pl.xticks(rotation=90, fontsize='medium')
	axmatrix.set_yticks(np.arange(D.shape[0]), minor=False)
	axmatrix.set_yticklabels(cl_gene,  minor=False)
	axmatrix.yaxis.set_label_position('right')
	axmatrix.yaxis.tick_right()
	if len(cl_gene) > 40:
		pl.yticks(fontsize='x-small')
	else:
		pl.yticks(fontsize='medium')
	plt.title(title)
	axcolor = fig.add_axes([0.928,0.23,0.02,0.70])
	cbar = pl.colorbar(im, cax=axcolor)
	cbar.set_label(met, labelpad=-1.1)
	fig.savefig(path+'.pdf')
	plt.close()
	sys.stderr.write('Hierarchical Heatmap saved on '+path+'.pdf \n')


def hier_clust(data, labels, title, path):
	'''Hierchical clustering.
	-data: DataFrame used.
	-labels: variable used to identify the data
	-title:
	-path: Destination of the plot.'''
	import sys
	#Hierarchical
	sys.setrecursionlimit(1000000000)
	D = copy.deepcopy(data)
	D1 = ssd.pdist(D,metric ='euclidean')
	fig = pl.figure(figsize=(15,13))
	Y1 = sch.linkage(D1, method='single')
	Z1 = sch.dendrogram(Y1, labels = labels)
	idx1 = Z1['leaves']
	D = D.iloc[idx1,:]
	#ticks for matrix
	plt.title(title)
	fig.savefig(path+'.pdf', dpi=360)
	plt.close()
	sys.stderr.write('Hierarchical Heatmap saved on '+path+'.pdf \n')
	return Z1
	
	