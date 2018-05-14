#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:38:40 2018

@author: mcamara
"""
import matplotlib.pyplot as plt
import numpy as np

def distrib_genes_per_tissue(W, tissue_list, colors, path):
	'''Plots the distribution of W across a tissue using it colors palette.
	- W: DataFrame of Weights.
	- tissue_list: List of classes.
	- colors: list of colors order by class.
	- path: Destination of the plot.'''
	i = 0
	color = colors[0].values.tolist()
	for tiss in W:
		plt.style.use('ggplot')
		plt.figure(figsize=(14.50,9))
		plt.hist(W.loc[:,tiss],bins=100, color = color[i])
		threshold=np.percentile(W.loc[:,tiss], 95)
		plt.title(tissue_list[i])
		plt.axvline(W.loc[:,tiss].mean(), color='black', linestyle='dashed', linewidth=1.5)
		plt.axvline(threshold, color='g', linestyle='dashed', linewidth=1.5)
		plt.tight_layout()
		plt.savefig(path+'_'+tissue_list[i]+'.pdf', dpi=360)
		plt.close()
		i+=1
