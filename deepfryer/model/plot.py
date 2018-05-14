# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('agg')
import sys
import numpy as np
import itertools

import matplotlib.pyplot as plt
import pylab as pl

import keras.backend as K

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def dim_red_plot(mode, model, score, test_data, y_lab, colordict, markdict, title, path):
	'''Performs dimensionality reduction (DR) using the model or not with different algorithms, such as TSNE and PCA, in order to see the data in 2D.
		-mode: String. 'pca', 'pca_tsne' or 'ran_tsne'.
		-model: model if you want to compress data using it previously. 
		-score: score metric of the model if your compressing to 2 dimensions the data.
		-test_data: data on which the DR it will be performed.
		-y_lab: labels used on the data
		-colordict: dictionary of color used.
		-markdict:dictionary of marks used.
		-title: title of the plot.
		-path: destination of the plot.'''
	if model is not None:
		i=len(model.layers)-4
		encoded_output = K.function([model.layers[0].input], [model.layers[i].output])
		x_test_tmp = encoded_output([test_data])[0]
	else:
		x_test_tmp = test_data
	y_t = np.vectorize(colordict.get)(y_lab[0])
	m_t = np.vectorize(markdict.get)(y_lab[0])
	if mode == 'pca':
		pca = PCA(n_components=2, random_state=1, svd_solver='auto', iterated_power='auto')
		x_test_tmp=pca.fit_transform(x_test_tmp)
	elif mode == 'pca_tsne':
		tsne=TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, random_state=1, verbose=1,method='exact',init='pca')
		x_test_tmp=tsne.fit_transform(x_test_tmp)
	elif mode == 'ran_tsne':
		tsne=TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, random_state=1, verbose=1,method='exact',init='random')
		x_test_tmp=tsne.fit_transform(x_test_tmp)
	else:
		sys.stderr.out("Model with 2 components\n")
	fig = pl.figure(figsize=(14.50, 12))
	ax = fig.add_axes([0.05,0.05,0.90,0.90])
	pl.setp(ax, xticks=(), yticks=())
	for i in range(x_test_tmp.shape[0]):
		pl.scatter(x_test_tmp[i, 0], x_test_tmp[i, 1],c=y_t[i], marker=m_t[i])
	pl.title(title)
	for i in y_lab[0].unique():
	   	# Position of each label.
		index=np.array(y_lab.index.values.tolist())
		samples=np.array(y_lab[y_lab[0]==i].index.values.tolist())
		indices=np.where(np.isin(index, samples))
		xtext, ytext = np.median(x_test_tmp[indices[0], :], axis=0)
		ax.text(xtext, ytext, str(i), fontsize=8)
	if mode == 'pca':
		plt.xlabel('PC1: {:{width}.{prec}f}'.format(pca.explained_variance_ratio_[0], width=5, prec=3), fontsize = 'medium')
		plt.ylabel('PC2: {:{width}.{prec}f}'.format(pca.explained_variance_ratio_[1], width=5, prec=3), fontsize='medium')
	elif mode == 'pca_tsne':
		ax.text(0.90,0.97, 'KL-divergence: {:{width}.{prec}f}'.format(tsne.kl_divergence_, width=5, prec=3), ha='center', va='center', transform=ax.transAxes, fontsize='medium')
	elif mode == 'ran_tsne':
		ax.text(0.90,0.97, 'KL-divergence: {:{width}.{prec}f}'.format(tsne.kl_divergence_, width=5, prec=3), ha='center', va='center', transform=ax.transAxes, fontsize='medium')
	else:
		ax.text(0.90,0.97, 'Model acc: {:{width}.{prec}f}'.format(score, width=5, prec=3), ha='center', va='center', transform=ax.transAxes, fontsize=10)
	plt.savefig(path+title+'.pdf', dpi = 360)
	plt.close()
	
def plot_confusion_matrix(cm, classes, path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	-cm: confusion matrix.
	-classes: classes used.
	-path: destination of the plot
	-normalize: Boolean. Normalization of values.
	-title: String with the name of the plot.
	-cmap: plt colormap instance. 
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		sys.stderr.write("Normalized confusion matrix \n")
	else:
		sys.stderr.write('Confusion matrix, without normalization \n')
	fig = plt.figure(figsize=(14.5, 14.5))
	axmatrix = fig.add_axes([0.115,0.23,0.95,0.70])
	im = axmatrix.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title,size='medium')
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=90)
	plt.yticks(tick_marks, classes)
	plt.ylabel('True label', size ='medium')
	plt.xlabel('Predicted label', size='medium')
	axcolor = fig.add_axes([0.95,0.23,0.022,0.70])
	cbar = pl.colorbar(im, cax=axcolor)
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		if cm.shape[0] > 40:
			axmatrix.text(j, i, round(cm[i, j],2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", size ='x-small')
		else:
			axmatrix.text(j, i, round(cm[i, j],2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", size ='medium')
	
	if normalize:
		fig.savefig(path+'_CM_Norm.pdf', dpi=360)
		plt.close()
	else:
		fig.savefig(path+'_CM.pdf', dpi = 360)
		plt.close()
		
def plot_roc(fpr, tpr, roc_auc, n_classes, colors, title, path):
	'''Plots Receiving Operator Characteristic curve.
		-fpr: false positive rate calculated for the model as a dictionary.
		-tpr: true positive rate calculated for the model as a dictionary.
		-roc_auc: roc_auc calculated for the model as a dictionary.
		-n_classes: list of classes.
		-colors: list of colors.
		-title: string for the name of the plot.
		-path: destination of the plot.'''
	lw = 2
	plt.figure(figsize=(14.50,9))
	plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
	plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
	#colors 
	for i, color in zip(n_classes, colors):
		if i == 'micro':
			color = '#6d3333'
		elif i == 'macro':
			color = '#3e663c'
		plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='{0} (AUC = {1:0.5f})'.format(i, roc_auc[i]))
	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate', size = 'large')
	plt.ylabel('True Positive Rate', size = 'large')
	plt.title('Receiver operating curve')
	plt.legend(loc="lower right", fontsize='medium', ncol=2)
	plt.tight_layout()
	plt.savefig(path+'_ROC.pdf', dpi=360)
	plt.close()

def plot_score(cvscore, title, path):
	'''Plot the CV score of model.
	-cvscore: cv score stored in a list.
	-title: name of the plot.
	-path: destination of the plot.'''
	plt.figure(figsize=(14.5,9))
	plt.title(title)
	plt.xlabel("CV steps")
	plt.ylabel("Score")
	plt.plot(range(len(cvscore)), cvscore, 'o-', color="g", label="Cross-validation score")
	plt.plot(range(len(cvscore)), np.repeat(cvscore.mean(), len(cvscore)), 'o-', color="r", label="Mean")
	plt.legend(loc="best")
	plt.tight_layout()
	plt.savefig(path+title+'_cv_curve.pdf', dpi =360)
	plt.close()
