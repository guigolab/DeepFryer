# -*- coding: utf-8 -*-
import sys
import numpy as np
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import roc_curve, auc

from scipy import interp

def conf_matrix(pred, test, tlist):
	'''Computes the confusion matrix over the predicitions from the model.
	-pred: set of predictions
	-test: set of ground truth
	-tlist: list of classes.'''
	test.loc[:,1] = pred
	test.loc[:,0] = [tlist[i] for i in test.loc[:,0]]
	test.loc[:,1] = [tlist[i] for i in test.loc[:,1]]
	classes = np.unique(test.loc[:,0])
	conf_mat = cm(test.loc[:,0], test.loc[:,1], classes)
	return conf_mat, classes

def roc_auc_class(y_test, y_score, tlist):
	'''Computes the roc_auc over a class.
	-y_test: set of real labels.
	-y_score: set of probabilities associated to each class prediction.
	-tlist: list of classes'''
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in tlist:
		fpr[i], tpr[i], _= roc_curve(y_test.loc[:,i], y_score.loc[:,i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	sys.stderr.write('ROC and AUC calculated \n')
	return fpr, tpr, roc_auc

def minor_major(y_test, y_score, tpr, fpr, roc_auc, tlist):
	'''computes the minor and major roc_aur for the model.
	-y_test: set of real labels.
	-y_score: set of probabilities associated to each class prediction.
	-tpr: true positive rates in a dictionary with class as keys.
	-fpr: false positive rates in a dictionary with class as keys.
	-roc_auc: dictionary of roc_auc with class as keys.
	-tlist: list of classes'''
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	all_fpr = np.unique(np.concatenate([fpr[i] for i in tlist]))
	mean_tpr = np.zeros_like(all_fpr)
	for i in tlist:
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])
	mean_tpr /= len(tlist)
	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
	sys.stderr.write('micro and macro ROC and AUC calculated \n')
	return fpr, tpr, roc_auc

			        