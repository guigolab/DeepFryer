#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 12:21:48 2018

@author: mcamara
"""
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

def run_svm(x_train, y_train, x_test, y_test, fold = 5, kernel = 'linear'):
	'''Run a SVM and K-fold cv it.
	-x_train: train values set.
	-y_train: train labels.
	-x_test: test values set.
	-y_test: test labels.
	-folds: N splits of set for CV round.
	-kernel: kernel used by svm'''
	svm = SVC(kernel = kernel)
	cvscore = cross_val_score(svm, x_train.as_matrix(), y_train.loc[:,0], cv = fold, scoring = 'accuracy')
	svm = SVC(kernel = kernel, probability = True)
	sys.stderr.write(kernel+' SVM Initialized\n')
	svm.fit(x_train, y_train)
	score = svm.score(x_test,y_test)
	pred = svm.predict_proba(x_test)
	sys.stderr.write(kernel+' Support Vector Machine Finished with a test acc of: '+str(score)+'\n')
	return cvscore, svm, score, pred


def grid_svm(x_train, y_train, x_test, y_test, fold = 5):
	'''Run a svm and K-fold cv it and looking for best parameters. After it gives you a report on performance on stdout.
	-x_train: train values set.
	-y_train: train labels.
	-x_test: test values set.
	-y_test: test labels.
	-folds: N splits of set for CV round.'''
	
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1 ,1e-1, 1e-2, 1e-3, 1e-4],'C': [0.01, 0.1, 1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]

	scores = ['accuracy']

	for score in scores:
		sys.stdout.write("# Tuning hyper-parameters for %s" % score)
		sys.stdout.write("\n")
		clf = GridSearchCV(SVC(), tuned_parameters, cv=fold, scoring=score)
		clf.fit(x_train, y_train)
		sys.stdout.write("Best parameters set found on development set:")
		sys.stdout.write("\n")
		sys.stdout.write(str(clf.best_params_))
		sys.stdout.write("\n")
		sys.stdout.write("Grid scores on development set:")
		sys.stdout.write("\n")
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			sys.stdout.write("%0.3f (+/-%0.03f) for %r \n"% (mean, std * 2, params))
		sys.stdout.write("\n")
		sys.stdout.write("Detailed classification report:")
		sys.stdout.write("\n")
		sys.stdout.write("The model is trained on the full development set.")
		sys.stdout.write("The scores are computed on the full evaluation set.")
		sys.stdout.write("\n")
		y_true, y_pred = y_test, clf.predict(x_test)
		sys.stdout.write(classification_report(y_true, y_pred))
		sys.stdout.write("\n")
		return clf
	