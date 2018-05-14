#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 12:22:28 2018

@author: mcamara
"""
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def run_randomf(x_train, y_train, x_test, y_test, fold=5):
	'''Run a random Forest and K-fold cv it.
	-x_train: train values set.
	-y_train: train labels.
	-x_test: test values set.
	-y_test: test labels.
	-folds: N splits of set for CV round.'''
	sys.stderr.write('Random Forest Initialized\n')
	rfc = RandomForestClassifier()
	cvscore = cross_val_score(rfc, x_train.as_matrix(), y_train.loc[:,0], cv = fold, scoring = 'accuracy')
	rfc = RandomForestClassifier()
	rfc.fit(x_train, y_train)
	score = rfc.score(x_test,y_test)
	pred = rfc.predict_proba(x_test)
	sys.stderr.write('Random Forest Finished with a test acc of: '+str(score)+'\n')
	return cvscore, rfc, score, pred
	