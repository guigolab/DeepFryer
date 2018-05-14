#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 13:07:28 2018

@author: mcamara
"""

import xgboost as xgb

def run_xgb(x_train, y_train, x_test, y_test, x_val, y_val, nb_classes):
	dtrain = xgb.DMatrix(x_train_tmp, label=y_train_tmp)
	dtest = xgb.DMatrix(x_test_tmp, label = y_test_tmp)
	dval = xgb.DMatrix(x_val_tmp, label = y_val_tmp)
	param = {'max_depth': 1, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic', 'tree_method':'exact', 'eval_metric':['auc', 'error']}
	evallist = [(dval, 'eval'), (dtrain, 'train')]
	num_round = 50
	bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=5)
	y_pred = bst.predict_proba(dtest, output_margin=False, ntree_limit=bst.best_ntree_limit)

	sys.stderr.write('Gradient Boosting Finished with a test acc of: '+str(score)+'\n')
	return cvscore, svm, score, pred
	