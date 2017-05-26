#!/usr/bin/env python
# coding: utf-8
# @Filename: make_traing_test
# @Date: 2017-05-15 20:30
# @Author: peike
# @Blog: http://www.peikeli.com

import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt
from sklearn import metrics
from sklearn.externals import joblib
from  sklearn.model_selection import RandomizedSearchCV
from make_train_test import *
from sklearn.metrics import precision_recall_curve



def model_pre_fit(clf, train_X, train_y, useTrainCV=True, cv_folds=5, early_stopping_rounds=100):
    # cross-validation train
    if useTrainCV:
        xgb_param = clf.get_xgb_params()
        xg_train = xgb.DMatrix(train_X.values, label=train_y.values)
        cv_result = xgb.cv(xgb_param, xg_train, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
                           metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True, show_stdv=True)
        clf.set_params(n_estimators = cv_result.shape[0])
        print "Best iteration: %d" % cv_result.shape[0]
    # Fit the algorithm on the data
    clf.fit(train_X.values, train_y.values)
    # Predict training set:
    prediction = clf.predict(train_X.values)
    prediction_proba = clf.predict_proba(train_X.values)[:, 1]
    # Print model report:
    print "Model Report\n----"
    print "AUC Score: %f" % metrics.roc_auc_score(train_y.values, prediction)
    print "F1 Score: %f" % metrics.f1_score(train_y.values, prediction_proba)
    # Plot feature importance
    feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    return clf


def model_random_search(clf, train_X, train_y):
    param_test = {
        'max_depth':range(3,10,1),
        'min_child_weight':range(1,6,1),
        'gamma':[i/10.0 for i in range(0,5)],
        'subsample':[i/100.0 for i in range(60,100,5)],
        'colsample_bytree':[i/100.0 for i in range(60,100,5)],
        'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
    }
    grid_search = RandomizedSearchCV(estimator=clf, n_iter=100, param_distributions = param_test, scoring='roc_auc', n_jobs=-1, cv=5, verbose=10)
    grid_search.fit(train_X.values, train_y.values)
    print "best parameter:", grid_search.best_params_
    print "best cv score:", grid_search.best_score_
    return grid_search


def model_post_fit():
    pass


def make_submission(clf, path):
    test_index, test_X = make_test_set('2016-03-17', '2016-04-16')
    submission = test_index[['user_id','sku_id']]
    submission['proba'] = clf.predict_proba(test_X.values)[:, 1]
    submission = submission[submission.proba>0.15]
    submission = submission.groupby('user_id',as_index=False).apply(lambda t:t[t.proba==t.proba.max()]).reset_index(drop=True)
    submission.drop(['proba'],inplace=True, axis=1)
    submission.to_csv(path, index=False)
    print 'submission saved in' + path
    return submission


def report(submission, label):
    #F1_1
    tp1 = sum(submission.user_id.isin(label.user_id))
    precision1 = tp1/float(submission.shape[0])
    recall1 = tp1/float(label.shape[0])
    F11 = 6*precision1*recall1 / (5*recall1 + precision1)
    #F1_2
    tp2 = pd.merge(submission, label, how='left', on=['user_id', 'sku_id']).label.sum()
    precision2 = tp2 / float(submission.shape[0])
    recall2 = tp2 / float(label.shape[0])
    F12 = 5*precision2*recall2 / (2*recall2 + 3*precision2)
    #Total
    F1 = 0.4*F11 + 0.6*F12
    print "----\nUser"
    print "predict user no: %d | truth user no: %d " %(submission.shape[0], label.shape[0])
    print "precision1: %.4f | recall1: %.4f | f11: %.4f" %(precision1, recall1, F11)
    print "----\nProduct"
    print "precision2: %.4f | recall2: %.4f | f12: %.4f" %(precision2, recall2, F12)
    print "----\ntotal f1: %.4f"%(F1)
    return F1



if __name__ == '__main__':
    train_set = joblib.load('../cache/train_set_2016-03-27_2016-04-11.set')
    train_label = train_set['label']
    train_set = train_set.drop(['label'], axis=1)
    xgb = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=6, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', scale_pos_weight=1)
    xgb = model_pre_fit(xgb, train_set, train_label)
    xgb = model_random_search(xgb,train_set, train_label)
    prediction_proba = xgb.predict_proba(train_set)[:, 1]
    precision, recall, thresholds = precision_recall_curve(train_label, prediction_proba, pos_label=1)
    f11 = 6 * precision * recall / (5 * recall + precision)
    f12 = 5 * precision * recall / (2 * recall + 3 * precision)
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(thresholds, precision[1:], color='navy', label='Precision curve')
    plt.plot(thresholds, recall[1:], color='red', label='Recall curve')
    plt.plot(thresholds, f11[1:], color='green', label='F11 curve')
    plt.plot(thresholds, f12[1:], color='yellow', label='F12 curve')
    plt.xlabel('Threshold')
    plt.ylabel('Precision-Recall')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.legend()
    plt.show()
    # Submission
    test_set = joblib.load('../cache/test_set_2016-04-01_2016-04-16.set')
    prediction_proba1 = xgb.predict_proba(test_set)[:, 1]
    submission = test_set[['user_id', 'sku_id']]
    submission['proba'] = prediction_proba1
    submission = submission.groupby('user_id', as_index=False).first()
    submission = submission.sort_values('proba', ascending=False).reset_index(drop=True).reset_index(drop=True)
    submission_csv = submission.iloc[0:600, :].drop(['proba'], axis=1)
    submission_csv.to_csv('../submission/submisson.csv', index=False)
    submission = make_submission(xgb, "../submission/xgb_total.model")