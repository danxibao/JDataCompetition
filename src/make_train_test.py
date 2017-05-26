#!/usr/bin/env python
# coding: utf-8
# @Filename: make_traing_test
# @Date: 2017-05-15 19:24
# @Author: peike
# @Blog: http://www.peikeli.com


from feature_extraction import *
import pandas as pd
import os
from sklearn.externals import joblib
from datetime import datetime, timedelta
import gc


def get_basic_ID(start_date, end_date):
    dump_path = '../cache/basic_ID_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        ID_df = joblib.load(dump_path)
        print 'basic id %s to %s loaded' % (start_date, end_date)
    else:
        action_df = get_action_dataframe(start_date, end_date)
        product_df = pd.read_csv('../data/JData_Product.csv')
        action_df = action_df[action_df.sku_id.isin(product_df.sku_id)]
        ID_df = action_df.groupby(['user_id', 'sku_id'], as_index=False).first()
        ID_df = ID_df[['user_id', 'sku_id','cate','brand']]
        # filter crawl users
        crawl_user_df = pd.read_csv('../data/crawl_user.csv')
        ID_df = ID_df[~ID_df.user_id.isin(crawl_user_df.user_id)].reset_index(drop=True)
        # filter low potential users
        low_potential_user = pd.read_csv('../data/low_potential_user.csv')
        ID_df = ID_df[~ID_df.user_id.isin(low_potential_user.user_id)].reset_index(drop=True)
        # filter cate8_bought_user
        cate8_bought_user = pd.read_csv('../data/cate8_bought_user.csv', parse_dates=['time'])
        cate8_bought_user = cate8_bought_user[cate8_bought_user.time<pd.Timestamp(end_date)]
        ID_df = ID_df[~ID_df.user_id.isin(cate8_bought_user.user_id)].reset_index(drop=True)
        joblib.dump(ID_df, dump_path)
        print('basic id %s to %s dumped' %(start_date, end_date))
    return ID_df

def get_test_basic_ID(start_date, end_date):
    dump_path = '../cache/basic_ID_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        ID_df = joblib.load(dump_path)
        print 'basic id %s to %s loaded' % (start_date, end_date)
    else:
        action_df = get_action_dataframe(start_date, end_date)
        product_df = pd.read_csv('../data/JData_Product.csv')
        action_df = action_df[action_df.sku_id.isin(product_df.sku_id)]
        ID_df = action_df.groupby(['user_id', 'sku_id'], as_index=False).first()
        ID_df = ID_df[['user_id', 'sku_id','cate','brand']]
        # filter low potential users
        crawl_user_df = pd.read_csv('../data/crawl_user.csv')
        ID_df = ID_df[~ID_df.user_id.isin(crawl_user_df.user_id)].reset_index(drop=True)
        # filter low potential users
        low_potential_user = pd.read_csv('../data/low_potential_user.csv')
        ID_df = ID_df[~ID_df.user_id.isin(low_potential_user.user_id)].reset_index(drop=True)
        # filter cate8_bought_user
        cate8_bought_user = pd.read_csv('../data/cate8_bought_user.csv', parse_dates=['time'])
        cate8_bought_user = cate8_bought_user[cate8_bought_user.time<pd.Timestamp(end_date)]
        ID_df = ID_df[~ID_df.user_id.isin(cate8_bought_user.user_id)].reset_index(drop=True)
        joblib.dump(ID_df, dump_path)
        print('basic id %s to %s dumped' %(start_date, end_date))
    return ID_df


def get_label(start_date, end_date):
    dump_path = '../cache/label_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        label_df = joblib.load(dump_path)
        print('label %s to %s loaded' % (start_date, end_date))
    else:
        action_df = get_action_dataframe(start_date, end_date)
        # all buy action
        label_df = action_df[(action_df['type'] == 4)&(action_df.cate==8)]
        label_df = label_df.groupby(['user_id', 'sku_id'], as_index=False).first()
        label_df['label'] = 1
        label_df = label_df[['user_id', 'sku_id', 'label']]
        joblib.dump(label_df, dump_path)
        print('label %s to %s dumped' % (start_date, end_date))
    return label_df


def report(pred, label):
    #F1_1
    tp1 = sum(pred.user_id.isin(label.user_id))
    precision1 = tp1/float(pred.shape[0])
    recall1 = tp1/float(label.shape[0])
    F11 = 6*precision1*recall1 / (5*recall1 + precision1)
    #F1_2
    tp2 = pd.merge(pred, label, how='left', on=['user_id', 'sku_id']).label.sum()
    precision2 = tp2 / float(pred.shape[0])
    recall2 = tp2 / float(label.shape[0])
    F12 = 5*precision2*recall2 / (2*recall2 + precision2)
    #Total
    F1 = 0.4*F11 + 0.6*F12
    print "Report\n-----"
    print "user"
    print "predict user no: %d | truth user no: %d " %(pred.shape[0], label.shape[0])
    print "precision1: %.4f | recall1: %.4f | f11: %.4f" %(precision1, recall1, F11)
    print "product"
    print "precision2: %.4f | recall2: %.4f | f12: %.4f" %(precision2, recall2, F12)
    print "total f1: %.4f" % F1


def make_train_set(train_start_date, train_end_date, label_start_date, label_end_date):
    dump_path = '../cache/train_set_%s_%s.set' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        train_set = joblib.load(dump_path)
        print('training set %s to %s loaded' % (train_start_date, train_end_date))
    else:
        train_set = get_basic_ID(train_start_date, train_end_date)
        basic_u = get_basic_user_feature(train_start_date, train_end_date)
        basic_i = get_basic_product_feature()
        basic_c = get_product_comment_feature(train_start_date, train_end_date)
        common_u = get_user_common_feature(train_start_date, train_end_date)
        user_chain = get_user_chain_feature(train_start_date, train_end_date)
        common_i = get_product_common_feature(train_start_date, train_end_date)
        common_b = get_brand_common_feature(train_start_date, train_end_date)
        common_ui = get_user_product_common_feature(train_start_date, train_end_date)
        common_uc = get_user_cate_common_feature(train_start_date, train_end_date)
        common_ub = get_user_brand_common_feature(train_start_date, train_end_date)
        label = get_label(label_start_date, label_end_date)
        train_set = pd.merge(train_set, basic_u, how='left', on='user_id')
        train_set = pd.merge(train_set, basic_i, how='left', on='sku_id')
        train_set = pd.merge(train_set, basic_c, how='left', on='sku_id')
        train_set = pd.merge(train_set, common_u, how='left', on='user_id')
        train_set = pd.merge(train_set, user_chain, how='left', on='user_id')
        train_set = pd.merge(train_set, common_i, how='left', on='sku_id')
        train_set = pd.merge(train_set, common_b, how='left', on='brand')
        train_set = pd.merge(train_set, common_ui, how='left', on=['user_id', 'sku_id'])
        train_set = pd.merge(train_set, common_uc, how='left', on=['user_id', 'cate'])
        train_set = pd.merge(train_set, common_ub, how='left', on=['user_id', 'brand'])
        train_set = pd.merge(train_set, label, how='left', on=['user_id', 'sku_id'])
        train_set['label'] = train_set['label'].fillna(0)
        del basic_u, basic_i, basic_c, common_u, common_i, common_b, common_ui, common_uc, common_ub
        gc.collect()
        for i in (1, 2, 3, 5, 7, 10, 15):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            stat_u = get_user_stat_feature(start_days, train_end_date, str(i))
            train_set = pd.merge(train_set, stat_u, how='left', on=['user_id'])
            stat_i = get_product_stat_feature(start_days, train_end_date, str(i))
            train_set = pd.merge(train_set, stat_i, how='left', on=['sku_id'])
            stat_b = get_brand_stat_feature(start_days, train_end_date, str(i))
            train_set = pd.merge(train_set, stat_b, how='left', on=['brand'])
            del stat_u, stat_i, stat_b
            gc.collect()
        train_set.fillna(-1,inplace=True)
        joblib.dump(train_set, dump_path)
        print('training set %s to %s dumped' % (train_start_date, train_end_date))
    train_label = train_set['label']
    train_set = train_set.drop(['label'], axis=1)
    return train_set, train_label


def make_test_set(test_start_date, test_end_date):
    dump_path = '../cache/test_set_%s_%s.set' % (test_start_date, test_end_date)
    if os.path.exists(dump_path):
        test_set = joblib.load(dump_path)
        print 'testing set %s to %s loaded' % (test_start_date, test_end_date)
    else:
        test_set = get_test_basic_ID(test_start_date, test_end_date)
        basic_u = get_basic_user_feature(test_start_date, test_end_date)
        basic_i = get_basic_product_feature()
        basic_c = get_product_comment_feature(test_start_date, test_end_date)
        common_u = get_user_common_feature(test_start_date, test_end_date)
        user_chain = get_user_chain_feature(test_start_date, test_end_date)
        common_i = get_product_common_feature(test_start_date, test_end_date)
        common_b = get_brand_common_feature(test_start_date, test_end_date)
        common_ui = get_user_product_common_feature(test_start_date, test_end_date)
        common_uc = get_user_cate_common_feature(test_start_date, test_end_date)
        common_ub = get_user_brand_common_feature(test_start_date, test_end_date)
        test_set = pd.merge(test_set, basic_u, how='left', on='user_id')
        test_set = pd.merge(test_set, basic_i, how='left', on='sku_id')
        test_set = pd.merge(test_set, basic_c, how='left', on='sku_id')
        test_set = pd.merge(test_set, common_u, how='left', on='user_id')
        test_set = pd.merge(test_set, user_chain, how='left', on='user_id')
        test_set = pd.merge(test_set, common_i, how='left', on='sku_id')
        test_set = pd.merge(test_set, common_b, how='left', on='brand')
        test_set = pd.merge(test_set, common_ui, how='left', on=['user_id', 'sku_id'])
        test_set = pd.merge(test_set, common_uc, how='left', on=['user_id', 'cate'])
        test_set = pd.merge(test_set, common_ub, how='left', on=['user_id', 'brand'])
        del basic_u, basic_i, basic_c, common_u, common_i, common_b, common_ui, common_uc, common_ub
        gc.collect()
        for i in (1, 2, 3, 5, 7, 10, 15):
            start_days = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            stat_u = get_user_stat_feature(start_days, test_end_date, str(i))
            test_set = pd.merge(test_set, stat_u, how='left', on=['user_id'])
            stat_i = get_product_stat_feature(start_days, test_end_date, str(i))
            test_set = pd.merge(test_set, stat_i, how='left', on=['sku_id'])
            stat_b = get_brand_stat_feature(start_days, test_end_date, str(i))
            test_set = pd.merge(test_set, stat_b, how='left', on=['brand'])
            del stat_u, stat_i, stat_b
            gc.collect()
        test_set.fillna(-1, inplace=True)
        joblib.dump(test_set, dump_path)
        print('testing set %s to %s dumped' % (test_start_date, test_end_date))
    return test_set


if __name__ == "__main__":
    make_train_set('2016-03-27', '2016-04-11', '2016-04-11', '2016-04-16')
    make_train_set('2016-03-22', '2016-04-06', '2016-04-06', '2016-04-11')
    make_train_set('2016-03-17', '2016-04-01', '2016-04-01', '2016-04-06')
    make_test_set('2016-04-01', '2016-04-16')