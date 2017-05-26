#!/usr/bin/env python
# coding: utf-8
# @Filename: feature_extraction
# @Date: 2017-04-20 19:30
# @Author: peike
# @Blog: http://www.peikeli.com


import pandas as pd
import numpy as np
import os
import gc
from sklearn.externals import joblib


action1_path = "../data/JData_Action_201602.csv"
action2_path = "../data/JData_Action_201603.csv"
action3_path = "../data/JData_Action_201604.csv"
comment_path = "../data/JData_Comment.csv"
product_path = "../data/JData_Product.csv"
user_path = "../data/JData_User.csv"


def convert_age(age_str):
    dic = {u"-1": -1, u'15岁以下': 1, u'16-25岁': 2, u'26-35岁': 3, u'36-45岁': 4, u'46-55岁': 5, u'56岁以上': 6}
    return dic[age_str]


def convert_reg_tm(reg_tm):
    if reg_tm <= 0:
        time = 0
    elif 0 < reg_tm <= 3:
        time = 1
    elif 3 < reg_tm <= 7:
        time = 2
    elif 7 < reg_tm <= 30:
        time = 3
    elif 30 < reg_tm <= 365:
        time = 4
    elif 365 < reg_tm:
        time = 5
    else:
        time = -1
    return time


def get_basic_user_feature(start_date, end_date):
    dump_path = '../cache/basic_user_feature_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        user_df = joblib.load(dump_path)
        print 'basic user features loaded'
    else:
        user_df = pd.read_csv(user_path, encoding='gbk', parse_dates=[-1])
        user_df.dropna(inplace=True, how="any")
        user_df['age'] = user_df['age'].map(convert_age)
        user_df['sex'] = user_df['sex'].astype('int64')
        user_df['user_reg_tm'] = (pd.Timestamp(end_date) - user_df['user_reg_tm']) / pd.Timedelta(1, 'D')
        user_df['user_reg_tm'] = user_df['user_reg_tm'].astype('int64')
        user_df['user_reg_tm'] = user_df['user_reg_tm'].map(convert_reg_tm)
        sex_df = pd.get_dummies(user_df["sex"], prefix="sex")
        user_df = pd.concat([user_df[['user_id', 'age', 'user_lv_cd', 'user_reg_tm']], sex_df], axis=1)
        joblib.dump(user_df, dump_path)
        print 'basic user features dumped'
    return user_df


def get_basic_product_feature():
    dump_path = '../cache/basic_product_feature.data'
    if os.path.exists(dump_path):
        product_df = joblib.load(dump_path)
        print 'basic product features loaded'
    else:
        product_df = pd.read_csv(product_path)
        attr1_df = pd.get_dummies(product_df["a1"], prefix="a1")
        attr2_df = pd.get_dummies(product_df["a2"], prefix="a2")
        attr3_df = pd.get_dummies(product_df["a3"], prefix="a3")
        product_df = pd.concat([product_df[['sku_id']], attr1_df, attr2_df, attr3_df], axis=1)
        joblib.dump(product_df, dump_path)
        print 'basic product features dumped'
    return product_df


def get_product_comment_feature(start_date, end_date):
    dump_path = '../cache/comments_feature_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        comment_df = joblib.load(dump_path)
        print 'basis product comment features loaded'
    else:
        comment_df = pd.read_csv(comment_path, parse_dates=[0])
        comment_date_list = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07",
                             "2016-03-14", "2016-03-21", "2016-03-28", "2016-04-04", "2016-04-11", "2016-04-15"]
        comment_date = "2016-02-01"
        for date in reversed(comment_date_list):
            if date < end_date:
                comment_date = date
                break
        comment_df = comment_df[comment_df.dt == comment_date]
        comment_df.drop('dt', axis=1, inplace=True)
        joblib.dump(comment_df, dump_path)
        print 'basic product comment features dumped'
    return comment_df


def get_action_dataframe(start_date, end_date):
    dump_path = '../cache/action_dataframe_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        action_df = joblib.load(dump_path)
        print('action dataframe %s to %s loaded' % (start_date, end_date))
    else:
        Dtype = {'user_id': np.int32, 'sku_id': np.int32, 'type': np.int32, 'cate': np.int32, 'brand': np.int32}
        action1_df = pd.read_csv(action1_path, dtype=Dtype, parse_dates=['time'])
        action2_df = pd.read_csv(action2_path, dtype=Dtype, parse_dates=['time'])
        action3_df = pd.read_csv(action3_path, dtype=Dtype, parse_dates=['time'])  
        action_df = pd.concat([action1_df, action2_df, action3_df], ignore_index=True)
        action_df = action_df[(action_df.time >= start_date) & (action_df.time < end_date)]
        joblib.dump(action_df, dump_path)
        print('action dataframe %s to %s dumped' % (start_date, end_date))
    return action_df


def get_user_stat_feature(start_date, end_date, suffix):
    dump_path = '../cache/user_stat_feature_%s_%s.data' %(start_date, end_date)
    if os.path.exists(dump_path):
        user_stat_feature_df = joblib.load(dump_path)
        print('user stat features %s to %s loaded' % (start_date, end_date))
    else:
        action_df = get_action_dataframe(start_date, end_date)
        # user active types
        type_df = pd.get_dummies(action_df['type'], prefix='user_'+suffix+'_type')
        user_type_count_df = pd.concat([action_df['user_id'], type_df], axis=1)
        user_type_count_df = user_type_count_df.groupby(['user_id'], as_index=False).sum()
        user_type_count_df['u_num_' + suffix] = user_type_count_df.drop(['user_id'], axis=1).sum(1)
        # user active days
        action_df['time_convert'] = action_df.time.dt.normalize()
        fs = ['user_id', 'user_active_day_'+suffix, 'user_active_day_ratio_'+suffix]
        user_day_count_df = action_df.groupby('user_id')['time_convert'].nunique()\
                                     .reset_index().rename(columns={'time_convert': 'user_active_day_'+suffix})
        period = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        user_day_count_df['user_active_day_ratio_'+suffix] = user_day_count_df['user_active_day_'+suffix]/float(period)
        # merge
        user_stat_feature_df = pd.merge(user_type_count_df, user_day_count_df[fs], how='left', on='user_id')
        joblib.dump(user_stat_feature_df, dump_path)
        print('user stat features %s to %s dumped' % (start_date, end_date))
    return user_stat_feature_df


def get_user_common_feature(start_date, end_date):
    dump_path = '../cache/user_common_feature_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        user_common_feature_df = joblib.load(dump_path)
        print('user common features %s to %s loaded') % (start_date, end_date)
    else:
        action_df = get_action_dataframe(start_date, end_date)
        # user act time gap features
        action_df['time_gap'] = (pd.Timestamp(end_date) - action_df['time'])/pd.Timedelta(1, 'D')
        t1 = action_df.groupby(['user_id'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap':'latest_user_act'})
        t2 = action_df[action_df.type == 4].groupby(['user_id'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap':'latest_user_buy_act'})
        t3 = action_df[action_df.type == 2].groupby(['user_id'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_cart_act'})
        t4 = action_df[action_df.type == 5].groupby(['user_id'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_favor_act'})
        # user act transfer ratio
        type_df = pd.get_dummies(action_df['type'], prefix='user_type')
        user_type_count_df = pd.concat([action_df['user_id'], type_df], axis=1)
        user_type_count_df = user_type_count_df.groupby(['user_id'], as_index=False).sum()
        user_type_count_df['user_type_1_ratio'] = user_type_count_df['user_type_4'] / user_type_count_df['user_type_1']
        user_type_count_df['user_type_2_ratio'] = user_type_count_df['user_type_4'] / user_type_count_df['user_type_2']
        user_type_count_df['user_type_3_ratio'] = user_type_count_df['user_type_4'] / user_type_count_df['user_type_3']
        user_type_count_df['user_type_5_ratio'] = user_type_count_df['user_type_4'] / user_type_count_df['user_type_5']
        user_type_count_df['user_type_6_ratio'] = user_type_count_df['user_type_4'] / user_type_count_df['user_type_6']
        user_type_count_df.replace([np.nan, np.infty], [0.0, 1.0], inplace = True)
        user_type_count_df = user_type_count_df[['user_id', 'user_type_1_ratio', 'user_type_2_ratio', 'user_type_3_ratio','user_type_5_ratio', 'user_type_6_ratio']]
        # user product & cate & brand count feature
        user_product_count_df = action_df.groupby('user_id')['sku_id'].nunique().reset_index().rename(columns={'sku_id':'user_product_count'})
        user_cate_count_df = action_df.groupby('user_id')['cate'].nunique().reset_index().rename(columns={'cate': 'user_cate_count'})
        user_brand_count_df = action_df.groupby('user_id')['brand'].nunique().reset_index().rename(columns={'brand': 'user_brand_count'})
        user_product_buy_count_df = action_df[action_df.type==4].groupby('user_id')['sku_id'].nunique().reset_index().rename(columns={'sku_id': 'user_product_buy_count'})
        user_cate_buy_count_df = action_df[action_df.type==4].groupby('user_id')['cate'].nunique().reset_index().rename(columns={'cate': 'user_cate_buy_count'})
        user_brand_buy_count_df = action_df[action_df.type == 4].groupby('user_id')['brand'].nunique().reset_index().rename(columns={'brand': 'user_brand_buy_count'})
        # merge
        user_common_feature_df = pd.merge(t1, t2, how='left', on=['user_id'])
        user_common_feature_df = pd.merge(user_common_feature_df, t3, how='left', on=['user_id'])
        user_common_feature_df = pd.merge(user_common_feature_df, t4, how='left', on=['user_id'])
        # user has open chain or not
        user_common_feature_df['user_open_chain'] = user_common_feature_df.latest_user_cart_act < user_common_feature_df.latest_user_buy_act.fillna(9999)
        user_common_feature_df = pd.merge(user_common_feature_df, user_type_count_df, how='left', on=['user_id'])
        user_common_feature_df = pd.merge(user_common_feature_df, user_product_count_df, how='left', on=['user_id'])
        user_common_feature_df = pd.merge(user_common_feature_df, user_cate_count_df, how='left', on=['user_id'])
        user_common_feature_df = pd.merge(user_common_feature_df, user_brand_count_df, how='left', on=['user_id'])
        user_common_feature_df = pd.merge(user_common_feature_df, user_product_buy_count_df, how='left', on=['user_id'])
        user_common_feature_df = pd.merge(user_common_feature_df, user_cate_buy_count_df, how='left', on=['user_id'])
        user_common_feature_df = pd.merge(user_common_feature_df, user_brand_buy_count_df, how='left', on=['user_id'])
        user_common_feature_df[['user_product_buy_count', 'user_cate_buy_count', 'user_brand_buy_count']] = user_common_feature_df[['user_product_buy_count','user_cate_buy_count','user_brand_buy_count']].fillna(0)
        user_common_feature_df.fillna(-1, inplace=True)
        joblib.dump(user_common_feature_df, dump_path)
        print('user common features %s to %s dumped' % (start_date, end_date))
    return user_common_feature_df


def get_product_stat_feature(start_date, end_date, suffix):
    dump_path = '../cache/product_stat_feature_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        product_stat_feature_df = joblib.load(dump_path)
        print('product stat features %s to %s loaded' % (start_date, end_date))
    else:
        action_df = get_action_dataframe(start_date, end_date)
        # product count features
        type_df = pd.get_dummies(action_df['type'], prefix='product_'+suffix+'_type')
        product_type_count_df = pd.concat([action_df['sku_id'], type_df], axis=1)
        product_type_count_df = product_type_count_df.groupby(['sku_id'], as_index=False).sum()
        product_type_count_df['i_num_'+suffix] = product_type_count_df.drop(['sku_id'], axis=1).sum(1)
        # product active features
        action_df['time_convert'] = action_df.time.dt.normalize()
        fs = ['sku_id', 'product_active_day_'+suffix, 'product_active_day_ratio_'+suffix]
        product_day_count_df = action_df.groupby('sku_id')['time_convert'].nunique().reset_index()\
                                        .rename(columns={'time_convert': 'product_active_day_'+suffix})
        period = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        product_day_count_df['product_active_day_ratio_'+suffix] = product_day_count_df['product_active_day_'+suffix] / float(period)
        # merge
        product_stat_feature_df = pd.merge(product_type_count_df, product_day_count_df[fs], how='left', on='sku_id')
        joblib.dump(product_stat_feature_df, dump_path)
        print('product stat features %s to %s dumped' % (start_date, end_date))
    return product_stat_feature_df


def get_product_common_feature(start_date, end_date):
    dump_path = '../cache/product_common_feature_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        product_common_feature_df = joblib.load(dump_path)
        print('product common features %s to %s loaded' % (start_date, end_date))
    else:
        action_df = get_action_dataframe(start_date, end_date)
        # product act time gap features
        action_df['time_gap'] = (pd.Timestamp(end_date) - action_df['time']) / pd.Timedelta(1, 'D')
        t1 = action_df.groupby(['sku_id'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_product_act'})
        t2 = action_df[action_df.type == 4].groupby(['sku_id'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_product_buy_act'})
        t3 = action_df[action_df.type == 2].groupby(['sku_id'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_product_cart_act'})
        t4 = action_df[action_df.type == 5].groupby(['sku_id'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_product_favor_act'})
        # product act transfer ratio
        type_df = pd.get_dummies(action_df['type'], prefix='product_type')
        product_type_count_df = pd.concat([action_df[['sku_id']], type_df], axis=1)
        product_type_count_df = product_type_count_df.groupby(['sku_id'], as_index=False).sum()
        product_type_count_df['product_type_1_ratio'] = product_type_count_df['product_type_4'] / product_type_count_df['product_type_1']
        product_type_count_df['product_type_2_ratio'] = product_type_count_df['product_type_4'] / product_type_count_df['product_type_2']
        product_type_count_df['product_type_3_ratio'] = product_type_count_df['product_type_4'] / product_type_count_df['product_type_3']
        product_type_count_df['product_type_5_ratio'] = product_type_count_df['product_type_4'] / product_type_count_df['product_type_5']
        product_type_count_df['product_type_6_ratio'] = product_type_count_df['product_type_4'] / product_type_count_df['product_type_6']
        product_type_count_df.replace([np.nan, np.infty], [0.0, 1.0], inplace=True)
        product_type_count_df = product_type_count_df[['sku_id', 'product_type_1_ratio', 'product_type_2_ratio', 'product_type_3_ratio', 'product_type_5_ratio','product_type_6_ratio']]
        # product user count feature
        product_user_count_df = action_df.groupby('sku_id')['user_id'].nunique().reset_index().rename(columns={'user_id': 'product_user_count'})
        product_user_buy_count_df = action_df[action_df.type == 4].groupby('sku_id')['user_id'].nunique().reset_index().rename(columns={'user_id': 'product_user_buy_count'})
        # ranking feature
        action_df['ci_num'] = action_df.groupby(['cate', 'sku_id'], as_index=False)['type'].transform(lambda x: len(x))
        action_df['ci_rank'] = action_df.groupby('cate')['ci_num'].rank(method='dense', ascending=False)
        action_df['bi_num'] = action_df.groupby(['brand', 'sku_id'], as_index=False)['type'].transform(lambda x: len(x))
        action_df['bi_rank'] = action_df.groupby('brand')['bi_num'].rank(method='dense', ascending=False)
        action_df['cb_num'] = action_df.groupby(['cate', 'brand'], as_index=False)['type'].transform(lambda x: len(x))
        action_df['cb_rank'] = action_df.groupby('cate')['cb_num'].rank(method='dense', ascending=False)
        product_rank_df = action_df[['sku_id', 'ci_rank', 'bi_rank', 'cb_rank']].drop_duplicates()
        # merge
        product_common_feature_df = pd.merge(t1, t2, how='left', on=['sku_id'])
        product_common_feature_df = pd.merge(product_common_feature_df, t3, how='left', on=['sku_id'])
        product_common_feature_df = pd.merge(product_common_feature_df, t4, how='left', on=['sku_id'])
        product_common_feature_df = pd.merge(product_common_feature_df, product_type_count_df, how='left', on=['sku_id'])
        product_common_feature_df = pd.merge(product_common_feature_df, product_user_count_df, how='left', on=['sku_id'])
        product_common_feature_df = pd.merge(product_common_feature_df, product_user_buy_count_df, how='left', on=['sku_id'])
        product_common_feature_df['product_user_buy_count'] = product_common_feature_df['product_user_buy_count'].fillna(0)
        product_common_feature_df = pd.merge(product_common_feature_df, product_rank_df, how='left', on=['sku_id'])
        product_common_feature_df.fillna(-1, inplace=True)
        joblib.dump(product_common_feature_df, dump_path)
        print "product common features %s to %s dumped" % (start_date, end_date)
    return product_common_feature_df


def get_brand_stat_feature(start_date, end_date, suffix):
    dump_path = '../cache/brand_stat_feature_%s_%s.data' %(start_date, end_date)
    if os.path.exists(dump_path):
        brand_stat_feature_df = joblib.load(dump_path)
        print('brand stat features %s to %s loaded' % (start_date, end_date))
    else:
        action_df = get_action_dataframe(start_date, end_date)
        # brand active types
        type_df = pd.get_dummies(action_df['type'], prefix='brand_'+suffix+'_type')
        brand_type_count_df = pd.concat([action_df['brand'], type_df], axis=1)
        brand_type_count_df = brand_type_count_df.groupby(['brand'], as_index=False).sum()
        brand_type_count_df['b_num_' + suffix] = brand_type_count_df.drop(['brand'], axis=1).sum(1)
        # brand active days
        action_df['time_convert'] = action_df.time.dt.normalize()
        fs = ['brand', 'brand_active_day_'+suffix, 'brand_active_day_ratio_'+suffix]
        brand_day_count_df = action_df.groupby('brand')['time_convert'].nunique()\
                                     .reset_index().rename(columns={'time_convert': 'brand_active_day_'+suffix})
        period = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        brand_day_count_df['brand_active_day_ratio_'+suffix] = brand_day_count_df['brand_active_day_'+suffix]/float(period)
        # merge
        brand_stat_feature_df = pd.merge(brand_type_count_df, brand_day_count_df[fs], how='left', on='brand')
        joblib.dump(brand_stat_feature_df, dump_path)
        print('brand stat features %s to %s dumped' % (start_date, end_date))
    return brand_stat_feature_df


def get_brand_common_feature(start_date, end_date):
    dump_path = '../cache/brand_common_feature_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        brand_common_feature_df = joblib.load(dump_path)
        print('brand common features %s to %s loaded' % (start_date, end_date))
    else:
        action_df = get_action_dataframe(start_date, end_date)
        # action type
        type_df = pd.get_dummies(action_df['type'], prefix='brand_type')
        brand_type_count_df = pd.concat([action_df['brand'], type_df], axis=1)
        brand_type_count_df = brand_type_count_df.groupby('brand',as_index=False).sum()
        brand_type_count_df['brand_type_1_ratio'] = brand_type_count_df['brand_type_4'] / brand_type_count_df['brand_type_1']
        brand_type_count_df['brand_type_2_ratio'] = brand_type_count_df['brand_type_4'] / brand_type_count_df['brand_type_2']
        brand_type_count_df['brand_type_3_ratio'] = brand_type_count_df['brand_type_4'] / brand_type_count_df['brand_type_3']
        brand_type_count_df['brand_type_5_ratio'] = brand_type_count_df['brand_type_4'] / brand_type_count_df['brand_type_5']
        brand_type_count_df['brand_type_6_ratio'] = brand_type_count_df['brand_type_4'] / brand_type_count_df['brand_type_6']
        brand_type_count_df.replace([np.nan, np.infty], [0.0, 1.0], inplace=True)
        brand_type_count_df = brand_type_count_df[['brand', 'brand_type_1_ratio', 'brand_type_2_ratio', 'brand_type_3_ratio', 'brand_type_5_ratio', 'brand_type_6_ratio']]
        # brand count feature
        brand_user_count_df = action_df.groupby('brand')['user_id'].nunique().reset_index().rename(columns={'user_id':'brand_user_count'})
        brand_user_buy_count_df = action_df[action_df.type==4].groupby('brand')['user_id'].nunique().reset_index().rename(columns={'user_id': 'brand_user_buy_count'})
        brand_product_count_df = action_df.groupby('brand')['sku_id'].nunique().reset_index().rename(columns={'sku_id': 'brand_product_count'})
        brand_product_buy_count_df = action_df[action_df.type==4].groupby('brand')['sku_id'].nunique().reset_index().rename(columns={'sku_id': 'brand_product_buy_count'})
        # time gap
        action_df['time_gap'] = (pd.Timestamp(end_date) - action_df['time'])/pd.Timedelta(1, 'D')
        t1 = action_df.groupby(['brand'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap':'latest_brand_act'})
        t2 = action_df[action_df.type == 4].groupby(['brand'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap':'latest_brand_buy_act'})
        t3 = action_df[action_df.type == 2].groupby(['brand'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_brand_cart_act'})
        t4 = action_df[action_df.type == 5].groupby(['brand'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_brand_favor_act'})
        # merge
        brand_common_feature_df = pd.merge(t1, t2, how='left', on=['brand'])
        brand_common_feature_df = pd.merge(brand_common_feature_df, t3, how='left', on=['brand'])
        brand_common_feature_df = pd.merge(brand_common_feature_df, t4, how='left', on=['brand'])
        brand_common_feature_df = pd.merge(brand_common_feature_df, brand_type_count_df, how='left', on=['brand'])
        brand_common_feature_df = pd.merge(brand_common_feature_df, brand_user_count_df, how='left', on=['brand'])
        brand_common_feature_df = pd.merge(brand_common_feature_df, brand_user_buy_count_df, how='left', on=['brand'])
        brand_common_feature_df = pd.merge(brand_common_feature_df, brand_product_buy_count_df, how='left', on=['brand'])
        brand_common_feature_df = pd.merge(brand_common_feature_df, brand_product_count_df, how='left', on=['brand'])
        brand_common_feature_df[['brand_user_count', 'brand_product_buy_count','brand_product_count', 'brand_user_buy_count']] \
            = brand_common_feature_df[['brand_user_count', 'brand_product_buy_count','brand_product_count', 'brand_user_buy_count']].fillna(0)
        brand_common_feature_df.fillna(-1, inplace=True)
        joblib.dump(brand_common_feature_df, dump_path)
        print "brand common features %s to %s dumped" % (start_date, end_date)
    return brand_common_feature_df


def get_user_product_common_feature(start_date, end_date):
    dump_path = '../cache/user_product_common_feature_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        user_product_common_feature_df = joblib.load(dump_path)
        print('user_product common features %s to %s loaded' % (start_date, end_date))
    else:
        action_df = get_action_dataframe(start_date, end_date)
        # user product count features
        type_df = pd.get_dummies(action_df['type'], prefix='user_product_type')
        user_product_type_count_df = pd.concat([action_df[['user_id','sku_id']], type_df], axis=1)
        user_product_type_count_df = user_product_type_count_df.groupby(['user_id','sku_id'], as_index=False).sum()
        # ranking features
        user_product_type_count_df['ui_num'] = user_product_type_count_df.drop(['user_id','sku_id'],axis=1).sum(1)
        user_product_type_count_df['ui_rank'] = user_product_type_count_df.groupby('user_id', as_index=False)['ui_num'].transform(lambda x: x.rank(method='dense', ascending=False))
        # user product active features
        action_df['time_convert'] = action_df.time.dt.normalize()
        user_product_day_count_df = action_df.groupby(['user_id','sku_id'])['time_convert'].nunique().reset_index().rename(columns={'time_convert': 'user_product_active_day'})
        # user product act time gap features
        action_df['time_gap'] = (pd.Timestamp(end_date) - action_df['time']) / pd.Timedelta(1, 'D')
        t1 = action_df.groupby(['user_id', 'sku_id'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_product_act'})
        t2 = action_df[action_df.type == 4].groupby(['user_id', 'sku_id'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_product_buy_act'})
        t3 = action_df[action_df.type == 2].groupby(['user_id', 'sku_id'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_product_cart_act'})
        t4 = action_df[action_df.type == 5].groupby(['user_id', 'sku_id'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_product_favor_act'})
        # merge
        user_product_common_feature_df = pd.merge(t1, t2, how='left', on=['user_id', 'sku_id'])
        user_product_common_feature_df = pd.merge(user_product_common_feature_df, t3, how='left', on=['user_id', 'sku_id'])
        user_product_common_feature_df = pd.merge(user_product_common_feature_df, t4, how='left', on=['user_id', 'sku_id'])
        # user has buy this product before or not
        user_product_common_feature_df['user_product_bought'] = ~user_product_common_feature_df.latest_user_product_buy_act.isnull()
        # user product has open chain or not
        user_product_common_feature_df['user_product_open_chain'] = user_product_common_feature_df.latest_user_product_cart_act < user_product_common_feature_df.latest_user_product_buy_act.fillna(9999)
        user_product_common_feature_df = pd.merge(user_product_common_feature_df, user_product_type_count_df, how='left', on=['user_id','sku_id'])
        user_product_common_feature_df = pd.merge(user_product_common_feature_df, user_product_day_count_df, how='left', on=['user_id', 'sku_id'])
        user_product_common_feature_df.fillna(-1, inplace=True)
        joblib.dump(user_product_common_feature_df, dump_path)
        print('user_product common features %s to %s dumped' % (start_date, end_date))
    return user_product_common_feature_df


def get_user_cate_common_feature(start_date, end_date):
    dump_path = '../cache/user_cate_common_feature_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        user_cate_common_feature_df = joblib.load(dump_path)
        print('user_cate common features %s to %s loaded' % (start_date, end_date))
    else:
        action_df = get_action_dataframe(start_date, end_date)
        # user cate count features
        type_df = pd.get_dummies(action_df['type'], prefix='user_cate_type')
        user_cate_type_count_df = pd.concat([action_df[['user_id', 'cate']], type_df], axis=1)
        user_cate_type_count_df = user_cate_type_count_df.groupby(['user_id', 'cate'], as_index=False).sum()
        #ranking features
        user_cate_type_count_df['uc_num'] = user_cate_type_count_df.drop(['user_id', 'cate'], axis=1).sum(1)
        user_cate_type_count_df['uc_rank'] = user_cate_type_count_df.groupby('user_id', as_index=False)['uc_num'].transform(lambda x: x.rank(method='dense', ascending=False))
        # user cate active features
        action_df['time_convert'] = action_df.time.dt.normalize()
        user_cate_day_count_df = action_df.groupby(['user_id', 'cate'])['time_convert'].nunique().reset_index().rename(columns={'time_convert': 'user_cate_active_day'})
        # user cate act time gap features
        action_df['time_gap'] = (pd.Timestamp(end_date) - action_df['time']) / pd.Timedelta(1, 'D')
        t1 = action_df.groupby(['user_id', 'cate'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_cate_act'})
        t2 = action_df[action_df.type == 4].groupby(['user_id', 'cate'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_cate_buy_act'})
        t3 = action_df[action_df.type == 2].groupby(['user_id', 'cate'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_cate_cart_act'})
        t4 = action_df[action_df.type == 5].groupby(['user_id', 'cate'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_cate_favor_act'})
        # merge
        user_cate_common_feature_df = pd.merge(t1, t2, how='left', on=['user_id', 'cate'])
        user_cate_common_feature_df = pd.merge(user_cate_common_feature_df, t3, how='left', on=['user_id', 'cate'])
        user_cate_common_feature_df = pd.merge(user_cate_common_feature_df, t4, how='left', on=['user_id', 'cate'])
        # user has buy this cate before or not
        user_cate_common_feature_df['user_cate_bought'] = ~user_cate_common_feature_df.latest_user_cate_buy_act.isnull()
        user_cate_common_feature_df = pd.merge(user_cate_common_feature_df, user_cate_type_count_df, how='left', on=['user_id', 'cate'])
        user_cate_common_feature_df = pd.merge(user_cate_common_feature_df, user_cate_day_count_df, how='left', on=['user_id', 'cate'])
        user_cate_common_feature_df.fillna(-1, inplace=True)
        joblib.dump(user_cate_common_feature_df, dump_path)
        print('user_cate common features %s to %s dumped' % (start_date, end_date))
    return user_cate_common_feature_df


def get_user_chain_feature(start_date, end_date):
    dump_path = '../cache/user_chain_feature_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        user_chain_df = joblib.load(dump_path)
        print('user chain features %s to %s loaded' % (start_date, end_date))
    else:
        action_df = get_action_dataframe(start_date, end_date)
        # 购买的数目
        action_df['buy_num'] = map(lambda x: 1 if x == 4 else 0, action_df['type'])
        buy_df = action_df.groupby(['user_id', 'sku_id'], as_index=False)['buy_num'].sum()
        buy_df = buy_df[buy_df['buy_num'] > 0]

        # 最早交互时间 最晚购买时间
        action_df['time_gap'] = (pd.Timestamp(end_date) - action_df['time']) / pd.Timedelta(1, 'D')
        t1 = action_df.groupby(['user_id', 'sku_id'], as_index=False)['time_gap'].agg(np.max).reset_index().rename(columns={'time_gap': 'first_ui_act'})
        t2 = action_df[action_df['type'] == 4].groupby(['user_id', 'sku_id'], as_index=False)['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'last_ui_buy'})
        user_chain_df = pd.merge(buy_df, t1, how='left', on=['user_id', 'sku_id'])
        user_chain_df = pd.merge(user_chain_df, t2, how='left', on=['user_id', 'sku_id'])
        # user购买周期的min avg max
        user_chain_df['buy_period'] = user_chain_df['first_ui_act']-user_chain_df['last_ui_buy']
        user_chain_df['buy_period'] = user_chain_df['buy_period']/user_chain_df['buy_num']
        t1 = user_chain_df.groupby('user_id', as_index=False)['buy_period'].agg(np.min).reset_index().rename(columns={'buy_period': 'min_buy_period'})
        t2 = user_chain_df.groupby('user_id', as_index=False)['buy_period'].agg(np.mean).reset_index().rename(columns={'buy_period': 'avg_buy_period'})
        t3 = user_chain_df.groupby('user_id', as_index=False)['buy_period'].agg(np.max).reset_index().rename(columns={'buy_period': 'max_buy_period'})
        user_chain_df = pd.merge(user_chain_df, t1, how='left', on=['user_id'])
        user_chain_df = pd.merge(user_chain_df, t2, how='left', on=['user_id'])
        user_chain_df = pd.merge(user_chain_df, t3, how='left', on=['user_id'])
        user = action_df[['user_id']].drop_duplicates()
        user_chain_df = user_chain_df[['user_id', 'min_buy_period', 'avg_buy_period', 'max_buy_period']].drop_duplicates()
        user_chain_df = pd.merge(user, user_chain_df, how='left', on=['user_id'])
        user_chain_df.fillna(-1, inplace=True)
        joblib.dump(user_chain_df, dump_path)
        print('user chain common features %s to %s dumped' % (start_date, end_date))
    return user_chain_df


def get_user_brand_common_feature(start_date, end_date):
    dump_path = '../cache/user_brand_common_feature_%s_%s.data' % (start_date, end_date)
    if os.path.exists(dump_path):
        user_brand_common_feature_df = joblib.load(dump_path)
        print('user_brand common features %s to %s loaded' % (start_date, end_date))
    else:
        action_df = get_action_dataframe(start_date, end_date)
        # user brand count features
        type_df = pd.get_dummies(action_df['type'], prefix='user_brand_type')
        user_brand_type_count_df = pd.concat([action_df[['user_id', 'brand']], type_df], axis=1)
        user_brand_type_count_df = user_brand_type_count_df.groupby(['user_id', 'brand'], as_index=False).sum()
        # ranking features
        user_brand_type_count_df['ub_num'] = user_brand_type_count_df.drop(['user_id','brand'],axis=1).sum(1)
        user_brand_type_count_df['ub_rank'] = user_brand_type_count_df.groupby('user_id', as_index=False)['ub_num'].transform(lambda x: x.rank(method='dense', ascending=False))
        # user brand active features
        action_df['time_convert'] = action_df.time.dt.normalize()
        user_brand_day_count_df = action_df.groupby(['user_id', 'brand'])['time_convert'].nunique().reset_index().rename(columns={'time_convert': 'user_brand_active_day'})
        # user cate act time gap features
        action_df['time_gap'] = (pd.Timestamp(end_date) - action_df['time']) / pd.Timedelta(1, 'D')
        t1 = action_df.groupby(['user_id', 'brand'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_brand_act'})
        t2 = action_df[action_df.type == 4].groupby(['user_id', 'brand'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_brand_buy_act'})
        t3 = action_df[action_df.type == 2].groupby(['user_id', 'brand'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_brand_cart_act'})
        t4 = action_df[action_df.type == 5].groupby(['user_id', 'brand'])['time_gap'].agg(np.min).reset_index().rename(columns={'time_gap': 'latest_user_brand_favor_act'})
        # merge
        user_brand_common_feature_df = pd.merge(t1, t2, how='left', on=['user_id', 'brand'])
        user_brand_common_feature_df = pd.merge(user_brand_common_feature_df, t3, how='left', on=['user_id', 'brand'])
        user_brand_common_feature_df = pd.merge(user_brand_common_feature_df, t4, how='left', on=['user_id', 'brand'])
        # user has buy this brand before or not
        user_brand_common_feature_df['user_brand_bought'] = ~user_brand_common_feature_df.latest_user_brand_buy_act.isnull()
        user_brand_common_feature_df = pd.merge(user_brand_common_feature_df, user_brand_type_count_df, how='left', on=['user_id', 'brand'])
        user_brand_common_feature_df = pd.merge(user_brand_common_feature_df, user_brand_day_count_df, how='left', on=['user_id', 'brand'])
        user_brand_common_feature_df.fillna(-1, inplace=True)
        joblib.dump(user_brand_common_feature_df, dump_path)
        print('user_cate common features %s to %s dumped' % (start_date, end_date))
    return user_brand_common_feature_df


if __name__ == '__main__':
    start_date = '2016-03-27'
    end_date = '2016-04-11'
    test = get_user_chain_feature(start_date, end_date)
    print test.head()