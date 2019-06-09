# -*- coding: utf-8 -*-

#@author: chenxinye

import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

train = pd.read_csv('../input/train.csv', parse_dates = ['activation_date'])
test = pd.read_csv('../input/test.csv', parse_dates = ['activation_date'])
train_active = pd.read_csv('../input/train_active.csv',parse_dates = ['activation_date'])
test = pd.read_csv('../input/test.csv')
test_active = pd.read_csv('../input/test_active.csv',parse_dates = ['activation_date'])
train_periods = pd.read_csv('../input/periods_train.csv', parse_dates=['activation_date','date_from', 'date_to'])
test_periods = pd.read_csv('../input/periods_test.csv', parse_dates=['activation_date','date_from', 'date_to'])

df_all = pd.concat([
    train,
    train_active,
    test,
    test_active
]).reset_index(drop=True)
df_all.drop_duplicates(['item_id'], inplace=True)


df_all["activation_date"] = pd.to_datetime(df_all["activation_date"])

import numpy as np
df_all['wday'] = df_all['activation_date'].dt.weekday

df_all['price'] = np.log1p(df_all['price'])

all_periods = pd.concat([
    train_periods,
    test_periods
])


all_periods['days_up'] = all_periods['date_to'].dt.dayofyear - all_periods['date_from'].dt.dayofyear
all_periods['days_total'] = all_periods['date_to'].dt.dayofyear - all_periods['activation_date'].dt.dayofyear


def agg(df,agg_cols):
    for c in tqdm(agg_cols):
        new_feature = '{}_{}_{}'.format('_'.join(c['groupby']), c['agg'], c['target'])
        gp = df.groupby(c['groupby'])[c['target']].agg(c['agg']).reset_index().rename(index=str, columns={c['target']:new_feature})
        df = df.merge(gp,on=c['groupby'],how='left')
    return df

agg_cols = [
    {'groupby': ['item_id'], 'target':'days_up', 'agg':'count'},
    {'groupby': ['item_id'], 'target':'days_up', 'agg':'sum'},   
    {'groupby': ['item_id'], 'target':'days_total', 'agg':'sum'},  
]

all_periods = agg(all_periods,agg_cols)

all_periods.drop_duplicates(['item_id'], inplace=True)
all_periods.drop(['activation_date','date_from','date_to','days_up','days_total'],axis=1, inplace=True)
all_periods.reset_index(drop=True,inplace=True)

df_all = df_all.merge(all_periods, on='item_id', how='left')

df_all.to_pickle('./data/df_all.pkl')

