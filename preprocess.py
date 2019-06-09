# -*- coding: utf-8 -*-

#@author: chenxinye

import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

train = pd.read_csv('../input/train.csv', parse_dates = ['activation_date'])
test = pd.read_csv('../input/test.csv', parse_dates = ['activation_date'])

df_all = pd.concat([train,test],axis=0).reset_index(drop=True)
df_all['wday'] = df_all['activation_date'].dt.weekday
df_all['price'] = np.log1p(df_all['price'])

df_all['param_123'] = (df_all['param_1'].fillna('') + ' ' + df_all['param_2'].fillna('') + ' ' + df_all['param_3'].fillna('')).astype(str)
df_all['title'] = df_all['title'].fillna('').astype(str)
df_all['text'] = df_all['description'].fillna('').astype(str) + ' ' + df_all['title'].fillna('').astype(str) + ' ' + df_all['param_123'].fillna('').astype(str)

text_vars = ['user_id','region', 'city', 'parent_category_name', 'category_name', 'user_type','param_1','param_2','param_3']
for col in tqdm(text_vars):
    lbl = LabelEncoder()
    lbl.fit(df_all[col].values.astype('str'))
    df_all[col] = lbl.transform(df_all[col].values.astype('str'))

    # for basic model
df_all.to_pickle('./data/basic.pkl')

# for text feature engineering
df_all[['deal_probability','title','param_123','text']].to_pickle('./data/df_text.pkl')