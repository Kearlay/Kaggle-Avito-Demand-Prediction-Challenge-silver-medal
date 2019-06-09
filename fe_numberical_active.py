# -*- coding: utf-8 -*-

#@author: chenxinye

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gc
import numpy as np
import pickle

test_id = pd.read_csv('../input/test.csv',usecols=['item_id'])
df_all = pickle.load(open('./data/df_all.pkl','rb'))


df_all_tmp = df_all.copy()
raw_columns = df_all_tmp.columns.values
gc.collect()

# numberical aggregate features
def agg(df,agg_cols):
    for c in tqdm(agg_cols):
        new_feature = '{}_{}_{}'.format('_'.join(c['groupby']), c['agg'], c['target'])
        gp = df.groupby(c['groupby'])[c['target']].agg(c['agg']).reset_index().rename(index=str, columns={c['target']:new_feature})
        df = df.merge(gp,on=c['groupby'],how='left')
    return df

agg_cols = [

############################unique aggregation##################################
    {'groupby': ['user_id'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['parent_category_name'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['category_name'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['region'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['city'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['image_top_1'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['wday'], 'target':'price', 'agg':'nunique'},
    {'groupby': ['param_1'], 'target':'price', 'agg':'nunique'},

    {'groupby': ['category_name'], 'target':'image_top_1', 'agg':'nunique'},
    {'groupby': ['parent_category_name'], 'target':'image_top_1', 'agg':'nunique'},    
    {'groupby': ['user_id'], 'target':'image_top_1', 'agg':'nunique'},
    
    
    {'groupby': ['user_id'], 'target':'parent_category_name', 'agg':'nunique'},
    {'groupby': ['user_id'], 'target':'category_name', 'agg':'nunique'},
    {'groupby': ['user_id'], 'target':'wday', 'agg':'nunique'},
    {'groupby': ['user_id'], 'target':'param_1', 'agg':'nunique'},
    
############################count aggregation##################################  
    {'groupby': ['user_id'], 'target':'item_id', 'agg':'count'},

    {'groupby': ['user_id','param_1'], 'target':'item_id', 'agg':'count'},

    {'groupby': ['user_id','region'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','city'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','parent_category_name'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','category_name'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','wday'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','image_top_1'], 'target':'item_id', 'agg':'count'},
 
    {'groupby': ['user_id','wday','category_name'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','wday','image_top_1'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','wday','parent_category_name'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','wday','city'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','wday','region'], 'target':'item_id', 'agg':'count'},
    
    {'groupby': ['user_id','category_name','city'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['user_id','wday','category_name','city'], 'target':'item_id', 'agg':'count'},
    
    {'groupby': ['price'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['price','user_id'], 'target':'item_id', 'agg':'count'},
    {'groupby': ['price','category_name'], 'target':'item_id', 'agg':'count'},
    
############################mean/median/sum/min/max aggregation##################################    
    
    {'groupby': ['image_top_1','user_id'], 'target':'price', 'agg':'mean'},
    {'groupby': ['image_top_1','user_id'], 'target':'price', 'agg':'median'},
    {'groupby': ['image_top_1','user_id'], 'target':'price', 'agg':'sum'},
    {'groupby': ['image_top_1','user_id'], 'target':'price', 'agg':'max'},

    {'groupby': ['param_2'], 'target':'price', 'agg':'mean'},
    {'groupby': ['param_2'], 'target':'price', 'agg':'max'},
    {'groupby': ['param_3'], 'target':'price', 'agg':'mean'},
    {'groupby': ['param_3'], 'target':'price', 'agg':'max'},
    
    {'groupby': ['user_id'], 'target':'price', 'agg':'mean'},
    {'groupby': ['user_id'], 'target':'price', 'agg':'median'},
    {'groupby': ['user_id'], 'target':'price', 'agg':'sum'},
    {'groupby': ['user_id'], 'target':'price', 'agg':'min'},
    {'groupby': ['user_id'], 'target':'price', 'agg':'max'},

    {'groupby': ['item_seq_number'], 'target':'price', 'agg':'mean'},
    {'groupby': ['item_seq_number'], 'target':'price', 'agg':'median'},
    {'groupby': ['item_seq_number'], 'target':'price', 'agg':'sum'},
    {'groupby': ['item_seq_number'], 'target':'price', 'agg':'min'},
    {'groupby': ['item_seq_number'], 'target':'price', 'agg':'max'},


    {'groupby': ['image_top_1'], 'target':'price', 'agg':'mean'},
    {'groupby': ['image_top_1'], 'target':'price', 'agg':'median'},
    {'groupby': ['image_top_1'], 'target':'price', 'agg':'sum'},
    {'groupby': ['image_top_1'], 'target':'price', 'agg':'max'},

    {'groupby': ['param_1'], 'target':'price', 'agg':'mean'},
    {'groupby': ['param_1'], 'target':'price', 'agg':'max'},


    {'groupby': ['region'], 'target':'price', 'agg':'mean'},
    {'groupby': ['region'], 'target':'price', 'agg':'max'},
    
    {'groupby': ['city'], 'target':'price', 'agg':'mean'},
    {'groupby': ['city'], 'target':'price', 'agg':'max'},
    
    {'groupby': ['parent_category_name'], 'target':'price', 'agg':'mean'},
    {'groupby': ['parent_category_name'], 'target':'price', 'agg':'sum'},
    {'groupby': ['parent_category_name'], 'target':'price', 'agg':'max'},

    {'groupby': ['category_name'], 'target':'price', 'agg':'mean'},
    {'groupby': ['category_name'], 'target':'price', 'agg':'sum'},
    {'groupby': ['category_name'], 'target':'price', 'agg':'max'},   
    
    {'groupby': ['wday','category_name','city'], 'target':'price', 'agg':'mean'},
    {'groupby': ['wday','category_name','city'], 'target':'price', 'agg':'median'},
    {'groupby': ['wday','category_name','city'], 'target':'price', 'agg':'sum'},
    {'groupby': ['wday','category_name','city'], 'target':'price', 'agg':'max'},
    
    
    {'groupby': ['user_id'], 'target':'item_id_sum_days_up', 'agg':'mean'},
    {'groupby': ['user_id'], 'target':'item_id_count_days_up', 'agg':'mean'},     
    

]

df_all_tmp = agg(df_all_tmp,agg_cols)
tmp_columns = df_all_tmp.columns.values

df_train = df_all_tmp[df_all_tmp['deal_probability'].notnull()]
df_test = test_id.merge(df_all_tmp,on='item_id',how='left')
del df_all_tmp
gc.collect()
for i in tmp_columns:
    if i not in raw_columns:
        print (i)
        df_train[i].to_pickle('./data/features/number_agg/train/' + str(i))
        df_test[i].to_pickle('./data/features/number_agg/test/' + str(i))  
        