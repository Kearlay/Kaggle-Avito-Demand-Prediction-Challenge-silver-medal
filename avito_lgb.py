# -*- coding: utf-8 -*-

#@author: chenxinye

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gc
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy import sparse
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix
import os
import glob

df_all = pickle.load(open('./data/basic.pkl','rb'))
df_train = df_all[df_all['deal_probability'].notnull()]
df_test = df_all[df_all['deal_probability'].isnull()].reset_index(drop=True)
y = df_all[df_all['deal_probability'].notnull()].deal_probability

# tfidf features
ready_df_train = sparse.load_npz('./data/features/nlp/ready_df_train_200000_new.npz')
ready_df_test = sparse.load_npz('./data/features/nlp/ready_df_test_200000_new.npz')
tfvocab = pickle.load(open('./data/features/nlp/tfvocab_200000_new.pkl', 'rb'))

# text agg features
for fn in glob.glob('./data/features/text_agg/train/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_train[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))
    
for fn in glob.glob('./data/features/text_agg/test/*'):
    tmp = pickle.load(open(fn,'rb')).reset_index(drop=True)
    df_test[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))  

# numberical agg features
for fn in glob.glob('./data/features/number_agg/train/*'):
    tmp = pickle.load(open(fn,'rb'))
    df_train[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))
    
for fn in glob.glob('./data/features/number_agg/test/*'):
    tmp = pickle.load(open(fn,'rb'))
    df_test[os.path.basename(fn)] = tmp
    del tmp
    gc.collect()
    #print (os.path.basename(fn))        

# price diff features
df_train['image_top_1_diff_price'] = df_train['price'] - df_train['image_top_1_median_price']
df_train['parent_category_name_diff_price'] = df_train['price'] - df_train['parent_category_name_mean_price']
df_train['category_name_diff_price'] = df_train['price'] - df_train['category_name_mean_price']
df_train['param_1_diff_price'] = df_train['price'] - df_train['param_1_mean_price']
df_train['param_2_diff_price'] = df_train['price'] - df_train['param_2_mean_price']
df_train['item_seq_number_diff_price'] = df_train['price'] - df_train['item_seq_number_mean_price']
df_train['user_id_diff_price'] = df_train['price'] - df_train['user_id_mean_price']
df_train['region_diff_price'] = df_train['price'] - df_train['region_mean_price']
df_train['city_diff_price'] = df_train['price'] - df_train['city_mean_price']

df_test['image_top_1_diff_price'] = df_test['price'] - df_test['image_top_1_median_price']
df_test['parent_category_name_diff_price'] = df_test['price'] - df_test['parent_category_name_mean_price']
df_test['category_name_diff_price'] = df_test['price'] - df_test['category_name_mean_price']
df_test['param_1_diff_price'] = df_test['price'] - df_test['param_1_mean_price']
df_test['param_2_diff_price'] = df_test['price'] - df_test['param_2_mean_price']
df_test['item_seq_number_diff_price'] = df_test['price'] - df_test['item_seq_number_mean_price']
df_test['user_id_diff_price'] = df_test['price'] - df_test['user_id_mean_price']
df_test['region_diff_price'] = df_test['price'] - df_test['region_mean_price']
df_test['city_diff_price'] = df_test['price'] - df_test['city_mean_price']

categorical = ['param_123']
for feature in categorical:
    print(f'Transforming {feature}...')
    encoder = LabelEncoder()
    encoder.fit(df_train[feature].append(df_test[feature]).astype(str))
    
list(df_train.columns)

df_train.param_123

# create train test 
df_train = df_train.drop(['item_id','user_id','title','text','param_123','description','activation_date','image',
                'deal_probability'],axis=1)   
df_train['item_seq_number'] = np.log1p(df_train['item_seq_number'])
df_train.fillna(-1,inplace=True)

df_test = df_test.drop(['item_id','user_id','title','text','param_123','description','activation_date','image',
                'deal_probability'],axis=1) 
df_test['item_seq_number'] = np.log1p(df_test['item_seq_number'])
df_test.fillna(-1,inplace=True)

df_train = df_train.astype(np.float)
df_test = df_test.astype(np.float)

X_tr = hstack([csr_matrix(df_train),ready_df_train]) # Sparse Matrix
X_test = hstack([csr_matrix(df_test),ready_df_test])

tfvocab = df_train.columns.tolist() + tfvocab
for shape in [X_tr,X_test]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))

# train lightgbm model
from sklearn.model_selection import StratifiedKFold

X = X_tr.tocsr()
#del X_tra
gc.collect()

test_pred = np.zeros(X_test.shape[0])
cat_features=['region','city','parent_category_name',
              'category_name','user_type','image_top_1','wday','param_1']

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.019,
    'num_leaves': 300,  
    #'max_depth': 15,  
    'max_bin': 256,  
    'subsample': 0.7,  
    'colsample_bytree': 0.4,  
    'reg_alpha': 0,  
    'reg_lambda': 0,  
    'verbose': 1
    }

MAX_ROUNDS = 15000
NFOLDS = 5
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=224)


for i,(train_index,val_index) in enumerate(kfold.split(X,y)):
    print("Running fold {} / {}".format(i + 1, NFOLDS))
    print("Train Index:",train_index,",Val Index:",val_index)
    X_tra,X_val,y_tra,y_val = X[train_index, :], X[val_index, :], y[train_index], y[val_index]
    if i >=0:

        dtrain = lgb.Dataset(
            X_tra, label=y_tra, feature_name=tfvocab, categorical_feature=cat_features)
        dval = lgb.Dataset(
            X_val, label=y_val, reference=dtrain, feature_name=tfvocab, categorical_feature=cat_features)    
        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dval], early_stopping_rounds=200, verbose_eval=200)

        del dtrain,dval
        del X_tra,y_tra,y_val,X_val
        gc.collect()

        test_pred = bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS)
        test_pred.dump('../models/kfold_' + str(i) + '.pkl')
        del test_pred
        gc.collect()

pred_1 = pickle.load(open('../models/kfold_0.pkl','rb'))
pred_2 = pickle.load(open('../models/kfold_1.pkl','rb'))
pred_3 = pickle.load(open('../models/kfold_2.pkl','rb'))
pred_4 = pickle.load(open('../models/kfold_3.pkl','rb'))
pred_5 = pickle.load(open('../models/kfold_4.pkl','rb'))

df_test = pd.read_csv("../input/test.csv")  
lgpred = (pred_1+pred_2+pred_3+pred_4+pred_5)/5
df_test.head()
lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=df_test['item_id'])
lgsub['deal_probability'] = np.clip(lgsub['deal_probability'], 0.0, 1.0) # Between 0 and 1
lgsub.to_csv("lgsub.csv",index=True,header=True)