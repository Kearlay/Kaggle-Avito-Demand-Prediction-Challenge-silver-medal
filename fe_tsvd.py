# -*- coding: utf-8 -*-

#@author: chenxinye

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gc
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

df_text = pickle.load(open('./data/df_text.pkl','rb'))
df_train_text = df_text[df_text['deal_probability'].notnull()]
df_test_text = df_text[df_text['deal_probability'].isnull()]

### title TFIDF Vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,1))
full_title_tfidf = tfidf_vec.fit_transform(df_text['title'].values.tolist() )
train_title_tfidf = tfidf_vec.transform(df_train_text['title'].values.tolist())
test_title_tfidf = tfidf_vec.transform(df_test_text['title'].values.tolist())

### SVD Components ###
n_comp = 5

svd_title_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_title_obj.fit(full_title_tfidf)
train_title_svd = pd.DataFrame(svd_title_obj.transform(train_title_tfidf))
test_title_svd = pd.DataFrame(svd_title_obj.transform(test_title_tfidf))
train_title_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
test_title_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
for i in train_title_svd.columns:
    print (i)
    test_title_svd[i].to_pickle('./data/features/tsvd/train/' + str(i))


### text TFIDF Vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=200000)

full_text_tfidf = tfidf_vec.fit_transform(df_text['text'].values.tolist() )
train_text_tfidf = tfidf_vec.transform(df_train_text['text'].values.tolist())
test_text_tfidf = tfidf_vec.transform(df_test_text['text'].values.tolist())

### SVD Components ###
n_comp = 40

svd_text_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_text_obj.fit(full_text_tfidf)
train_text_svd = pd.DataFrame(svd_text_obj.transform(train_text_tfidf))
test_text_svd = pd.DataFrame(svd_text_obj.transform(test_text_tfidf))
train_text_svd.columns = ['svd_text_'+str(i+1) for i in range(n_comp)]
test_text_svd.columns = ['svd_text_'+str(i+1) for i in range(n_comp)]
for i in train_text_svd.columns:
    print (i)
    train_text_svd[i].to_pickle('./data/features/tsvd/train/' + str(i))
    test_text_svd[i].to_pickle('./data/features/tsvd/test/' + str(i))  