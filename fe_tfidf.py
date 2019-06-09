# -*- coding: utf-8 -*-

#@author: chenxinye

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gc
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 
import pickle
from scipy import sparse
from nltk.tokenize.toktok import ToktokTokenizer 
from nltk.stem.snowball import RussianStemmer
from nltk import sent_tokenize # should be multilingual
from string import punctuation
from nltk import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import FastText
import re
from string import punctuation

punct = set(punctuation)  

# load data
df_text = pickle.load(open('./data/df_text.pkl','rb'))

# Tf-Idf
def clean_text(s):
    s = re.sub('м²|\d+\\/\d|\d+-к|\d+к', ' ', s.lower())
    s = re.sub('\\s+', ' ', s)
    s = s.strip()
    return s   
    
russian_stop = set(stopwords.words('russian'))

df_text['param_123'] = df_text['param_123'].apply(lambda x: clean_text(x))
df_text['title'] = df_text['title'].apply(lambda x: clean_text(x))
df_text["text"] = df_text["text"].apply(lambda x: clean_text(x))

df_train_text = df_text[df_text['deal_probability'].notnull()]
df_test_text = df_text[df_text['deal_probability'].isnull()]

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "lowercase": True,
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}

def get_col(col_name): return lambda x: x[col_name]
vectorizer = FeatureUnion([
        ('text',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=200000,
            **tfidf_para,
            preprocessor=get_col('text'))),
        ('title',TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words = russian_stop,
            preprocessor=get_col('title'))),
        ('param_123',TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words = russian_stop,
            preprocessor=get_col('param_123')))    
    ])  

vectorizer.fit(df_text.to_dict('records'))
ready_df_train = vectorizer.transform(df_train_text.to_dict('records'))
ready_df_test = vectorizer.transform(df_test_text.to_dict('records'))
tfvocab = vectorizer.get_feature_names()

sparse.save_npz('./data/features/nlp/ready_df_train_200000_new.npz', ready_df_train)
sparse.save_npz('./data/features/nlp/ready_df_test_200000_new.npz', ready_df_test)

with open('./data/features/nlp/tfvocab_200000_new.pkl', 'wb') as tfvocabfile:  
    pickle.dump(tfvocab, tfvocabfile)