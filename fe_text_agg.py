# -*- coding: utf-8 -*-

#@author: chenxinye

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 
import re
import string
from nltk.stem.snowball import RussianStemmer
import pickle

def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))
	
# load data
df_text = pickle.load(open('./data/df_text.pkl','rb'))

stopwords = {x: 1 for x in stopwords.words('russian')}
punct = set(string.punctuation)
emoji = set()
for s in df_text['text'].fillna('').astype(str):
    for c in s:
        if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
            continue
        emoji.add(c)

all = df_text.copy()

# Meta Text Features
textfeats = ['param_123']
for cols in textfeats:   
    all[cols] = all[cols].astype(str) 

    all[cols + '_num_cap'] = all[cols].apply(lambda x: count_regexp_occ('[А-ЯA-Z]', x))
    all[cols + '_num_low'] = all[cols].apply(lambda x: count_regexp_occ('[а-яa-z]', x))
    all[cols + '_num_rus_cap'] = all[cols].apply(lambda x: count_regexp_occ('[А-Я]', x))
    all[cols + '_num_eng_cap'] = all[cols].apply(lambda x: count_regexp_occ('[A-Z]', x))    
    all[cols + '_num_rus_low'] = all[cols].apply(lambda x: count_regexp_occ('[а-я]', x))
    all[cols + '_num_eng_low'] = all[cols].apply(lambda x: count_regexp_occ('[a-z]', x))
    all[cols + '_num_dig'] = all[cols].apply(lambda x: count_regexp_occ('[0-9]', x))   
    all[cols + '_num_pun'] = all[cols].apply(lambda x: sum(c in punct for c in x))
    all[cols + '_num_space'] = all[cols].apply(lambda x: sum(c.isspace() for c in x))
    all[cols + '_num_chars'] = all[cols].apply(len) # Count number of Characters
    all[cols + '_num_words'] = all[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    all[cols + '_num_unique_words'] = all[cols].apply(lambda comment: len(set(w for w in comment.split())))
    all[cols + '_ratio_unique_words'] = all[cols+'_num_unique_words'] / (all[cols+'_num_words']+0.0001)
    
textfeats = ['text']
for cols in textfeats:   
    all[cols] = all[cols].astype(str)
    all[cols + '_num_cap'] = all[cols].apply(lambda x: count_regexp_occ('[А-ЯA-Z]', x))
    all[cols + '_num_low'] = all[cols].apply(lambda x: count_regexp_occ('[а-яa-z]', x))
    all[cols + '_num_rus_cap'] = all[cols].apply(lambda x: count_regexp_occ('[А-Я]', x))
    all[cols + '_num_eng_cap'] = all[cols].apply(lambda x: count_regexp_occ('[A-Z]', x))    
    all[cols + '_num_rus_low'] = all[cols].apply(lambda x: count_regexp_occ('[а-я]', x))
    all[cols + '_num_eng_low'] = all[cols].apply(lambda x: count_regexp_occ('[a-z]', x))
    all[cols + '_num_dig'] = all[cols].apply(lambda x: count_regexp_occ('[0-9]', x))
    all[cols + '_num_pun'] = all[cols].apply(lambda x: sum(c in punct for c in x))
    all[cols + '_num_space'] = all[cols].apply(lambda x: sum(c.isspace() for c in x))
    all[cols + '_num_emo'] = all[cols].apply(lambda x: sum(c in emoji for c in x))
    all[cols + '_num_row'] = all[cols].apply(lambda x: x.count('/\n'))
    all[cols + '_num_chars'] = all[cols].apply(len) # Count number of Characters
    all[cols + '_num_words'] = all[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    all[cols + '_num_unique_words'] = all[cols].apply(lambda comment: len(set(w for w in comment.split())))
    all[cols + '_ratio_unique_words'] = all[cols+'_num_unique_words'] / (all[cols+'_num_words']+1) # Count Unique Words    
    all[cols +'_stopword_ratio'] = all[cols].apply(lambda x: len([w for w in x.split() if w in stopwords])) / all[cols].apply(lambda comment: len(comment.split()))
    all[cols +'_num_stopwords'] = all[cols].apply(lambda x: len([w for w in x.split() if w in stopwords]))
    all[cols +'_num_words_upper'] = all[cols].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    all[cols +'_num_words_lower'] = all[cols].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
    all[cols +'_num_words_title'] = all[cols].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    
textfeats = ['title']
for cols in textfeats:   
    all[cols] = all[cols].astype(str)
    all[cols + '_num_cap'] = all[cols].apply(lambda x: count_regexp_occ('[А-ЯA-Z]', x))
    all[cols + '_num_low'] = all[cols].apply(lambda x: count_regexp_occ('[а-яa-z]', x))
    all[cols + '_num_rus_cap'] = all[cols].apply(lambda x: count_regexp_occ('[А-Я]', x))
    all[cols + '_num_eng_cap'] = all[cols].apply(lambda x: count_regexp_occ('[A-Z]', x))    
    all[cols + '_num_rus_low'] = all[cols].apply(lambda x: count_regexp_occ('[а-я]', x))
    all[cols + '_num_eng_low'] = all[cols].apply(lambda x: count_regexp_occ('[a-z]', x))
    all[cols + '_num_dig'] = all[cols].apply(lambda x: count_regexp_occ('[0-9]', x))
    all[cols + '_num_pun'] = all[cols].apply(lambda x: sum(c in punct for c in x))
    all[cols + '_num_space'] = all[cols].apply(lambda x: sum(c.isspace() for c in x))
    all[cols + '_num_chars'] = all[cols].apply(len) # Count number of Characters
    all[cols + '_num_words'] = all[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    all[cols + '_num_unique_words'] = all[cols].apply(lambda comment: len(set(w for w in comment.split())))
    all[cols + '_ratio_unique_words'] = all[cols+'_num_unique_words'] / (all[cols+'_num_words']+1)


df_train = all[all['deal_probability'].notnull()]
df_test = all[all['deal_probability'].isnull()]
df_all_tmp = all.drop(['deal_probability','param_123','title','text'],axis=1)
tmp_columns = df_all_tmp.columns.values
for i in tmp_columns:
    print (i)
    df_train[i].to_pickle('./data/features/text_agg/train/' + str(i))
    df_test[i].to_pickle('./data/features/text_agg/test/' + str(i))  
        