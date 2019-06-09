# -*- coding: utf-8 -*-

#@author: chenxinye

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
#TF_CPP_MIN_LOG_LEVEL=2
#os.environ["MKL_THREADING_LAYER"] = 'GNU'
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import gc
import string
import random
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import Imputer
from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, Flatten
from keras.layers import Input, SpatialDropout1D,Dropout, GlobalAveragePooling1D, GRU, Bidirectional, Dense, Embedding, CuDNNGRU
from keras.layers.merge import concatenate, dot, multiply, add
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Nadam, RMSprop, adam
from keras.layers.noise import AlphaDropout, GaussianNoise
from keras import backend as K
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, MaxPooling2D, GlobalMaxPool1D, BatchNormalization
from keras.preprocessing import text, sequence
import warnings
from AttLayer import Attention
warnings.filterwarnings("ignore")
train = pd.read_csv('../feature_engineering/fe_0614_train.csv',index_col=0)
test = pd.read_csv('../feature_engineering/fe_0614_test.csv',index_col=0)

train['aggregate'] = pd.read_pickle('../stack_sub/aggregate_train.pkl')
test['aggregate'] = pd.read_pickle('../stack_sub/aggregate_test.pkl')

img_train = pd.read_pickle('../image_meta_feature/aggregate3_train_exp16.pkl')
img_test = pd.read_pickle('../image_meta_feature/aggregate3_test_exp16.pkl')

img_train = pd.DataFrame(img_train,columns='Image_'+pd.Index(range(16)).astype(str))
img_test = pd.DataFrame(img_test,columns='Image_'+pd.Index(range(16)).astype(str))

train = pd.concat([train,img_train],axis=1)
test = pd.concat([test,img_test],axis=1)

train_data = pd.concat([train,test],ignore_index=True)
del train,test

categorical = [
    # 'wday',
    # 'image_top_1',
    'param_1',
    'param_2',
    'param_3',
    'city',
    'region',
    'category_name',
    'parent_category_name',
    'user_type'
]


remove_list = [
    'wday',
    'user_id',
    'item_id',
    'title',
    'description',
    'activation_date',
    'image',
    'deal_probability'
]

predictors = [x for x in train_data.columns if x not in remove_list]
numerical = [x for x in predictors if x not in categorical]

train_numerical = train_data[numerical]

## set threshold 
## for every numerical column, first substract the min (skip na values)
## if max - min > threshold, take natural log of that column
## in case log(0), add 1 after substracting min, i.e., col_val = col_val - min + 1
## then do normalization: (col_val - min) / (max - min)
## after normalization, impute na with mean

threshold = 1000
use_boxcox_cols = (train_data[numerical].max() -  train_data[numerical].min() > 1000).index.values
for col in use_boxcox_cols:    
#    train_numerical.loc[(-train_numerical[col].isnull()), col] = train_numerical.loc[(-train_numerical[col].isnull()), col] - train_numerical[col].min(skipna = True) + 1
#    if (train_numerical[col].max(skipna = True) >= threshold):        
    #train_numerical.loc[(-train_numerical[col].isnull()), col] = np.log1p(train_numerical.loc[(-train_numerical[col].isnull()), col])        
    train_numerical[col] = np.log1p(train_numerical[col])
#    train_numerical.loc[(-train_numerical[col].isnull()), col] = (train_numerical.loc[(-train_numerical[col].isnull()), col]) / (train_numerical[col].max(skipna = True))
#    train_numerical.loc[train_numerical[col].isnull(), col] = train_numerical[col].mean()   
sc = StandardScaler()
train_numerical[train_numerical == np.Inf] = 0.0
train_numerical[train_numerical == np.NINF] = 0.0
train_numerical.fillna(0.0, inplace = True)
train_numerical = sc.fit_transform(train_numerical)
#train_numerical.isnull().sum()


def fill_seq(text, maxlen=100):
    s = text.replace(',', ' ').replace('(', ' ').replace(')',' ').replace('.', ' ').strip().split()
    return ' '.join(s)

##================split into x_train/x_val. No stratification requried probably
train_data['title'] = train_data.title.fillna('missing').str.lower()
train_data['description'] = train_data.description.fillna('missing').str.lower()
train_data['title_description'] = (train_data['title']+" "+train_data['description']).astype(str)

train_data.title_description = train_data.title_description.map(lambda x: fill_seq(x,100))
print train_data.title_description[0]

from tqdm import tqdm
EMBEDDING_FILE = '../input/fasttext.selftrained.300.model.vec'

max_features = 200000
maxlen = 100
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
print('fitting tokenizer')
tokenizer.fit_on_texts(list(train_data['title_description']))

print('getting embeddings')
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in tqdm(open(EMBEDDING_FILE)))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in tqdm(word_index.items()):
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


print('OOV embeddings: %d' % np.mean(np.sum(embedding_matrix, axis=1) == 0))
print('convert to sequences')
des_train = tokenizer.texts_to_sequences(train_data['title_description'])

print('padding')
des_train = sequence.pad_sequences(des_train, maxlen=maxlen)

## ================ Create the Tokenizers
train_data['param123'] = (train_data['param_1']+'_'+train_data['param_2']+'_'+train_data['param_3']).astype(str)
param123_tk = {x:i+1 for i, x in enumerate(train_data.param123.unique())}
region_tk = {x:i+1 for i, x in enumerate(train_data.region.unique())}
city_tk =  {x:i+1 for i, x in enumerate(train_data.city.unique())}
cat1_tk =  {x:i+1 for i, x in enumerate(train_data.parent_category_name.unique())}
cat2_tk =  {x:i+1 for i, x in enumerate(train_data.category_name.unique())}
param1_tk =  {x:i+1 for i, x in enumerate(train_data.param_1.unique())}
param2_tk =  {x:i+1 for i, x in enumerate(train_data.param_2.unique())}
param3_tk =  {x:i+1 for i, x in enumerate(train_data.param_3.unique())}
seqnum_tk =  {x:i+1 for i, x in enumerate(train_data.item_seq_number.unique())}
usertype_tk = {x:i+1 for i, x in enumerate(train_data.user_type.unique())}
imgtype_tk = {x:i+1 for i, x in enumerate(train_data.image_top_1.unique())}
tokenizers = [region_tk, city_tk, cat1_tk, cat2_tk, param1_tk, param2_tk, \
         param3_tk, seqnum_tk, usertype_tk, imgtype_tk, param123_tk]

## ================ These functions are going to get repeated on train, val, and test data
def tokenize_data(data, tokenizers, train_numerical):

    region_tk, city_tk, cat1_tk, cat2_tk, param1_tk, param2_tk, param3_tk, seqnum_tk, usertype_tk, imgtype_tk, param123_tk = tokenizers
    x_reg = np.asarray([region_tk.get(key, 0) for key in data.region], dtype=int)
    x_city  = np.asarray([city_tk.get(key, 0) for key in data.city], dtype=int)
    x_cat1  = np.asarray([cat1_tk.get(key, 0) for key in data.parent_category_name], dtype=int)
    x_cat2  = np.asarray([cat2_tk.get(key, 0) for key in data.category_name], dtype=int)
    x_prm1 = np.asarray([param1_tk.get(key, 0) for key in data.param_1], dtype=int)
    x_prm2 = np.asarray([param2_tk.get(key, 0) for key in data.param_2], dtype=int)
    x_prm3 = np.asarray([param3_tk.get(key, 0) for key in data.param_3], dtype=int)
    x_sqnm = np.asarray([seqnum_tk.get(key, 0) for key in data.item_seq_number], dtype=int)
    x_usr = np.asarray([usertype_tk.get(key, 0) for key in data.user_type], dtype=int)
    x_itype = np.asarray([imgtype_tk.get(key, 0) for key in data.image_top_1], dtype=int)
    x_prm123 = np.asarray([param123_tk.get(key, 0) for key in data.param123], dtype=int)
 
    #return [
    #    x_reg, x_city, x_cat1, x_cat2,
    #    x_prm1, x_prm2, x_prm3, x_sqnm,
    #    x_usr, x_itype, dow, feat_mat
    #]
    return [
        x_reg, x_city, x_cat1, x_cat2,
        x_prm1, x_prm2, x_prm3,              #### not use x_itype, x_sqnm, 
        x_usr, x_prm123, train_numerical
    ]


## ================================================================================
#===================Final Processing on x, y train, val, test data
x_train = tokenize_data(train_data, tokenizers, train_numerical)
x_train.append(des_train)
y_train = train_data.deal_probability.as_matrix()
len_train = train_data.deal_probability.notnull().sum()
print(len_train)

## ================================================================================
#=================== train, test, validation split

x_test = [x_train[i][len_train : ] for i in range(0, len(x_train))]
print(len(x_test[0]))

x_train = [x_train[i][ : len_train] for i in range(0, len(x_train))]

print(len(x_train[0]))
#indices = np.random.permutation(len(x_train[0]))
#print (len(indices))

def get_train_val(training_idx, val_idx):
    
    #training_idx, val_idx = indices[ : (len_train - int(len_train * 0.1))], indices[(len_train - int(len_train * 0.1)):]
    _xval = [x_train[i][val_idx] for i in range(0, len(x_train))]
    _xtrain = [x_train[i][training_idx] for i in range(0, len(x_train))]

    _yval = y_train[val_idx]
    _ytrain = y_train[training_idx]

    print(len(_xval[0]))
    print(len(_xtrain[0]))

    print(len(_yval))
    print(len(_ytrain))
    
    return _xtrain,_ytrain,_xval,_yval


##================Beginning of the NN Model Outline.
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

drop_rate = 0.2
emb_size = 12
batch_size = 512
#for drop_rate in [0, 0.1, 0.2]:
#for batch_size in [500, 1000, 2000, 5000]:

def build_model(emb_size = emb_size):
# embedding
    input_reg = Input(shape=(1,))
    input_city = Input(shape=(1,))
    input_cat1 = Input(shape=(1,))
    input_cat2 = Input(shape=(1,))
    input_prm1 = Input(shape=(1,))
    input_prm2 = Input(shape=(1,))
    input_prm3 = Input(shape=(1,))
    #input_sqnm = Input(shape=(1,))
    input_usr = Input(shape=(1,))
    #input_itype = Input(shape=(1,))
    input_prm123 = Input(shape=(1,))
    #input_weekday = Input(shape=(1,))
    input_hc_feat = Input(shape=(x_train[-2].shape[1],),dtype='float32')

    input_des = Input(shape = (maxlen, ))
    emb_des = Embedding(nb_words,
                    embed_size,
                    weights = [embedding_matrix],
                    input_length = maxlen,
                    trainable = False)(input_des)

    # emb_des= SpatialDropout1D(0.2)(emb_des)

    #warppers = []
    emb_des_bigru= GRU(units=300,
                                return_sequences = True)(emb_des)
    emb_des_bigru = Attention(100)(emb_des_bigru)
    #warppers.append(emb_des_bigru)
    
    #for kernel_size in [2,3,4]:
    #    emb_conv = Conv1D(kernel_size=kernel_size,filters=64,padding='same')(emb_des)
    #    emb_conv_gmp = GlobalMaxPool1D()(emb_conv)
    #    #emb_conv_gap = GlobalAveragePooling1D()(emb_conv)
    #    warppers.append(emb_conv_gmp)
        #warppers.append(emb_conv_gap)
    ## maxpooling for description features
    #emb_des= MaxPooling1D()(emb_des)

    # emb_des = Dense(64, activation="relu")(emb_des)

    # nsy_price = GaussianNoise(0.1)(input_price)

    emb_reg  = Embedding(len(region_tk)+1, emb_size)(input_reg)
    emb_city = Embedding(len(city_tk)+1, emb_size)(input_city)
    emb_cat1 = Embedding(len(cat1_tk)+1, emb_size)(input_cat1)
    emb_cat2 = Embedding(len(cat2_tk)+1, emb_size)(input_cat2)
    emb_prm1 = Embedding(len(param1_tk)+1, emb_size)(input_prm1)
    emb_prm2 = Embedding(len(param2_tk)+1, emb_size)(input_prm2)
    emb_prm3 = Embedding(len(param3_tk)+1, emb_size)(input_prm3)
    #emb_sqnm = Embedding(len(seqnum_tk)+1, emb_size)(input_sqnm)
    emb_usr  = Embedding(len(usertype_tk)+1, emb_size)(input_usr)
    #emb_itype= Embedding(len(imgtype_tk)+1, emb_size)(input_itype)
    emb_prm123 = Embedding(len(param123_tk)+1, emb_size)(input_usr)

    x = concatenate([Flatten() (emb_reg),
                     Flatten() (emb_city),
                     Flatten() (emb_cat1),
                     Flatten() (emb_cat2),
                     Flatten() (emb_prm1),
                     Flatten() (emb_prm2),
                     Flatten() (emb_prm3),
                     #Flatten() (emb_sqnm),
                     Flatten() (emb_usr),
                     #Flatten() (emb_itype),
                     Flatten() (emb_prm123),
                     input_hc_feat,
                     emb_des_bigru
                     ]) # Do not want to dropout price, its noised up instead.

    # x = BatchNormalization()(x)
    
    x = Dense(512, activation="relu")(x)
    x = Dropout(drop_rate)(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(drop_rate)(x)
    y = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[input_reg, input_city, input_cat1,\
                          input_cat2, input_prm1, input_prm2,\
                          input_prm3, input_usr,input_prm123,\
                          input_hc_feat,input_des], outputs=y)

# def rmse(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
## https://github.com/keras-team/keras/issues/1170
# optim = keras.optimizers.SGD(lr=0.01, momentum=0.9)
    #model.summary()

    return model

from sklearn.cross_validation import KFold

save_path = 'baseline_6_16_v1'
pred_train = np.zeros((len_train,1))
pred_test = np.zeros((len(x_test[0]),1))
skf = KFold(len_train,n_folds=5,shuffle=True,random_state=42)
for fold,(tr_idx,te_idx) in enumerate(skf):
    
        xtrain, ytrain, xval, yval = get_train_val(tr_idx, te_idx)
        model = build_model()
        #optim = keras.optimizers.Adam(lr=0.0005)
        model.compile(optimizer='adam', loss=root_mean_squared_error) 
        earlystop = EarlyStopping(monitor="val_loss",mode="auto",
                              patience=2,
                          verbose=0)
    
        checkpt = ModelCheckpoint(monitor="val_loss",
                              mode="auto",
                              filepath='../weights/{0}_{1}.hdf5'.format(save_path,fold),
                              verbose=0,
                              save_best_only=True)
    
        rlrop = ReduceLROnPlateau(monitor='val_loss',
                              mode='auto',
                              patience=2,
                              verbose=1,
                              factor=0.33,
                              cooldown=0,
                              min_lr=1e-6)

        train_history = model.fit(xtrain, ytrain,
                              batch_size=batch_size,
                              validation_data=(xval, yval),
                              epochs=100,
                              callbacks =[checkpt, earlystop])

        model.load_weights('../weights/{0}_{1}.hdf5'.format(save_path,fold))
        _pred_test = model.predict(x_test, batch_size=batch_size,verbose=1)
        _pred_val = model.predict(xval, batch_size=batch_size,verbose=1)

        pred_test += _pred_test.reshape((-1,1))
        pred_train[te_idx] = _pred_val.reshape((-1,1))

pred_test/=5.0
pd.to_pickle(pred_test,'../stack_sub/{}_test.pkl'.format(save_path))
pd.to_pickle(pred_train,'../stack_sub/{}_train.pkl'.format(save_path))

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_train[:len_train],pred_train))

##================Beginning of the NN Model Outline.
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

drop_rate = 0.2
emb_size = 12
batch_size = 512
#for drop_rate in [0, 0.1, 0.2]:
#for batch_size in [500, 1000, 2000, 5000]:

def build_model_v2(emb_size = emb_size):
# embedding
    input_reg = Input(shape=(1,))
    input_city = Input(shape=(1,))
    input_cat1 = Input(shape=(1,))
    input_cat2 = Input(shape=(1,))
    input_prm1 = Input(shape=(1,))
    input_prm2 = Input(shape=(1,))
    input_prm3 = Input(shape=(1,))
    #input_sqnm = Input(shape=(1,))
    input_usr = Input(shape=(1,))
    #input_itype = Input(shape=(1,))
    input_prm123 = Input(shape=(1,))
    #input_weekday = Input(shape=(1,))
    input_hc_feat = Input(shape=(x_train[-2].shape[1],),dtype='float32')

    input_des = Input(shape = (maxlen, ))
    emb_des = Embedding(nb_words,
                    embed_size,
                    weights = [embedding_matrix],
                    input_length = maxlen,
                    trainable = False)(input_des)

    # emb_des= SpatialDropout1D(0.2)(emb_des)

    warppers = []
    #emb_des_bigru= Bidirectional(GRU(units=150,
    #                            return_sequences = True))(emb_des)
    
    #emb_des_bigru = Attention(100)(emb_des_bigru)
    #warppers.append(emb_des_bigru)
    
    for kernel_size in [2,3,4]:
        emb_conv = Conv1D(kernel_size=kernel_size,filters=128,padding='same')(emb_des)
        emb_conv_gmp = GlobalMaxPool1D()(emb_conv)
    #    #emb_conv_gap = GlobalAveragePooling1D()(emb_conv)
        warppers.append(emb_conv_gmp)
        #warppers.append(emb_conv_gap)
    ## maxpooling for description features
    #emb_des= MaxPooling1D()(emb_des)

    # emb_des = Dense(64, activation="relu")(emb_des)

    # nsy_price = GaussianNoise(0.1)(input_price)

    emb_reg  = Embedding(len(region_tk)+1, emb_size)(input_reg)
    emb_city = Embedding(len(city_tk)+1, emb_size)(input_city)
    emb_cat1 = Embedding(len(cat1_tk)+1, emb_size)(input_cat1)
    emb_cat2 = Embedding(len(cat2_tk)+1, emb_size)(input_cat2)
    emb_prm1 = Embedding(len(param1_tk)+1, emb_size)(input_prm1)
    emb_prm2 = Embedding(len(param2_tk)+1, emb_size)(input_prm2)
    emb_prm3 = Embedding(len(param3_tk)+1, emb_size)(input_prm3)
    #emb_sqnm = Embedding(len(seqnum_tk)+1, emb_size)(input_sqnm)
    emb_usr  = Embedding(len(usertype_tk)+1, emb_size)(input_usr)
    #emb_itype= Embedding(len(imgtype_tk)+1, emb_size)(input_itype)
    emb_prm123 = Embedding(len(param123_tk)+1, emb_size)(input_usr)

    x = concatenate([Flatten() (emb_reg),
                     Flatten() (emb_city),
                     Flatten() (emb_cat1),
                     Flatten() (emb_cat2),
                     Flatten() (emb_prm1),
                     Flatten() (emb_prm2),
                     Flatten() (emb_prm3),
                     #Flatten() (emb_sqnm),
                     Flatten() (emb_usr),
                     #Flatten() (emb_itype),
                     Flatten() (emb_prm123),
                     input_hc_feat,
                     #emb_des_bigru
                     concatenate(warppers)
                     ]) # Do not want to dropout price, its noised up instead.

    # x = BatchNormalization()(x)
    
    x = Dense(512, activation="relu")(x)
    x = Dropout(drop_rate)(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(drop_rate)(x)
    y = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[input_reg, input_city, input_cat1,\
                          input_cat2, input_prm1, input_prm2,\
                          input_prm3, input_usr,input_prm123,\
                          input_hc_feat,input_des], outputs=y)

# def rmse(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
## https://github.com/keras-team/keras/issues/1170
# optim = keras.optimizers.SGD(lr=0.01, momentum=0.9)
    #model.summary()

    return model

save_path = 'baseline_6_15_v4_2'
pred_train = np.zeros((len_train,1))
pred_test = np.zeros((len(x_test[0]),1))
skf = KFold(len_train,n_folds=5,shuffle=True,random_state=42)
for fold,(tr_idx,te_idx) in enumerate(skf):
    
        xtrain, ytrain, xval, yval = get_train_val(tr_idx, te_idx)
        model = build_model_v2()
        #optim = keras.optimizers.Adam(lr=0.0005)
        model.compile(optimizer='adam', loss=root_mean_squared_error) 
        earlystop = EarlyStopping(monitor="val_loss",mode="auto",
                              patience=2,
                          verbose=0)
    
        checkpt = ModelCheckpoint(monitor="val_loss",
                              mode="auto",
                              filepath='../weights/{0}_{1}.hdf5'.format(save_path,fold),
                              verbose=0,
                              save_best_only=True)
    
        rlrop = ReduceLROnPlateau(monitor='val_loss',
                              mode='auto',
                              patience=2,
                              verbose=1,
                              factor=0.33,
                              cooldown=0,
                              min_lr=1e-6)

        train_history = model.fit(xtrain, ytrain,
                              batch_size=batch_size,
                              validation_data=(xval, yval),
                              epochs=100,
                              callbacks =[checkpt, earlystop])

        model.load_weights('../weights/{0}_{1}.hdf5'.format(save_path,fold))
        _pred_test = model.predict(x_test, batch_size=batch_size,verbose=1)
        _pred_val = model.predict(xval, batch_size=batch_size,verbose=1)

        pred_test += _pred_test.reshape((-1,1))
        pred_train[te_idx] = _pred_val.reshape((-1,1))

pred_test/=5.0
pd.to_pickle(pred_test,'../stack_sub/{}_test.pkl'.format(save_path))
pd.to_pickle(pred_train,'../stack_sub/{}_train.pkl'.format(save_path))

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_train[:len_train],pred_train))