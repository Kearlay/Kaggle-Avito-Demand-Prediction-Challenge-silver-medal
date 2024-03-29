#@author: chenxinye

library(tidyverse)
library(lubridate)
library(magrittr)
library(text2vec)
library(tokenizers)
library(stopwords)
library(xgboost)
library(Matrix)
library(stringr)
library(stringi)
library(forcats)
set.seed(0)

#---------------------------
setwd("E:\\Avito\\input")
cat("Loading data...\n")
tr <- read_csv("../input/train.csv") 
te <- read_csv("../input/test.csv")

#---------------------------
cat("Preprocessing...\n")
tri <- 1:nrow(tr)
y <- tr$deal_probability

tr_te <- tr %>% 
  select(-deal_probability) %>% 
  bind_rows(te) %>% 
  mutate(no_img = is.na(image) %>% as.integer(),
         no_dsc = is.na(description) %>% as.integer(),
         no_p1 = is.na(param_1) %>% as.integer(), 
         no_p2 = is.na(param_2) %>% as.integer(), 
         no_p3 = is.na(param_3) %>% as.integer(),
         titl_len = str_length(title),
         desc_len = str_length(description),
         titl_capE = str_count(title, "[A-Z]"),
         titl_capR = str_count(title, "[А-Я]"),
         desc_capE = str_count(description, "[A-Z]"),
         desc_capR = str_count(description, "[А-Я]"),
         titl_cap = str_count(title, "[A-ZА-Я]"),
         desc_cap = str_count(description, "[A-ZА-Я]"),
         titl_pun = str_count(title, "[[:punct:]]"),
         desc_pun = str_count(description, "[[:punct:]]"),
         titl_dig = str_count(title, "[[:digit:]]"),
         desc_dig = str_count(description, "[[:digit:]]"),
         user_type = factor(user_type),
         category_name = factor(category_name) %>% as.integer(),
         parent_category_name = factor(parent_category_name) %>% as.integer(), 
         region = factor(region) %>% as.integer(),
         param_1 = factor(param_1) %>% as.integer(),
         param_2 = factor(param_2) %>% as.integer(),
         param_3 = factor(param_3) %>% fct_lump(prop = 0.00005) %>% as.integer(),
         city =  factor(city) %>% fct_lump(prop = 0.0003) %>% as.integer(),
         user_id = factor(user_id) %>% fct_lump(prop = 0.000025) %>% as.integer(),
         price = log1p(price),
         txt = paste(title, description, sep = " "),
         mday = mday(activation_date),
         wday = wday(activation_date)) %>% 
  select(-item_id, -image, -title, -description, -activation_date) %>% 
  replace_na(list(image_top_1 = -1, price = -1, 
                  param_1 = -1, param_2 = -1, param_3 = -1, 
                  desc_len = 0, desc_cap = 0, desc_pun = 0, 
                  desc_dig = 0, desc_capE = 0, desc_capR = 0)) %T>% 
  glimpse()

rm(tr, te); gc()

#---------------------------
cat("Parsing text...\n")
it <- tr_te %$%
  str_to_lower(txt) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  tokenize_word_stems(language = "russian") %>% 
  itoken()

vect <- create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("ru")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.4, vocab_term_max = 12500) %>% 
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <-  create_dtm(it, vect) %>% 
  fit_transform(m_tfidf)

rm(it, vect, m_tfidf); gc()

#---------------------------
cat("Preparing data...\n")
X <- tr_te %>% 
  select(-txt) %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(tfidf)

rm(tr_te, tfidf); gc()

dtest <- xgb.DMatrix(data = X[-tri, ])
X <- X[tri, ]; gc()
tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = X[tri, ], label = y[tri])
dval <- xgb.DMatrix(data = X[-tri, ], label = y[-tri])
cols <- colnames(X)

rm(X, y, tri); gc()

#---------------------------
cat("Training model...\n")
p <- list(objective = "reg:logistic",
          booster = "gbtree",
          eval_metric = "rmse",
          nthread = 8,
          eta = 0.05,
          max_depth = 18,
          min_child_weight = 11,
          gamma = 0,
          subsample = 0.8,
          colsample_bytree = 0.7,
          alpha = 2.25,
          lambda = 0,
          nrounds = 5000)

m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 10, early_stopping_rounds = 50)

xgb.importance(cols, model = m_xgb) %>%   
  xgb.plot.importance(top_n = 35)

#---------------------------
cat("Creating submission file...\n")
read_csv("../input/sample_submission.csv")  %>%  
  mutate(deal_probability = predict(m_xgb, dtest)) %>%
  write_csv(paste0("xgb_tfidf", m_xgb$best_score, ".csv"))