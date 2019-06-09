#@author: chenxinye

rm(list = ls())

lambda <- 1 # the penalty parameter

setwd("E:\\Avito\\submission\\0626")
files <- c("lgb_row0.1_col1.0_leaves600_score0.21357.csv",
           "lgb-82-18.csv",
           "stack_NN_21367.csv",
           "sub_lgb_2186.csv",
           "LGB.csv")

s0 <- 0.3032  # ALL ZERO benchmark
s <- c(0.2187, 0.2170, 0.2187, 0.2186, 0.2174)
## P data frame
for(i in 1:length(files)){
  if(i == 1){
    P_frame <- read.csv(files[i], col.names = c('item_id',unlist(strsplit(files[i],'\\.'))[3]))
  }
  else{
    tmp <- read.csv(files[i], col.names = c('item_id',unlist(strsplit(files[i],'\\.'))[3]))
    P_frame <- merge(P_frame, tmp, by = 'item_id')
  }
}
item_id <- P_frame[,1]
P_frame$item_id <- NULL

## P matrix
P <- as.matrix(P_frame)

## the first part
first_part <- solve(t(P) %*% P + lambda * diag(ncol(P)))

## second part
second_part <- c()
N <- dim(P)[1]
for(i in 1:length(files)){
  tmp <- (N*(s0^2 - s[i]^2) + as.numeric(t(P[,i]) %*% P[,i]))/2
  second_part <- c(second_part, tmp)
}

## the coef!
w <- first_part %*% second_part

## the result!
result <- P %*% w

## transform back!
sub <- result[,1]
## write csv!
submission <- data.frame(item_id = item_id, deal_probability = sub)
write.csv(submission, file = '0626-prob.csv', row.names = F, quote = F)
