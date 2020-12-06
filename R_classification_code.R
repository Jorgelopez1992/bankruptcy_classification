#install.packages('gridExtra')
#install.packages("caret")
#install.packages("pROC")
#install.packages("ROCR")
#install.packages("randomForest")
library(tidyverse)
library(glmnet)
library(DescTools)
library(gridExtra)
library(ROCR)
library(caret)
library(pROC)
library(randomForest)  
library(gridExtra)

#downloading data directly from github repository 
data <- read_csv('https://raw.githubusercontent.com/Jorgelopez1992/bankruptcy_classification/master/bankruptcy_Train.csv',
               col_names = T)

#splitting data intro X matrix and Y vector
X <- as.matrix(data[,c(1:64)])
y <- as.matrix(data[,65])


#info for splitting into train/test in loop
n        =    nrow(X)
p        =    ncol(X)
n.train        =     floor(0.9*n)
n.test         =     n-n.train

M              =     50   #total loops 
num_thrs       =     100  #Number of threshholds for classification for logistic regression (0.01,0.02 etc.)


#empty vectors to store time values
elas.cv.time<-c(rep(0,M)) 
elas.time<-c(rep(0,M)) 
rid.cv.time<-c(rep(0,M))
rid.time<-c(rep(0,M)) 
las.cv.time<-c(rep(0,M))
las.time<-c(rep(0,M)) 
rf.time<-c(rep(0,M))

elas.cv.time<-c(rep(0,M)) 
elas.time<-c(rep(0,M)) 
rid.cv.time<-c(rep(0,M))
rid.time<-c(rep(0,M)) 
las.cv.time<-c(rep(0,M))
las.time<-c(rep(0,M)) 
rf.time<-c(rep(0,M))


#Vectors to store AUC
elas.train.error <- c(rep(0,M))
elas.test.error <-c(rep(0,M))
las.train.error <- c(rep(0,M))
las.test.error <-c(rep(0,M))
rid.train.error <- c(rep(0,M))
rid.test.error <-c(rep(0,M))
rf.train.error <- c(rep(0,M))
rf.test.error <-c(rep(0,M))

#Vectors to store AUC
elas.train.auc <- c(rep(0,M))
elas.test.auc <-c(rep(0,M))
las.train.auc <- c(rep(0,M))
las.test.auc <-c(rep(0,M))
rid.train.auc <- c(rep(0,M))
rid.test.auc <-c(rep(0,M))
rf.train.auc <- c(rep(0,M))
rf.test.auc <-c(rep(0,M))


for (m in c(1:M)) {
  #randomly sampling rows 
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  weight.vec <- ifelse(y.train==0, sum(y.train)/n.train, sum(y.train==0)/n.train) #WEIGHTS BASED ON FREQUENCY
  
  ######################################################################################  
  ###############  ELASTIC NET   #######################################################
  ###################################################################################### 
  
  start.time <- Sys.time()
  elas.cv.fit      =     cv.glmnet(X.train, y.train, family = "binomial",  # finding optimal lambda with cv
                                   alpha=0.5,nfolds=10, weights=weight.vec,type.measure = "auc")
  end.time<-Sys.time()    
  elas.cv.time[m]<-elas.time[m]+(end.time-start.time)    # track of time 
  
  start.time <- Sys.time()
  elas.fit         =     glmnet(X.train, y.train, alpha = 0.5, lambda = elas.cv.fit$lambda.min, weights=weight.vec)
  end.time<-Sys.time()
  elas.time[m]<-elas.time[m]+(end.time-start.time)
  
  #getting beta coefficients and probabilities
  beta0.elas.hat          =        elas.fit$a0
  beta.elas.hat           =        as.vector(elas.fit$beta)
  elas.prob.train         =        exp(X.train %*% beta.elas.hat +  beta0.elas.hat  )/(1 + exp(X.train %*% beta.elas.hat +  beta0.elas.hat  ))
  
  #creating vectors to store loop data for each threshold 
  tr.thrs                 =       c(0:num_thrs)
  tr.TPR                  =       c(0:num_thrs)
  tr.FPR                  =       c(0:num_thrs)
  tr.type1error           =       c(0:num_thrs)
  tr.type2error           =       c(0:num_thrs)
  tr.error                =       c(0:num_thrs)
  tr.FNR                  =       c(0:num_thrs)
  tr.FP                   =       c(0:num_thrs)
  tr.FN                   =       c(0:num_thrs)
  min.dif.thrs            =       NULL
  
  #loop to find the probability threshold where type1error==type2error (or as close as possible)
  for (i in 0:num_thrs){
    if (i==0){
      thrs=0
    } else {
      thrs=i/num_thrs
    }
    y.hat.train             =        ifelse(elas.prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    FN.train                =        sum(y.train[y.hat.train==0] == 1) #false negatives in the data
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train         =        FPR.train
    typeII.err.train        =        1 - TPR.train
    
    tr.thrs[i+1] <- thrs
    tr.TPR[i+1] <- TPR.train
    tr.FPR[i+1] <- FPR.train
    tr.FNR[i+1] <- FN.train/P.train
    tr.FP[i+1] <- FP.train
    tr.FN[i+1] <- FN.train
    tr.error[i+1] <- (FP.train+FN.train)/n.train
    tr.type1error[i+1]         =       typeI.err.train
    tr.type2error[i+1]         =       typeII.err.train
    
  }
  
  #dataframe of all thresholds and errors
  df.elas=data.frame(threshold=tr.thrs,false_positive_rate=tr.FPR,false_negative_rate=tr.FNR,
                     false_positive=tr.FP,false_negative=tr.FN,
                     type1error=tr.type1error,type2error=tr.type2error,error_rate=tr.error
                     )
  
  elas.train.auc[m] <- AUC(x=tr.FPR, y=tr.TPR)

  
  ##################################        Elastic net test set           ##############################################
  elas.prob.test               =        exp(X.test %*%beta.elas.hat +  beta0.elas.hat  )/(1 + exp(X.test %*% beta.elas.hat +  beta0.elas.hat   ))
  
  #creating vectors to store loop data for each threshold 
  test.thrs                 =       c(0:num_thrs)
  test.TPR                  =       c(0:num_thrs)
  test.FPR                  =       c(0:num_thrs)
  test.type1error           =       c(0:num_thrs)
  test.type2error           =       c(0:num_thrs)
  test.error                =       c(0:num_thrs)
  test.FNR                  =       c(0:num_thrs)
  test.FP                   =       c(0:num_thrs)
  test.FN                   =       c(0:num_thrs)
  min.dif.thrs            =       NULL
  
  #loop to find the probability threshold where type1error==type2error (or as close as possible)
  for (i in 0:num_thrs){
    if (i==0){
      thrs=0
    } else {
      thrs=i/num_thrs
    }
    y.hat.test             =        ifelse(elas.prob.test > thrs, 1, 0) #table(y.hat.train, y.train)
    FP.test                =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test                =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    FN.test                =        sum(y.test[y.hat.test==0] == 1) #false negatives in the data
    P.test                 =        sum(y.test==1) # total positives in the data
    N.test                 =        sum(y.test==0) # total negatives in the data
    FPR.test               =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
    TPR.test               =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.test         =        FPR.test
    typeII.err.test        =        1 - TPR.test
    
    test.thrs[i+1] <- thrs
    test.TPR[i+1] <- TPR.test
    test.FPR[i+1] <- FPR.test
    test.FNR[i+1] <- FN.test/P.test
    test.FP[i+1] <- FP.test
    test.FN[i+1] <- FN.test
    test.error[i+1] <- (FP.test+FN.test)/n.test
    test.type1error[i+1]         =       typeI.err.test
    test.type2error[i+1]         =       typeII.err.test
    
  }
  
  #dataframe of all thresholds and errors
  df.test.elas=data.frame(threshold=test.thrs,false_positive_rate=test.FPR,false_negative_rate=test.FNR,
                     false_positive=test.FP,false_negative=test.FN,
                     type1error=test.type1error,type2error=test.type2error,error_rate=test.error
  )
  elas.test.auc[m] <- AUC(x=test.FPR, y=test.TPR)

  cat('\n','loop ',toString(m),'elastic results:','\n','Train:',toString(elas.train.auc[m]),
      '\n','Test:',toString(elas.test.auc[m]))
  
  
  ######################################################################################  
  ###############        RIDGE   #######################################################
  ###################################################################################### 
  start.time<-Sys.time()
  
  rid.cv.fit      =     cv.glmnet(X.train, y.train, family = "binomial", alpha=0,nfolds=10,
                                  weights=weight.vec,type.measure = "auc")
  end.time<-Sys.time()    
  rid.cv.time[m]<-rid.time[m]+(end.time-start.time)    # track of time 
  
  start.time <- Sys.time()
  rid.fit         =     glmnet(X.train, y.train, alpha = 0, lambda = rid.cv.fit$lambda.min,weights=weight.vec)
  
  end.time<-Sys.time()
  rid.time[m]<-rid.time[m]+(end.time-start.time)
  
  #getting beta coefficients and probabilities
  beta0.rid.hat          =        rid.fit$a0
  beta.rid.hat           =        as.vector(rid.fit$beta)
  rid.prob.train         =        exp(X.train %*% beta.rid.hat +  beta0.rid.hat  )/(1 + exp(X.train %*% beta.rid.hat +  beta0.rid.hat  ))
  
  #creating vectors to store loop data for each threshold 
  tr.thrs                 =       c(0:num_thrs)
  tr.TPR                  =       c(0:num_thrs)
  tr.FPR                  =       c(0:num_thrs)
  tr.type1error           =       c(0:num_thrs)
  tr.type2error           =       c(0:num_thrs)
  tr.error                =       c(0:num_thrs)
  tr.FNR                  =       c(0:num_thrs)
  tr.FP                   =       c(0:num_thrs)
  tr.FN                   =       c(0:num_thrs)
  min.dif.thrs            =       NULL
  
  #loop to find the probability threshold where type1error==type2error (or as close as possible)
  for (i in 0:num_thrs){
    if (i==0){
      thrs=0
    } else {
      thrs=i/num_thrs
    }
    y.hat.train             =        ifelse(rid.prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    FN.train                =        sum(y.train[y.hat.train==0] == 1) #false negatives in the data
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train         =        FPR.train
    typeII.err.train        =        1 - TPR.train
    
    tr.thrs[i+1] <- thrs
    tr.TPR[i+1] <- TPR.train
    tr.FPR[i+1] <- FPR.train
    tr.FNR[i+1] <- FN.train/P.train
    tr.FP[i+1] <- FP.train
    tr.FN[i+1] <- FN.train
    tr.error[i+1] <- (FP.train+FN.train)/n.train
    tr.type1error[i+1]         =       typeI.err.train
    tr.type2error[i+1]         =       typeII.err.train
    
  }
  
  #dataframe of all thresholds and errors
  df.rid=data.frame(threshold=tr.thrs,false_positive_rate=tr.FPR,false_negative_rate=tr.FNR,
                    false_positive=tr.FP,false_negative=tr.FN,
                    type1error=tr.type1error,type2error=tr.type2error,error_rate=tr.error
  )
  rid.train.auc[m] <- AUC(x=tr.FPR, y=tr.TPR)
  
  #finding the probability threshold where type1error & type2error  are as close as possible
  min.dif.index <- which.min(abs(df.rid$type1error-df.rid$type2error)) #saving it into index so it can be saved automatically
  min.dif.thrs <- df.rid[min.dif.index,1]
  #rid.train.error[m] <- df.rid[min.dif.index,8]
  
  rid.prob.test               =        exp(X.test %*%beta.rid.hat +  beta0.rid.hat  )/(1 + exp(X.test %*% beta.rid.hat +  beta0.rid.hat   ))
  
  #creating vectors to store loop data for each threshold 
  test.thrs                 =       c(0:num_thrs)
  test.TPR                  =       c(0:num_thrs)
  test.FPR                  =       c(0:num_thrs)
  test.type1error           =       c(0:num_thrs)
  test.type2error           =       c(0:num_thrs)
  test.error                =       c(0:num_thrs)
  test.FNR                  =       c(0:num_thrs)
  test.FP                   =       c(0:num_thrs)
  test.FN                   =       c(0:num_thrs)
  min.dif.thrs            =       NULL
  
  #loop to find the probability threshold where type1error==type2error (or as close as possible)
  for (i in 0:num_thrs){
    if (i==0){
      thrs=0
    } else {
      thrs=i/num_thrs
    }
    y.hat.test             =        ifelse(rid.prob.test > thrs, 1, 0) #table(y.hat.train, y.train)
    FP.test                =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test                =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    FN.test                =        sum(y.test[y.hat.test==0] == 1) #false negatives in the data
    P.test                 =        sum(y.test==1) # total positives in the data
    N.test                 =        sum(y.test==0) # total negatives in the data
    FPR.test               =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
    TPR.test               =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.test         =        FPR.test
    typeII.err.test        =        1 - TPR.test
    
    test.thrs[i+1] <- thrs
    test.TPR[i+1] <- TPR.test
    test.FPR[i+1] <- FPR.test
    test.FNR[i+1] <- FN.test/P.test
    test.FP[i+1] <- FP.test
    test.FN[i+1] <- FN.test
    test.error[i+1] <- (FP.test+FN.test)/n.test
    test.type1error[i+1]         =       typeI.err.test
    test.type2error[i+1]         =       typeII.err.test
    
  }
  
  #dataframe of all thresholds and errors
  df.test.rid=data.frame(threshold=test.thrs,false_positive_rate=test.FPR,false_negative_rate=test.FNR,
                          false_positive=test.FP,false_negative=test.FN,
                          type1error=test.type1error,type2error=test.type2error,error_rate=test.error
  )
  rid.test.auc[m] <- AUC(x=test.FPR, y=test.TPR)

  cat('\n','ridge results:','\n','Train:',toString(rid.train.auc[m]),
      '\n','Test:',toString(rid.test.auc[m]))
  
  
  ######################################################################################  
  ###############        lasso   #######################################################
  ###################################################################################### 
  start.time<-Sys.time()
  
  las.cv.fit      =     cv.glmnet(X.train, y.train, family = "binomial", alpha=1,nfolds=10,
                                  type.measure = "auc",weights=weight.vec)
  end.time<-Sys.time()    
  las.cv.time[m]<-elas.time[m]+(end.time-start.time)    # track of time 
  
  start.time <- Sys.time()
  las.fit         =     glmnet(X.train, y.train, alpha = 1, lambda = las.cv.fit$lambda.min,weights=weight.vec)
  
  end.time<-Sys.time()
  las.time[m]<-las.time[m]+(end.time-start.time)
  
  #getting beta coefficients and probabilities
  beta0.las.hat          =        las.fit$a0
  beta.las.hat           =        as.vector(las.fit$beta)
  las.prob.train         =        exp(X.train %*% beta.las.hat +  beta0.las.hat  )/(1 + exp(X.train %*% beta.las.hat +  beta0.las.hat  ))
  
  #creating vectors to store loop data for each threshold 
  tr.thrs                 =       c(0:num_thrs)
  tr.TPR                  =       c(0:num_thrs)
  tr.FPR                  =       c(0:num_thrs)
  tr.type1error           =       c(0:num_thrs)
  tr.type2error           =       c(0:num_thrs)
  tr.error                =       c(0:num_thrs)
  tr.FNR                  =       c(0:num_thrs)
  tr.FP                   =       c(0:num_thrs)
  tr.FN                   =       c(0:num_thrs)
  min.dif.thrs            =       NULL
  
  #loop to find the probability threshold where type1error==type2error (or as close as possible)
  for (i in 0:num_thrs){
    if (i==0){
      thrs=0
    } else {
      thrs=i/num_thrs
    }
    y.hat.train             =        ifelse(las.prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    FN.train                =        sum(y.train[y.hat.train==0] == 1) #false negatives in the data
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train         =        FPR.train
    typeII.err.train        =        1 - TPR.train
    
    tr.thrs[i+1] <- thrs
    tr.TPR[i+1] <- TPR.train
    tr.FPR[i+1] <- FPR.train
    tr.FNR[i+1] <- FN.train/P.train
    tr.FP[i+1] <- FP.train
    tr.FN[i+1] <- FN.train
    tr.error[i+1] <- (FP.train+FN.train)/n.train
    tr.type1error[i+1]         =       typeI.err.train
    tr.type2error[i+1]         =       typeII.err.train
    
  }
  
  #dataframe of all thresholds and errors
  df.las=data.frame(threshold=tr.thrs,false_positive_rate=tr.FPR,false_negative_rate=tr.FNR,
                    false_positive=tr.FP,false_negative=tr.FN,
                    type1error=tr.type1error,type2error=tr.type2error,error_rate=tr.error
  )
  
  las.train.auc[m] <- AUC(x=tr.FPR, y=tr.TPR)
  
  
  #finding the probability threshold where type1error & type2error  are as close as possible
  min.dif.index <- which.min(abs(df.las$type1error-df.las$type2error)) #saving it into index so it can be saved automatically
  min.dif.thrs <- df.las[min.dif.index,1]
  las.train.error[m] <- df.las[min.dif.index,8]
  
  ##################################        lasso net test set           ##############################################
  las.prob.test               =        exp(X.test %*%beta.las.hat +  beta0.las.hat  )/(1 + exp(X.test %*% beta.las.hat +  beta0.las.hat   ))
  

  #creating vectors to store loop data for each threshold 
  test.thrs                 =       c(0:num_thrs)
  test.TPR                  =       c(0:num_thrs)
  test.FPR                  =       c(0:num_thrs)
  test.type1error           =       c(0:num_thrs)
  test.type2error           =       c(0:num_thrs)
  test.error                =       c(0:num_thrs)
  test.FNR                  =       c(0:num_thrs)
  test.FP                   =       c(0:num_thrs)
  test.FN                   =       c(0:num_thrs)
  min.dif.thrs            =       NULL
  
  #loop to find the probability threshold where type1error==type2error (or as close as possible)
  for (i in 0:num_thrs){
    if (i==0){
      thrs=0
    } else {
      thrs=i/num_thrs
    }
    y.hat.test             =        ifelse(las.prob.test > thrs, 1, 0) #table(y.hat.train, y.train)
    FP.test                =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test                =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    FN.test                =        sum(y.test[y.hat.test==0] == 1) #false negatives in the data
    P.test                 =        sum(y.test==1) # total positives in the data
    N.test                 =        sum(y.test==0) # total negatives in the data
    FPR.test               =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
    TPR.test               =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.test         =        FPR.test
    typeII.err.test        =        1 - TPR.test
    
    test.thrs[i+1] <- thrs
    test.TPR[i+1] <- TPR.test
    test.FPR[i+1] <- FPR.test
    test.FNR[i+1] <- FN.test/P.test
    test.FP[i+1] <- FP.test
    test.FN[i+1] <- FN.test
    test.error[i+1] <- (FP.test+FN.test)/n.test
    test.type1error[i+1]         =       typeI.err.test
    test.type2error[i+1]         =       typeII.err.test
    
  }
  
  #dataframe of all thresholds and errors
  df.test.las=data.frame(threshold=test.thrs,false_positive_rate=test.FPR,false_negative_rate=test.FNR,
                         false_positive=test.FP,false_negative=test.FN,
                         type1error=test.type1error,type2error=test.type2error,error_rate=test.error
  )
  las.test.auc[m] <- AUC(x=test.FPR, y=test.TPR)

  cat('\n','lasso results:','\n','Train:',toString(las.train.auc[m]),
      '\n','Test:',toString(las.test.auc[m]),'\n')
  
  ######################################################################################  
  ###########################   random forest   ########################################
  ###################################################################################### 
#  mtry <- tuneRF(X.train,as.factor(y.train), ntreeTry=1000,                      #
#                 stepFactor=1.5,improve=0.01, trace=F, plot=F) #tuning 
#  best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]                     #
  
  
  start.time<-Sys.time()
  rf.model   =     randomForest(X.train,as.factor(y.train), ntree=1000,
                                importance = TRUE, proximity = TRUE)
  #  rf.model   =     randomForest(X.train,as.factor(y.train), ntree=500,
  #                                importance = TRUE, proximity = TRUE)
  
  end.time<-Sys.time()
  rf.time[m]<-rf.time[m]+(end.time-start.time)
  
  
  
  y.train.probs     =     predict(rf.model, X.train,type='prob')
  
  #creating vectors to store loop data for each threshold 
  tr.thrs                 =       c(0:num_thrs)
  tr.TPR                  =       c(0:num_thrs)
  tr.FPR                  =       c(0:num_thrs)
  tr.type1error           =       c(0:num_thrs)
  tr.type2error           =       c(0:num_thrs)
  tr.error                =       c(0:num_thrs)
  tr.FNR                  =       c(0:num_thrs)
  tr.FP                   =       c(0:num_thrs)
  tr.FN                   =       c(0:num_thrs)
  min.dif.thrs            =       NULL
  
  #loop to find the probability threshold where type1error==type2error (or as close as possible)
  for (i in 0:num_thrs){
    if (i==0){
      thrs=0
    } else {
      thrs=i/num_thrs
    }
    y.hat.train             =        ifelse(y.train.probs[,2] > thrs, 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    FN.train                =        sum(y.train[y.hat.train==0] == 1) #false negatives in the data
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train         =        FPR.train
    typeII.err.train        =        1 - TPR.train
    
    tr.thrs[i+1] <- thrs
    tr.TPR[i+1] <- TPR.train
    tr.FPR[i+1] <- FPR.train
    tr.FNR[i+1] <- FN.train/P.train
    tr.FP[i+1] <- FP.train
    tr.FN[i+1] <- FN.train
    tr.error[i+1] <- (FP.train+FN.train)/n.train
    tr.type1error[i+1]         =       typeI.err.train
    tr.type2error[i+1]         =       typeII.err.train
    
  }
  
  #dataframe of all thresholds and errors
  df.rf=data.frame(threshold=tr.thrs,false_positive_rate=tr.FPR,false_negative_rate=tr.FNR,
                   false_positive=tr.FP,false_negative=tr.FN,
                   type1error=tr.type1error,type2error=tr.type2error,error_rate=tr.error
  )
  
  rf_p_train <- predict(rf.model, type="prob")[,2]
  rf_pr_train <- prediction(rf_p_train, y.train)
  r_auc_train1 <- performance(rf_pr_train, measure = "auc")@y.values[[1]]
  
  rf.train.auc[m] <- r_auc_train1
  
  
  ################################### TEST ##########################################
  y.test.probs       =     predict(rf.model, X.test,type='prob')
  
  #creating vectors to store loop data for each threshold 
  test.thrs                 =       c(0:num_thrs)
  test.TPR                  =       c(0:num_thrs)
  test.FPR                  =       c(0:num_thrs)
  test.type1error           =       c(0:num_thrs)
  test.type2error           =       c(0:num_thrs)
  test.error                =       c(0:num_thrs)
  test.FNR                  =       c(0:num_thrs)
  test.FP                   =       c(0:num_thrs)
  test.FN                   =       c(0:num_thrs)
  min.dif.thrs            =       NULL
  
  
  #loop to find the probability threshold where type1error==type2error (or as close as possible)
  for (i in 0:num_thrs){
    if (i==0){
      thrs=0
    } else {
      thrs=i/num_thrs
    }
    y.hat.test             =        ifelse(y.test.probs[,2] > thrs, 1, 0) #table(y.hat.train, y.train)
    FP.test                =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test                =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    FN.test                =        sum(y.test[y.hat.test==0] == 1) #false negatives in the data
    P.test                 =        sum(y.test==1) # total positives in the data
    N.test                 =        sum(y.test==0) # total negatives in the data
    FPR.test               =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
    TPR.test               =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.test         =        FPR.test
    typeII.err.test        =        1 - TPR.test
    
    test.thrs[i+1] <- thrs
    test.TPR[i+1] <- TPR.test
    test.FPR[i+1] <- FPR.test
    test.FNR[i+1] <- FN.test/P.test
    test.FP[i+1] <- FP.test
    test.FN[i+1] <- FN.test
    test.error[i+1] <- (FP.test+FN.test)/n.test
    test.type1error[i+1]         =       typeI.err.test
    test.type2error[i+1]         =       typeII.err.test
    
  }
  
  #dataframe of all thresholds and errors
  df.test.rf=data.frame(threshold=test.thrs,false_positive_rate=test.FPR,false_negative_rate=test.FNR,
                        false_positive=test.FP,false_negative=test.FN,
                        type1error=test.type1error,type2error=test.type2error,error_rate=test.error
  )
  
  rf_p_test <- predict(rf.model, type="prob",newdata = X.test)[,2]
  rf_pr_test <- prediction(rf_p_test, y.test)
  r_auc_test1 <- performance(rf_pr_test, measure = "auc")@y.values[[1]] 
  
  rf.test.auc[m] <- r_auc_test1
  
  cat('\n','random forest results:','\n','Train:',toString(rf.train.auc[m]),
      '\n','Test:',toString(rf.test.auc[m]),'\n','--------------------------------------------------------------------')
}


#creating dataframes to with auc data
elas.train.df<-data.frame(elas.train.auc,c(rep('Elastic Net',M)),c(rep('Train',M)))
colnames(elas.train.df)<-c('auc','model','set')
elas.test.df<-data.frame(elas.test.auc,c(rep('Elastic Net',M)),c(rep('Test',M)))
colnames(elas.test.df)<-c('auc','model','set')

rid.train.df<-data.frame(rid.train.auc,c(rep('Ridge',M)),c(rep('Train',M)))
colnames(rid.train.df)<-c('auc','model','set')
rid.test.df<-data.frame(rid.test.auc,c(rep('Ridge',M)),c(rep('Test',M)))
colnames(rid.test.df)<-c('auc','model','set')

las.train.df<-data.frame(las.train.auc,c(rep('Lasso',M)),c(rep('Train',M)))
colnames(las.train.df)<-c('auc','model','set')
las.test.df<-data.frame(las.test.auc,c(rep('Lasso',M)),c(rep('Test',M)))
colnames(las.test.df)<-c('auc','model','set')

rf.train.df<-data.frame(rf.train.auc,c(rep('Random Forest',M)),c(rep('Train',M)))
colnames(rf.train.df)<-c('auc','model','set')
rf.test.df<-data.frame(rf.test.auc,c(rep('Random Forest',M)),c(rep('Test',M)))
colnames(rf.test.df)<-c('auc','model','set')


#merging dataframes
auc.df<-rbind(elas.train.df,elas.test.df,
              rid.train.df,rid.test.df,
              las.train.df,las.test.df,
              rf.train.df,rf.test.df)
auc.df$auc<-as.numeric(auc.df$auc)

auc.df$set<-factor(auc.df$set, levels=c('Train','Test'))
auc.df$model<- factor(auc.df$model, levels=c('Elastic Net','Lasso','Ridge','Random Forest'))


ggplot(auc.df,aes(x=model,y=auc,fill=model))+geom_boxplot()+
  facet_wrap(~set)+theme_light()+
  ggtitle('')+ylab('AUC')+xlab('Model')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.title.x=element_blank())+
  theme(axis.text.x=element_blank())


#### CV PLOTS
par(mfrow=c(1,3))
plot(elas.cv.fit, main = "Elastic Net")
plot(las.cv.fit,main= 'Lasso') 
plot(rid.cv.fit, main = "Ridge")
par(mfrow=c(1,1))

### TIMES REPORTS####
cat('\n','elastic net cv time',toString(elas.cv.time[50]),'\n',
    'elastic net time',toString(elas.time[50]),'\n',
    'ridge cv time',toString(rid.cv.time[50]),'\n',
    'ridge net  time',toString(rid.time[50]),'\n',
    'lasso net cv time',toString(las.cv.time[50]),'\n',
    'lasso net  time',toString(las.time[50]),'\n', 
    'random forest time',toString(rf.time[50]),'\n')

#90% interval AUC
quantile( elas.test.auc,probs=c(0.05,0.95))
quantile( rid.test.auc,probs=c(0.05,0.95))
quantile( las.test.auc,probs=c(0.05,0.95))
quantile( rf.test.auc,probs=c(0.05,0.95))

# BAR PLOTS FOR COEFFICIENTS
elas.betas <- as.vector(elas.fit$beta)
las.betas <- as.vector(las.fit$beta)
rid.betas <- as.vector(rid.fit$beta)
rf.betas <- as.vector(rf.model$importance[,2])

features<-c('net profit / total assets',
  'total liabilities / total assets',
  'working capital / total assets','current assets / short-term liabilities',
  '[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365',
  'retained earnings / total assets','EBIT / total assets','book value of equity / total liabilities',
  'sales / total assets','equity / total assets','(gross profit + extraordinary items + financial expenses) / total assets',
  'gross profit / short-term liabilities','(gross profit + depreciation) / sales',
  '(gross profit + interest) / total assets','(total liabilities * 365) / (gross profit + depreciation)',
  '(gross profit + depreciation) / total liabilities','total assets / total liabilities','gross profit / total assets',
  'gross profit / sales','(inventory * 365) / sales','sales (n) / sales (n-1)',
  'profit on operating activities / total assets','net profit / sales','gross profit (in 3 years) / total assets',
  '(equity - share capital) / total assets','(net profit + depreciation) / total liabilities',
  'profit on operating activities / financial expenses','working capital / fixed assets','logarithm of total assets',
  '(total liabilities - cash) / sales','(gross profit + interest) / sales',
  '(current liabilities * 365) / cost of products sold','operating expenses / short-term liabilities',
  'operating expenses / total liabilities','profit on sales / total assets','total sales / total assets',
  '(current assets - inventories) / long-term liabilities','constant capital / total assets',
  'profit on sales / sales','(current assets - inventory - receivables) / short-term liabilities',
  'total liabilities / ((profit on operating activities + depreciation) * (12/365))',
  'profit on operating activities / sales','rotation receivables + inventory turnover in days',
  '(receivables * 365) / sales','net profit / inventory','(current assets - inventory) / short-term liabilities',
  '(inventory * 365) / cost of products sold',
  'EBITDA (profit on operating activities - depreciation) / total assets',
  'EBITDA (profit on operating activities - depreciation) / sales',
  'current assets / total liabilities','short-term liabilities / total assets',
  '(short-term liabilities * 365) / cost of products sold)','equity / fixed assets',
  'constant capital / fixed assets','working capital','(sales - cost of products sold) / sales',
  '(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)',
  'total costs /total sales','long-term liabilities / equity','sales / inventory','sales / receivables',
  '(short-term liabilities *365) / sales','sales / short-term liabilities','sales / fixed assets')

betas.df<-data.frame(elas.betas,las.betas,rid.betas,rf.betas,features,stringsAsFactors=FALSE)

ordered.df<-betas.df%>%arrange(desc(abs(betas.df$elas.betas)))

elas.betas <- ordered.df$elas.betas
las.betas <- ordered.df$las.betas
rid.betas <- ordered.df$rid.betas
rf.betas <- ordered.df$rf.betas
features <- ordered.df$features
colors <- rep(ifelse(elas.betas>1,1,0),4)

Betas <- c(elas.betas,las.betas,rid.betas,rf.betas)
Features <- c(rep(features,4))

Models <- c(rep('Elastic Net',64),rep('Lasso',64),rep('Ridge',64),rep('Random Forest',64))
coeffs <- data.frame(Betas,Models,Features,colors)

colnames(coeffs)<-c('Coefficient','Model','Feature','Colors')

ggplot(coeffs , aes(x = reorder(Feature,-abs(rep(elas.betas,4))), y=Coefficient,fill=as.factor(Colors))) +
  geom_bar(stat = "identity", colour="black")+facet_grid(Model~.)+theme_minimal()+
  theme(axis.title.x=element_blank(),legend.position = "none",axis.text.x =element_blank())


#### COEFFICIENT PLOTS #######
p1 <-ggplot(coeffs%>%filter(Model=='Elastic Net'),
       aes(x = reorder(Feature,-abs(elas.betas)), y=Coefficient,fill=as.factor(Colors)))+
         geom_bar(stat = "identity", fill='white', colour="black")+theme_minimal()+ylab('Elastic Net')+
  theme(axis.title.x=element_blank(),legend.position = "none",axis.text.x =element_blank())+
  ggtitle('Coefficient Values')

p2 <-ggplot(coeffs%>%filter(Model=='Lasso'),
       aes(x = reorder(Feature,-abs(elas.betas)), y=Coefficient,fill=as.factor(Colors)))+
  geom_bar(stat = "identity", fill='white',colour="black")+theme_minimal()+ylab('Lasso')+
  theme(axis.title.x=element_blank(),legend.position = "none",axis.text.x =element_blank())

p3 <-ggplot(coeffs%>%filter(Model=='Ridge'),
       aes(x = reorder(Feature,-abs(elas.betas)), y=Coefficient,fill=as.factor(Colors)))+
  geom_bar(stat = "identity", fill='white', colour="black")+theme_minimal()+ylab('Ridge')+
  theme(axis.title.x=element_blank(),legend.position = "none",axis.text.x =element_blank())

p4 <-ggplot(coeffs%>%filter(Model=='Random Forest'),
       aes(x = reorder(Feature,-abs(elas.betas)), y=Coefficient,fill=as.factor(Colors)))+
  geom_bar(stat = "identity", fill='white', colour="black")+theme_minimal()+ylab('Random Forest')+
  theme(axis.title.x=element_blank(),legend.position = "none",axis.text.x =element_blank())

grid.arrange(p1, p2,p3,p4, ncol = 1)

features

reorder(Feature,-abs(elas.betas))

View(cbind(features,ifelse(elas.betas>0,1,0)))
