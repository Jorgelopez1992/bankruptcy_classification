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

M              =     10   #total loops 
num_thrs       =     100  #Number of threshholds for classification for logistic regression (0.01,0.02 etc.)


#empty vectors to store time values
elas.cv.time<-c(rep(0,M)) 
elas.time<-c(rep(0,M)) 
rid.cv.time<-c(rep(0,M))
rid.time<-c(rep(0,M)) 
las.cv.time<-c(rep(0,M))
las.time<-c(rep(0,M)) 
rf.time<-c(rep(0,M))

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
  rid.train.error[m] <- df.rid[min.dif.index,8]
  
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
      '\n','Test:',toString(las.test.auc[m]))
  
  ######################################################################################  
  ###########################   random forest   ########################################
  ###################################################################################### 
#  mtry <- tuneRF(train.data[1:64],train.data$class, ntreeTry=500,
#                 stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
#  best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
  
  
  
}


#creating dataframes to with auc data
elas.train.df<-data.frame(elas.train.auc,c(rep('elastic_net',M)),c(rep('train',M)))
colnames(elas.train.df)<-c('auc','model','set')
elas.test.df<-data.frame(elas.test.auc,c(rep('elastic_net',M)),c(rep('test',M)))
colnames(elas.test.df)<-c('auc','model','set')

rid.train.df<-data.frame(rid.train.auc,c(rep('ridge',M)),c(rep('train',M)))
colnames(rid.train.df)<-c('auc','model','set')
rid.test.df<-data.frame(rid.test.auc,c(rep('ridge',M)),c(rep('test',M)))
colnames(rid.test.df)<-c('auc','model','set')

las.train.df<-data.frame(las.train.auc,c(rep('lasso',M)),c(rep('train',M)))
colnames(las.train.df)<-c('auc','model','set')
las.test.df<-data.frame(las.test.auc,c(rep('lasso',M)),c(rep('test',M)))
colnames(las.test.df)<-c('auc','model','set')


#merging dataframes 
auc.df<-rbind(elas.train.df,elas.test.df,
               rid.train.df,rid.test.df,
               las.train.df,las.test.df)
auc.df$auc<-as.numeric(auc.df$auc)


ggplot(auc.df,aes(x=model,y=auc,fill=model))+geom_boxplot()+
  facet_wrap(~set)+theme_light()+
  ggtitle('')+ylab('AUC')+xlab('Model')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.title.x=element_blank(),axis.text.x=element_blank())+
  ggtitle('AUC Comparison')

elas_plot<-plot(elas.cv.fit)
rid_plot<-plot(rid.cv.fit)
las_plot<-plot(las.cv.fit)
