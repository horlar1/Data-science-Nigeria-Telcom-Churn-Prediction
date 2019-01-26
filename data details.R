# setting working directory
setwd("~/R/course work/kaggle/datasciencenig/customer churn")

# libraries
library(plyr)
library(tidyquant)
library(recipes)
library(rsample)
library(lime)
library(yardstick)
library(caret)
library(pROC)

# model lib
library(randomForest)
library(rpart)
library(rpart.plot)
library(e1071)
library(class)
library(ipred)
library(party)
library(ada)
library(xgboost)
#############################


# import the dataset
train_set <- read.csv('TRAIN.csv',stringsAsFactors = F)
test <- read.csv('TEST.csv', stringsAsFactors = F)

testid <- test$Customer.ID



# drop empty row
emp.row <- ""
emp.row <- train_set$Customer.ID %in% emp.row
train_set <- train_set[!emp.row,]

# target variable
churn <- train_set$Churn.Status
table(churn)

# Add a "churn" variable to the test set to allow for combining data sets
test.churn <- data.frame(Churn.Status = rep("None", nrow(test)), test[,])

# Combine data sets
df <- rbind(train_set, test.churn)


#check for missing values
library(Amelia)
df[df== ""] <- NA  #there are some ? values changed to Na
# test[test==""] <- NA
missmap(df, col=c('yellow','Black'), main='Missing values', legend = F)


#check for columns with missing values
na.cols <- which(colSums(is.na(df)) > 0)
na.cols <- sort(colSums(sapply(df[na.cols], is.na)), decreasing = TRUE)
paste('There are', length(na.cols), 'columns with missing values')


colnames(df) <- c('id','network_age','customer.tenure','t.spend.months',
                  't.sms','t.data.spend','t.data.consum','t.uni.calls',
                  't.onnet','t.offnet','t.complaint','network.type1',
                  'network.type2','mostlove1','mostlove2','churn')


# Most.Loved.Competitor.network.in.in.Month.1
table(df$mostlove1)
sort(unique(df$mostlove1))
df$mostlove1[is.na(df$mostlove1)] <- 0
df$mostlove1 <- mapvalues(df$mostlove1,
                          from = c(0),
                          to = c('None'))
df$mostlove1 <- as.factor(df$mostlove1)
# Most.Loved.Competitor.network.in.in.Month.2
table(df$mostlove2)
sort(unique(df$mostlove2))
df$mostlove2[is.na(df$mostlove2)] <- "Uxaa"         # most common
df$mostlove2 <- as.factor(df$mostlove2)
# Network.type.subscription.in.Month.1
summary(as.factor(df$network.type1))
df$network.type1[is.na(df$network.type1)] <- "3G"
df$network.type1 <- as.factor(df$network.type1)
# Network.type.subscription.in.Month.2
summary(as.factor(df$network.type2))
df$network.type2[is.na(df$network.type2)] <- "3G"
df$network.type2 <- as.factor(df$network.type2)

###
df$spend.per.data <- df$t.data.spend/df$t.data.consum
df$t.calls <- df$t.onnet + df$t.offnet

#### Finding correlation
#get numeric columns
corr.df <- cbind(df[1:1400,], churn)
num.cols <- sapply(corr.df,is.numeric)
corr <- corr.df[,num.cols] %>% cor()
# Visuals
library(corrplot)
corrplot(corr, type = 'upper', method='color', addCoef.col = 'black', tl.cex = .7,cl.cex = .7, number.cex=.7)

################
## Id's of training observations that fail to fit
out <- c(451, 1392, 459, 920, 887,897,1360,978,1212,884)
ids <- df[out,'id']
ids <- df$id %in% ids
df <- df[!ids, ]

churn <- churn[-out]

#####################
df <- df %>% within(rm('network_age','churn','id'))

# customer tenure
out <- which(df$customer.tenure<0)
df[out,"customer.tenure"] <- 0
#### Grouping the customer tenure
group_tenure <- function(tenure){
  if (tenure >= 0 & tenure <= 12){
    return('0-12 Month')
  }else if(tenure > 12 & tenure <= 24){
    return('12-24 Month')
  }else if (tenure > 24 & tenure <= 48){
    return('24-48 Month')
  }else if (tenure > 48 & tenure <=60){
    return('48-60 Month')
  }else if (tenure > 60){
    return('> 60 Month')
  }
}
# applying the group function
df$customer.tenure <- sapply(df$customer.tenure,group_tenure)
df$customer.tenure <- as.factor(df$customer.tenure)


df$t.complaint <- as.factor(df$t.complaint)



# get the feature class
feature_classes <- sapply(names(df), function(x) {
  class(df[[x]])
})

numeric_feats <- names(feature_classes[feature_classes != c("factor")])
df_numeric <- df[,numeric_feats]

# library(moments)
# library(MASS)
# # linear models assume normality from dependant variables
# # transform any skewed data into normal
# skewed <- apply(df_numeric, 2, skewness)
# skewed <-skewed[abs(skewed) > 0.75]
# ## Transform skewed features with boxcox or log transformation
# for (x in names(skewed)) {
#   df_numeric[[x]] <- log(df_numeric[[x]] + 1)
# }


#### CATEGORICAL FEATURES
# ONE HOT ENCODING FOR CATEGORICAL VARIABLES
# get names of categorical features
categorical_feats <- names(feature_classes[feature_classes == "factor"])
df.categories <- df[,categorical_feats]

# # # one hot encoding for categorical data
dummy <- dummyVars(" ~ .",data=df.categories)
df.categories <- data.frame(predict(dummy,newdata=df.categories))

# combine the feature
df <- cbind(df_numeric,df.categories)


# normalise numeric feartures
scale.n <- preProcess(df, method = c('scale','center'))
df <- predict(scale.n, df)
str(df)

# check for near-zeero variance
# library(caret)
nzv.data <- nearZeroVar(df, saveMetrics = TRUE)
# take any of the near-zero-variance perdictors
drop.cols <- rownames(nzv.data)[nzv.data$nzv == TRUE]

df <- df[,!names(df) %in% drop.cols]
#
paste('The dataframe now has', dim(df)[1], 'rows and', dim(df)[2], 'columns')


# Dropping unwanted feature(s)
cols <- c('t.spend.months')#,'t.data.spend','t.onnet','t.offnet')
df <- df[,!names(df) %in% cols]




# Split data set into training and test set
df_train <- df[1:1390,]
df_test <- df[1391:nrow(df),]

library(Matrix)
trainSparse <- sparse.model.matrix(~. , data = df_train)[,-1]
testSparse <- sparse.model.matrix(~., data = df_test)[,-1]




############### fconwd CONNECT TO STORED R OBJECTS IN subdir
############### subdir WILL BE CREATED, IF IT DOES NOT EXIST IN FIRST CALL 
library(SOAR)
fconwd=function(subdir){
  oldLC <- Sys.getenv("R_LOCAL_CACHE", unset = ".R_Cache")
  Sys.setenv(R_LOCAL_CACHE=subdir) }


fconwd("rstore")
tmp = Objects()

Store(df_train)
Store(df_test)
Store(trainSparse)
Store(testSparse)
Store(testid)
gc()

rm(train_set,test,corr,corr.df, df_numeric,df.categories,feature_classes,dummy,test.churn,num.cols,numeric_feats,drop.cols,nzv.data,df,categorical_feats,ids,out,emp.row)


library(EFS)
selection <- ensemble_fs(tr, 27 ,cor_threshold = 0.7,runs = 2,
      selection = c(TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE,FALSE))

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
set.seed(1235)
results <- rfe(df_train, as.factor(churn), rfeControl=control,ntree = 1000)
a <- predictors(results)
plot(results, type = c("g", "o"))
varImp(results)
