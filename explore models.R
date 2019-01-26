# setting working directory
setwd("~/R/course work/kaggle/datasciencenig/customer churn")


submit <- data.frame(cusomerId = testid, churn = pred.gbm)
colnames(submit) <- c('Customer ID', 'Churn Status')
write.csv(submit, file = "churn_GCVsubmnew1.csv", row.names = F)



churn <- ifelse(churn == 0,1,0)

tr <- cbind(df_train,churn)
tr <- tr[-4]

#Split the training set
library(caTools)
#set a seed
set.seed(1235)
# create a split ratio
samples <- sample.split(tr$churn,SplitRatio = 0.8)
trainset <- subset(tr,samples == TRUE)
testset <- subset(tr, samples == FALSE)

# train control
ctrl <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = T
)

# MODELLING
churn ~ t.spend.months + t.data.consum + mostlove2.Uxaa
# randomforest
set.seed(1235)
rf.model <- randomForest(as.factor(b)~ t.spend.months +t.sms+ mostlove2.Uxaa,
                         data = trainset, ntree = 1000,importance =TRUE)
rf.model

tune.mod <- tuneRF(x = vv[-4], y = vv$churn, ntreeTry = 1000)
# prediction
pred <- predict(rf.model, newdata = testset, type = 'class')
# Evaluation
auc1 <- roc(as.numeric(testset$b), as.numeric(pred))
plot(auc1, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc1$auc[[1]],3)),col = 'blue')

### try cv
set.seed(1235)
folds = createFolds(tr$churn, k = 10)
cv = lapply(folds, function(x) {
  set.seed(1235)
  training_fold = tr[-x, ]
  test_fold = tr[x, ]
  rf.mod <- randomForest(as.factor(churn) ~ t.spend.months + t.data.consum + t.sms  + mostlove2.Uxaa, data = training_fold,
                         ntree = 1000,importance = TRUE)
  pred <- predict(rf.mod , newdata = test_fold[-27], type = 'class')
  # xgb <- xgboost(data = as.matrix(training_fold[-1]), params = p,
  #                label = training_fold$Survived,nrounds =521, print_every_n = 100,
  #                early_stopping_rounds = 400)
  #
  # reddd <- predict(xgb, newdata = as.matrix(test_fold[-1]))
  # reddd <- ifelse(reddd>0.5,1,0)
  cm = table(test_fold[, 27], pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
#cv
paste('Average CV Accuracy',mean(as.numeric(cv)))

# gbm
g.grid <- expand.grid(degree =c(1,2,3,4,5))
set.seed(1235)
gbm.mod <- train(as.factor(churn) ~., data = tr, method = 'deepboost',
                 trControl = ctrl)#,tuneGrid = g.grid)
gbm.mod
plot(gbm.mod)
ggplot(varImp(gbm.mod))

pred.gbm <- predict(gbm.mod, newdata =  testset[-27], type = 'raw')
# Evaluation
auc1 <- roc(as.numeric(testset$churn), as.numeric(pred.gbm))
plot(auc1, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc1$auc[[1]],3)),col = 'blue')

predd <- as.data.frame(gbm.mod$pred) %>% filter(num_iter ==  50 ,tree_depth == 2, beta == 0.00390625,  lambda ==0.015625) %>% arrange(rowIndex)
confusionMatrix(predd$obs, predd$pred)

####
### method = Linda
set.seed(1235)
ada.mod <- train(as.factor(churn) ~., data = tr, method = 'ada',
                 trControl = ctrl)#,tuneGrid = g.grid)
ada.mod
ggplot(varImp(ada.mod))

pred.ada <- predict(ada.mod, newdata =  testset, type = 'raw')
# Evaluation
auc1 <- roc(as.numeric(testset$churn), as.numeric(pred.ada))
plot(auc1, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc1$auc[[1]],3)),col = 'blue')


predd <- as.data.frame(ada.mod$pred) %>% filter(iter ==  150 ,maxdepth == 3, nu ==0.1) %>% arrange(rowIndex)
confusionMatrix(predd$obs, predd$pred)


ada.mod$finalModel
### Robust Regularized LDA
set.seed(1235)
rr.mod <- train(as.factor(churn) ~., data = tr, method = 'RSimca',
                 trControl = ctrl)#,tuneGrid = g.grid)
rr.mod

ggplot(varImp(rr.mod))

# feature selection
library(FSelector)

library(rpart)
evaluator <- function(subset) {
  #k-fold cross validation
  k <- 10
  splits <- runif(nrow(tr))
  results = sapply(1:k, function(i) {
    set.seed(1235)
    test.idx <- (splits >= (i - 1) / k) & (splits < i / k)
    train.idx <- !test.idx
    test <- tr[test.idx, , drop=FALSE]
    train <- tr[train.idx, , drop=FALSE]
    set.seed(1235)
    tree <- rpart(as.simple.formula(subset, "churn"), train)
    #tree <- randomForest(as.simple.formula(subset, "churn"), train)
    error.rate = sum(test$churn != predict(tree, test, type="c"))/nrow(test)
    return(1 - error.rate)
  })
  print(subset)
  print(mean(results))
  return(mean(results))
}

subset <- best..search(names(tr)[-27], evaluator)
f <- as.simple.formula(subset, "churn")
print(f)

# Fit
set.seed(1235)
rp.tr <- rpart(churn ~ t.spend.months + t.sms+ t.calls  + mostlove2.Uxaa,
               data = tr, method = 'class')

prp(rp.tr)

# predict
pred <- predict(rp.tr, newdata = testset, type = 'class')
# Evaluation
auc1 <- roc(as.numeric(testset$churn), as.numeric(pred))
plot(auc1, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc1$auc[[1]],3)),col = 'blue')



# Adaboost
set.seed(101)
mod.ada <- ada(churn ~ t.spend.months + t.sms+ t.calls  + mostlove2.Uxaa , data = tr, iter = 50, bag.frac = 0.5,
               verbose = T)
summary(mod.ada)
varplot(mod.ada)
set.seed(101)
mod.ada <- addtest(mod.ada, test.x = testset, test.y = testset$churn)
plot(mod.ada,test = TRUE)

a <- predict(mod.ada, newdata = testset)
b <- predict(mod.ada, newdata = df_test)




# # logistic reg
# log_model <- glm(as.factor(churn) ~ ., family = binomial(link = 'logit'), trainset)
# 
# summary(log_model)
# anova(log_model)
# #predict
# test_predict <- predict(log_model, testset, type = 'response')
# head(test_predict)
# result <- ifelse(test_predict>0.5,1,0)
# misclasserr <- mean(result != testset$churn)
# 1-misclasserr
# 
# # Auc evaluation
# auc1 <- roc(as.numeric(testset$churn), as.numeric(test_predict))
# plot(auc1, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc1$auc[[1]],3)),col = 'blue')
