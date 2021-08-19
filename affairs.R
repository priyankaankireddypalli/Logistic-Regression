# 1
library(readr)
# Importing the dataset
Affairs <- read.csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Logistic_regression\\Affairs.csv")
summary(Affairs)
# Checking for NA values
sum(is.na(Affairs))
# There are no NA values in our dataset.
Affairs1 <- Affairs[ , -1] # Removing the first column which is is an Index
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
# since the output values should be in the range of 0 to 1 we are doing discretization of the dataset
# if the person is having an external affairs they will be recorded as 1 and if he is not having
# any affairs they will be recorded as 0. 
Affairs1$naffairs<-cut(Affairs1$naffairs,breaks =c(-1,0.9,13), labels = c("0","1"))
# here the output will be in form of factor
# if we need output as it is we can use the below method 
# Affairs1$naffairs[Affairs1$naffairs>0]<-1
# Affairs1$naffairs[Affairs1$naffairs<0]<-0
sum(is.na(Affairs1))
summary(Affairs1)
colnames(Affairs1)
attach(Affairs1)
fullmodel <- glm(naffairs ~ ., data = Affairs1, family = "binomial")
summary(fullmodel)
prob_full <- predict(fullmodel, Affairs1, type = "response")
prob_full
# Decide on optimal prediction probability cutoff for the model
library(InformationValue)
optCutOff <- optimalCutoff(Affairs1$naffairs, prob_full)
optCutOff
# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(Affairs1$naffairs, prob_full, threshold = optCutOff)
# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(Affairs1$naffairs, prob_full)
# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)
results <- confusionMatrix(predvalues, Affairs1$naffairs)
sensitivity(predvalues, Affairs1$naffairs)
confusionMatrix(actuals = Affairs1$naffairs, predictedScores = predvalues)
# Data Partitioning
n <- nrow(Affairs1)
n1 <- n * 0.80
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- Affairs1[train_index, ]
test <- Affairs1[-train_index, ]
# Train the model using Training data
finalmodel <- glm(naffairs ~ ., data = train, family = "binomial")
summary(finalmodel)
# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test
# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$naffairs)
confusion
# Model Accuracy 
Acc_test <- sum(diag(confusion)/sum(confusion))
Acc_test 
# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)
# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values
table(test$naffairs, test$pred_values)
# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train
# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$naffairs)
confusion_train
# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train
# Creating empty vectors to store predicted classes based on threshold value
pred_values_Tr <- NULL
pred_values_Tr <- ifelse(prob_train > optCutOff, 1, 0)
# Creating new column to store the above values
train[,"prob"] <- prob_train
train[,"pred_values"] <- pred_values_Tr
table(train$naffairs, train$pred_values)
# We are having right fight here, there is no huge difference between test and train data.

