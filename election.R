# 3
library(readr)
# Importing the dataset
el1 <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Logistic_regression\\election_data.csv') 
# Checking for NA values
sum(is.na(el1)) 
# Omitting NA values from the Data 
el11 <- na.omit(el1) 
dim(el11) 
# Removing the first column which is is an Index
attach(el11)
el11 <- el11[-1]
# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
model <- glm(Result ~ ., data = el11, family = "binomial")
summary(model)
# We are going to use NULL and Residual Deviance to compare the between different models
# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))
# Prediction to check model validation
prob <- predict(model, el11, type = "response")
prob
# or use plogis for prediction of probabilities
prob <- plogis(predict(model, el11))
# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, el11$Result)
confusion
# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc
# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(prob > 0.5, 1, 0)
library(caret)
# Confusion Matrix
confusionMatrix(factor(el11$Result, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))
# Build Model on 100% of data
# claimants1 <- claimants1[ , -1] # Removing the first column which is is an Index
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
# Decide on optimal prediction probability cutoff for the model
library(InformationValue)
optCutOff <- optimalCutoff(Result, prob)
optCutOff
# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(Result, prob, threshold = optCutOff)
# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(Result, prob)
# Confusion Matrix
results <- confusionMatrix(pred_values, Result)
sensitivity(pred_values, Result)
confusionMatrix(actuals =Result, predictedScores = pred_values)
# Data Partitioning
n <- nrow(el11)
n1 <- n * 0.85
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- el11[train_index, ]
test <- el11[-train_index, ]
# Train the model using Training data
finalmodel <- glm(Result ~ ., data = train, family = "binomial")
summary(finalmodel)
# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test
# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$Result)
confusion
# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 
# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)
# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values
table(test$Result, test$pred_values)
# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train
# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$Result)
confusion_train
# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train

