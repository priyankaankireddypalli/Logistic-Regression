# 4
library(readr)
# Importing the dataset
Bank <- read.csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Logistic_regression\\bank_data.csv")
summary(Bank)
sum(is.na(Bank))
# there are no NA values in our dataset.
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
attach(Bank)
fullmodel <- glm(Bank$y ~ ., data = Bank, family = "binomial")
summary(fullmodel)
prob_full <- predict(fullmodel, Bank, type = "response")
prob_full
# Decide on optimal prediction probability cutoff for the model
library(InformationValue)
optCutOff <- optimalCutoff(Bank$y, prob_full)
optCutOff
# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(Bank$y, prob_full, threshold = optCutOff)
# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(Bank$y, prob_full)
# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)
results <- confusionMatrix(predvalues, Bank$y)
sensitivity(predvalues, Bank$y)
confusionMatrix(actuals = Bank$y, predictedScores = predvalues)
# Data Partitioning
n <- nrow(Bank)
n1 <- n * 0.80
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- Bank[train_index, ]
test <- Bank[-train_index, ]
# Train the model using Training data
finalmodel <- glm(train$y ~ ., data = train, family = "binomial")
summary(finalmodel)
# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test
# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$y)
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
table(test$y, test$pred_values)
# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train
# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$y)
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
table(train$y, train$pred_values)

# We are having right fight here, there is no huge difference between test and train data.
