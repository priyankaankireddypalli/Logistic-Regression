# 2
library(readr)
# Importing the dataset
Advertisement <- read.csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Logistic_regression\\advertising.csv")
summary(Advertisement)
# Checking for NA values
sum(is.na(Advertisement))
# there are no NA values in our dataset.
attach(Advertisement)
unique(Ad_Topic_Line)
# We are getting 1000 unique datas in the Ad_Topic_Line column
unique(City)
# we are getting 966 unique datas in the City column
unique(Country)
#we are getting 237 unique datas in the Country column
unique(Timestamp)
# we are getting 995 unique datas in the Timestamp column
# these unique values won't help our model to train so we will drop these columns
Advertisement<-Advertisement[c(-5,-6,-8,-9)]
fullmodel <- glm(Clicked_on_Ad ~ ., data = Advertisement, family = "binomial")
summary(fullmodel)
prob_full <- predict(fullmodel, Advertisement, type = "response")
prob_full
# Decide on optimal prediction probability cutoff for the model
library(InformationValue)
optCutOff <- optimalCutoff(Clicked_on_Ad, prob_full)
optCutOff
# Check multicollinearity in the model
library(car)
vif(fullmodel)
# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(Clicked_on_Ad, prob_full, threshold = optCutOff)
# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(Clicked_on_Ad, prob_full)
# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)
results <- confusionMatrix(predvalues, Clicked_on_Ad)
sensitivity(predvalues, Clicked_on_Ad)
confusionMatrix(actuals = Clicked_on_Ad, predictedScores = predvalues)
# Data Partitioning
n <- nrow(Advertisement)
n1 <- n * 0.80
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- Advertisement[train_index, ]
test <- Advertisement[-train_index, ]
# Train the model using Training data
finalmodel <- glm(Clicked_on_Ad ~ ., data = train, family = "binomial")
summary(finalmodel)
# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test
# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$Clicked_on_Ad)
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
table(test$Clicked_on_Ad, test$pred_values)
# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train
# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$Clicked_on_Ad)
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
table(train$Clicked_on_Ad, train$pred_values)
# We are having right fight here, there is no huge difference between test and train data.
