# 4
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
# Importing the dataset
Bank = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Logistic_regression\\bank_data.csv")
Bank.columns #to get the column names
Bank.describe()
Bank.isna().sum()
#there are no NA values in our dataset.
# Rename the columns
Bank = Bank.rename(columns = {'joself.employed': 'joselfemployed','joadmin.': 'joadmin','joblue.collar': 'jobluecollar'}, inplace = False)
Bank.columns
# The output of sigmoid function lies in between 0-1
# Model building 
from statsmodels.formula.api import logit
logistic_model = logit('y ~ age + default + balance + housing + loan + duration + campaign + pdays + previous + poutfailure + poutother + poutsuccess + poutunknown + con_cellular + con_telephone + con_unknown + divorced + married + single + joadmin + jobluecollar + joentrepreneur + johousemaid + jomanagement + joretired + joselfemployed + joservices + jostudent + jotechnician + jounemployed + jounknown ',Bank)
result = logistic_model.fit()
pred = result.predict()
from sklearn import metrics
fpr, tpr, thresholds = roc_curve(Bank.y, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
import pylab as pl
i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]
# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
# filling all the cells with zeroes
Bank["pred"] = np.zeros(45211)
# taking threshold value and above the prob value will be treated as correct value 
Bank.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(Bank["pred"], Bank["y"])
classification
# Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(Bank, test_size = 0.3) # 30% test data
# Model building 
# import statsmodels.formula.api as sm
model = logit('y ~ age + default + balance + housing + loan + duration + campaign + pdays + previous + poutfailure + poutother + poutsuccess + poutunknown + con_cellular + con_telephone + con_unknown + divorced + married + single + joadmin + jobluecollar + joentrepreneur + johousemaid + jomanagement + joretired + joselfemployed + joservices + jostudent + jotechnician + jounemployed + jounknown ', data = train_data).fit()
#summary
model.summary2() # for AIC
model.summary()
# Prediction on Test data set
test_pred = model.predict(test_data)
# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(13564)
# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1
# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['y'])
confusion_matrix
accuracy_test = (9772 + 1304)/(13564) 
accuracy_test
# classification report
classification_test = classification_report(test_data["test_pred"], test_data["y"])
classification_test
#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["y"], test_pred)
#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test
# prediction on train data
train_pred = model.predict(train_data)
# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(31647)
# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1
# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['y'])
confusion_matrx
accuracy_train = (22965 + 2997)/(31647)
print(accuracy_train)
# classification report
classification_train = classification_report(train_data["train_pred"], train_data["y"])
classification_train
#ROC CURVE AND AUC
fprT, tprT, threshold = metrics.roc_curve(train_data["y"], train_pred)
#PLOT OF ROC
plt.plot(fprT, tprT);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc_train = metrics.auc(fprT, tprT)
roc_auc_train

# We are having right fight here, there is no huge difference between test and train data.
