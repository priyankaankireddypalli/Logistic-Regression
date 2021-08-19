# 3
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
Election = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Logistic_regression\\election_data.csv")
Election.columns #to get the column names
Election.describe()
Election.isna().sum()
#there are NA values in our dataset.
# NA values are present in Election.id , Result, year, Amount spent,popularity rank
# Removing the first column which is is an Index
Election1 = Election.drop('Election-id', axis = 1)
# Rename the columns
Election1 = Election1.rename(columns = {'Amount Spent': 'AmountSpent', 'Popularity Rank': 'PopularityRank'}, inplace = False)
Election1.columns
# Imputating the missing values           
# Mean Imputation for continuous data
mean_value1 = Election1.Year.mean()
mean_value1
Election1.Year = Election1.Year.fillna(mean_value1)
Election1.Year.isna().sum()
from sklearn.impute import SimpleImputer
mean_imputation=SimpleImputer(missing_values=np.nan,strategy='mean') #defining mode imputation
Election1['AmountSpent']=pd.DataFrame(mean_imputation.fit_transform(Election1[['AmountSpent']]))
Election1.AmountSpent.isna().sum()
# For Mode - for Discrete variables
mode_value = Election1.Result.mode()
mode_value
Election1.Result = Election1.Result.fillna((mode_value)[0])
Election1.Result.isna().sum()
mode_imputation=SimpleImputer(missing_values=np.nan,strategy='most_frequent') #defining mode imputation
Election1['PopularityRank']=pd.DataFrame(mode_imputation.fit_transform(Election1[['PopularityRank']]))
Election1.PopularityRank.isna().sum()
Election1.info()
# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('Result ~ Year + AmountSpent + PopularityRank', data = Election1).fit(method='bfgs')
#summary
logit_model.summary2() # for AIC
logit_model.summary()
pred = logit_model.predict(Election1.iloc[ :, 1: ])
from sklearn import metrics
fpr, tpr, thresholds = roc_curve(Election1.Result, pred)
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
Election1["pred"] = np.zeros(11)
# taking threshold value and above the prob value will be treated as correct value 
Election1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(Election1["pred"], Election1["Result"])
classification
# Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(Election1, test_size = 0.4) # 40% test data
# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('Result ~ Year + AmountSpent + PopularityRank', data = train_data).fit(method='bfgs')
#summary
model.summary2() # for AIC
model.summary()
# Prediction on Test data set
test_pred = logit_model.predict(test_data)
# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(5)
# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1
# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Result'])
confusion_matrix
accuracy_test = (2 + 2)/(5) 
accuracy_test
# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Result"])
classification_test
#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Result"], test_pred)
#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test
# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])
# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(6)
# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1
# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Result'])
confusion_matrx
accuracy_train = (2 + 3)/(6)
print(accuracy_train)
# classification report
classification_train = classification_report(train_data["train_pred"], train_data["Result"])
classification_train
#ROC CURVE AND AUC
fprT, tprT, threshold = metrics.roc_curve(train_data["Result"], train_pred)
#PLOT OF ROC
plt.plot(fprT, tprT);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc_train = metrics.auc(fprT, tprT)
roc_auc_train

# We are facing overfitting issue here since there is huge difference in test and train data

