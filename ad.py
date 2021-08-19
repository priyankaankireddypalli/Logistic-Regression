# 2
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
Advertisement = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Logistic_regression\\advertising.csv")
Advertisement.columns #to get the column names
Advertisement.describe()
Advertisement.isna().sum()
#there are no NA values in our dataset.
Advertisement.info()
# Rename the columns
Advertisement = Advertisement.rename(columns = {'Daily_Time_ Spent _on_Site': 'DailyTimeSpent', 'Area_Income': 'AreaIncome','Daily Internet Usage':'DailyInternetUsage'}, inplace = False)
Advertisement.columns
pd.crosstab(index=Advertisement['Ad_Topic_Line'], columns='count').sort_values(['count'], ascending=False)
#we are getting 1000 unique datas in the Ad_Topic_Line column
pd.crosstab(index=Advertisement['City'], columns='count').sort_values(['count'], ascending=False)
#we are getting 969 unique datas in the City column
pd.crosstab(index=Advertisement['Country'], columns='count').sort_values(['count'], ascending=False)
#we are getting 237 unique datas in the Country column
pd.crosstab(index=Advertisement['Timestamp'], columns='count').sort_values(['count'], ascending=False)
#we are getting 997 unique datas in the Timestamp column
#these unique values won't help our model to train so we will drop these columns
Advertisement.drop(['Ad_Topic_Line','City','Country','Timestamp'],axis=1,inplace=True)
Advertisement.columns
# Model building 
from statsmodels.formula.api import logit
logistic_model = logit('Clicked_on_Ad ~ DailyTimeSpent + Age + AreaIncome + DailyInternetUsage + Male',Advertisement)
result = logistic_model.fit()
pred = result.predict()
# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(Advertisement.Clicked_on_Ad, pred)
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
Advertisement["pred"] = np.zeros(1000)
# taking threshold value and above the prob value will be treated as correct value 
Advertisement.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(Advertisement["pred"], Advertisement["Clicked_on_Ad"])
classification
# Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(Advertisement, test_size = 0.3) # 30% test data
# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('Clicked_on_Ad ~ DailyTimeSpent + Age + AreaIncome + DailyInternetUsage + Male', data = train_data).fit()
#summary
model.summary2() # for AIC
model.summary()
# Prediction on Test data set
test_pred = model.predict(test_data)
# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(300)
# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1
# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Clicked_on_Ad'])
confusion_matrix
accuracy_test = (135 + 154)/(300) 
accuracy_test
# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Clicked_on_Ad"])
classification_test
#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Clicked_on_Ad"], test_pred)
#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test
# prediction on train data
train_pred = model.predict(train_data)
# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(700)
# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1
# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Clicked_on_Ad'])
confusion_matrx
accuracy_train = (360 + 318)/(700)
print(accuracy_train)
# classification report
classification_train = classification_report(train_data["train_pred"], train_data["Clicked_on_Ad"])
classification_train
#ROC CURVE AND AUC
fprT, tprT, threshold = metrics.roc_curve(train_data["Clicked_on_Ad"], train_pred)
#PLOT OF ROC
plt.plot(fprT, tprT);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc_train = metrics.auc(fprT, tprT)
roc_auc_train

# We are having right fight here, there is no huge difference between test and train data.

