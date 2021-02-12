#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and install sklearn on your computer (e.g. download the Anaconda distribution) or 
use a Jupyter notebook on Google Colab (free).
 Download the sample dataset assigData4.csv. This file has 1200 positive and 6000 negative samples. Each sample 215 features 
 (assume the first feature is “feature 1” below). The final value on each line of the file is the class (1 = positive, 
 0 = negative) of that sample. These data represent features extracted from genomic windows that do (positive) and do not (negative) 
 correspond to microRNA. Use Weka to do the following:
a) Data visualization:
i. Load the data (note that there is no header line and that the ‘last’ attribute should be considered as the nominal class).
 Suggest you use the pandas library for this.
ii. Plot the distribution of feature 15 for the two classes on a single histogram. The seaborn library may be useful here.
iii. Plot a scatterplot illustrating the correlation between features 3 and 8, colouring the data by class. Again,
 the seaborn library is useful here.
b) Preprocessing: sklearn implements several filter type (i.e. not wrapper type) feature selection methods.
i. Describe the SelectKBest approach using the chi metric. (~50 words, don’t just copy)
ii. Run a different filter-type feature selection approach on your data (i.e. other than SelectKBest with chi).
i. Briefly describe which and what parameters you used.
ii. Summarize the results: how many features were selected and which features selected? If your method simply returns 
a ranked list of all 215 features, choose a subset by applying an arbitrary cutoff score to the ranked list. Describe 
your approach.
c) Classification: using a naïve Bayes classifier:
i. Which parameters must be set by the user (briefly describe their meaning)
ii. When creating a hold-out test set, what is stratified sampling and how is it applicable here? (~20 words)
iii. For the original feature set (215 features): Conduct a 5-fold cross-validation test. Provide 
the confusion matrix, the accuracy, the precision, the sensitivity, and the specificity. 
Generate a ROC curve and a precision-recall curve.
iv. Repeat iii using your optimal feature set from b-iii) above.
v. Which feature set led to the best performance? (discuss difference in observed performance metrics; ~50 words)
"""
import sklearn as sk
import pandas as pa
import seaborn as se
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
#a)i
names=[]
for i in range(1,216):
    name="feature"+str(i)
    names.append(name)
names.append('classes')
data=pa.read_csv('assigData4.csv',names=names)
#a)ii
feature15class0=[]
for j in range(0,7200):
    if data.classes[j]==0:
        feature15class0.append(data.feature15[j])
se.distplot(feature15class0,bins=50,kde=False,label="class=0")
plt.legend()
feature15class1=[]
for k in range(0,7200):
    if data.classes[k]==1:
        feature15class1.append(data.feature15[k])
se.distplot(feature15class1,color='red',bins=50,kde=False,label="class=1")
plt.legend()
#a)iii# we could see from the graph, there is a weak correlation between the two features in each class.
plt.figure()
se.scatterplot(x = "feature3", y = "feature8", data = data,hue='classes')
#b)i.
"""chi2 is used to express the correlation between the feature with the class,
and then we could use selectkBest method to get k features which have good correlation
with the class"""
#b)ii
"""I use the SelectPercentile with f_classif method, which use chi2 , and then choose the features of
the highest percentile of the scores. the parameter : score_func=<function f_classif>, percentile"""
#ii
Y=pa.Series(data.classes).values
dataupdate=data.drop(columns='classes')
X=dataupdate.values
X_new =SelectPercentile(f_classif, percentile=70)
X_new.fit_transform(X,Y)
selectedfeatureindices=X_new.get_support(indices=True)
a=[]
for l in list(range(len(selectedfeatureindices))):
    b=("feature"+str(selectedfeatureindices[l]+1))
    a.append(b)
print(a)
"""I select 150 features, 'feature1', 'feature2', 'feature3', 'feature4', 'feature5',
'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12',
'feature13', 'feature14', 'feature15', 'feature16', 'feature17', 'feature18', 'feature19',
'feature20', 'feature21', 'feature22', 'feature23', 'feature24', 'feature25', 'feature26',
'feature27', 'feature28', 'feature31', 'feature32', 'feature33', 'feature34', 'feature35',
'feature36', 'feature37', 'feature38', 'feature39','feature40', 'feature41', 'feature42',
'feature43', 'feature44', 'feature45', 'feature47', 'feature49', 'feature54', 'feature55',
'feature56', 'feature59', 'feature60', 'feature61', 'feature65', 'feature66', 'feature67',
'feature68', 'feature69', 'feature70', 'feature71', 'feature72', 'feature73', 'feature74',
'feature75', 'feature76', 'feature77', 'feature78', 'feature79', 'feature80', 'feature84',
'feature85', 'feature89', 'feature90', 'feature91', 'feature93', 'feature95', 'feature96',
'feature97', 'feature98', 'feature99', 'feature100', 'feature103', 'feature104', 'feature105',
'feature106', 'feature108', 'feature109', 'feature110', 'feature112', 'feature116', 'feature119',
'feature120', 'feature121', 'feature122', 'feature123', 'feature124', 'feature126', 'feature129',
'feature132', 'feature136', 'feature137', 'feature138', 'feature139', 'feature140', 'feature142',
'feature143', 'feature145', 'feature146', 'feature147', 'feature148', 'feature149', 'feature150',
'feature151', 'feature154', 'feature155', 'feature156', 'feature157', 'feature159', 'feature161',
'feature162', 'feature163', 'feature164', 'feature165', 'feature167', 'feature169', 'feature170',
'feature171', 'feature172', 'feature173', 'feature174', 'feature175', 'feature178', 'feature179',
'feature180', 'feature182', 'feature184', 'feature185', 'feature194', 'feature196', 'feature197',
'feature200', 'feature201', 'feature202', 'feature203', 'feature204', 'feature209', 'feature210',
'feature211', 'feature212', 'feature213', 'feature214', 'feature215'."""
#c)
#i
#ii
"""we partition the data into training set,validation set and test set"""
#iii
dataarray=dataupdate.values
classarray=data.classes.values
c=sk.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in c.split(dataarray, classarray):
    print("TRAIN:", train_index, "TEST:", test_index)
X_train, X_test = dataarray[train_index], dataarray[test_index]
y_train, y_test = classarray[train_index], classarray[test_index]
gnb = GaussianNB()
gnb.fit(X_train,y_train)
score=gnb.predict_proba(X_train)[:, 1]
y_pred=sk.model_selection.cross_val_predict(gnb,X_train , y_train, groups=None, cv=5,verbose=2)
confusionmatrix=sk.metrics.confusion_matrix(y_train, y_pred)
#4621, 179
# 156, 804
TP=confusionmatrix[1,1]
FP=confusionmatrix[0,1]
FN=confusionmatrix[1,0]
TN=confusionmatrix[0,0]
print("confusion matrix is ",([TP,FP],[FN,TN]))
trainaccuracy= (TP+TN)/(TP+TN+FN+FP)
trainprecision= TP/(TP+FP)
trainsensitivity=TP/(TP+FN)
trainspecificity=TN/(TN+FP)
print("the accuracy is",trainaccuracy)
print("the precision is",trainprecision)
print("the sensitivity is",trainsensitivity)
print("the trainspecificity is",trainspecificity)
fpr, tpr, thresholds =sk.metrics.roc_curve(y_train, score)
plt.figure()
plt.scatter(fpr,tpr)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("trainFPR")
plt.ylabel("trainTPR")
plt.show()
y_score = gnb.predict_proba(X_train)
precision, recall, threshold = sk.metrics.precision_recall_curve(y_train,y_score[:,1])
plt.figure()
plt.plot(recall, precision)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('trainrecall')
plt.ylabel('trainprecision')
plt.show()
#iv
dataarraynew=data[a].values
classarraynew=data.classes.values
s=sk.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indexnew, test_indexnew in s.split(dataarraynew, classarraynew):
    print("TRAIN:", train_indexnew, "TEST:", test_indexnew)
X_trainnew, X_testnew = dataarraynew[train_indexnew], dataarraynew[test_indexnew]
y_trainnew, y_testnew = classarraynew[train_indexnew], classarraynew[test_indexnew]
gnbnew = GaussianNB()
gnbnew.fit(X_trainnew,y_trainnew)
scorenew=gnbnew.predict_proba(X_trainnew)[:, 1]
y_prednew=sk.model_selection.cross_val_predict(gnbnew,X_trainnew , y_trainnew, groups=None, cv=5,verbose=2)
confusionmatrixnew=sk.metrics.confusion_matrix(y_trainnew, y_prednew)
#4621, 179
# 156, 804
TPnew=confusionmatrixnew[1,1]
FPnew=confusionmatrixnew[0,1]
FNnew=confusionmatrixnew[1,0]
TNnew=confusionmatrixnew[0,0]
print("confusion matrix is ",([TPnew,FPnew],[FNnew,TNnew]))
trainaccuracynew= (TPnew+TNnew)/(TPnew+TNnew+FNnew+FPnew)
trainprecisionew= TPnew/(TPnew+FPnew)
trainsensitivitynew=TPnew/(TPnew+FNnew)
trainspecificitynew=TNnew/(TNnew+FPnew)
print("the accuracy is",trainaccuracynew)
print("the precision is",trainprecisionew)
print("the sensitivity is",trainsensitivitynew)
print("the trainspecificity is",trainspecificitynew)
fprnew, tprnew, thresholdsnew =sk.metrics.roc_curve(y_trainnew, scorenew)
plt.figure()
plt.scatter(fprnew,tprnew)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("trainFPRnew")
plt.ylabel("trainTPRnew")
plt.show()
y_scorenew = gnbnew.predict_proba(X_trainnew)
precisionnew, recallnew, thresholdnew = sk.metrics.precision_recall_curve(y_trainnew,y_scorenew[:,1])
plt.figure()
plt.plot(recallnew, precisionnew)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('trainrecallnew')
plt.ylabel('trainprecisionnew')
plt.show()
y_trainpredict=gnb.predict(X_train)
ytrainaccuracy=sk.metrics.accuracy_score(y_train, y_trainpredict)
print("the accuracy on the train set of the original data is",ytrainaccuracy)
y_trainnewpredict=gnbnew.predict(X_trainnew)
ytrainnewaccuracy=sk.metrics.accuracy_score(y_trainnew, y_trainnewpredict)
print("the accuracy on the train set of the selecting data is",ytrainnewaccuracy)
y_testpredict=gnb.predict(X_test)
ytestaccuracy=sk.metrics.accuracy_score(y_test, y_testpredict)
print("the accuracy on the test set of the original data is",ytestaccuracy)
y_testnewpredict=gnbnew.predict(X_testnew)
ytestnewaccuracy=sk.metrics.accuracy_score(y_testnew, y_testnewpredict)
print("the accuracy on the test set of the selecting data is",ytestnewaccuracy)
