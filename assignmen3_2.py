#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assume that you have developed a system which screens passengers as they disembark from cruise
ships and generates scores indicating how likely it is that a passenger has contracted the Ebola
virus at the buffet. You can tune the threshold, T, to achieve either high sensitivity or high
specificity. For a given value of T, your decision rule is “if (score>T) select YES” and the
passenger will be quarantined. You have collected data and predictions for 2000 cases, along with
the actual class of each passenger (i.e. did the passenger actually have Ebola or not). The data is
given in the following file (column 1 = score, column 2 = true class where 1=YES):
assigData3.tsv
i) Plot a ROC curve and compute the AUC for the given data.
ii) Given the cost of false negatives, you decide that a sensitivity of at least 90% is required. What
is the maximum specificity can we achieve?
iii) Plot a precision-recall curve for the given data.
iv) What precision can you obtain for a sensitivity of 90%? (highlight this point on your curve)
<continued on next page…>
v) Repeat part iv) using a bootstrap test to obtain a 95% confidence interval on the precision at a
recall rate of 90%. Follow Procedure 5.6 from Cohen’s text:
1) Construct a distribution from K bootstrap samples for a statistic u; *
2) Sort the values in the distribution
3) The lower bound of the 95% confidence interval is the (K*0.025)th value, the
upper bound is the (K*0.975) value in the sorted distribution.
*Here, u is the observed precision at a recall of 90% and a bootstrap sample will consist of 2000
samples drawn with replacement.
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from get_method import get_fpr_tpr_precision,get_auc

f=np.loadtxt('assigData3.txt')
score = f[: , 0]
trueclass = f[: , 1]
FPR,TPR,precision = get_fpr_tpr_precision(score,trueclass)
plt.scatter(FPR, TPR)
plt.xlim(0, 1) 
plt.ylim(0, 1)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()
auc,b = get_auc(FPR,TPR)
print (f"the auc is %{auc}.")

"""ii"""
c = []
for m in range(0, 2000):
    if TPR[b[m]] >= 0.90:
        c.append(FPR[b[m]])
m = np.min(c)
specificity = 1 - m
print(f"the maximum specificity is {specificity}")

"""iii"""
plt.figure()
plt.scatter(TPR, precision)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("recall")
plt.ylabel("precision")
plt.show()

"""iv"""
for l in range(0, 2000):
    if TPR[b[l]] == 0.9:
        print (precision[b[l]])
plt.plot(TPR[b[l]], precision[b[l]], 'r*')

"""v"""
precisionv = []
for f in range(0, 2000):
    if TPR[f] == 0.9:
        precisionv.append(precision[f])
boot=sklearn.utils.resample(precisionv, replace = True, n_samples = 2000, random_state = 1)
boot = np.sort(boot)
lower = boot[50]
upper = boot[1950]