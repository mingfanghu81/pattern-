#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
use this method to get FPR,TPR and precision,auc. this method is for assignmen3_2.py
"""
import numpy as np

def get_fpr_tpr_precision(score,real_class):
    FPR = []
    TPR = []
    precision = []
    
    for i in range(0 , 2000):
        T = score[i]
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for j in range(0 , 2000):
            if score[j] > T and real_class[j] == 1:
                TP=TP+1
            elif score[j] > T and real_class[j] == 0:
                FP=FP+1
            elif score[j] <= T and real_class[j] == 1:
                FN = FN + 1
            else:
                TN = TN + 1
        FPR.append(FP / (FP + TN))
        TPR.append(TP / (TP + FN))
        if TP == 0 and FP == 0:
            precision.append(0)
        else:
            precision.append(TP / (TP + FP))
    return FPR, TPR,precision

def get_auc(FPR,TPR):
    b = np.argsort(FPR)
    auc = 0
    prev_x = 0
    for k in range(0, 2000):
        if FPR[b[k]] != prev_x:
            auc += (FPR[b[k]] - prev_x) * TPR[b[k]]
            prev_x = FPR[b[k]]
    return auc,b
