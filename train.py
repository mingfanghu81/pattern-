#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
use this method to get trainaccuracy, trainprecision, trainsensitivity, trainspecificity
"""
def get_train(TP, FP, TN, FN):
    trainaccuracy = (TP + TN)/ (TP + TN + FN + FP)
    trainprecision = TP/ (TP + FP)
    trainsensitivity = TP/ (TP + FN)
    trainspecificity = TN/ (TN + FP)
    return trainaccuracy, trainprecision, trainsensitivity, trainspecificity
    