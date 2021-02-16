#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get accuracy 
"""
import sklearn as sk

def get_accuracy(gnb, X, y):
    y_predict = gnb.predict(X)
    y_accuracy = sk.metrics.accuracy_score(y, y_predict)
    return y_accuracy
