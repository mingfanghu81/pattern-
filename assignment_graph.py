#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this method is used to get the graph of the relationship between recall and precision. 
"""
import matplotlib.pyplot as plt
def recall_precision(recall, precision, n):
    plt.plot(recall,precision)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.plot(recall[0],precision[0],'r*')
    plt.plot(recall[n - 1],precision[n - 1],'r*')
    plt.plot(recall[n // 2],precision[n // 2],'r*')
    plt.show()