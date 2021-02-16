#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assume that you have developed a new classifier that achieves a sensitivity of 0.75 and a
specificity of 0.80. Create a confusion matrix for three test sets comprising a) 100 samples in each
class, b) 100 positive samples and 1000 negative samples, c) 20 samples in each class.
Plot three precision-recall curves (on a single plot) for a random classifier applied in
each of the three cases. Here, a “random classifier” will classify each test point with
50% likelihood of assigning the positive class and 50% likelihood of negative.
iv) Add three points to the precision-recall plot from part iii), illustrating your classifier’s
performance in each of the three cases.
"""

import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from assignment_graph import recall_precision
"for the case, 100 positive samples and 100 negative samples "
true=[0] * 100+[1] * 100
numpy.random.shuffle(true)
estimate = numpy.random.random(200)
precision,recall,threshold=precision_recall_curve(true,estimate)
sample_number = len(true)
recall_precision(recall, precision, sample_number)

"for the case, 100 positive samples and 1000 negative samples"
trueb=[1] * 100+[0] * 1000
numpy.random.shuffle(trueb)
estimateb= numpy.random.random(1100)
precisionb,recallb,thresholdb=precision_recall_curve(trueb, estimateb)
plt.figure()
sample_numberb = len(trueb)
recall_precision(recallb, precisionb, sample_numberb)

"for the case, 20 positive samples and 20 negative samples "
truec=[0] * 20+[1] * 20
numpy.random.shuffle(truec)
estimatec= numpy.random.random(40)
precisionc,recallc,thresholdc=precision_recall_curve(truec,estimatec)
plt.figure()
sample_numberc = len(truec)
recall_precision(recallc, precisionc, sample_numberc)

