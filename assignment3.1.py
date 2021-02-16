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


from assignment_graph import recall_precision

"for the case, 100 positive samples and 100 negative samples "
positive_number = 100
negative_number = 100
recall_precision(positive_number, negative_number)

"for the case, 100 positive samples and 1000 negative samples"
positive_numbera = 100
negative_numbera = 1000
recall_precision(positive_numbera, negative_numbera)

"for the case, 20 positive samples and 20 negative samples "
positive_numberb = 20
negative_numberb = 20
recall_precision(positive_numberb, negative_numberb)

