# -*- coding: utf-8 -*-
def accuracy(pred,true_values):
    result=sum(pred==true_values)
    return float(result)/len(true_values)
