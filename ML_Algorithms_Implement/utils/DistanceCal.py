# -*- coding: utf-8 -*-

import numpy as np
# calculate Euclidean Distances
def euclideanDis(train_data, test_data):
    distance=[]
    distance= np.sqrt(np.sum(np.subtract(train_data,test_data) ** 2,axis=1))
    return distance