# -*- coding: utf-8 -*-
import sys
import utils.csv_read_write as crw
from utils.datasplit import train_test_split 
from utils.Accuracy import accuracy
from utils.DistanceCal import euclideanDis
import numpy as np
from collections import Counter 

# load data
data=np.array(crw.read_csv("dataset/iris2.csv"))

# randomize data
np.random.seed(1)
np.random.shuffle(data)

#separate features and labels
labels=data[:,4]
selector= np.ones((data[1].shape),dtype=bool)
selector[4]=False
features=data[:,selector].astype(float)

#set k value
k=3

#Train Test data split
train_data,train_labels,test_data,test_labels = train_test_split(features,labels,split=.6)



# Get k nearest neighbors
def getkneighbors(train_data,test_data,k):
    index= np.arange(train_data.shape[0])
    dis= euclideanDis(train_data,test_data)
    neigh=dict(zip(index,dis))
    kneigh=[]
    for i in range(k):
        ky=min(neigh, key=neigh.get)          
        kneigh.append(ky)
        del neigh[ky]
    return kneigh
        
    
# make Predictions
pred=[]
for t in test_data:
    counter={}
    kneigh=getkneighbors(train_data,t,k)
    counter=dict(Counter(train_labels[kneigh]))
    pred.append(max(counter,key=counter.get))

#Accuracy    
print("Accuracy",accuracy(pred,test_labels)) 

# comparision with sklearn KNN    
from sklearn.neighbors import KNeighborsClassifier
ne = KNeighborsClassifier(n_neighbors=3)
ne.fit(train_data,train_labels)
print("Acc sklearn:",ne.score(test_data,test_labels))
            