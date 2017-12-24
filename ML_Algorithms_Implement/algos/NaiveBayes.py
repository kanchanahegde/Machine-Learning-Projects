# -*- coding: utf-8 -*-

import utils.csv_read_write as crw
from utils.datasplit import train_test_split 
from utils.Accuracy import accuracy
import numpy as np
from collections import defaultdict , Counter 

# load data
data=np.array(crw.read_csv("dataset/bc.csv"))

# randomize data
np.random.seed(1)
np.random.shuffle(data)

#separate features and labels
labels=data[:,9]
selector= np.ones((data[1].shape),dtype=bool)
selector[9]=False
features=data[:,selector]

#Train Test data split
train_data,train_labels,test_data,test_labels = train_test_split(features,labels,split=.66)


#  calculate a priori probabilities
prob = dict(Counter(train_labels))
for key,value in prob.items():
    prob[key]=value/train_labels.shape[0]

print(prob)

# calculating probabilities of event given class
# create a separate dict for each class
cls = np.unique(train_labels)
class_dict={}
for c in cls:
     class_dict[c]=defaultdict(dict)

for c in cls:
     train_cls= train_data[train_labels==c,:]
     for col in range(train_cls.shape[1]):
            class_dict[c][col]=dict(Counter(train_cls[:,col]))
            for key,value in class_dict[c][col].items():
                class_dict[c][col][key]=value/train_cls.shape[0]
                
                
#prediction for test data
pred=[]
for t in test_data:
      result={}
      for c in cls:
         prob_rel=prob[c]
         for col in range(len(t)):
             if t[col] in class_dict[c][col].keys():
                 prob_rel *= class_dict[c][col][t[col]]
             else:
                 prob_rel*=1
         result[c]=prob_rel
      pred.append(max(result,key=result.get))  

# Accuracy
print("Accuracy:",accuracy(pred,test_labels))

              