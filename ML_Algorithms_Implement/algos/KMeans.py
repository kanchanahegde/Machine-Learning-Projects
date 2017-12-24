# -*- coding: utf-8 -*-
import utils.csv_read_write as crw
from utils.DistanceCal import euclideanDis
import numpy as np
from collections import Counter 

# load data
data=np.array(crw.read_csv("dataset/iris2.csv"))

# randomize data
np.random.seed(42)
np.random.shuffle(data)


# remove class labels from the dataset
test=data[:,4]
selector=np.ones((data.shape[1]),dtype=bool)
selector[4]=False
train_data=data[:,selector].astype(float)


# initialize k
print("Please enter the value of k:")

k=int(input())

# randomly choose k points in the dataset as initial centroid
centroids=train_data[:k,:]

print("Initial starting random points:")
print(centroids)
#def updateCentroids(train_data,centroid):
def assignCentroid(train_data,centroids):
    dis=euclideanDis(centroids,train_data)
    return np.argmin(dis)
    
def calNewCentroids(train_data,labels,k):
    newCentroid=np.zeros((k,train_data.shape[1]))
    for i in range(k):
        select=np.array(labels)==i
        if np.any(select):
           newCentroid[i]=np.mean(train_data[select],axis=0)
    return newCentroid   
    
def determineUpdate(oldCentroids,newCentroids):
    updateCentriods=True
    if np.all(np.equal(oldCentroids,newCentroids)):
        updateCentriods=False
    return updateCentriods     

def sumSquareError(train_data,centroid):
    error = np.mean(np.sum(np.subtract(train_data,centroid) ** 2,axis=1),axis=0)
    return error
        
    


iterations=0       
updateCentriods=True
while (updateCentriods):
      labels=[]
      newCentroids=[]
      iterations +=1
      for i,t in enumerate(train_data):
          labels.append(assignCentroid(t,centroids))
      newCentroids=calNewCentroids(train_data,labels,k)
      updateCentriods=determineUpdate(centroids,newCentroids)
      if updateCentriods:         
          centroids=newCentroids
      else:
          clusterError=[]
          clusterClass={}
          clusterC=[]
          errorCount=0
          for i in range(k):
              select=np.array(labels)==i
              clusterError.append(sumSquareError(train_data[select],centroids[i]))
              count=[]
              count=Counter(test[select])
              clusterClass[i]=count
              if bool(count) :
                 clusterC.append(max(count,key=count.get))
                 errorCount += sum(count.values())-max(count.values())

# Display Run Information
print("\nIterations:",iterations)
[print("Cluster: {0}, SSE within cluster: {1}".format(i , j)) for i, j in enumerate(clusterError) ]
for k, v in clusterClass.items():
    print("\nIn Cluster:{0}".format(k))
    for ke,va in v.items():
        print("{:<20}:{:<5}".format(ke,va))
        
[print("Cluster {0}: {1}".format(i,j)) for i,j in enumerate(clusterC)]
print("Incorrectly clustered Instances: {0}  {1:.2f}%".format(errorCount,errorCount/len(labels)*100 ))
    
   