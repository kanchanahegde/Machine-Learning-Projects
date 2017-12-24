# -*- coding: utf-8 -*-

import utils.csv_read_write as crw
from utils.datasplit import train_test_split 
from utils.DataFile import getFileName,getFileHeader
import numpy as np
from collections import Counter
import math

# load data
print("Please enter the file name:")
dataFile=str(input())
data=np.array(crw.read_csv(getFileName(dataFile)))
header=getFileHeader(dataFile)

# randomize data
#np.random.seed(42)
np.random.shuffle(data)

#Train Devlopment Test data split
train_data,test_data = train_test_split(data,[],split=.7)
train_data,dev_data = train_test_split(train_data,[],split=.7) # dev data is 30% of remaining training data


# function to calculate entropy
def entropy(data):
    count=Counter(data[:,-1])  # dictionary with key : class name , Value: number of samples that belongs to that class
    dataLen=data.shape[0]      # total number of samples
    return -1 * sum([(v/dataLen)*(math.log(v/dataLen,2)) for v in count.values()])    # entropy calculation

def informationGain(featureEntropy,currentEntropy):
    return currentEntropy - featureEntropy       # currentEntropy is the parent node entropy ,and featureEntropy is children node entropy
    
    
def find_best_split(data):
    bestGain = 0  # keep track of the best information gain
    bestSplitAttr = None  # keep track of the best feature column number
    currentEntropy= entropy(data)  # Entropy of parent node
    n_features = data.shape[1] - 1  # number of columns

    for col in range(n_features):   # for each feature
        if all(data[:,col]) != -1:       # checks if the coloumn is not already used for split
            featureCount=Counter(data[:,col])      # get the feature values and their count
            featureEntropy = 0
            lenData=data.shape[0]
            for k,v in featureCount.items():       # for each feature value, calculate entropy, and add it feature Entropy 
                featureEntropy += (v/lenData) * entropy(data[data[:,col]==k])
            infoGain = informationGain(featureEntropy,currentEntropy)       # get the information gain
            if infoGain >= bestGain:         
                bestGain, bestSplitAttr  = infoGain, col

    return bestGain, bestSplitAttr

# partitions the data based on feature(col) values
def partition(data,col):
    partData={}
    values=np.unique(data[:,col])
    for v in values:
        tempData=data[data[:,col]==v]
        tempData[:,col]=-1
        partData[v]=tempData
    return partData
    
# class to create Internal and Leaf node of decision tree
# if isLeaf is True, then the object is Leaf node otherwise Internal node
class DecisionNode:
    def __init__(self,isLeaf,splitCol,maxClass,confd,branches):
        self.isLeaf = isLeaf
        self.splitCol = splitCol
        self.classPred =  maxClass
        self.confd = confd
        self.branches = branches  
    def __repr__(self):
        if self.isLeaf is True:
            return "<Leafnode:%s %s %s>" % (self.splitCol,self.classPred,self.confd)
        else:
            return "<Internalnode:%s %s>" % (self.splitCol,self.branches)
            

def buildTree(data):
   
    gain, col = find_best_split(data)  # get the bestGain and best attribute to split

# gain is zero, when no more attribute to split on / when all the samples belong to same class
    if gain == 0:                       # make Leaf node, if gain is zero
        dictClass = Counter(data[:,-1]) 
        maxClass = max(dictClass,key=dictClass.get)
        confd = float(dictClass[maxClass])/sum(dictClass.values())
        return DecisionNode(True,None,maxClass,confd,{})

    partitionedData = partition(data, col) # partition the data on the Best split attribute
    count=Counter(data[:,-1])
    maxClass = max(count,key=count.get)
    confd = float(count[maxClass])/sum(count.values())
    branch={}
    for k,v in partitionedData.items():   #For each partition, build the tree
        branch[k] = buildTree(v)

    return DecisionNode(False,col,maxClass,confd,branch)  # internal  node

def printTree(node, spacing=""):

    # Base case: reached a leaf node
    if node.isLeaf == True:
        print (spacing + "Predict", node.classPred,round(node.confd,3))
        return

    # print Internal Node
    print (spacing + header[node.splitCol].upper())

    # Call this function recursively on the each branch
    for k, v in node.branches.items():
        print (spacing + k + '--->')
        printTree(v, spacing + "       ")
 

# Classify the test sample, using the constructed decision tree
def classify(decisionTree,test_data):
    if isinstance(decisionTree,DecisionNode):  
        if decisionTree.isLeaf == True:
            return decisionTree.classPred
        else:
            if test_data[decisionTree.splitCol] in decisionTree.branches:
              return classify(decisionTree.branches[test_data[decisionTree.splitCol]],test_data)
    
# function classifies each test data using constructed decision tree and then calculate the accuracy     
def predict(decisionTree,test_data):
    trueLabel = test_data[:,-1]
    test_features = test_data[:,:-1]
    i = 0
    correctPred = 0 
    for t in test_features:
        pred=classify(decisionTree,t)
        if (trueLabel[i] == pred):
            correctPred += 1
        i = i + 1    
    return float(correctPred)/len(trueLabel)

'''
for each Internal node
1. initialaccuracy is determined 
2. Internal node is temporarily changed to leaf node
3. Accuracy is again determined 
4. If the cuurent accuracy is more or equal to initial accuracy, internal node is permanently changed to leaf node, 
   else it will be reverted to its original state
''' 
def pruneTree(decisionTree,devData):
      if decisionTree.isLeaf == False:
         accinit=predict(decisionTree,devData)
         decisionTree.isLeaf = True
         acc=predict(decisionTree,devData)
         if acc >= accinit:
             decisionTree.branches=None
             return 
         else:
             decisionTree.isLeaf = False
             for k,v in decisionTree.branches.items():
                 pruneTree(v,devData)
      else:
          return
      
# count the number of Leaf nodes
def treeLeaves(decisionTree):
    if decisionTree.isLeaf == True:
        return 1
    else:
        sum=0
        for k,v in decisionTree.branches.items():
            sum += treeLeaves(v)
        return sum

# count the number of Internal nodes
def treeNodes(decisionTree):
    if decisionTree.isLeaf == True:
        return 0
    else:
        sum=1
        for k,v in decisionTree.branches.items():
            sum += treeNodes(v)
        return sum  
    

# Build the tree 
decisionTree = buildTree(train_data)

print("Unpruned Tree")
printTree(decisionTree)
print("Testing_data Accuracy on Unpruned Tree:",predict(decisionTree,test_data))
print("Number of Leaves:",treeLeaves(decisionTree))
print("Size of the Tree:",treeLeaves(decisionTree) + treeNodes(decisionTree))      

# Prune the tree
pruneTree(decisionTree,dev_data)

print("\n\n")
print("Pruned Tree")
printTree(decisionTree)
print("Testing_data Accuracy on Pruned Tree:",predict(decisionTree,test_data))
print("Number of Leaves in Pruned Tree:",treeLeaves(decisionTree))
print("Size of the Pruned Tree:",treeLeaves(decisionTree) + treeNodes(decisionTree))      


