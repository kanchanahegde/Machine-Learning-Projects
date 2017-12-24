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
np.random.seed(1)
np.random.shuffle(data)

#Train Test data split
train_data,test_data = train_test_split(data,[],split=.7)

def entropy(data):
    count=Counter(data[:,-1])
    dataLen=data.shape[0]
    return -1 * sum([(v/dataLen)*(math.log(v/dataLen,2)) for v in count.values()])

def informationGain(featureEntropy,currentEntropy):
    return currentEntropy - featureEntropy
    
    
def find_best_split(data):
    bestGain = 0  # keep track of the best information gain
    bestSplitAttr = None  
    currentEntropy= entropy(data)
    n_features = data.shape[1] - 1  # number of columns

    for col in range(n_features):  # for each feature
        if data[:,col] != -1:
            featureCount=Counter(data[:,col])
            featureEntropy = 0
            lenData=data.shape[0]
            for k,v in featureCount.items():
                featureEntropy += (v/lenData) * entropy(data[data[:,col]==k])
            infoGain = informationGain(featureEntropy,currentEntropy)
            if infoGain >= bestGain:
                bestGain, bestSplitAttr  = infoGain, col

    return bestGain, bestSplitAttr

def partition(data,col):
    partData={}
    values=np.unique(data[:,col])
    for v in values:
        tempData=data[data[:,col]==v]
        tempData[:,col]=-1
        partData[v]=tempData
    return partData
    
class Leaf:
    def __init__(self,maxClass,val,confd,dictClass):
        self.classPred =  maxClass
        self.val = val
        self.confd = confd
        self.dictClass = dictClass
    def __repr__(self):
        return "<Leaf:%s %.5s>" % (self.classPred,self.confd)
    
class DecisionNode:
    def __init__(self,splitCol,branches):
        self.splitCol = splitCol
        self.branches = branches  
    def __repr__(self):
        return "<Inode:%s b:%s>" % (self.splitCol, self.branches)

def buildTree(data):
   
    gain, col = find_best_split(data)

    if gain == 0:   # Stop further spliting ,create Leaf node here
        dictClass = Counter(data[:,-1]) 
        maxClass = max(dictClass,key=dictClass.get)
        Confd = float(dictClass[maxClass])/sum(dictClass.values())
        return Leaf(maxClass,dictClass[maxClass],Confd,dictClass)

    partitionedData = partition(data, col)
    
    branch={}
    for k,v in partitionedData.items():
        branch[k] = buildTree(v)

    return DecisionNode(col,branch)


def printTree(node, spacing=""):

    # Base case: Leaf node
    if isinstance(node, Leaf):
        print (spacing + "Predict {0} {1}".format( node.classPred,round(node.confd,3)))
        return

    print (spacing + header[node.splitCol])

    # Call this function recursively on the each branch
    for k, v in node.branches.items():
        print (spacing + k + '--->')
        printTree(v, spacing + "       ")
 

def classify(DecisionTree,test_data):
    if isinstance(DecisionTree,Leaf):
        return DecisionTree.classPred
    else:
        if test_data[DecisionTree.splitCol] in DecisionTree.branches:
         return  classify(DecisionTree.branches[test_data[DecisionTree.splitCol]],test_data)
    
    
def predict(DecisionTree,test_data):
    trueLabel = test_data[:,-1]
    test_features = test_data[:,:-1]
    i = 0
    correctPred = 0 
    for t in test_features:
        pred=classify(DecisionTree,t)
        if (trueLabel[i] == pred):
            correctPred += 1
        i = i + 1
        
    return float(correctPred)/len(trueLabel)
    

# get total Leaves in the tree
def treeLeaves(DecisionTree):
    if isinstance(DecisionTree,Leaf):
        return 1
    else:
         sum=0
         for k,v in DecisionTree.branches.items():
             sum += treeLeaves(v)
         return sum

# get total internal nodes in the tree
def treeNodes(DecisionTree):
    if isinstance(DecisionTree,Leaf):
        return 0
    else:
         sum=1
         for k,v in DecisionTree.branches.items():
             sum += treeNodes(v)
         return sum     
 
    
DecisionTree = buildTree(train_data) # build the tree using train_data
printTree(DecisionTree)              # print the constructed decision tree

print("Testing_data Accuracy:",predict(DecisionTree,test_data))
print("Leaves Nodes:",treeLeaves(DecisionTree))
print("Total number of Nodes:",treeLeaves(DecisionTree) + treeNodes(DecisionTree))      

