import numpy as np

def train_test_split(features,labels,split=.7):
    split_index= round((split*features.shape[0]))
    train_data=np.array(features[:split_index])
    test_data=np.array(features[split_index:])
    if len(labels) :
        train_labels=np.array(labels[:split_index])
        test_labels=np.array(labels[split_index:])
        return train_data,train_labels,test_data,test_labels
    return train_data,test_data
    
    

       

