import numpy as np
from statistics import mode
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd



def split_train_test_breast_cancer():
    handle = open('breast-cancer-wisconsin.data', 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[i for i in r.split(',')] for r in rows if r])
    out=out[:,1:]
    for col in range(9):
        out[:,col][out[:,col]=='?']=mode(out[:,col])
    np.random.shuffle(out)
    test_data=out[:199,:]
    train_data=out[199:,:]
    train_features=np.array(train_data[:,:-1],dtype=int)
    train_labels=np.array(train_data[:,-1],dtype=int)
    train_labels[train_labels==4]=1
    train_labels[train_labels==2]=0
    test_features=np.array(test_data[:,:-1],dtype=int)
    test_labels=np.array(test_data[:,-1],dtype=int)
    test_labels[test_labels==4]=1
    test_labels[test_labels==2]=0
    return (train_features,train_labels,test_features,test_labels)
    
def split_train_test_spam():
    handle = open('spambase.data', 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[i for i in r.split(',')] for r in rows if r])
    np.random.shuffle(out)
    test_data=out[:601,:]
    train_data=out[601:,:]
    train_features=np.array(train_data[:,:-1],dtype=float)
    train_labels=np.array(train_data[:,-1],dtype=int)
    test_features=np.array(test_data[:,:-1],dtype=float)
    test_labels=np.array(test_data[:,-1],dtype=int)
    return (train_features,train_labels,test_features,test_labels)
    
'''
train_features,train_labels,test_features,test_labels=split_train_test_breast_cancer()
print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)
print(test_labels.shape)

dt=DecisionTreeClassifier()
clf=GridSearchCV(dt,param_grid={'ccp_alpha':[0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050]},cv=5)

for i in range(1,11):
    index=60*i
    clf.fit(train_features[:index,:],train_labels[:index])
    print(clf.best_params_)
    acc_train=clf.score(train_features[:index,:],train_labels[:index])
    acc_test=clf.score(test_features,test_labels)
    print(acc_train,acc_test)


train_features,train_labels,test_features,test_labels=split_train_test_spam()
for i in range(1,11):
    index=430*i
    clf_result=clf.fit(train_features[:index,:],train_labels[:index])
    print(i,clf_result.get_depth())
    acc_train=clf_result.score(train_features[:index,:],train_labels[:index])
    acc_test=clf_result.score(test_features,test_labels)
    print(acc_train,acc_test)
'''
