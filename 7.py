import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split
import graphviz

train_dataset = r'C:\Users\98306\Desktop\data\train_dataset.csv'
test_dataset = r'C:\Users\98306\Desktop\data\test_dataset.csv'

with open(train_dataset, encoding = 'utf-8') as trainset:
    traindatas = np.loadtxt(trainset, str, delimiter = "\t", skiprows = 1)

with open(test_dataset, encoding = 'utf-8') as testset:
    testdatas = np.loadtxt(testset, str, delimiter = "\t", skiprows = 1)

index = 0
traindata = traindatas[:, [3, 4, 6, 7, 9, 10, 17]]
trainlabel = traindatas[:, [18]]
testdata = testdatas[:, [3, 4, 6, 7, 9, 10, 17]]
encoder_map = { 'login': 1, 'sso': 2,
                'pwd': 1, 'sms': 2, 'otp': 3, 'qr': 4,
                '家庭宽带': 1, '代理IP': 2, '内网': 3, '公共宽带': 4,
                '1级': 1, '2级': 2, '3级': 3,
                'app': 1, 'web': 2,
                'desktop': 1, 'mobile': 2,
                'sales': 1, 'finance': 2, 'management': 3, 'hr': 4,
                '': 0 }
# 将字段编码成数字
for iter in range(len(traindata)):
    for item in range(len(traindata[iter])):
        # print(traindata[iter])
        traindata[iter][item] = encoder_map[traindata[iter][item]]
        # print(traindata[iter])

for iter in range(len(testdata)):
    for item in range(len(testdata[iter])):
        testdata[iter][item] = encoder_map[testdata[iter][item]]
# 转换为nparray
data_X = np.array(traindata,dtype='int')
data_Y = np.array(trainlabel,dtype='int')
data_Z=np.array(testdata,dtype='int')
Xtrain=data_X[:10000]
Ytrain=data_Y[:10000]
Xtest=data_X[10000:]
Ytest=data_Y[10000:]


tr=[]
te=[]
for i in range(1,30):
    clf=DTC(random_state=0,max_depth=i)
    clf=clf.fit(Xtrain,Ytrain)
    score_tr=clf.score(Xtrain,Ytrain)
    socle_te=clf.score(Xtest,Ytest)
    tr.append(score_tr)
    te.append(socle_te)
print(max(te))
plt.plot(range(1,30),tr,color="red",label="train")
plt.plot(range(1,30),te,color="blue",label="test")
plt.xticks(range(1,30))
plt.legend()
plt.show()
# max_depth=2

tr=[]
te=[]
for i in range(1,300):
    clf=DTC(random_state=0,max_depth=2,min_samples_leaf=i)
    clf=clf.fit(Xtrain,Ytrain)
    score_tr=clf.score(Xtrain,Ytrain)
    socle_te=clf.score(Xtest,Ytest)
    tr.append(score_tr)
    te.append(socle_te)
print(max(te))
plt.plot(range(1,300),tr,color="red",label="train")
plt.plot(range(1,300),te,color="blue",label="test")
plt.xticks(range(1,300))
plt.legend()
plt.show()
# min_samples_leaf=#
tr=[]
te=[]
for i in range(1,7):
    clf=DTC(random_state=0,max_depth=2,max_features=i)
    clf=clf.fit(Xtrain,Ytrain)
    score_tr=clf.score(Xtrain,Ytrain)
    socle_te=clf.score(Xtest,Ytest)
    tr.append(score_tr)
    te.append(socle_te)
print(max(te))
plt.plot(range(1,7),tr,color="red",label="train")
plt.plot(range(1,7),te,color="blue",label="test")
plt.xticks(range(1,7))
plt.legend()
plt.show()