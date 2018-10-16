
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 22:11:59 2018
@author: Rock
"""
"""
sex: insurance contractor gender, female, male
bmi: Body mass index, providing an understanding of body, 
     weights that are relatively high or low relative to height, 
     objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
children: Number of children covered by health insurance / Number of dependents
smoker: Smoking
region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
charges: Individual medical costs billed by health insurance
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing,linear_model
from sklearn.tree import export_graphviz
import math
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn import ensemble, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
import matplotlib.pyplot as plt

data_url="insurance.csv"
df=pd.read_csv(data_url)
predictors=['age','sex','bmi','children']

'''(2)更改類別值'''
label_encoder=preprocessing.LabelEncoder()
#性別(male=1,female=0)
df['sex']=label_encoder.fit_transform(df['sex'])
##抽菸(yes=1,no=0)
df['smoker']=label_encoder.fit_transform(df['smoker'])
##地區改為0,1(sw=3,se=2,nw=1,ne=0)
df['region']=label_encoder.fit_transform(df['region'])

X = pd.DataFrame(df,columns=['age','sex','charges','bmi'])
y = df["smoker"]

#CART ALGO
XTrain_Gini, XTest_Gini, yTrain_Gini, yTest_Gini = train_test_split(X, y, test_size=0.25,random_state=1)
CART_tree = DecisionTreeClassifier(criterion='gini')
CART_tree.fit(XTrain_Gini, yTrain_Gini)
CART_predict=CART_tree.predict(XTest_Gini)
print("(1)使用CART準確率:",CART_tree.score(XTest_Gini,yTest_Gini))

#Logistic ALGO
XTrain_Log, XTest_Log, yTrain_Log, yTest_Log = train_test_split(X, y, test_size=0.25,random_state=1)
logistic = linear_model.LogisticRegression()
logistic.fit(XTrain_Log,yTrain_Log)
logistic_predict = logistic.predict(XTest_Log)
print("(2)使用Logistic準確率:",logistic.score(XTest_Log,yTest_Log))



#KNN ALGO
print("\nk值選擇:")
XTrain_KNN, XTest_KNN, yTrain_KNN, yTest_KNN = train_test_split(X, y, test_size=0.25,random_state=1)
ks=np.arange(1,round(0.2*len(XTest_KNN))+1)
accuracies=[]
best_kvalue=0
for k in ks:
    knn=neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
    accuracy=knn.score(XTrain_KNN,yTrain_KNN)
    accuracies.append(accuracy)
Best_k_value=max(accuracies)
Index=accuracies.index(Best_k_value)+1 
print("(3)使用KNN準確率:",knn.score(XTest_KNN,yTest_KNN))
print("最佳k值:",Index)
plt.plot(ks,accuracies)
plt.show()


#交叉驗證k最佳化
print("\nk-fold交叉驗證法:")
ks=np.arange(1,round(0.2*len(X)+1))
accuracies=[]
best_kvalue=0
for k in ks:
    knn=neighbors.KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X,y,scoring="accuracy",cv=10)
    accuracies.append(scores.mean())
Best_k_value=max(accuracies)
Index=accuracies.index(Best_k_value) +1  
print("最佳k值:",Index)
plt.plot(ks,accuracies)
plt.show()


