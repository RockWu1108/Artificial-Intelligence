# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 22:35:45 2018

@author: Rock
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
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

#切割訓練測試集
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3,random_state=0)

#kernel ：核函数 
#gamma ：核函数参数
#C懲罰值

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
   
    print("# Tuning hyper-parameters for %s" % score)
   

     # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
     #cv=交叉驗證生成器(default=3)
     #scoring=評估測試集上的預測
    clf = GridSearchCV(SVC(), tuned_parameters, cv=3,
                       scoring='%s_macro' % score)
    # 用训练集训练这个学习器 clf
    clf.fit(XTrain, yTrain)

    print("Best parameters set found on development set:")
    print()

    # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    print(clf.best_params_)

    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score'] #平均值
    stds = clf.cv_results_['std_test_score'] #標準差

    # 看一下具体的参数间不同数值的组合后得到的分数是多少
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))

        predict = clf.predict(XTest)
        accuracy = metrics.accuracy_score(yTest, predict)

        print("%0.3f (+/-%0.03f) for %r"
              % (means, scores.std() / 2, params),"accuracy: %.2f%%" % (100 * accuracy))








