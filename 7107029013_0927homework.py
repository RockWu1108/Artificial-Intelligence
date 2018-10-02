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

import pandas as pd
from sklearn import preprocessing
from sklearn.tree import export_graphviz

from sklearn.model_selection import train_test_split
from sklearn import ensemble, metrics
from sklearn.tree import DecisionTreeClassifier
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



'''決策樹'''
X = pd.DataFrame(df,columns=['age','sex','charges','bmi'])
y = df["smoker"]
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.25,
                                                random_state=1)
dtree = DecisionTreeClassifier(criterion='entropy' ,max_depth=4)
dtree.fit(XTrain, yTrain)
#preds = dtree.predict_proba(X=XTest)
#print(pd.crosstab(preds[:,0],columns=XTest['age']))
print("決策樹準確率:", dtree.score(XTest, yTest))
#print(dtree.predict(XTest))
#print(yTest.values)
with open("tree3.dot", "w") as f:
    f = export_graphviz(dtree,
                             feature_names=['age','sex','charges','bmi'],
                             out_file=f)


'''隨機森林'''
forest = ensemble.RandomForestClassifier(n_estimators = 100)#森林裡的樹木數量
forest_fit = forest.fit(XTrain, yTrain)
test_y_predicted = forest.predict(XTest)

# 績效
accuracy = metrics.accuracy_score(yTest, test_y_predicted)
print("隨機森林準確率:",accuracy)


