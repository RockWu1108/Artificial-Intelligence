import pandas as pd
from sklearn import preprocessing
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import ensemble, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
data_url="insurance.csv"
df=pd.read_csv(data_url)

#更改類別值
label_encoder=preprocessing.LabelEncoder()
#性別(male=1,female=0)
df['sex']=label_encoder.fit_transform(df['sex'])
##抽菸(yes=1,no=0)
df['smoker']=label_encoder.fit_transform(df['smoker'])
##地區改為0,1(sw=3,se=2,nw=1,ne=0)
df['region']=label_encoder.fit_transform(df['region'])



# 建立訓練與測試資料
X = pd.DataFrame(df,columns=['age','sex','charges','bmi'])
y = df["smoker"]
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)

# 建立 boosting 模型
boost = ensemble.AdaBoostClassifier(n_estimators = 100)
boost_fit = boost.fit(XTrain, yTrain)

# 預測
test_y_predicted = boost.predict(XTest)

# 績效
accuracy = metrics.accuracy_score(yTest, test_y_predicted)
print("績效:",accuracy)


#AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None) 
#min_samples_split=表示在分解内部结点时最少的样本数
#min_samples_leaf=表示每个叶结点最小的样本数目
bdt = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=20,min_samples_leaf=5),algorithm='SAMME.R') 
param_test1 = {'n_estimators': range(250,300,10),"learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
gsearch1 = GridSearchCV(bdt,param_test1,cv=10)
gsearch1.fit(X,y)
print(gsearch1.best_params_, gsearch1.best_score_ )









