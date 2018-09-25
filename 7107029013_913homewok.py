# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
"""
x=np.linspace(0, 2*np.pi, 100)#linspace(起始點,結束點,(0~6.28)平分100個點)
y=np.sin(x)#將X所有點依序代進y方程式(散布圖)
#plt.subplot(2, 2, 1)#切割繪製圖(2*2)subplot(行,列,第幾個圖繪製)
plt.scatter(x, y) #scatter()用來繪製散布圖
plt.show()
"""

'''原始數據'''
x1= np.linspace(0, 2*np.pi, 100)#linspace(起始點,結束點,(0~6.28)平分100個點)
y1=np.sin(x1)+np.random.randn(len(x1))/5.0#將X所有點依序代進y方程式(散布圖)
#plt.subplot(2, 2, 2)#切割繪製圖(2*2)subplot(行,列,第幾個圖繪製)
plt.scatter(x1,y1)#scatter()用來繪製散布圖
#plt.show()



'''一次線性回歸'''
slr = LinearRegression()#開一台線性回歸機
#x1=pd.DataFrame(x1)
x1=x1.reshape(-1, 1)#reshape()重塑x1改為二維，列為1但行未知(-1)
slr.fit(x1, y1)#將x1,y1資料餵給機器學習
print("迴歸係數:", slr.coef_)
print("截距:", slr.intercept_)
predicted_y1 = slr.predict(x1) #取出機器學習後的成果
#plt.subplot(2, 2, 3)
plt.scatter(x1,y1,color='blue',label='N=100')
plt.plot(x1, predicted_y1,'r-',label='Degree=1')#印出線性回歸機做出的結果





'''三次線性回歸'''
poly_feature_3= PolynomialFeatures(degree=3, include_bias= False)
X_poly_3= poly_feature_3.fit_transform(x1)

lin_reg_3= LinearRegression()#開啟線性回歸機
lin_reg_3.fit(X_poly_3, y1)
print("(1)迴歸係數(Degree=3):", lin_reg_3.coef_)
print("截距:",lin_reg_3.intercept_)
X_plot= np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly= poly_feature_3.fit_transform(X_plot)
y_plot= np.dot(X_plot_poly, lin_reg_3.coef_.T)+ lin_reg_3.intercept_
#plt.subplot(2, 2, 4)
plt.plot(X_plot, y_plot, 'y-',label='Degree=3')
#plt.plot(x1, y1, 'b.') 


'''九次線性回歸'''
poly_feature_9= PolynomialFeatures(degree=9, include_bias= False)
X_poly_9= poly_feature_9.fit_transform(x1)

lin_reg_9= LinearRegression()#開啟線性回歸機
lin_reg_9.fit(X_poly_9, y1)
print("(2)迴歸係數(Degree=9):", lin_reg_9.coef_)
print("截距:",lin_reg_9.intercept_)
X_plot= np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly= poly_feature_9.fit_transform(X_plot)
y_plot= np.dot(X_plot_poly, lin_reg_9.coef_.T)+ lin_reg_9.intercept_
#plt.subplot(2, 2, 4)
plt.plot(X_plot, y_plot, 'c-',label='Degree=9')
#plt.plot(x1, y1, 'b.') 
plt.legend()
plt.show()
