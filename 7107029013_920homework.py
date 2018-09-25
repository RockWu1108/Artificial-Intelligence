import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=np.linspace(0,1000,50)+np.random.randint(0,100,50)
data=data.reshape(2,25)
train_x=data[0]
train_y=data[1]

plt.scatter(train_x,train_y)
plt.show()

mu = train_x.mean() #平均值
sigma = train_x.std()#標準差
#z分數(縮小比例)
def s(x):
    return (x - mu) / sigma 
#重新繪製z分數後數值
train_z = s(train_x)
print(train_z)
plt.scatter(train_z, train_y)
plt.show()

#----------------------------------

#找出 y=ax+b 最接近個點的方程式
theta0 = np.random.rand() #隨機變數a
theta1 = np.random.rand() #隨機變數b

def f(x):
    return theta0 + theta1 * x

def E(x,y):
    return 0.5 * np.sum((y - f(x)) ** 2)

ETA = 1e-3

diff = 1

count = 0

error = E(train_z, train_y)
while diff > 1e-2:
    tmp_theta0 = theta0 -ETA * np.sum((f(train_z) - train_y))
    tmp_theta1 = theta1 -ETA * np.sum((f(train_z) - train_y) * train_z)
    
    theta0 = tmp_theta0
    theta1 = tmp_theta1
    
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error
    
    count += 1
    log = '{}次: theta0 = {:.3f}, theta1 = {:.3f}. 差分 = {:.4f}'
    print(log.format(count, theta0, theta1, diff))

    x = np.linspace(-3, 3, 100)
    plt.plot(train_z,  train_y, 'o')
    plt.plot(x, f(x))
    plt.show()