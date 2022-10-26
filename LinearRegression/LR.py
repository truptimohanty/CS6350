#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math
import random


# In[169]:


def calculate_cost(X,Y,W):
    cost = 0
    for i in range(len(X)):
        cost = cost + (Y[i] - np.dot(W,X[i]))**2
    return 0.5*cost 

def batch_gradient(X,Y,r):
    cost =[]
    w = np.zeros(X.shape[1])
    tol = math.inf
    error = 0
    count = 0
    while tol >1e-6:
        
        grad_w = np.zeros(X.shape[1])
        
        for j in range(len(X[0])):
            count = count+1
            sum = 0
            for i in range(len(X)):
                sum = sum + X[i][j]*(Y[i]-np.dot(w,X[i]))
            grad_w[j] = sum 
        w_next = w + r*grad_w
        tol = np.linalg.norm(w-w_next)
        error = tol
        cost.append(calculate_cost(X,Y,w))
        w = w_next
        
        if calculate_cost(X,Y,w)>100:
            print('diverge')
            break;
    cost.append(calculate_cost(X,Y,w))
    
    return w,cost,error,count

def stochastic_gradient(X, Y, r):

    W = np.zeros(X.shape[1])
    err = math.inf

    costs = [calculate_cost(X, Y, W)]

    while err > 10e-10:
        i = random.randrange(len(X))
        grad_w = np.zeros(X.shape[1])
        for j in range(len(X[0])): 
            grad_w[j] = X[i][j] *(Y[i] - np.dot(W, X[i]))

        next_W = W + r*grad_w
        W = next_W
        next_cost = calculate_cost(X, Y, W) 
        err = abs(next_cost - costs[-1])
        costs.append(next_cost)
    return W, costs


df_train = pd.read_csv('./concrete/train.csv',header = None)
df_test = pd.read_csv('./concrete/test.csv')

print('############# Question 4 ##############')
X_train = df_train.iloc[:,:-1]
one_column = np.ones(X_train.shape[0])
X_train_full = np.column_stack((one_column,X_train))
y_train = df_train.iloc[:,-1]


X_test = df_test.iloc[:,:-1]
one_column = np.ones(X_test.shape[0])
X_test_full = np.column_stack((one_column,X_test))
y_test = df_test.iloc[:,-1]

r = [1,0.5,0.25,0.125,0.065,0.03125,0.015]
for r in r :
    
    W,costs,er,count = batch_gradient(X_train_full,y_train,r)
    print('learning_rate {} has epsilon value {}'.format(r,er))

r = 0.01
plt.figure()
W,costs,er,count = batch_gradient(X_train_full,y_train,r)
plt.plot(costs,color='red')
plt.xlabel('iteration')
plt.ylabel('Error rate')
plt.title('Batch Gradient Descent')
train_er_bgd = calculate_cost(X_train_full,y_train,W)
test_er_bgd = calculate_cost(X_test_full,y_test,W)
print('###########################################')
print('BGD trained weights at learning rate 0.01',W)
print('Training error',train_er_bgd)
print('Test error',test_er_bgd)

plt.savefig('bgd.jpg', dpi=300, bbox_inches='tight')



r = 0.001 
plt.figure()
W,costs = stochastic_gradient(X_train_full,y_train,r)
plt.plot(costs,color='blue')
plt.xlabel('iteration')
plt.ylabel('Error rate')
plt.title('Stochastic Gradient Descent')
train_er_sgd = calculate_cost(X_train_full,y_train,W)
test_er_sgd = calculate_cost(X_test_full,y_test,W)
print('###########################################')
print('SGD trained weights at learning rate 0.001',W)
print('Training error SGD',train_er_sgd)
print('Test error SGD',test_er_sgd)
plt.savefig('sgd.jpg', dpi=300, bbox_inches='tight')

X_train_new = X_train_full.T
xxt = np.matmul(X_train_new,X_train_new.T)
inv_calc = np.linalg.inv(xxt)
w_analytical = np.matmul(np.matmul(inv_calc,X_train_new),y_train)
test_cost_ana = calculate_cost(X_test_full,y_test,w_analytical)
train_cost_ana = calculate_cost(X_train_full,y_train,w_analytical)
print('###########################################')
print('Analytical weights',w_analytical)
print('Training error analytical',train_cost_ana)
print('Test error analytical',test_cost_ana)


# In[ ]:




