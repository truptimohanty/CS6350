#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


### Read the data and preprocessing 
df_train = pd.read_csv('bank-note/train.csv', header = None)
df_test = pd.read_csv('bank-note/test.csv',header = None)
column = ['var','skew','curtosis','entropy','labels']
df_train.columns = column
df_test.columns = column

X_train = np.array(df_train[['var','skew','curtosis','entropy']])
X_test = np.array(df_test[['var','skew','curtosis','entropy']])
y_train = df_train['labels']
y_test = df_test['labels']

y_train =np.array([-1 if y == 0 else 1 for y in y_train])
y_test = np.array([-1 if y==0 else 1 for y in y_test])



 #Augmenting a one column vector as a last column of X

one_column = np.ones(X_train.shape[0])
D_train = np.column_stack((X_train,one_column))



one_column = np.ones(X_test.shape[0])
D_test = np.column_stack((X_test,one_column))




lr = 0.01 # learning rate 
d = 0.1 # d value for schedule gamma 
T = 100 # no of epochs 

var_list = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]


def SGD_MAP( x, y, v, lr):
    num_sample = x.shape[0]
    dim = x.shape[1]
    w = np.zeros([1, dim]) # set initial W
    idx = np.arange(num_sample)
    for t in range(T):
        np.random.shuffle(idx) # radom Shuffle 
        x = x[idx,:]
        y = y[idx]
        for i in range(num_sample):
            x_i = x[i,:].reshape([1, -1])
            tmp = y[i] * np.sum(np.multiply(w, x_i))
            g = - num_sample * y[i] * x_i / (1 + np.exp(tmp)) + w / v #(MAP depends on Variance)
            lr = lr / (1 + lr / d * t)
            w = w - lr * g
    return w.reshape([-1,1])


def SGD_MLE(x, y, lr):
    num_sample = x.shape[0]
    dim = x.shape[1]
    w = np.zeros([1, dim]) 
    idx = np.arange(num_sample)
    for t in range(T):
        np.random.shuffle(idx) # radom Shuffle
        x = x[idx,:]
        y = y[idx]
        for i in range(num_sample):
            tmp = y[i] * np.sum(np.multiply(w, x[i,:]))
            g = - num_sample * y[i] * x[i,:] / (1 + np.exp(tmp))
            lr = lr / (1 + lr / d * t)
            w = w - lr * g
    return w.reshape([-1,1])


print("********** Part 3(a) Logistic Regression MAP Implementation **********")
print("********** lr = 0.01, d = 0.1, Epoch = 100 **********")
print()
print("Variance Value \t Train Error \tTest Error")


for v in var_list:

    w= SGD_MAP(D_train, y_train, v, lr)
    pred = np.matmul(D_train, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    train_err = np.sum(np.abs(pred - np.reshape(y_train,(-1,1)))) / 2 / y_train.shape[0]
    
    pred = np.matmul(D_test, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    test_err = np.sum(np.abs(pred - np.reshape(y_test,(-1,1)))) / 2 / y_test.shape[0]
    print(f"{v}\t\t{train_err:.8f}\t{test_err:.8f}")

print()

print("********** Part 3(b) Logistic Regression MLE Implementation **********")
print("********** lr = 0.01, d = 0.1, Epoch = 100 **********")


w =SGD_MLE(D_train, y_train, lr)


pred = np.matmul(D_train, w)
pred[pred > 0] = 1
pred[pred <= 0] = -1
train_err = np.sum(np.abs(pred - np.reshape(y_train,(-1,1)))) / 2 / y_train.shape[0]

pred = np.matmul(D_test, w)
pred[pred > 0] = 1
pred[pred <= 0] = -1

test_err = np.sum(np.abs(pred - np.reshape(y_test,(-1,1)))) / 2 /y_test.shape[0]
print()
print("Train Error \tTest Error")

print(f"{train_err:.8f}\t{test_err:.8f}")


# In[ ]:




