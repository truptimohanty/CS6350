#!/usr/bin/env python
# coding: utf-8

# In[34]:


import torch
from torch import nn
import pandas as pd
from torch.nn import Module, ModuleList, Parameter, Tanh,ReLU
import numpy as np

## Read the data and preprocessing 
df_train = pd.read_csv('bank-note/train.csv', header = None)
df_test = pd.read_csv('bank-note/test.csv',header = None)
column = ['var','skew','curtosis','entropy','labels']
df_train.columns = column
df_test.columns = column

X_train = np.array(df_train[['var','skew','curtosis','entropy']])
X_test = np.array(df_test[['var','skew','curtosis','entropy']])
y_train = df_train['labels']
y_test = df_test['labels']

y_train =np.matrix([-1 if y == 0 else 1 for y in y_train]).T
y_test = np.matrix([-1 if y==0 else 1 for y in y_test]).T



 #Augmenting a one column vector as a first column of X
one_column = np.ones(X_train.shape[0])
X_train_aug = np.column_stack((X_train,one_column))


one_column = np.ones(X_test.shape[0])
X_test_aug = np.column_stack((X_test,one_column))


# Convert the data to PyTorch tensors
x = torch.tensor(X_train_aug)
y = torch.tensor(y_train)

x_test_tensor = torch.tensor(X_test_aug)
y_test_tensor = torch.tensor(y_test)




class Linear_Module(Module): # defining forward pass and initilizaiton 
    def __init__(self, n_in, n_out):
        super(Linear_Module, self).__init__()
        # Use He initialization
        w = torch.empty(n_out, n_in)
        nn.init.kaiming_uniform_(w) 
        self.weight = Parameter(w.double())
        b = torch.empty(1, n_out)
        nn.init.kaiming_uniform_(b)
        self.bias = Parameter(b.double())

    def forward(self, X):
        return X @ self.weight.T  + self.bias
    

class Neural_Net(Module): # defining the neural network 
    def __init__(self, layers):
        super(Neural_Net, self).__init__()
        self.act = ReLU() # relu activation 
        self.fcs = ModuleList()
        self.layers = layers

        for i in range(len(self.layers)-1): # adding layers 
            self.fcs.append(Linear_Module(self.layers[i], self.layers[i+1]))

    def forward(self, X): # evaluating forward pass 
        for fc in self.fcs[:-1]:
            X = fc(X)
            X = self.act(X)
        X = self.fcs[-1](X)
        return X



depth = [3,5,9]
width = [5,10,25,50,100]
T =35

print()
print("PyTorch using ReLu activation function with He Initialization")

print("Depth\tWidth\tTrain Error\tTest Error")
for d in depth:
    
    for w in width:
        
        layers = [x.shape[1]]
        layers += ([w for i in range(d)])
        layers += [1]
        model = Neural_Net(layers)
        optimizer = torch.optim.Adam(model.parameters()) # default Adam optimizer 

        train_error = [] # to save the train error with epochs
        
        #for t in range(1,30):
        
        for epoch in range(T):
            optimizer.zero_grad() # making the grad = 0
            L = ((model(x) - y)**2).sum() # calclating loss
            L.backward() # grad calc
            optimizer.step() # update the weights 

        with torch.no_grad():
            y_train_pred = (model(x)).detach().numpy()
            y_test_pred = (model(x_test_tensor)).detach().numpy()

        y_train_pred[y_train_pred >= 0] = 1
        y_train_pred[y_train_pred < 0] = -1

        wrong_pred = np.sum(np.abs(y_train_pred-y_train) / 2)
        train_err = wrong_pred/y_train.shape[0]
        train_error.append(train_err)
        y_test_pred[y_test_pred >= 0] = 1
        y_test_pred[y_test_pred < 0] = -1
        wrong_pred = np.sum(np.abs(y_test_pred-y_test) / 2)
        test_err = wrong_pred/y_test.shape[0]

        print(f"{d}\t{w}\t{train_err:.8f}\t{test_err:.8f}")
    print('#################################################')


# In[ ]:




