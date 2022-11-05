#!/usr/bin/env python
# coding: utf-8

# In[140]:


import numpy as np 
import pandas as pd


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

def average_error(y_true, y_pred):
    '''
    calculating the average error
    '''
    if len(y_true) != len(y_pred):
        print('length mismatch')
        return None  
    
    error = sum(abs(y_true-y_pred)/2)/len(y_true)
    return(error)



def standard_perceptron(X,y,r,no_epoch):
    #Augmenting a one column vector as a first column of X
    one_column = np.ones(X.shape[0])
    X_aug = np.column_stack((one_column,X))
    
    # Initialize W
    W = np.zeros(X_aug.shape[1])
    
    index = np.arange(X_aug.shape[0])
        
    for e in range(no_epoch):
        ## Setting the seed to value e 
        np.random.seed(e)
        np.random.shuffle(index)
       
        X_aug = X_aug[index,:]
        y = y[index]
        
        #print(y)
        for i in range(X_aug.shape[0]):

            y_pred = np.dot(W,X_aug[i])
         
            if y[i] * y_pred <= 0 :
                #print('print not equal')
            
                W = W + r*(y[i]*X_aug[i])
    return W


def voted_perceptron(X,y,r,no_epoch):
    
    #Augmenting a one column vector as a first column of X
    one_column = np.ones(X.shape[0])
    X_aug = np.column_stack((one_column,X))
    W = np.zeros(X_aug.shape[1])
    
    all_w = []
    all_c = [] 
    Cm = 0
    index = np.arange(X_aug.shape[0])
        
    for e in range(no_epoch):
        ## Setting the seed to value e 
        np.random.seed(e)
        np.random.shuffle(index)
       
        X_aug = X_aug[index,:]
        y = y[index]
        
        #print(y)
        for i in range(X_aug.shape[0]):

            y_pred = np.dot(W,X_aug[i])
         
            if y[i] * y_pred <= 0 :
                #print('print not equal')
                all_w.append(W)
                all_c.append(Cm)
                
                W = W + r*(y[i]*X_aug[i])
                Cm = 1
                
            else:
                Cm = Cm+1
                
    return all_w,all_c




def average_perceptron(X,y,r,no_epoch):
    
    #Augmenting a one column vector as a first column of X
    one_column = np.ones(X.shape[0])
    X_aug = np.column_stack((one_column,X))
    W = np.zeros(X_aug.shape[1])
    a = np.zeros(X_aug.shape[1])
    
    index = np.arange(X_aug.shape[0])
        
    for e in range(no_epoch):
        ## Setting the seed to value e 
        np.random.seed(e)
        np.random.shuffle(index)
       
        X_aug = X_aug[index,:]
        y = y[index]
        
        #print(y)
        for i in range(X_aug.shape[0]):

            y_pred = np.dot(W,X_aug[i])
         
            if y[i] * y_pred <= 0 :
                #print('print not equal')
            
                W = W + r*(y[i]*X_aug[i])
            a = a+W
                
    return a




def predict(X,W):
    
    one_column = np.ones(X.shape[0])
    #Augmenting a one column vector as a first column of X
    X_aug = np.column_stack((one_column,X))
    
    y_pred = np.matmul(X_aug,W)
    
    y_pred [y_pred >0 ] = 1
    y_pred [y_pred <=0 ] = -1
    
    return y_pred
    
    

    
def predict_voted(X,W,C):
    
    one_column = np.ones(X.shape[0])
    X_aug = np.column_stack((one_column,X))
    
    y_pred = np.matmul(X_aug,np.transpose(W))
    
    y_pred [y_pred>0 ] = 1
    y_pred [y_pred<=0 ] = -1
    
    voted_pred = np.matmul(y_pred,C)
    voted_pred [voted_pred>0 ] = 1
    voted_pred [voted_pred<=0 ] = -1
    
    return voted_pred



print('############ Q2a Standard Perceptron ###################')

W =standard_perceptron(X_train,y_train,r=0.1,no_epoch = 10)
y_pred = predict(X_test,W)
#print(y_pred.shape)
print('The Weight Vector is :',W)
print('Average Prediction Error on Test dataset',average_error(y_test, y_pred))

print('\n')
print('############ Q2b Voted Perceptron ###################')

W,C = voted_perceptron(X_train,y_train,r=0.1,no_epoch = 10)
y_pred = predict_voted(X_test,W,C)

final_wc = pd.concat([pd.DataFrame(W),pd.Series(C)],axis=1)
final_wc.columns=['b','w0','w1','w2','w3','counts']
print('Weight Vector and counts ')
print(final_wc)
print('Weights based on maximum counts\n',final_wc.sort_values(by='counts',ascending=False) )
print('Average Prediction Error on Test dataset:',average_error(y_test, y_pred))

print('\n')
print('############ Q2c Average Perceptron ###################')
W =average_perceptron(X_train,y_train,r=0.1,no_epoch = 10)
y_pred = predict(X_test,W)
print('The Weight Vector is :',W)
print('Average Prediction Error on Test dataset:',average_error(y_test, y_pred))



# In[ ]:




