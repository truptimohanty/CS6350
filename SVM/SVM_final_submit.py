#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np 
import pandas as pd
from scipy import optimize
import math


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



 #Augmenting a one column vector as a first column of X
one_column = np.ones(X_train.shape[0])
X_train_aug = np.column_stack((X_train,one_column))


one_column = np.ones(X_train.shape[0])
X_train_aug = np.column_stack((X_train,one_column))



one_column = np.ones(X_test.shape[0])
X_test_aug = np.column_stack((X_test,one_column))
 

    
def predict(X,W):
    '''
    Predict output with W and X given
    '''
    y_pred = np.matmul(X,W)
    
    y_pred [y_pred >0 ] = 1
    y_pred [y_pred <=0 ] = -1
    
    return y_pred


def average_error(y_true, y_pred):
    '''
    calculating the average error
    '''
    if len(y_true) != len(y_pred):
        print('length mismatch')
        return None  
    
    error = sum(abs(y_true-y_pred)/2)/len(y_true)
    return(error)



def calculate_w_sch_a(X,y,lr,C,no_epoch):
    
    '''
    Calculate the weights with schedule learning rate using a
    '''
    N = X.shape[0]
    no_features = X.shape[1]
    W = np.zeros(no_features)
    index = np.arange(N)
        
    for t in range(no_epoch):
        ## Setting the seed to value t 
        #np.random.seed(t)
        np.random.shuffle(index)
       
        X = X[index,:]
        y = y[index]
        
        #print(y)
        for i in range(N):
            g = np.copy(W)
            # setting the last index zeo to calculate deltab
            g[no_features-1]=0 
             # test subgradient condition based on max function      
            if(y[i]*np.dot(W,X[i]))<= 1:  
                g = g-C*N*y[i]*X[i,:]
             
            lr = lr/(1+lr*t/a)# update lr
            W = W - lr*g
    return W



# Calculate the weights with schedule learning rate without a  

def calculate_w_sch(X,y,lr,C,no_epoch):
   
    N = X.shape[0]
    no_features = X.shape[1]
    W = np.zeros(no_features)
    index = np.arange(N)
        
    for t in range(no_epoch):
        ## Setting the seed to value t 
        #np.random.seed(t)
        np.random.shuffle(index)
       
        X = X[index,:]
        y = y[index]
        
        #print(y)
        for i in range(N):
            g = np.copy(W)
            # setting the last index zeo to calculate deltab
            g[no_features-1]=0
            
            if(y[i]*np.dot(W,X[i]))<= 1:         
                g = g-C*N*y[i]*X[i,:]
            lr = lr/(1+t)
            W = W - lr*g
    return W

#define constraint function for the optimization 
def constraint(alpha,y): 
    con = np.matmul(np.reshape(alpha,(1,-1)),np.reshape(y,(-1,1)))
    return con[0]

# define objective function (Dual SVM)   
def obj_fun(alpha,X,y):
    alphayx = np.multiply(np.multiply(np.reshape(alpha,(-1,1)),np.reshape(y,(-1,1))),X)
    total_loss = 0.5*np.sum(np.matmul(alphayx,np.transpose(alphayx)))-np.sum(alpha)
    return total_loss


# define dual_svm to optimize alpha and return W
def dual_svm(X,y,C):
    N = X.shape[0] # total number of samples
    bounds = [(0,C)]*N # considering alpha bounds between 0 to C
    constraints = ({'type':'eq', 'fun': lambda alpha: constraint(alpha,y)}) #defining constraints
    alpha0 = np.zeros(N) # initial guess for alpha
#     using SLSQP Sequential Least SQuares Programming optimizer.
    optimum = optimize.minimize(lambda alpha: obj_fun(alpha,X,y), alpha0, method='SLSQP',bounds=bounds, constraints=constraints,options={'disp':False})
    W = np.sum(np.multiply(np.multiply(np.reshape(optimum.x,(-1,1)),np.reshape(y,(-1,1))),X),axis=0) # calculate W
    index = np.where((optimum.x>0) & (optimum.x<C)) # index  alpha {0,C}
    b = np.mean(y[index]-np.matmul(X[index,:],np.reshape(W,(-1,1)))) 
    W = W.tolist()
    W.append(b)
    return np.array(W)

#define Gaussian Kernel 
def gaussian_kernel(X1,X2,gamma):
   
    N1 = X1.shape[0]
    N2 = X2.shape[0]
    fv1 = X1.shape[1]
    fv2 = X2.shape[1]
    
    # repeating rowwise
    Xi = np.tile(X1, (1,N2))
    Xi = np.reshape(Xi,(-1,fv1))
   
    Xj = np.tile(X2, (N1,1))
 
    k_rbf = np.exp(np.sum(np.square(Xi-Xj),axis=1)/-gamma)
    # reshape as per x1 x2 dimension 
    k_rbf = np.reshape(k_rbf,(N1,N2))
    
    return k_rbf
                                                                                   
# define objective function gaussina kernel 

def obj_fun_gaussian(alpha,k,y):
    '''
    objective function for Dual SVM with Gaussian Kernel
    '''
    alphay = np.multiply(np.reshape(alpha,(-1,1)),np.reshape(y,(-1,1)))
    alphay_alphay = np.matmul(alphay, np.transpose(alphay))
    total_loss = 0.5*np.sum(np.multiply(alphay_alphay,k))-np.sum(alpha)
    return total_loss
                                                                                 

    
# define dual_svm with gaussian kernel (optimize)
def dual_svm_gaussian(X,y,C,gamma):
    N = X.shape[0] # total number of samples
    bounds = [(0,C)]*N # considering alpha bounds between 0 to C
    constraints = ({'type':'eq', 'fun': lambda alpha: constraint(alpha,y)}) #defining constraints
    alpha0 = np.zeros(N) # initial guess for alpha
    k = gaussian_kernel(X,X,gamma)
    optimum = optimize.minimize(lambda alpha: obj_fun_gaussian(alpha,k,y), alpha0, method='SLSQP',bounds=bounds, constraints=constraints,options={'disp':False})
    return optimum.x


def predict_dual_svm_gaussian(alpha,X0,y0,X,gamma):
    k_rbf = gaussian_kernel(X0,X,gamma)
    k_rbf = np.multiply(np.reshape(y0,(-1,1)),k_rbf)
    y_pred = np.sum(np.multiply(np.reshape(alpha,(-1,1)),k_rbf),axis=0)
    y_pred = np.reshape(y_pred,(-1,1))
    y_pred [y_pred >0 ] = 1
    y_pred [y_pred <=0 ] = -1

    return y_pred.reshape(1,-1)[0]


def perceptron_gaussian(X_train,y_train,gamma,no_epoch):
    '''
    Define perceptron Gaussian with mistake count 
    '''
    c = np.zeros(X_train.shape[0])
    index = np.arange(X_train.shape[0])
    
    k_rbf = gaussian_kernel(X_train,X_train,gamma)
    
    for t in range(no_epoch):
#         np.random.seed(t)
        np.random.shuffle(index)
        
        
        for i in index:
            temp = np.sum(c*y_train*k_rbf[:,i]) 
            out = 1 if temp>0 else -1
            if out != y_train[i]:
                c[i] = c[i]+1
    return c
        

def predict_perceptron_gaussian(X_test,c,X_train,y_train,gamma):
    '''
    Predict perceptron Gaussian with mistake count 
    '''
    y_pred = []
    for x in X_test:
        temp = 0
        for i in range(c.shape[0]):
            k_rbf = math.exp(-1*np.linalg.norm(X_train[i]-x)**2/gamma)
            temp = temp + (c[i]*y_train[i]*k_rbf).item()
        if temp<0:
            y_pred.append(-1)
        else:
            y_pred.append(1)
    return y_pred
 
a = 0.1
C_values = [float(100/873),float(500/873),float(700/873)]
gamma_values = [0.1,0.5,1,5,100]


print('################# 2a #########################')
for c in C_values:
    print('C = ', c)
    W = calculate_w_sch_a(X_train_aug,y_train,lr=0.1,C=c,no_epoch=100)
    y_train_pred = predict(X_train_aug,W)
#     print(y_train_pred)
    y_test_pred = predict(X_test_aug,W)
    print('Learned Weight = ',W)
    print('Average Train Error = ',average_error(y_train, y_train_pred),'Average Test Error = ',average_error(y_test, y_test_pred))
print('\n')  

print('################# 2b #########################')
for c in C_values:
    print('C = ', c)
    W = calculate_w_sch(X_train_aug,y_train,lr=0.1,C=c,no_epoch=100)
    y_train_pred = predict(X_train_aug,W)
    y_test_pred = predict(X_test_aug,W)
    print('Learned Weight = ',W)
    print('Average Train Error = ',average_error(y_train, y_train_pred),'Average Test Error = ',average_error(y_test, y_test_pred))
print('\n')  


print('################# 3a Dual SVM #########################')
for c in C_values:
    print('C = ', c)
    W = dual_svm(X_train_aug[:,[x for x in range(4)]],y_train,C=c)
    y_train_pred = predict(X_train_aug,W)
    y_test_pred = predict(X_test_aug,W)
    print('Learned Weight = ',W)
    print('Average Train Error = ',average_error(y_train, y_train_pred),'Average Test Error = ',average_error(y_test, y_test_pred))
print('\n')  
   

print('################# 3b and 3c Dual SVM Gaussian Kernel (Non Linear)#########################')
for C in C_values:
    count = 0
    for gamma in gamma_values:
        print('C = ', C, 'gamma = ',gamma)
    
        alpha = dual_svm_gaussian(X_train_aug[:,[x for x in range(4)]],y_train,C=C,gamma=gamma)
        index = np.where(alpha>0)[0]

        y_train_pred = predict_dual_svm_gaussian(alpha,X_train_aug[:,[x for x in range(4)]],y_train,
                                             X_train_aug[:,[x for x in range(4)]],gamma)
        y_test_pred = predict_dual_svm_gaussian(alpha,X_train_aug[:,[x for x in range(4)]],y_train,
                                             X_test_aug[:,[x for x in range(4)]],gamma)

        print('no of Support Vectors =',len(index),'Average Train Error = ',average_error(y_train, y_train_pred),'Average Test Error = ',average_error(y_test, y_test_pred))
    
        if count>0:
            intersect_vectors = len(np.intersect1d(index,prev_index))
            print('no of support vectors are same = ', intersect_vectors)
        count = count +1
        prev_index = index

print('\n')        
        
print('################# 3d Perceptron Gaussian Kernel (Non Linear)#########################')
    
for gamma in gamma_values:
    c =  perceptron_gaussian(X_train,y_train,gamma,no_epoch=100)
    y_train_pred = predict_perceptron_gaussian(X_train,c,X_train,y_train,gamma)
    y_test_pred = predict_perceptron_gaussian(X_test,c,X_train,y_train,gamma)
    
    print('gamma =',gamma, 'Average Train Error = ',average_error(y_train, y_train_pred),'Average Test Error = ',average_error(y_test, y_test_pred))
    
    
            
                            


# In[ ]:




