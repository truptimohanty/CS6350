#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np 
import pandas as pd
from DecisionTreeModels import decision_tree_model, predict, average_error,create_tree
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import repeat
import multiprocessing as mp
from AdaboostImplementation import adaboost_fit, adaboost_predict_test

df = pd.read_csv('default of credit card clients.csv')
#print(df.head())
df = df.drop(columns=['ID'])
df.columns

df['SEX'] = df['SEX'].astype(object)
df['EDUCATION'] = df['EDUCATION'].astype(object)
df['MARRIAGE'] = df['MARRIAGE'].astype(object)
df['PAY_0'] =  df['PAY_0'].astype(object)
df['PAY_2'] = df['PAY_2'].astype(object)
df['PAY_3'] = df['PAY_3'].astype(object)
df['PAY_4'] = df['PAY_4'].astype(object)
df['PAY_5'] = df['PAY_5'].astype(object)
df['PAY_6'] = df['PAY_6'].astype(object)
df.dtypes

# preprocessed the input data to convert the integer attributes to categorical using median as threshold


def preprocessed_train_input(X):
    '''
    preprocessed the training input data to convert the 
    integer attributes to categorical using median as threshold
    return the modified train data and the median value to be used for test data
    '''
    X_num = X.select_dtypes(include=[np.int64])
    X_object = X.select_dtypes(include='object')
    binary = pd.DataFrame()
    median = {}
    for c in X_num.columns:
        med = np.median(X_num[c])
        # update the median dictinary with median value
        median[c]=med
        # if > median value True otherwise False
        binary[c]=X_num[c]>med
    X_new = pd.concat([X_object,binary],axis=1)
    return(X_new,median)


def preprocessed_test_input(X,train_median):
    '''
    processed the numerical test data with median value of the train data
    '''
    X_num = X.select_dtypes(include=[np.int64])
    X_object = X.select_dtypes(include='object')
    binary = pd.DataFrame()
    for c in X_num.columns:
        #print(c)
        #print(train_median[c])
        # if > median value True otherwise False
        binary[c]=X_num[c]>train_median[c]
    X_new = pd.concat([X_object,binary],axis=1)
    return(X_new)


df_train = df.sample(frac=0.8,replace=False,random_state=20)
#print(df_train.head())

df_test = df[~df.index.isin(df_train.index)]
#print(df_test.head())
df_train = df_train.reset_index(drop=True)
#df_train

df_test= df_test.reset_index(drop=True)
#df_test

X_train_b = df_train[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]

 
 #X_train after preprocessing
X_train=preprocessed_train_input(X_train_b)[0] # 0th index 
X_train
y_train = df_train['default payment next month']
train_median= preprocessed_train_input(X_train_b)[1] # 1st index return median value 


X_test_b = df_train[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]

# # X_test after preprocessing
X_test=preprocessed_test_input(X_test_b,train_median)

y_test = df_test['default payment next month']


y_test_n = pd.Series([ 1 if y==1 else -1 for y in y_test])
y_train_n = pd.Series([ 1 if y==1 else -1 for y in y_train])


def bagging_np_sampling(X_train, y_train, X_test,y_test, size_sample, n_trees=10, replace_tr = True, verbose = False):

    
    
    index = np.arange(X_train.shape[0])
    
   
    sum_train_error = []
    sum_test_error = []
    y_test_pred = {}
    y_train_pred = {}
    
    
    for i in tqdm(range(n_trees)):
        
        #np.random.seed(i)
    
        
        #sample from X_train, y_train
        index_sampled = np.random.choice(index, size=size_sample, replace=replace_tr)

        X_train_sample = X_train.loc[index_sampled,:]
        y_train_sample = y_train.loc[index_sampled]
        if(verbose):
             print('index_sampled=',index_sampled)
        
        m = decision_tree_model(X_train_sample, y_train_sample,model_name='IG',max_depth =16)
        #m = decision_tree_model(df_sampled.iloc[:,:-1], df_sampled['y'],model_name='IG',max_depth =16 )
        #print('model =========\n',m)
        y_train_pred[i] = predict(X_train,m)
        #print(y_train_pred[i].value_counts())
        y_test_pred[i] = predict(X_test,m)
        #print(y_train_pred[i].value_counts())
        tr_avg = average_error(y_train,pd.DataFrame(y_train_pred).mode(axis=1)[0])
        te_avg = average_error(y_test,pd.DataFrame(y_test_pred).mode(axis=1)[0])
        #print('errror',te_avg)
        sum_train_error.append(tr_avg)
        sum_test_error.append(te_avg)

    
    return(sum_train_error,sum_test_error)


def randomforest_np_sampling(X_train, y_train, X_test,y_test, size_sample=len(X_train),size_features=6, n_trees=10, replace_tr = True, verbose = False):

    
    
    index = np.arange(X_train.shape[0])
    
   
    sum_train_error = []
    sum_test_error = []
    y_test_pred = {}
    y_train_pred = {}
    
    
    for i in tqdm(range(n_trees)):
        
        #np.random.seed(i)   
        idx = np.random.choice(index, size=size_sample, replace=replace_tr)
        #print('index',idx)
        X_train_sample = X_train.loc[idx,:]
        y_train_sample = y_train.loc[idx]
        
        feature_sampled = np.random.choice(X_train.columns, size=size_features, replace=False)
        #print('feature_sampled',feature_sampled)

        X_train_rf = X_train_sample[feature_sampled]
        if(verbose):
             print('index_sampled=',index_sampled)
        
        m = decision_tree_model(X_train_rf, y_train_sample,model_name='IG',max_depth =size_features)
        #m = decision_tree_model(df_sampled.iloc[:,:-1], df_sampled['y'],model_name='IG',max_depth =16 )
        #print('model =========\n',m)
        y_train_pred[i] = predict(X_train,m)
        #print(y_train_pred[i].value_counts())
        y_test_pred[i] = predict(X_test,m)
        #print(y_train_pred[i].value_counts())
        tr_avg = average_error(y_train,pd.DataFrame(y_train_pred).mode(axis=1)[0])
        te_avg = average_error(y_test,pd.DataFrame(y_test_pred).mode(axis=1)[0])
        #print('errror',te_avg)
        sum_train_error.append(tr_avg)
        sum_test_error.append(te_avg)

    
    return(sum_train_error,sum_test_error)



print('######### SINGLE TREE WITH MAX DEPTH TRAIN AND TEST ERROR')
model = decision_tree_model(X_train,y_train_n,max_depth=23)

y_tr_pred = predict(X_train,model)
y_te_pred = predict(X_test,model)
print('average error Training single tree maximum depth',average_error(y_train_n,y_tr_pred))    
print('average error Testing single tree maximum depth',average_error(y_test_n,y_te_pred))  

print('###########AdaboostTree#############################################################')

model,alpha = adaboost_fit(X_train,y_train_n,no_iter=50)
train_ind_iter_err,test_ind_iter_err,train_iter_err,test_iter_err = adaboost_predict_test(X_train, X_test, y_train_n, y_test_n, model,alpha)


fig, (ax1, ax2) = plt.subplots(2, 1)
fig.tight_layout(h_pad=2)

ax1.plot(train_ind_iter_err, label='Train')
ax1.set_ylabel('Error Rate')
ax1.plot(test_ind_iter_err,  label='Test')

ax1.set_title("Individual Tree Error Rate")
ax1.legend(loc='upper right')

ax2.plot(train_iter_err, label='Train')
ax2.set_ylabel('Error Rate')
ax2.set_xlabel('Number of Trees')
ax2.plot(test_iter_err,  label='Test')

ax2.set_title("All decision trees prediction results")
ax2.legend()

fig.savefig('CreditDefault_adaboost')

print('########### BAGGING #############################################################')


bag_tr_err,bag_te_err = bagging_np_sampling(X_train,y_train_n,X_test,y_test_n,size_sample=len(X_train),
                                             n_trees=500)
    

    
plt.plot(np.arange(len(bag_tr_err)),bag_tr_err)
plt.plot(np.arange(len(bag_te_err)),bag_te_err)
plt.xlabel('number of trees')
plt.ylabel('Error Value')
plt.title('Bagging Performance Credit Default')
plt.savefig('bagging_creditdefault.jpg')


print('########### RANDOM FOREST #############################################################')


rf_tr_err,rf_te_err = randomforest_np_sampling(X_train,y_train_n,X_test,y_test_n,n_trees=10,
                                               size_sample=len(X_train),size_features=23)

plt.plot(np.arange(len(rf_tr_err)),rf_tr_err)
plt.plot(np.arange(len(rf_tr_err)),rf_te_err)
plt.xlabel('number of trees')
plt.ylabel('Error Value')
plt.title('Random Forest credit default Performance')
plt.savefig('bagging_creditdefault.jpg')


# In[ ]:




