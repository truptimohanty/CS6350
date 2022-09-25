#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np 
import pandas as pd
from DecisionTreeModels import decision_tree_model, predict, average_error
from pprint import pprint
from tqdm import tqdm


# In[111]:


## Reading the car data files and setting the X_train/X_test and y_train/y_test
df_train = pd.read_csv('car/train.csv')
df_train.columns = ['buying', 'maint', 'doors',
                        'persons', 'lug_boot', 'safety', 'label']
X_train = df_train[['buying', 'maint',
                        'doors', 'persons', 'lug_boot', 'safety']]
y_train = df_train['label']

df_test = pd.read_csv('car/test.csv')
df_test.columns = ['buying', 'maint', 'doors',
                       'persons', 'lug_boot', 'safety', 'label']
X_test = df_test[['buying', 'maint', 'doors',
                      'persons', 'lug_boot', 'safety']]
y_test = df_test['label']


# ## Q2a 
# 
# 
# ## decision_tree_model: Notes the user can pass the parameters X_train,y_train, model_name, and max_depth. model_name can be selectes as 'IG' for information gain, 'ME' for majority error and 'GI' for Gini Index

# In[112]:


model_name = 'IG'
max_depth = 6

# train the model 
m = decision_tree_model(X_train,y_train,model_name=model_name, max_depth=max_depth)
#pprint(m)

# predict the values
y_train_pred = predict(X_train,m)
y_test_pred = predict(X_test,m)

#calculate the average error
train_error = average_error(y_train,y_train_pred)
test_error = average_error(y_test,y_test_pred)
print('Q2a depth : ',max_depth, 'model : ',model_name,'train_error : ',train_error,'test_error : ',test_error)


# ## Q2b
# Train Error/Test Error for Information Gain, Majority Error and GiniIndex is mentioned below with no of depth

# In[114]:


# define empty list to store the calculated error
train_error_InfoGain = []
test_error_InfoGain = []
train_error_ME = []
test_error_ME=[]
train_error_GI = []
test_error_GI = []
depth = []


for i in tqdm(range(1,7,1)):
    max_depth = i
    depth.append(i)
    model_name = 'IG'
    m = decision_tree_model(X_train,y_train,model_name=model_name, max_depth=max_depth)
    y_train_pred = predict(X_train,m)
    y_test_pred = predict(X_test,m)
    train_error = average_error(y_train,y_train_pred)
    test_error = average_error(y_test,y_test_pred)
    train_error_InfoGain.append(train_error)
    test_error_InfoGain.append(test_error)
    #print('depth : ',max_depth, 'model : ',model_name,'train_error : ',train_error,'test_error : ',test_error)

    model_name = 'ME'
    m = decision_tree_model(X_train,y_train,model_name=model_name, max_depth=max_depth)
    y_train_pred = predict(X_train,m)
    y_test_pred = predict(X_test,m)
    train_error = average_error(y_train,y_train_pred)
    test_error = average_error(y_test,y_test_pred)
    train_error_ME.append(train_error)
    test_error_ME.append(test_error)
    #print('depth : ',max_depth, 'model : ',model_name,'train_error : ',train_error,'test_error : ',test_error)

    model_name = 'GI'
    m = decision_tree_model(X_train,y_train,model_name=model_name, max_depth=max_depth)
    y_train_pred = predict(X_train,m)
    y_test_pred = predict(X_test,m)
    train_error = average_error(y_train,y_train_pred)
    test_error = average_error(y_test,y_test_pred)
    train_error_GI.append(train_error)
    test_error_GI.append(test_error)
#     print('depth : ',max_depth, 'model : ',model_name,'train_error : ',train_error,'test_error : ',test_error)
    
#     print('#############################')
df_error = pd.DataFrame({
                         'depth':depth,
                         'tr_err_IG':train_error_InfoGain,
                         'te_err_IG':test_error_InfoGain,
                         'tr_err_ME':train_error_ME,
                         'te_err_ME':test_error_ME,
                         'tr_err_GI':train_error_GI,
                         'te_err_GI':test_error_GI,
                        })
print('Q2b')
print(df_error)


# In[115]:

# ## Q3a
# Let us consider “unknown” as a particular attribute value, and hence
# we do not have any missing attributes for both training and test. Vary the
# maximum tree depth from 1 to 16 — for each setting, run your algorithm to learn
# a decision tree, and use the tree to predict both the training and test examples.
# Again, if your tree cannot grow up to 16 levels, stop at the maximum level. Report
# in a table the average prediction errors on each dataset when you use information
# gain, majority error and gini index heuristics, respectively

# In[117]:


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

df_train = pd.read_csv('bank/train.csv',header=None)
column_name = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
df_train.columns=column_name

#print(df_train.dtypes)
# define X_train original 
X_train_b = df_train[['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']]

# X_train after preprocessing
X_train=preprocessed_train_input(X_train_b)[0] # 0th index 
y_train = df_train['y']
train_median= preprocessed_train_input(X_train_b)[1] # 1st index return median value 

df_test = pd.read_csv('bank/test.csv',header=None)
column_name = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
df_test.columns=column_name
X_test_b= df_test[['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']]

# X_test after preprocessing
X_test=preprocessed_test_input(X_test_b,train_median)
y_test = df_test['y']

#print(X_train.head(4))


# In[118]:


# define empty list to store the calculated error
train_error_InfoGain = []
test_error_InfoGain = []
train_error_ME = []
test_error_ME=[]
train_error_GI = []
test_error_GI = []
depth = []


for i in tqdm(range(1,17,1)):
    max_depth = i
    depth.append(i)
    model_name = 'IG'
    m = decision_tree_model(X_train,y_train,model_name=model_name, max_depth=max_depth)
    y_train_pred = predict(X_train,m)
    y_test_pred = predict(X_test,m)
    train_error = average_error(y_train,y_train_pred)
    test_error = average_error(y_test,y_test_pred)
    train_error_InfoGain.append(train_error)
    test_error_InfoGain.append(test_error)
    #print('depth : ',max_depth, 'model : ',model_name,'train_error : ',train_error,'test_error : ',test_error)

    model_name = 'ME'
    m = decision_tree_model(X_train,y_train,model_name=model_name, max_depth=max_depth)
    y_train_pred = predict(X_train,m)
    y_test_pred = predict(X_test,m)
    train_error = average_error(y_train,y_train_pred)
    test_error = average_error(y_test,y_test_pred)
    train_error_ME.append(train_error)
    test_error_ME.append(test_error)
    #print('depth : ',max_depth, 'model : ',model_name,'train_error : ',train_error,'test_error : ',test_error)

    model_name = 'GI'
    m = decision_tree_model(X_train,y_train,model_name=model_name, max_depth=max_depth)
    y_train_pred = predict(X_train,m)
    y_test_pred = predict(X_test,m)
    train_error = average_error(y_train,y_train_pred)
    test_error = average_error(y_test,y_test_pred)
    train_error_GI.append(train_error)
    test_error_GI.append(test_error)
#     print('depth : ',max_depth, 'model : ',model_name,'train_error : ',train_error,'test_error : ',test_error)
    
#     print('#############################')
df_error = pd.DataFrame({
                         'depth':depth,
                         'tr_err_IG':train_error_InfoGain,
                         'te_err_IG':test_error_InfoGain,
                         'tr_err_ME':train_error_ME,
                         'te_err_ME':test_error_ME,
                         'tr_err_GI':train_error_GI,
                         'te_err_GI':test_error_GI,
                        })
print('Q3a')
print(df_error)

    

# In[119]:


# ## Q3b 
# Let us consider ”unknown” as attribute value missing. Here we
# simply complete it with the majority of other values of the same attribute in the
# training set. Vary the maximum tree depth from 1 to 16 — for each setting,
# run your algorithm to learn a decision tree, and use the tree to predict both the
# training and test examples. Report in a table the average prediction errors on each
# dataset when you use information gain, majority error and gini index heuristics,
# respectively

# In[120]:


## replace the unknown value with maximum value

# first replace the unknown with nan value
X_train.replace('unknown', np.NaN, inplace=True)
X_test.replace('unknown', np.NaN, inplace=True)

# function to get maximum value for each column in X_train
def get_max(X_train):
    maximum = {}
    for c in X_train.columns:  
        maximum[c] = X_train[c].value_counts().index[0]
    return(maximum)

# function to set nan to maximum value both for train and test 
def set_max(X,maximum):
    for c in X.columns:
        X[c] = X[c].fillna(maximum[c])
    return(X)


# In[121]:


# get maximum of train data 
maximum = get_max(X_train)
# set nan to max value of the training data
X_train_m = set_max(X_train,maximum)
X_test_m = set_max(X_test,maximum)


# In[122]:


# define empty list for the errors
train_error_InfoGain = []
test_error_InfoGain = []
train_error_ME = []
test_error_ME=[]
train_error_GI = []
test_error_GI = []
depth = []


for i in tqdm(range(1,17,1)):
    max_depth = i
    depth.append(i)
    model_name = 'IG'
    m = decision_tree_model(X_train_m,y_train,model_name=model_name, max_depth=max_depth)
    y_train_pred = predict(X_train_m,m)
    y_test_pred = predict(X_test_m,m)
    train_error = average_error(y_train,y_train_pred)
    test_error = average_error(y_test,y_test_pred)
    train_error_InfoGain.append(train_error)
    test_error_InfoGain.append(test_error)
    #print('depth : ',max_depth, 'model : ',model_name,'train_error : ',train_error,'test_error : ',test_error)

    model_name = 'ME'
    m = decision_tree_model(X_train_m,y_train,model_name=model_name, max_depth=max_depth)
    y_train_pred = predict(X_train_m,m)
    y_test_pred = predict(X_test_m,m)
    train_error = average_error(y_train,y_train_pred)
    test_error = average_error(y_test,y_test_pred)
    train_error_ME.append(train_error)
    test_error_ME.append(test_error)
    #print('depth : ',max_depth, 'model : ',model_name,'train_error : ',train_error,'test_error : ',test_error)

    model_name = 'GI'
    m = decision_tree_model(X_train_m,y_train,model_name=model_name, max_depth=max_depth)
    y_train_pred = predict(X_train_m,m)
    y_test_pred = predict(X_test_m,m)
    train_error = average_error(y_train,y_train_pred)
    test_error = average_error(y_test,y_test_pred)
    train_error_GI.append(train_error)
    test_error_GI.append(test_error)
#     print('depth : ',max_depth, 'model : ',model_name,'train_error : ',train_error,'test_error : ',test_error)
    
#     print('#############################')
df_error = pd.DataFrame({
                         'depth':depth,
                         'tr_err_IG':train_error_InfoGain,
                         'te_err_IG':test_error_InfoGain,
                         'tr_err_ME':train_error_ME,
                         'te_err_ME':test_error_ME,
                         'tr_err_GI':train_error_GI,
                         'te_err_GI':test_error_GI,
                        })
print('Q3b')
print(df_error)


# In[105]:



# In[ ]:




