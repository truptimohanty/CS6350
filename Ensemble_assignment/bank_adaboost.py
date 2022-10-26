#!/usr/bin/env python
# coding: utf-8

# In[10]:



import numpy as np
import pandas as pd
from tqdm import tqdm 
from DecisionTreeModels import average_error, decision_tree_model, predict
import matplotlib.pyplot as plt
from AdaboostImplementation import adaboost_fit,adaboost_predict_test

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



y_test_n = pd.Series([1 if (y =='yes') else -1 for y in y_test])
y_train_n = pd.Series([1 if (y=='yes') else -1 for y in y_train])
y_train_n.value_counts()


print('###########SIngle Decision Tree#############')
m = decision_tree_model(X_train,y_train_n,max_depth=16)
y_train_pred = predict(X_train,m)

y_test_pred = predict(X_test,m)

print('Train  error single tree', average_error(y_train_n,y_train_pred))
print('Test error single tree', average_error(y_test_n,y_test_pred))

print('###########AdaboostTree#############')

model,alpha = adaboost_fit(X_train,y_train_n,no_iter=500)
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

fig.savefig('Bank_adaboost')

        


# In[ ]:




