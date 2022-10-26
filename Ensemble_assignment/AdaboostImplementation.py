#!/usr/bin/env python
# coding: utf-8

# In[10]:



import numpy as np
import pandas as pd
from tqdm import tqdm 
from DecisionTreeModels import average_error
import matplotlib.pyplot as plt
def entropy_calc_root_weighted(y,D):
    '''
    Calcualting the entropy by knowing the labels
    
    y : labels
    D : Weights
    '''
    # calcualting entropy sum(-plog2p)
    sum = 0
    
    for v in y.unique():
        pv = np.sum(D[y==v])/np.sum(D)
        sum = sum - pv*np.log2(pv)
    return sum

def expected_entropy_weighted(x, y, D):
    '''
    Calculating the expected entropy at each node (sv/s)*H(Sv)
    x : input data for a features
    y : labels
    D : Weights
    '''
    vals = dict(x.value_counts())
    sum = 0.0
    for k, v in vals.items():
        y_v = y.loc[x == k]
        D_v = D.loc[x == k]
        #sum = sum+(v/len(x))*entropy_calc_root_weighted(y_v,D_v)
        sum = sum + (np.sum(D_v)/np.sum(D)) * entropy_calc_root_weighted(y_v,D_v)
        
    return sum

def info_gain_weighted(x,y,D):
    '''
    return gain for x is series H(S)-H(Sv) 
    '''
    return entropy_calc_root_weighted(y,D) - expected_entropy_weighted(x,y,D)


def info_gain_calc_weighted(X,y,D):
    '''
    return information gain while handling a X as a dataframe
    '''
    d = dict()
    for c in X.columns:
        # print(expected_entropy(X[c],y))
        d[c] = info_gain_weighted(X[c], y,D)
    return d

def get_attr_maxgain_weighted(X, y,D):
    '''
    return the attribute which has max gain
    '''
    d = info_gain_calc_weighted(X, y,D)
    split_attr = max(d, key=d.get)
    return split_attr

def get_default_pred(y,D):
    default_val = ''
    max_weight = 0
    for y_u in y.unique():
        w = sum(D[y == y_u])
        if w >= max_weight:
            max_weight = w
            default_val = y_u
    return default_val
        
def create_tree_weighted(X, y, D):
    '''
    Create tree with X as attributes and y as labels, 
    '''
    split_attr = get_attr_maxgain_weighted(X,y,D)
    node = {}
    node[split_attr] = {}
    # testing for unique value in the split attribute
    for c in X[split_attr].unique():
            y1 = y.loc[X[split_attr]==c]
            Dv = D.loc[X[split_attr]==c]
            #node[split_attr][c] = y1.value_counts().index[0]  
            node[split_attr][c] = get_default_pred(y1, Dv) #y1.loc[Dv ==np.max(Dv)] 
    node[split_attr]['default_return'] =  get_default_pred(y, D)
    
    return node


def get_pred_value(x, model):
    '''
    predict y vlaue based on x as a single instance using the trained model
    '''

    if isinstance(model, dict): 
        # reset the index value to 0 
        x = x.reset_index(drop=True)
        attr = [k for k in list(model.keys()) if k !=
                'depth' or k != 'default_return'][0]
        # attr_val of the single instance
        attr_val = x.at[0, attr]
        if(attr_val in model[attr]):
            model = model[attr][attr_val]
        else:
            model = model[attr]['default_return']
        return get_pred_value(x, model)

    else:
        return model


def predict(X, model):
    '''
    predict y vlaue based on x as a dataframe using the trained model
    '''
    pred_values = list()
    for i in np.arange(len(X)):
        pred_values.append(get_pred_value(X.iloc[[i]], model))

    return pd.Series(pred_values)




def adaboost_fit(X_train,y_train,no_iter=50):

    model = {}
    D = {} # contains weights for each iteration
    I = {}
    alpha={}
    D[1] = pd.Series(np.ones(len(X_train))*(1/(len(X_train))))

    for t in tqdm(range(1,no_iter)):
        # print('D[{}]'.format(t),D[t])
        model[t] = create_tree_weighted(X_train,y_train,D[t])
        # print(model[t])
        y_train_pred = predict(X_train,model[t])
        # print('ytrain_pred=====',y_train_pred)
        #print('mul=======',np.sum(y_train*y_train_pred) )
        error = (1/2)-1/2*np.sum(D[t]*y_train*y_train_pred) 
        # print('error======',error)
        alpha[t] = (1/2)*np.log((1-error)/error)
        # print('alpha',alpha[t])
        I[t+1] = D[t]*np.exp(-alpha[t]*y_train*y_train_pred)
        D[t+1] = pd.Series(I[t+1]/np.sum(I[t+1]))
        
    return model,alpha


def adaboost_predict_test(X_train, X_test, y_train, y_test, model,alpha, fig_file_name="Adaboost_performance_bank.png"):
    y_pred_weighted_train = {}
    y_pred_weighted_test = {}
    
    train_iter_err = []
    test_iter_err = []
    train_ind_iter_err = []
    test_ind_iter_err = []
    
    for t in tqdm(model.keys()):
        current_train_pred = predict(X_train,model[t])
        train_ind_iter_err.append(average_error(y_train, current_train_pred))
        
        current_test_pred = predict(X_test,model[t])
        test_ind_iter_err.append(average_error(y_test, current_test_pred))
        
        y_pred_weighted_train[t] = alpha[t]*current_train_pred
        y_pred_weighted_test[t] = alpha[t]*current_test_pred
        
        y_train_pred = np.sign(np.sum(pd.DataFrame(y_pred_weighted_train),axis=1))
        train_iter_err.append(average_error(y_train, y_train_pred))
        
        y_test_pred = np.sign(np.sum(pd.DataFrame(y_pred_weighted_test),axis=1))
        test_iter_err.append(average_error(y_test, y_test_pred))
        
    return  train_ind_iter_err,test_ind_iter_err,train_iter_err,test_iter_err



# In[ ]:




