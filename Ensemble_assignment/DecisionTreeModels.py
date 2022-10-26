#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pprint import pprint
import pandas as pd
import numpy as np


def entropy_calc_root(y):
    '''
    Calcualting the entropy by knowing the labels
    '''
    # creating a dict based on labels unique values
    labels = dict(y.value_counts())
    sum = 0
    # calcualting entropy sum(-plog2p)
    for v in labels.values():
        sum = sum + -v/len(y)*np.log2(v/len(y))  # calc
    return(sum)


def expected_entropy(x, y):
    '''
    Calculating the expected entropy at each node (sv/s)*H(Sv)
    '''
    vals = dict(x.value_counts())
    sum = 0
    for k, v in vals.items():
        y_v = y.loc[x == k]
        sum = sum+(v/len(x))*entropy_calc_root(y_v)
    return (sum)


def info_gain(x, y):
    '''
    return gain for x is series H(S)-H(Sv) 
    '''
    return(entropy_calc_root(y)-expected_entropy(x, y))


def info_gain_calc(X, y):
    '''
    return information gain while handling a X as a dataframe
    '''
    d = dict()
    for c in X.columns:
        # print(expected_entropy(X[c],y))
        d[c] = info_gain(X[c], y)
    return(d)


def get_attr_maxgain(X, y):
    '''
    return the attribute which has max gain
    '''
    d = info_gain_calc(X, y)
    max_value = max(d, key=d.get)
    return(max_value)


def stop_criteria(y, current_depth, max_depth):
    '''
    Stopping Criteria :
    if current depth == max_depth
    pure leaf label i.e entropy(node) =0
    missing attributes values then return the value which label is most common at that node
    '''
    # if current depth == max_depth
    if current_depth >= max_depth:
        return True, y.value_counts().index[0]
    # pure leaf label i.e entropy(node) =0
    if entropy_calc_root(y) == 0:
        return True, y.unique()[0]
     # missing attributes values then return the value which label is most common at that node
    return False, y.value_counts().index[0]


def create_tree(X, y, current_depth=0,max_depth=3):
    '''
    Create tree with X as attributes and y as labels, 
    parameters :
    max depth
    '''
    #print('maxdepth inside function',max_depth)
    # recursive fuction base case stopping criteria
    stopping, leaf_label = stop_criteria(y, current_depth, max_depth)
    if stopping:
        return leaf_label

    else:
        current_depth = current_depth+1
        # split attribute based on maxgain
        split_attr = get_attr_maxgain(X, y)
        node = {}
        node[split_attr] = {}
        node['depth'] = current_depth

        # testing for unique value in the split attribute
        for c in X[split_attr].unique():
            cols = [c_in for c_in in X.columns if c_in != split_attr]
            sel_rows = X[split_attr].values == c
            # creating new X and y based on attribure value
            X1 = X.loc[sel_rows, cols]
            y1 = y.loc[sel_rows]
            # recursive function calling it self at new sub datasets 
            node[split_attr][c] = create_tree(X1, y1, current_depth, max_depth)
            # if attribute value is not there then update as default_return i.e
            # missing attributes values then return the value which label is most common at that node
            node[split_attr]['default_return'] = leaf_label
            #print('node ===',node)
    #print('Current depth', current_depth)
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
   
    return(pd.Series(pred_values))


def average_error(y_true, y_pred):
    '''
    calculating the average error
    '''
    sum = 0
    for i, j in zip(y_true, y_pred):
        if i != j:
            sum = sum+1

    error = sum/(len(y_true))
    return(error)


# In[28]:


def majority_error_root(y):
    '''
    calculate majority error with y lables
    '''
    labels = dict(y.value_counts())
    total = np.sum(list(labels.values()))
    most_common = max(labels.values())
    return((total-most_common)/total)

def expected_majority_error(x,y):
    '''
    expected majority error(sum(sv/s)*me(sv))
    '''
    vals = dict(x.value_counts())
    sum = 0
    for k,v in vals.items():
        y_v = y.loc[x == k]
        sum = sum+(v/len(x))*majority_error_root(y_v)
    return (sum) 

def majority_gain(x,y):
    '''
    calculate the majority gain for single column
    '''
    return(majority_error_root(y)-expected_majority_error(x,y))

def majority_gain_calc(X,y):
    '''
    calculate majority error for dataframe
    '''
    d = dict()
    for c in X.columns:
        #print(expected_majority_error(X[c],y))
        d[c]= majority_gain(X[c],y)
    return(d)


def get_attr_me_gain(X,y):
    '''
    return attribute with highest gain
    '''
    d = majority_gain_calc(X,y)
    max_value = max(d, key=d.get)
    return(max_value)

def stop_criteria_me(y, current_depth, max_depth):
    '''
    Stopping Criteria :
    if current depth == max_depth
    pure leaf label i.e majority_error =0
    missing attributes values then return the value which label is most common at that node
    '''
    # if current depth == max_depth
    if current_depth >= max_depth:
        return True, y.value_counts().index[0]
    # pure leaf label i.e me(node) =0
    if majority_error_root(y) == 0:
        return True, y.unique()[0]
     # missing attributes values then return the value which label is most common at that node
    return False, y.value_counts().index[0]


def create_tree_me(X, y, current_depth=0,max_depth=3):
    '''
    Create tree with X as attributes and y as labels, 
    parameters :
    max depth
    '''
    #print('maxdepth inside function',max_depth)
    # recursive fuction base case stopping criteria
    stopping, leaf_label = stop_criteria_me(y, current_depth, max_depth)
    if stopping:
        return leaf_label

    else:
        current_depth = current_depth+1
        # split attribute based on maxgain
        split_attr = get_attr_me_gain(X, y)
        node = {}
        node[split_attr] = {}
        node['depth'] = current_depth

        # testing for unique value in the split attribute
        for c in X[split_attr].unique():
            cols = [c_in for c_in in X.columns if c_in != split_attr]
            sel_rows = X[split_attr].values == c
            # creating new X and y based on attribure value
            X1 = X.loc[sel_rows, cols]
            y1 = y.loc[sel_rows]
            # recursive function calling it self at new sub datasets 
            node[split_attr][c] = create_tree_me(X1, y1, current_depth, max_depth)
            # if attribute value is not there then update as default_return i.e
            # missing attributes values then return the value which label is most common at that node
            node[split_attr]['default_return'] = leaf_label
            #print('node ===',node)
    #print('Current depth', current_depth)
    return node


def Gini_root(y):
    '''
    return GI if labels are known
    '''
    labels = dict(y.value_counts())
    sum = 0
    for v in labels.values():
        sum = sum + (v/len(y))**2
    
    return(1-sum)

def expected_GiniIndex(x,y):
    '''
    calculate the expected Gini index
    '''
    vals = dict(x.value_counts())
    sum = 0
    for k,v in vals.items():
        y_v = y.loc[x == k]
        sum = sum+(v/len(x))*Gini_root(y_v)
    return (sum) 

def Gini_gain(x,y):
    '''
    return gain due to Gini index for a column
    '''
    return(Gini_root(y)-expected_GiniIndex(x,y))


def Gini_gain_calc(X,y):
    '''
    return gain due to Gini index for a dataframe
    '''
    d = dict()
    for c in X.columns:
        
        #print(expected_GiniIndex(X[c],y))
        d[c]= Gini_gain(X[c],y)
    return(d)


def get_attr_GI_gain(X,y):
    '''
    return the Gini gain 
    '''
    d = Gini_gain_calc(X,y)
    max_value = max(d, key=d.get)
    return(max_value)

def stop_criteria_GI(y, current_depth, max_depth):
    '''
    Stopping Criteria :
    if current depth == max_depth
    pure leaf label i.e GI_error =0
    missing attributes values then return the value which label is most common at that node
    '''
    # if current depth == max_depth
    if current_depth >= max_depth:
        return True, y.value_counts().index[0]
    # pure leaf label i.e me(node) =0
    if Gini_root(y) == 0:
        return True, y.unique()[0]
     # missing attributes values then return the value which label is most common at that node
    return False, y.value_counts().index[0]


def create_tree_GI(X, y, current_depth=0,max_depth=3):
    '''
    Create tree with X as attributes and y as labels, 
    parameters :
    max depth
    '''
    #print('maxdepth inside function',max_depth)
    # recursive fuction base case stopping criteria
    stopping, leaf_label = stop_criteria_GI(y, current_depth, max_depth)
    if stopping:
        return leaf_label

    else:
        current_depth = current_depth+1
        # split attribute based on maxgain
        split_attr = get_attr_GI_gain(X, y)
        node = {}
        node[split_attr] = {}
        node['depth'] = current_depth

        # testing for unique value in the split attribute
        for c in X[split_attr].unique():
            cols = [c_in for c_in in X.columns if c_in != split_attr]
            sel_rows = X[split_attr].values == c
            # creating new X and y based on attribure value
            X1 = X.loc[sel_rows, cols]
            y1 = y.loc[sel_rows]
            # recursive function calling it self at new sub datasets 
            node[split_attr][c] = create_tree_GI(X1, y1, current_depth, max_depth)
            # if attribute value is not there then update as default_return i.e
            # missing attributes values then return the value which label is most common at that node
            node[split_attr]['default_return'] = leaf_label
            #print('node ===',node)
    #print('Current depth', current_depth)
    return node



def decision_tree_model(X,y,model_name='IG',max_depth=16):
    if model_name =='IG':
        model = create_tree(X, y, max_depth=max_depth)
    elif model_name =='GI':
        model = create_tree_GI(X, y, max_depth=max_depth)
    elif model_name =='ME':
        model = create_tree_me(X, y, max_depth=max_depth)
    else:
        raise ValueError("Not a valid entry")
        
    return(model)

