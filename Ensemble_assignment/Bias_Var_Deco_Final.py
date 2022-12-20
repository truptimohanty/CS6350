#!/usr/bin/env python
# coding: utf-8


import numpy as np 
import pandas as pd
from DecisionTreeModels import decision_tree_model, predict, average_error,create_tree, get_pred_value
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from itertools import repeat
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
    return X_new,median


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
    return X_new


#### HELPER FUNCTIONS ##############
def bagging_model_parallel(X_train, y_train, size_sample, n_trees=500, replace_tr = True, verbose = False):
    
    index = np.arange(X_train.shape[0])
    X_train_all_samples = []
    y_train_all_samples = []
    for i in range(n_trees):
        
        #np.random.seed(i)
        #sample from X_train, y_train
        idx = np.random.choice(index, size=size_sample, replace=replace_tr)

        X_train_all_samples.append(X_train.loc[idx,:])
        y_train_all_samples.append(y_train.loc[idx])
        
        if(verbose):
             print('index_sampled=',idx)
    
    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)
    outputs_async = pool.starmap_async(decision_tree_model, zip(X_train_all_samples, y_train_all_samples, 
                                                                repeat('IG',n_trees), repeat(16, n_trees)))
    outputs = outputs_async.get()
    
    return outputs


def bagging_predict_parallel(X,model):
#     y_pred = {}
    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)
    outputs_async = pool.starmap_async(predict, zip(repeat(X,len(model)),model))
    y_pred = outputs_async.get()
    #print(y_pred)
    return pd.DataFrame(y_pred).T.mode(axis=1)[0]


def rf_model_parallel(X_train, y_train, size_sample, size_features, n_trees=3, replace_tr = False,verbose = False):

   
    ## set the seed to repeat the boot strapping 
    index = np.arange(X_train.shape[0])
    
  
    X_train_all_samples = []
    y_train_all_samples = []
    
    for i in tqdm(range(n_trees)):
        #sample from X_train, y_train
        idx = np.random.choice(index, size=size_sample, replace=replace_tr)
        #print('index',idx)
        X_train_sample = X_train.loc[idx,:]
        y_train_sample = y_train.loc[idx]
        
        feature_sampled = np.random.choice(X_train.columns, size=size_features, replace=False)
        #print('feature_sampled',feature_sampled)

        X_train_rf = X_train_sample[feature_sampled]
        X_train_all_samples.append(X_train_rf)
        y_train_all_samples.append(y_train_sample)
        
    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)
    outputs_async = pool.starmap_async(decision_tree_model,zip(X_train_all_samples,y_train_all_samples,
                                                              repeat('IG',n_trees),repeat(size_features,n_trees)))
         
    outputs =  outputs_async.get()
    return outputs

def rf_predict_parallel(X,model):
    
    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)
    outputs_async = pool.starmap_async(predict,zip(repeat(X,len(model)),model))
    y_pred = outputs_async.get()  
    return pd.DataFrame(y_pred).T.mode(axis=1)[0]


def bias_var_new(y_true,y_pred):
    avg_bias = np.mean(np.square(y_true-np.mean(y_pred,axis=1)))
    var = ((np.sum(np.square(np.transpose(y_pred) - y_pred.mean(axis=1))))/(len(np.transpose(y_pred))-1))
    avg_var = np.mean(var)
    return avg_bias,avg_var


def bias_var_calc(y_true,y_pred):
    
    mean_h = np.mean(y_pred,axis=1)
    
    avg_bias = np.mean(np.square(y_true-mean_h))
    var = (np.sum(np.square(np.transpose(y_pred) - mean_h)))/(len(np.transpose(y_pred))-1)
    
    avg_var = np.mean(var)
    return avg_bias,avg_var


if __name__ == "__main__":
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

    y_pred_bagged = {}
    y_pred_singlelearner_bg = {}

    print('CALCULATING THE BIAS AND VARIANCE OF BAGGING AND SINGLE LEARNER TREE')

    for i in tqdm(range(100)): ##########$$$$$$$$$$$$$$$$$$$change to 100 TODO
        
        index = np.arange(X_train.shape[0])
        idx = np.random.choice(index, size=1000, replace=False)
        
        X_train_sample = X_train.loc[idx,:]
        y_train_sample = y_train_n.loc[idx]
        
        X_train_sample_new = X_train_sample.reset_index(drop = True)
        y_train_sample_new = y_train_sample.reset_index(drop = True)
        
        ##########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$change to 500 TODO
        
        models = bagging_model_parallel(X_train_sample_new,y_train_sample_new,size_sample = 1000,
                                        n_trees = 1,replace_tr = True, verbose = False)
        
        y_pred_bagged[i] = bagging_predict_parallel(X_test,models)
        y_pred_singlelearner_bg[i] = predict(X_test,models[0])
        
    y_pred_bagged = pd.DataFrame(y_pred_bagged)
    y_pred_singlelearner_bg = pd.DataFrame(y_pred_singlelearner_bg)

    print('bias and variance of bagged tree 100 times 500 trees',bias_var_calc(y_test_n,y_pred_bagged))   
    print('bias and variance of bagged  SingleLearner',bias_var_calc(y_test_n,y_pred_singlelearner_bg))   

    print('***************************************************************************')

    # print('CALCULATING THE BIAS AND VARIANCE OF RANDOM FOREST AND SINGLE LEARNER TREE')


    # y_pred_rf = {}
    # y_pred_singlelearner_rf = {}


    # print('CALCULATING THE BIAS AND VARIANCE OF RANDOM FOREST AND SINGLE LEARNER TREE')

    # for i in tqdm(range(100)):##########$$$$$$$$$$$$$$$$$$$change to 100
        
    #     index = np.arange(X_train.shape[0])
    #     idx = np.random.choice(index, size=1000, replace=False)
        
    #     X_train_sample = X_train.loc[idx,:]
    #     y_train_sample = y_train_n.loc[idx]
        
    #     X_train_sample_new = X_train_sample.reset_index(drop = True)
    #     y_train_sample_new = y_train_sample.reset_index(drop = True)
        
    #                                         ##########$$$$$$$$$$$$$$$$$$$change to n_trees 500
    #     models_rf = rf_model_parallel(X_train_sample_new,y_train_sample_new,size_sample = 1000,
    #                                     size_features= 16,n_trees = 100,replace_tr = True, verbose = False)
        
    #     y_pred_rf[i] =rf_predict_parallel(X_test,models_rf)
    #     y_pred_singlelearner_rf[i] = predict(X_test,models_rf[0])
        
    # y_pred_rf = pd.DataFrame(y_pred_rf)
    # y_pred_singlelearner_rf = pd.DataFrame(y_pred_singlelearner_rf)

        
    # print('bias and variance of Random Forest 100 times 500 trees',bias_var_calc(y_test_n,y_pred_rf))   
    # print('bias and variance of Random Forest single learner',bias_var_calc(y_test_n,y_pred_singlelearner_rf))   



