#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from DecisionTreeModels import decision_tree_model, predict, average_error, create_tree
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt


def bagging_np_sampling(X_train, y_train, X_test, y_test, size_sample, n_trees=10, replace_tr=True,  verbose=False):

    index = np.arange(X_train.shape[0])
    sum_train_error = []
    sum_test_error = []
    
    y_train_pred = {}
    y_test_pred = {}

    for i in tqdm(range(n_trees)):
        # sample from X_train, y_train
        index_sampled = np.random.choice(
            index, size=size_sample, replace=replace_tr)
            
        X_train_sample = X_train.loc[index_sampled, :]
        y_train_sample = y_train.loc[index_sampled]
        if verbose:
            print("ITERATION ::::", i)
            print('index_sampled=', index_sampled[0:10])
            print("X_train_sample", X_train_sample.head())
            print("y_train_sample", y_train_sample.head())

        m = decision_tree_model(X_train_sample, y_train_sample, model_name='IG', max_depth=16)
        
        # if verbose:
        #     print(m)
        
        y_train_pred[i] = predict(X_train, m)
        y_test_pred[i] = predict(X_test, m)
        
        train_avg = average_error(y_train, pd.DataFrame(y_train_pred).mode(axis=1)[0])
    
        test_avg = average_error(y_test, pd.DataFrame(y_test_pred).mode(axis=1)[0])
        
        sum_train_error.append(train_avg)
        sum_test_error.append(test_avg)
        
        if verbose:
            print("train actual", y_train.head(10))
            print("train Predict", y_train_pred[i].head(10))
            print("train....eval----------\n\n\n")
            print("train 1 counts::", np.sum(y_train == 1))
            print("train pred 1 ith tree", np.sum(y_train_pred[i] == 1))
            print("train performance ith tree", np.sum(np.sum(y_train_pred[i] != y_train)))
            print("train pred 1 count", np.sum(pd.DataFrame(y_train_pred).mode(axis=1)[0] == 1))
            print("train pred not matched", np.sum(pd.DataFrame(y_train_pred).mode(axis=1)[0] != y_train))
            print("train_avg:: ", train_avg)
            
            

    return (sum_train_error, sum_test_error)


def randomforest_np_sampling(X_train, y_train, X_test, y_test, size_sample, size_features, n_trees=10, replace_tr=True, verbose=False):

    index = np.arange(X_train.shape[0])
    sum_train_error = []
    sum_test_error = []
    y_test_pred = {}
    y_train_pred = {}

    for i in tqdm(range(n_trees)):
        # np.random.seed(i)
        index_sampled = np.random.choice(index, size=size_sample, replace=replace_tr)
        # print('index',idx)
        X_train_sample = X_train.loc[index_sampled, :]
        y_train_sample = y_train.loc[index_sampled]
        
        if verbose:
            print("\n\n\n\n\n ITERATION ::::", i)
            print('index_sampled=', index_sampled[0:10])
            print("X_train_sample", X_train_sample.head())
            print("y_train_sample", y_train_sample.head())

        feature_sampled = np.random.choice(
            X_train.columns, size=size_features, replace=False)
        if verbose:
            print('feature_sampled',feature_sampled)

        X_train_rf = X_train_sample[feature_sampled]
        if (verbose):
            print('index_sampled=', index_sampled)

        m = decision_tree_model(
            X_train_rf, y_train_sample, model_name='IG', max_depth=size_features)
        
        if verbose:
            print(m)
        
        #m = decision_tree_model(df_sampled.iloc[:,:-1], df_sampled['y'],model_name='IG',max_depth =16 )
        #print('model =========\n',m)
        y_train_pred[i] = predict(X_train, m)
        # print(y_train_pred[i].value_counts())
        y_test_pred[i] = predict(X_test, m)
        # print(y_train_pred[i].value_counts())
        tr_avg = average_error(y_train, pd.DataFrame(
            y_train_pred).mode(axis=1)[0])
        te_avg = average_error(
            y_test, pd.DataFrame(y_test_pred).mode(axis=1)[0])
        # print('errror',te_avg)
        sum_train_error.append(tr_avg)
        sum_test_error.append(te_avg)
        
        if verbose:
            print("train actual", y_train.head(10))
            print("train Predict", y_train_pred[i].head(10))
            print("train....eval----------\n\n\n")
            print("train 1 counts::", np.sum(y_train == 1))
            print("train pred 1 ith tree", np.sum(y_train_pred[i] == 1))
            print("train performance ith tree", np.sum(np.sum(y_train_pred[i] != y_train)))
            print("train pred 1 count", np.sum(pd.DataFrame(y_train_pred).mode(axis=1)[0] == 1))
            print("train pred not matched", np.sum(pd.DataFrame(y_train_pred).mode(axis=1)[0] != y_train))
            print("train_avg:: ", tr_avg)

    return (sum_train_error, sum_test_error)


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
        median[c] = med
        # if > median value True otherwise False
        binary[c] = X_num[c] > med
    X_new = pd.concat([X_object, binary], axis=1)
    return (X_new, median)


def preprocessed_test_input(X, train_median):
    '''
    processed the numerical test data with median value of the train data
    '''
    X_num = X.select_dtypes(include=[np.int64])
    X_object = X.select_dtypes(include='object')
    binary = pd.DataFrame()
    for c in X_num.columns:
        # print(c)
        # print(train_median[c])
        # if > median value True otherwise False
        binary[c] = X_num[c] > train_median[c]
    X_new = pd.concat([X_object, binary], axis=1)
    return (X_new)


if __name__ == "__main__":
    np.random.seed(100)
    df_train = pd.read_csv('bank/train.csv', header=None)
    column_name = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                   'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    df_train.columns = column_name

    # define X_train original
    X_train_b = df_train[['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                          'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']]

    # X_train after preprocessing
    X_train, train_median = preprocessed_train_input(X_train_b)
    y_train = df_train['y']
    # train_median= preprocessed_train_input(X_train_b)[1] # 1st index return median value

    df_test = pd.read_csv('bank/test.csv', header=None)
    df_test.columns = column_name
    X_test_b = df_test[['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']]

    # X_test after preprocessing
    X_test = preprocessed_test_input(X_test_b, train_median)
    y_test = df_test['y']

    # print(X_train.head(4))
    y_train_n = [1 if y == 'yes' else 0 for y in y_train]

    y_train_n = pd.Series([1 if y == 'yes' else -1 for y in y_train])
    y_test_n = pd.Series([1 if y == 'yes' else -1 for y in y_test])

    print('**************BAGGING ERROR CALCULATION WITH NUMBER OF TREES **************************')
    np.random.seed(100)
    train_err_bagging, test_err_bg = bagging_np_sampling(
        X_train, y_train_n, X_test, y_test_n, size_sample=len(X_train), n_trees=500, verbose= False)
    pd.DataFrame([train_err_bagging, test_err_bg]).to_csv('bagged_error.csv')

    plt.figure()
    plt.plot(train_err_bagging, label='Training Error')
    plt.plot(test_err_bg, label='Test Error')
    plt.xlabel('number of trees')
    plt.ylabel('Error Rate')
    plt.title('Bagging performance with number of trees')
    plt.savefig('Bagging_bank.png')

    print('************** RANDOM FOREST ERROR CALCULATION WITH NUMBER OF TREES **************************\n\n\n')
    
    
    print('************** RANDOM FOREST ERROR CALCULATION WITH NUMBER OF TREES 2 features**************************')
    np.random.seed(100)
    RF_trzin_err_2, RF_test_err_2 = randomforest_np_sampling(
        X_train, y_train_n, X_test, y_test_n, size_sample=len(X_train), n_trees=500, size_features=2, verbose= False)
    pd.DataFrame([RF_trzin_err_2, RF_test_err_2]).to_csv('RF_2feature.csv')

    plt.figure()
    plt.plot(np.arange(len(RF_trzin_err_2)),
             RF_trzin_err_2, label='Training Error')
    plt.plot(np.arange(len(RF_test_err_2)), RF_test_err_2, label='Test Error')

    plt.xlabel('number of trees')
    plt.ylabel('Error Value')
    plt.title('Random Forest Performance for 2 features')
    plt.legend()
    plt.savefig('RF_2features.png')
    
    
    
    print('************** RANDOM FOREST ERROR CALCULATION WITH NUMBER OF TREES 4 features**************************')
    np.random.seed(100)
    RF_traing_err_4, RF_test_err_4 = randomforest_np_sampling(
        X_train, y_train_n, X_test, y_test_n, size_sample=len(X_train), n_trees=500, size_features=4)
    pd.DataFrame([RF_traing_err_4, RF_test_err_4]).to_csv(
        'RF_4feature_seedless.csv')

    plt.figure()
    plt.plot(np.arange(len(RF_traing_err_4)),
             RF_traing_err_4, label='Training Error')
    plt.plot(np.arange(len(RF_test_err_4)), RF_test_err_4, label='Test Error')

    plt.xlabel('number of trees')
    plt.ylabel('Error Value')
    plt.title('Random Forest Performance for 4 features')
    plt.legend()
    plt.savefig('RF_4_features.png')

    print('************** RANDOM FOREST ERROR CALCULATION WITH NUMBER OF TREES 6 features**************************')
    np.random.seed(100)
    RF_train_err_6, RF_teest_err_6 = randomforest_np_sampling(
        X_train, y_train_n, X_test, y_test_n, size_sample=len(X_train), n_trees=500, size_features=6)

    pd.DataFrame([RF_train_err_6, RF_teest_err_6]).to_csv(
        'RF_6feature_seedless.csv')

    plt.figure()
    plt.plot(np.arange(len(RF_train_err_6)),
             RF_train_err_6, label='Training Error')
    plt.plot(np.arange(len(RF_teest_err_6)),
             RF_teest_err_6, label='Test Error')

    plt.xlabel('number of trees')
    plt.ylabel('Error Value')

    plt.legend()
    plt.savefig('RF_6_features.png')
    
    
    