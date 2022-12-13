#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from math import exp
from dataclasses import dataclass
from copy import deepcopy

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
X_train_aug = np.column_stack((one_column,X_train))


one_column = np.ones(X_test.shape[0])
X_test_aug = np.column_stack((one_column,X_test))
 
D_train =np.matrix(X_train_aug)
D_test =np.matrix( X_test_aug)




def average_error(y_true, y_pred):
    '''
    calculating the average error
    '''
    if len(y_true) != len(y_pred):
        print('length mismatch')
        return None  
    
    error = sum(abs(y_true-y_pred)/2)/len(y_true)
    return(error)


def sigmoid(x):
    return 1 / (1 + exp(-1*x))


def sigmoid_deriv(x): #as x = sigmoid(z)
    return x * (1 - x)


class Neural_Net:
    def __init__(self, layers, num_inputs, hidden_nodes, rand_init):
        
        self.layerCount = layers # defining total layers 

        # defining architecture
        self.layer_nodes = np.concatenate([np.array([num_inputs]),
                                               np.array(hidden_nodes)+1, 
                                               np.array([2])])
         # output at nodes = 0 
        self.nodes = np.zeros((layers, np.amax(self.layer_nodes)))
        # assign first node in each layer = 1
        self.nodes[:,0] = np.ones(layers) 
        # default 0 weights 
        self.weights = np.zeros((layers, np.amax(self.layer_nodes), 
                                 np.amax(self.layer_nodes)))
        if rand_init == True:
            # 3da array (layers x nodes x nodes)
            self.weights = np.random.normal(size=(layers,np.amax(self.layer_nodes), 
                                                  np.amax(self.layer_nodes)))
        
        self.grad_weights = np.zeros((layers,np.amax(self.layer_nodes),
                                  np.amax(self.layer_nodes)))
        self.y = None

        
        
def forward(x, nn): # calculate y 
    nn.nodes[0,:x.shape[1]] = np.copy(x) ## setting the input values 
    for layer in range(1, len(nn.layer_nodes)): 
        for node in range(1, nn.layer_nodes[layer]): 
            # calculate the linear sum ()
            layerSum = np.sum(np.multiply(nn.nodes[layer-1,:], nn.weights[layer-1,node,:]))
            # if reached the last layer just the sum no sigmoid
            if layer == nn.layerCount: 
                nn.y = layerSum
            else: 
                nn.nodes[layer, node] = sigmoid(layerSum)



def backward(y, nn):

    dLdy = nn.y - y
    # make zero as per no of nodes and total layers
    grad_node = np.zeros((len(nn.layer_nodes), np.amax(nn.layer_nodes)))

    for target in reversed(range(1, len(nn.layer_nodes))):
        if target != 0 and target == nn.layerCount: 
            for to in range(1, nn.layer_nodes[target]):
                grad_node[target, to] = dLdy
               
                for from_Node in range(nn.layer_nodes[target-1]):
                    nn.grad_weights[target-1,to,from_Node] = grad_node[target, to] * nn.nodes[target-1, from_Node]
        else: 
            for to in range(1, nn.layer_nodes[target]):
                grad_node[target, to] = 0
                 # calculate derivatives
                for connected in range(1, nn.layer_nodes[target+1]):
                    grad_node[target, to] += grad_node[target+1, connected] * nn.weights[target, connected, to] * sigmoid_deriv(nn.nodes[target, to])
    
            for to in range(nn.layer_nodes[target]):
                for from_Node in range(nn.layer_nodes[target-1]):
                    nn.grad_weights[target-1,to,from_Node] = grad_node[target, to] * nn.nodes[target-1, from_Node]
@dataclass
class GammaSchedule:
    gamma0: float
    d: float                   
                    
def SGD(x, y, nn, GammaSchedule, T):
    # initialize weights
    idxs = np.arange(x.shape[0])
    # set Gamma
    gamma = GammaSchedule.gamma0
    iterations = 1

    for epoch in range(T):
    # shuffle data
        np.random.shuffle(idxs)

        for i in idxs:
        # calculate updated gamma
            gamma = GammaSchedule.gamma0 / (1 + (GammaSchedule.gamma0 * iterations / GammaSchedule.d))

            forward(x[i], nn)
            backward(y[i], nn)
            # update weights 
            nn.weights = np.subtract(nn.weights, x.shape[0]*gamma*nn.grad_weights)
            iterations += 1

    return deepcopy(nn)


def SGD_predict(x, nn):
    predictions = []
    for s in x:
        forward(s, nn)
        pred = nn.y
        if pred < 0:
            predictions.append(-1)
        else:
            predictions.append(1)
    return np.array(predictions)



## Instantiate Class with 3 layers, 3 input ,  with 2 hidden nodes (1  will be added)
nn = Neural_Net(3, 3, [2, 2], False)

## given weights for problem 2

w = [
        [ 
            [ 0,  0,  0], 
            [-1, -2, -3], 
            [ 1,  2,  3]  
        ],
        [ 
            [ 0,  0,  0], 
            [-1, -2, -3], 
            [ 1,  2,  3]
        ],
        [
            [0, 0, 0],
            [-1, 2, -1.5],
            [0, 0, 0]
        ]
]
nn.weights = np.array(w)

n = [
        [1, 1, 1], # input
        [1, 0.00247, 0.9975], # hidden layer 1 z value after calculation
        [1, 0.01803, 0.98197] # hidden layer 2 z value after calculation
]
nn.nodes = np.array(n)

nn.y = -2.4369

print('########## 2(a) gradients with respect to all edge weights ###################')
print()
forward(np.matrix([1,1,1]), nn)
print("Forward Pass value at each node :\n", nn.nodes)
print()
print('Output y = ', nn.y)

backward(1, nn)
print()
print(f"gradient dw at all edges at layer 1\n",nn.grad_weights[0])
print()
print(f"gradient dw at all edges at layer 2\n",nn.grad_weights[1])
print()
print(f"gradient dw at all edges at layer 3\n",nn.grad_weights[2])




# specifying gamma and d value 
gammas_d = [                 
    GammaSchedule(1/8000, 40),GammaSchedule(1/17000, 25), 
    GammaSchedule(1/34000, 35),GammaSchedule(7/87000, 25),
    GammaSchedule(1/87000, 10) 
    ]


width_list = [5, 10, 25, 50, 100]



T = 1

print()
print("*** Part 2(b) Back Propagation SGD Random Initialization ***")


print("Width\tTrain Error\tTest Error")
for i, width in enumerate(width_list):
    nn = Neural_Net(3, D_train.shape[1], [width, width], True)

    nn_learned = SGD(D_train, y_train, nn, gammas_d[i], T)

    train_predicts = SGD_predict(D_train, nn_learned)
    train_err = average_error(train_predicts,y_train)
  
    test_predicts =SGD_predict(D_test, nn_learned)
    test_err = average_error(test_predicts,y_test)
   
    print(f"{width}\t{train_err:.8f}\t{test_err:.8f}")

    
T=10
print()
print("*** Part 2(C) Back Propagation SGD Zero Initialization ***")


print("Width\tTrain Error\tTest Error")
for i, width in enumerate(width_list):
    nn = Neural_Net(3, D_train.shape[1], [width, width], False)

    nn_learned = SGD(D_train, y_train, nn, gammas_d[i], T)

    train_predicts = SGD_predict(D_train, nn_learned)
    train_err = average_error(train_predicts,y_train)
  
    test_predicts =SGD_predict(D_test, nn_learned)
    test_err = average_error(test_predicts,y_test)
   
    print(f"{width}\t{train_err:.8f}\t{test_err:.8f}")

    


# In[ ]:




