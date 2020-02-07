# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 08:54:55 2019

"""


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

 
def LoadData1(filename):
    data_df = pd.read_csv(filename, header=0, sep=';')
    correlations = data_df.corr()['quality'].drop('quality')    # new added
    print(correlations)
    Data = data_df.values  # shuffle here
    Data = shuffle_data(Data)
    X = np.array(Data[:, 0:-1])
    Y = Data[:, -1]
    Y = np.asarray([1 if i > 5 else 0 for i in Y])  # if yi = 0-5 -> class 0 // otherwise if yi = 6-10 -> class 1
    Y = Y.transpose()
    return X, Y


def LoadData2(filename):
    with open(filename) as input_file:
        lines = input_file.readlines()
        data_clean = []

        for line in lines:
            newline = line.strip().split(',')
            temp = newline[1:]
            # remove ? from data /// clean the data
            for j in range(len(temp)):
                if temp[j] == '?':
                    temp[j] = -1
                else:
                    temp[j] = int(temp[j])
                    data_clean.append(temp)
        data_clean = np.array(data_clean)   # ndarray shuffle here
        data_clean = shuffle_data(data_clean)
        data_df = pd.DataFrame(data_clean)
        correlations = data_df.corr() # new added
        print(type(data_clean))
        data_x = data_clean[:, 0: -1]
        data_y = data_clean[:, -1]
        data_y = [1 if i == 2 else 0 for i in data_y ]
    return data_x, data_y,correlations

def split_train_test(X,Y):
    Num = len(Y)
    train_x, test_x = X[0:int(0.8*Num), :], X[int(0.8*Num): -1, :]
    train_y, test_y = Y[0:int(0.8*Num)], Y[int(0.8*Num): -1] 
    return train_x,test_x,train_y,test_y   

def dummy_feature(X):
    n1, m1 = np.shape(X)
    X0 = np.ones((n1, 1))
    # # add an additional line to X : X0 = 1 with n*1 dimensional
    X_new = np.column_stack((X0, X))
    return X_new

def shuffle_data(Data):
    number_of_rows = Data.shape[0]
    index = list(np.arange(number_of_rows))
    np.random.shuffle(index)
    Data = Data[index,:]
    return Data


# data statistics
def Data_Stat(X1,Y1):

    # Calculate data statistics
    Mean = np.mean(X1, axis=0)
    Median = np.median(X1, axis=0)
    Std = np.std(X1, axis=0)
    Var = np.var(X1, axis=0)
    Min = np.min(X1, axis=0)
    Max = np.max(X1, axis=0)
    CorreMat = np.corrcoef(X1.T)
    Data = np.column_stack((X1, Y1))
    CorreMat1 = np.corrcoef(Data.T)
#   print (CorreMat)
#    data_stat = pd.DataFrame({'Mean': Mean, 'Median': Median, 'Std': Std, 'Var': Var, 'Min': Min, 'Max': Max})
#    print(data_stat) 
    return  data_stat, CorreMat1

# for dataset wine
def Plot_scatter(): 
    Data = np.column_stack((X1, Y1)) # combine X and Y 
    pos = np.where(Data[:,-1] == 1)[0]# divide data into two groups  
    neg = np.where(Data[:,-1] == 0)[0]
    X_pos = Data[pos,0:-1]
    X_neg = Data[neg,0:-1]
    Y_pos = Data[pos,-1]
    Y_neg = Data[neg,-1]
    print(type(Y_neg))

    plt.figure(1)
    plt.subplots(figsize = (15,14))
    ax = plt.subplot(431)
    ax.scatter(X_pos[:,0], X_pos[:,1], s=40, c='tab:pink',alpha = 0.6, marker='s',label = 'positive')
    ax.scatter(X_neg[:,0], X_neg[:,1], s=40, c='tab:blue', alpha = 0.6,cmap='Set1',label = 'negtive')  
    ax.legend(loc="upper right", title="Classes")
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)  
    
    ax = plt.subplot(432) 
    ax.scatter(X_pos[:,2], X_pos[:,3], s=40, c='tab:pink',alpha = 0.6, marker='s',label = 'positive')
    ax.scatter(X_neg[:,2], X_neg[:,3], s=40, c='tab:blue', alpha = 0.6,cmap='Set1',label = 'negtive')  
    ax.legend(loc="upper right", title="Classes")
    ax.set_xlabel('$x_3$', fontsize=12)
    ax.set_ylabel('$x_4$', fontsize=12)
    
    ax = plt.subplot(433) 
    ax.scatter(X_pos[:,4], X_pos[:,5], s=40, c='tab:pink',alpha = 0.6, marker='s',label = 'positive')
    ax.scatter(X_neg[:,4], X_neg[:,5], s=40, c='tab:blue', alpha = 0.6,cmap='Set1',label = 'negtive')  
    ax.legend(loc="upper right", title="Classes")
    ax.set_xlabel('$x_5$', fontsize=12)
    ax.set_ylabel('$x_6$', fontsize=12)
    
    ax = plt.subplot(434) 
    ax.scatter(X_pos[:,6], X_pos[:,7], s=40, c='tab:pink',alpha = 0.6, marker='s',label = 'positive')
    ax.scatter(X_neg[:,6], X_neg[:,7], s=40, c='tab:blue', alpha = 0.6,cmap='Set1',label = 'negtive')  
    ax.legend(loc="upper right", title="Classes")
    ax.set_xlabel('$x_7$', fontsize=12)
    ax.set_ylabel('$x_8$', fontsize=12) 
    
    ax = plt.subplot(435) 
    ax.scatter(X_pos[:,8], X_pos[:,9], s=40, c ='tab:pink',alpha = 0.6, marker='s',label = 'positive')
    ax.scatter(X_neg[:,8], X_neg[:,9], s=40, c ='tab:blue', alpha = 0.6,cmap='Set1',label = 'negtive')  
    ax.legend(loc="upper right", title="Classes")
    ax.set_xlabel('$x_9$', fontsize=12)
    ax.set_ylabel('$x_{10}$', fontsize=12) 
    
    ax = plt.subplot(436) 
    ax.scatter(X_pos[:,9],X_pos[:,10], s=40,  c= 'tab:pink', marker='s', label='positive')
    ax.scatter(X_neg[:,9], X_neg[:,10], s=40, c ='tab:blue', alpha = 0.6,cmap='Set1',label = 'negtive')  
    ax.legend(loc="upper right", title="Classes")
    ax.set_xlabel('$x_{10}$', fontsize=12)
    ax.set_ylabel('$x_{11}$', fontsize=12) 
    plt.show()
    
    plt.figure(2)
    
    # Thermal map
    plt.imshow(CorreMat, origin='low', cmap='jet')

    plt.colorbar()
    plt.show()
    
# dataset wine
def plot_data1(X,Y):
    Data = np.column_stack((X, Y)) # combine X and Y 
    pos = np.where(Data[:,-1] == 1)[0]# divide data into two groups  
    neg = np.where(Data[:,-1] == 0)[0]
    X_pos = Data[pos,0:-1]
    X_neg = Data[neg,0:-1]
    n,m = np.shape(X_pos)
    
    plt.figure(1)
    plt.subplots(figsize = (15,15))   
    for i in range(m):
        ax = plt.subplot(4,3,i+1)
        ax.scatter(X_pos[:,i],Data[pos,-1], s=40, c='tab:blue',alpha = 0.6, marker='s',label = 'positive')
        ax.scatter(X_neg[:,i],Data[neg,-1], s=40, c='tab:pink',alpha = 0.6,label = 'negative')
        plt.legend()
        label = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
        ax.set_xlabel(label[i], fontsize=12)
    plt.show()
       
    plt.figure(2)
    plt.boxplot(X_pos)
    plt.show()
    
    plt.figure(3)
    plt.subplots(figsize = (15,15))   
    for i in range(m):
        ax = plt.subplot(4,3,i+1)
        classes = ['positive','negative']
        plt.hist([X_pos[:,i],X_neg[:,i]], density=True,  histtype='bar',label = classes)
        label = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
        xlabel = label[i]
        #xlabel = '%s%s'%('Feature ',{i+1})          
        ax.set_xlabel(xlabel, fontsize=12)
        plt.legend()  
    #plt.savefig('D:\ML\Project/Hist1.eps')
    plt.show()   
    return None

def plot_data2(X,Y):
    Data = np.column_stack((X, Y)) # combine X and Y 
    pos = np.where(Data[:,-1] == 1)[0]# divide data into two groups  
    neg = np.where(Data[:,-1] == 0)[0]
    X_pos = Data[pos,0:-1]
    X_neg = Data[neg,0:-1]
    n,m = np.shape(X_pos)
    
    plt.figure(1)
    plt.subplots(figsize = (15,15))   
    for i in range(m):
        ax = plt.subplot(4,3,i+1)
        ax.scatter(X_pos[:,i],Data[pos,-1], s=40, c='tab:blue',alpha = 0.6, marker='s',label = 'positive')
        ax.scatter(X_neg[:,i],Data[neg,-1], s=40, c='tab:pink',alpha = 0.6,label = 'negative')
        plt.legend()
        label = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
       # xlabel = label[i]
       #xlabel = '%s%s'%('Feature ',{i+1})     
        ax.set_xlabel(label[i], fontsize=12)
    plt.show()
       
    plt.figure(2)
    plt.boxplot(X_pos)
    plt.show()
    
    plt.figure(3)
    plt.subplots(figsize = (15,15))   
    for i in range(m):
        ax = plt.subplot(4,3,i+1)
        classes = ['positive','negative']
        plt.hist([X_pos[:,i],X_neg[:,i]], density=True,  histtype='bar',label = classes)
        label = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
        xlabel = label[i]
        #xlabel = '%s%s'%('Feature ',{i+1})          
        ax.set_xlabel(xlabel, fontsize=12)
        plt.legend()  
    #plt.savefig('D:\ML\Project/Hist1.eps')
    plt.show()
