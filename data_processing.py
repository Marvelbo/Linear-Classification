import numpy as np
import pandas as pd
import time


########################################################################
####      Task 1: Acquire, preprocess, and analyze the data        #####
########################################################################
def LoadData1(filename):     # read red wine data
    data_df = pd.read_csv(filename, header=0, sep=';')
    Data = data_df.values
    Data = shuffle_data(Data)    # data shuffle
    X = np.array(Data[:, 0:-1])
    Y = Data[:, -1]
    Y = np.asarray([1 if i > 5 else 0 for i in Y])     # if yi = 0-5 -> class 0 // otherwise if yi = 6-10 -> class 1
    Y = Y.transpose()
    return X, Y


def LoadData2(filename):   # read breast cancer data
    with open(filename) as input_file:
        lines = input_file.readlines()
        data_clean = []

        for line in lines:
            newline = line.strip().split(',')
            temp = newline[1:]
            # remove missing data points
            for j in range(len(temp)):
                if temp[j] == '?':
                    temp[j] = -1
                else:
                    temp[j] = int(temp[j])
                    data_clean.append(temp)

        data_clean = np.array(data_clean)
        data_clean = shuffle_data(data_clean)   # data shuffle
        print(type(data_clean))
        data_x = data_clean[:, 0: -1]
        data_y = data_clean[:, -1]
        data_y = [1 if i == 2 else 0 for i in data_y]    # '2' -> class 1 and  '4' -> class 0
    return data_x, data_y

# add a column of 1 (X0) to the feature matrix for the logistic regression model
def dummy_feature(X):
    n1, m1 = np.shape(X)
    X0 = np.ones((n1, 1))
    X_new = np.column_stack((X0, X))
    return X_new

# add intersecion terms or square terms into feature matrix
def add_inters(X, col1, col2):
    n, m = np.shape(X)
    added = np.ones((n, 1))
    for i in range(n):
        added[i] = X[i, col1]*X[i, col2]
    Xfea_add = np.column_stack((X, added))
    return Xfea_add

# we may want to remove some features
def remove(X, i):
    X = np.delete(X, i, axis = 1 )
    return X

# split training set, validation set and test set
def split_train_test(X,Y):
    Num = len(Y)
    train_x, test_x = X[0:int(0.9*Num), :], X[int(0.9*Num): -1, :]
    train_y, test_y = Y[0:int(0.9*Num)], Y[int(0.9*Num): -1]
    return train_x,train_y,test_x,test_y

# shuffle the data for generalization
def shuffle_data(Data) :
    number_of_rows = Data.shape[0]
    index = list(np.arange(number_of_rows))
    np.random.shuffle(index)
    # print('\nShuffled row index:')
    # print (index)
    # Apply index
    Data = Data[index,:]
    # print('\nArray with shuffled rows:')
    return Data
