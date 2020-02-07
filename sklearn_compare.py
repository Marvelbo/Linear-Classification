import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
# author: Ningbo Zhu
########################################################################
####      Task 1: Acquire, preprocess, and analyze the data        #####
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
        data_x = data_clean[:, 0: -1]
        data_y = data_clean[:, -1]
        data_y = np.asarray([1 if i == 2 else 0 for i in data_y ] )  # '2' -> class 1 and  '4' -> class 0
    return data_x, data_y

# add a column of 1 (X0) to the feature matrix for the logistic regression model
def dummy_feature(X):
    n1, m1 = np.shape(X)
    X0 = np.ones((n1, 1))
    X_new = np.column_stack((X0, X))
    return X_new

def split_train_test(X,Y):
    Num = len(Y)
    train_x, test_x = X[0:int(0.9*Num), :], X[int(0.9*Num): -1, :]
    train_y, test_y = Y[0:int(0.9*Num)], Y[int(0.9*Num): -1]
    return train_x,train_y,test_x,test_y

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


def sk_learn_test(X, Y):

    fold = 5
    kf = KFold(n_splits=fold, random_state=None, shuffle=True)
    accuracy_LR = np.zeros(fold)    # output a value each fold
    accuracy_LDA = np.zeros(fold)
    k = 0

# training and validation set
    for train_index, test_index in kf.split(X):
        trainx,valx = X[train_index, :], X[test_index, :]  # corresponding rows
        trainy,valy = Y[train_index], Y[test_index]


    # run logistic regression
        lg = LogisticRegression(random_state=0, max_iter=1000, penalty='l2', solver='liblinear', C=1).fit(trainx, np.ravel(trainy))
        accuracy_LR[k] = lg.score(valx, valy) * 100


    # run LDA
        clf = LinearDiscriminantAnalysis()
        clf.fit(trainx, trainy)
        accuracy_LDA[k] = clf.score(valx,valy)*100

        k += 1

    print('Logistic Regression')
    print(np.array(accuracy_LR))
    print('average = ', sum(accuracy_LR/fold))
    print('Linear Discriminant Analysis')
    print(np.array(accuracy_LDA))
    print('average = ', sum(accuracy_LDA / fold))

# red-wine data
X1, Y1 = LoadData1('winequality-red.csv')
X1 , Y1, x1_test, y1_test = split_train_test(X1,Y1)
# # breast data
X2, Y2 = LoadData2('breast-cancer-wisconsin.data')
X2 , Y2, x2_test, y2_test = split_train_test(X2,Y2)
print('red-wine data')
# sk_learn_test(X1, Y1)
print('brest-cancer data')
sk_learn_test(X2, Y2)