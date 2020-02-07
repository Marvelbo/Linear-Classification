#########################################################
###############LDA Classification########################
#########################################################

class LDA_classification(object):
    def __init__(self):
        self.w0 = None
        self.weights = None

    def fit_lda(self, train_x, train_y):
        N, M = train_x.shape    # N -> num of points , M -> num of features

        N1 = np.sum(train_y)    # P(y = 0) & P(y = 1)
        P1 = N1 / N
        P0 = 1 - P1

        # mu 0 & mu 1 & covariance
        sum_u0, sum_u1 = np.zeros(M).astype(int), np.zeros(M).astype(int)
        for i in range(N):
            if train_y[i] == 0:
                sum_u0 = np.add(sum_u0, train_x[i, :])
            else:
                sum_u1 = np.add(sum_u1, train_x[i, :])
        u0 = sum_u0 / (N-N1)
        u1 = sum_u1 / N1

        # sigma
        sigma = np.zeros((M, M))
        for i in range(N):
            if train_y[i] == 0:
                diff0 = train_x[i, :] - u0
                sigma = np.add(sigma, np.outer(diff0, diff0))
            else:
                diff1 = train_x[i, :] - u1
                sigma = np.add(sigma, np.outer(diff1, diff1))
        sigma = sigma / (N - 2)
        sigma = sigma / (N - 2)

        # calculate
        sigma_inv = np.linalg.inv(sigma)
        self.w0 = np.log(P1 / P0) - 0.5 * np.dot(u1, np.dot(sigma_inv, u1)) + 0.5 * np.dot(u0, np.dot(sigma_inv,u0))
                                                                                                       # possible err 2
        self.weights = np.dot(sigma_inv, (u1 - u0))

    def predict_lda(self, test_x):
        log_odds = self.w0 + np.dot(test_x, self.weights)
        predicty = np.array([1 if i > 0 else 0 for i in log_odds])
        return predicty

    def evaluate_acc(self, test_x, test_y):

        predict_y = self.predict_lda(test_x)  # predict
        accuracy = (sum(1 for i in range(len(predict_y)) if predict_y[i] == test_y[i])) * 100 / len(predict_y)
        # print(accuracy)
        return accuracy

    def testing(self, x_testing, y_testing):
        accuracy = self.evaluate_acc(x_testing, y_testing)
        return accuracy


#############################################################################
##############Logistic Regression Model######################################
#############################################################################

class Log_Model(object):
    def __init__(self):
        self.Weights = None  # feature weights
        self.alpha = None  # Stepsize

    def sigmoid(self, a):
        sigmoid_a = 1.0 / (1.0 + np.exp(-a))
        return sigmoid_a

    def fit_log(self, train_x, train_y):
        N, M = np.shape(train_x)  # N -> num of points , M -> num of features
        # w_update = self.Weights
        w_update = np.mat(np.zeros((M, 1)))
        MSE = 100
        i = 0
        train_y = np.mat(train_y).T  # convert to matrix

        while MSE > 1e-4:
            self.alpha = 1/ (1 + i) + 0.00001
            h = self.sigmoid(np.dot(train_x, w_update))
            w = w_update
            loss = train_y - h
            w_update = w_update + self.alpha * np.dot(np.transpose(train_x), loss)  # self.Weights to be a m*1 matrix
            MSE = np.dot((w_update - w).T, (w_update - w)) / M
            # print(MSE)
            # h = Log_Model.sigmoid(np.dot(train_x, w_update))
            i = i + 1
        #     print(MSE)
        # print(w_update)
        self.Weights = w_update
        return self.Weights



    def predict_log(self, test_x):
        log_odds = np.dot(test_x, self.Weights)
        log_function = self.sigmoid(log_odds)
        predicty = np.array([1 if i > 0.5 else 0 for i in log_function])
        return predicty

    def evaluate_acc(self, test_x, test_y):
        predict_y = self.predict_log(test_x)  # predict
        accuracy = (sum(1 for i in range(len(predict_y)) if predict_y[i] == test_y[i])) * 100 / len(predict_y)
        # print(accuracy)
        return accuracy

    def testing(self, x_testing, y_testing):
        accuracy = self.evaluate_acc(x_testing, y_testing)
        return accuracy