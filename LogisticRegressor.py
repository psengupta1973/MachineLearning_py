# Using Logistic Regression for binary classification

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressor:

    def __init__(self, numOfIterations=100, learningRate=0.3, regularizer=1, biasNeeded=False, scalingNeeded=False, verbose=False):
        self.epoch = numOfIterations
        self.theta = None
        self.biasNeeded = biasNeeded
        self.alpha = learningRate
        self.scalingNeeded = scalingNeeded
        self.verbose = verbose
        self._lambda = regularizer

    # add a column (with all ones) as the BIAS to X (at 0th index)
    def __addBias(self, X):
        m = X.shape[0]
        return np.concatenate([np.ones((m, 1)), X], axis=1)                            # add a column as bias in X at index 0 with all ones

    # the COST method is the cost function, it calculates the mean squared errors from entire dataset
    def __cost(self, X, y):
        np.seterr(over='raise')
        rows = X.shape[0]
        if self.theta is None:
            self.theta = np.random.rand(1, X.shape[1])
        hx   = np.zeros(rows)
        loss = np.zeros(rows)
        cost = 0.0
        for r in range(0, rows):
            hx[r] = self.__hypothesis(X[r])
            loss[r] = self.__loss(hx[r], y[r, 0])
            cost += loss[r]
        cost = (-1/rows) * cost
        cost += self.__regularizeCost(self.theta, rows)                 # regularize theta
        return cost

    # HYPOTHESIS function to calculate hx for a row
    def __hypothesis(self, Xrow):
        cols = len(Xrow)
        z = 0.0
        for c in range(0, cols):
            z += (self.theta[0, c] * Xrow[c])
        hx = self.__sigmoid(z)
        return hx

    # Logistic / SIGMOID function
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # LOSS function
    def __loss(self, hx, y):
        #np.seterr(divide='ignore')
        #np.seterr(invalid='ignore')
        hx += (10**-10)
        _1_hx = (1 - hx)
        _1_hx += (10**-10)                                              # avoid zeros from going to log()
        loss = (y * np.log(hx) + (1-y) * np.log(_1_hx))
        return loss

    # regularize theta to reduce overfitting
    def __regularizeCost(self, theta, rows):
        thetaExcept0 = theta[0, 1:]
        sum = np.sum(np.square(thetaExcept0))
        return (self._lambda/(2*rows)) * sum

    # GRADIENTDESCENT (loops) method adjusts theta parameters and returns a minimized theta
    def __gradientDescent(self, X, y):
        costPath = self.__cost(X, y)
        rows, cols = X.shape
        hx, derivative = 0.0, 0.0
        for i in range(0, self.epoch):
            for c in range(0, cols):
                for r in range(0, rows):
                    hx = self.__hypothesis(X[r])
                    derivative += (hx - y[r,0]) * X[r,c]
                derivative = derivative / rows
                if c > 0:
                    derivative += (self._lambda/rows) * self.theta[0,c]     # regularize except theta-0
                self.theta[0, c] -= self.alpha * derivative
            costPath = np.append(costPath, self.__cost(X, y))
        return self.theta, costPath

    # GRADIENTDESCENT (vectorized) method adjusts theta parameters and returns a minimized theta
    def x__gradientDescent(self, X, y):
        costPath = self.__cost(X, y)
        rows = X.shape[0]
        for c in range(0, self.epoch):
            z = np.dot(X, np.transpose(self.theta))
            hx = self.__sigmoid(z)
            derivative = 1/rows * np.dot(X.T, (hx - y))
            self.theta = self.theta - (self.alpha * derivative)
            costPath = np.append(costPath, self.__cost(X, y))
        return self.theta, costPath

    # Preprocessing: adjust values in X (feature scaling)
    def __scaleFeatues(self, X):
        m, n = X.shape
        featMean = np.zeros(n)
        featStd  = np.zeros(n)
        for i in range(1, n):
            featMean[i] = np.mean(X[:,i])
            featStd[i]  = np.std(X[:,i])
            X[:,i]   = ((X[:,i] - featMean[i]) / featStd[i]).reshape(m)
        return X

    # the TRAIN method goes through the training dataset and train the model and reduce cost
    def train(self, X, y):
        if self.biasNeeded:
            X = self.__addBias(X)
        if self.scalingNeeded:
            X = self.__scaleFeatues(X)
        if self.theta is None:
            self.theta = np.random.rand(1, X.shape[1])
        self.theta, costPath = self.__gradientDescent(X, y)
        #costPath = costPath[~np.isnan(costPath)]                                                 # remove nan values if any
        costSteps = len(costPath)
        if (costSteps > 1) & self.verbose == True:
            # VISUALIZE improvement of model after training
            print('Training iterations: ', self.epoch, ' \nCost minization: ', costPath[0],' --> ', np.min(costPath), ' \nTheta: ', self.theta, '\n')
            plt.plot(np.linspace(1, costSteps, costSteps, endpoint=True), costPath)
            plt.title("Iteration vs Cost ")
            plt.xlabel("# of iterations")
            plt.ylabel("theta")
            plt.show()
        return self.theta

    # PREDICT method predicts a y values (0 - Standard or 1 - Premium) for a given x value & minimizing theta
    def predict(self, X):
        if self.biasNeeded:
            X = self.__addBias(X)
        if self.scalingNeeded:
            X = self.__scaleFeatues(X)
        rows = X.shape[0]
        yPred = np.zeros(rows)
        for r in range(0, rows):
            yPred[r] = self.__hypothesis(X[r])
        yPred = yPred.round()
        yPred = yPred.reshape(X.shape[0],1)
        return yPred

    # VALIDATE method measures accuracy of model by predicting with training data 
    def validate(self, X, y):
        yPred = self.predict(X)
        if self.verbose == True:
            print('Accuracy: ',(len(y[y== yPred])/len(y)) * 100, '%')
        return yPred

    # saveModel method saves the model(i.e. theta) in a file for later use
    def saveModel(self, fileName):
        print('Saving model: ',self.theta, ' ', self.theta.shape)
        np.savetxt(fileName, self.theta, fmt='%.8e', delimiter=',')

    # loadModel method loads the model(i.e. theta) to be reused 
    def loadModel(self, fileName):
        self.theta = np.genfromtxt(fileName, delimiter=',')
        if len(self.theta) > 1:
            self.theta = self.theta.reshape(1, len(self.theta))
            print('Loading model: ',self.theta)

