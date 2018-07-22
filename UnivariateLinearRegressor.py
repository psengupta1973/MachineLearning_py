# Given two data sets with test scores vs hours of study
# We can run a linear regression to predict test score for a given hour and see the graph

import numpy as np
import matplotlib.pyplot as plt

class UnivariateLinearRegressor:

    def __init__(self, learnRate, iterationCount):
        self.learnRate = learnRate
        self.iterCount = iterationCount

    # the learn method goes over the list of data, finds error and runs gradient descent minimize error
    def learn(self, iterCount, learn_rate, x, y):
        theta = np.zeros(2)
        for c in range(0, iterCount):
            y_hat, theta, sumError = self.getMeanSquaredError(x, y, theta)
            theta = self.gradientDescent(x, y, theta, learn_rate)
        return y_hat, theta, sumError

    # the getMeanSquaredError method is the cost function, it calculates the mean squared errors from entire dataset
    def getMeanSquaredError(self, x, y, theta):
        m = len(x)
        y_hat = np.zeros(m)
        sumError = 0
        for i in range(0, m):
            y_hat[i] = theta[0] + theta[1] * x[i]
            sumError += (y_hat[i] - y[i]) **2
        sumError = sumError / (2*m)
        return y_hat, theta, sumError

    # gradientDescent method adjusts theta parameters and returns a rectified theta
    def gradientDescent(self, x, y, theta, l_rate):
        m = len(x)
        y_hat = np.zeros(m)
        for i in range(0, m):
            y_hat[i] = theta[0] + theta[1] * x[i]
            theta[0] = theta[0] - (l_rate * (1/m) * (y_hat[i] - y[i]))
            theta[1] = theta[1] - (l_rate * (1/m) * (y_hat[i] - y[i]) * x[i])
        return theta

    # predict method predicts a y values for a given x value and rectified theta parameters
    def predict(self, x, theta):
        print("Theta used for prediction: ", theta)
        m = len(x)
        y = np.zeros(m)
        for i in range(0, m):
            y[i] = theta[0] + theta[1] * x[i]
        return y

# readData method reads data from file in columns and loads the columns in x and y data arrays
def readData(fileName, delim=','):
    points = np.genfromtxt(fileName, delimiter=delim)
    x = points[:,0]
    y = points[:,1]
    return x, y

# main method runs the steps of linear regression in sequence 
def main():
    iterCount = 100
    learnRate = 0.0001

    ulr = UnivariateLinearRegressor(learnRate, iterCount)

    # test scores are loaded in y while hours of study are loaded in x
    y, x = readData("input/test_score_vs_hour_studied.csv")
    theta = np.zeros(2)

    print("Initial theta: ", theta)
    
    # Check initial errors before training
    y_hat, theta, err = ulr.getMeanSquaredError(x, y, theta)
    print("Initial Error: {0}  theta: [{1}, {2}]".format(err, theta[0], theta[1]))
    
    # Learn from data to train the model (i.e. Theta here)
    y_hat, theta, err = ulr.learn(iterCount, learnRate, x, y)
    print("After {0} iterations, Error: {1}  theta: [{2}, {3}]".format(iterCount, err, theta[0], theta[1]))
    
    # plot initial values of x and y as read from data file
    plt.subplot(1,2,1)
    plt.scatter(x, y)
    plt.xlabel('hours')
    plt.ylabel('scores')

    # plot values of x against predicted values of y after training
    plt.subplot(1,2,2)
    plt.scatter(x, y_hat)
    plt.plot(x, y_hat)
    plt.xlabel('hours')
    plt.ylabel('scores')
    plt.show()

    # Setting up a series of values for hours to predict corresponding scores
    hours = np.arange(100, 20, -12)
    scores = ulr.predict(hours, theta)
    print('Hours: ',hours)
    print('Scores: ',scores)

    plt.scatter(hours, scores)
    plt.plot(hours, scores, 'r--')
    plt.xlabel('Given hours')
    plt.ylabel('Predicted scores')
    plt.show()


if True:
    main()