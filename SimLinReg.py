# Given two data sets with test scores vs hours of study
# We can run a linear regression to predict test score for a given hour and see the graph

import numpy as np
import matplotlib.pyplot as plt

# the checkError method goes over the list of data once and returns a sum of errors
def checkError(x, y, theta):
    N = len(x)
    y_hat = np.zeros(N)
    sumError = 0
    for i in range(0, N):
        y_hat[i] = theta[0] + theta[1] * x[i]
        sumError += (y_hat[i] - y[i]) **2
    return y_hat, theta, sumError

# the tain method goes over the list of data, finds error and runs gradient descent for correction of error
def train(iterCount, learn_rate, x, y):
    theta = np.zeros(2)
    N = len(x)
    y_hat = np.zeros(N)
    for c in range(0, iterCount):
        sumError = 0
        for i in range(0, N):
            y_hat[i] = theta[0] + theta[1] * x[i]
            sumError += (y_hat[i] - y[i]) **2
        theta = gradientDescent(x, y, theta, learn_rate)
    return y_hat, theta, sumError

# gradientDescent method adjusts theta parameters and returns a rectified theta
def gradientDescent(x, y, theta, l_rate):
    M = len(x)
    y_hat = np.zeros(M)
    for i in range(0, M):
        y_hat[i] = theta[0] + theta[1] * x[i]
        theta[0] = theta[0] - (l_rate * (1/M) * (y_hat[i] - y[i]))
        theta[1] = theta[1] - (l_rate * (1/M) * (y_hat[i] - y[i]) * x[i])
    return theta

# predict method predicts a y values for a given x value and rectified theta parameters
def predict(x, theta):
    print("Theta used for prediction: ", theta)
    y = theta[0] + theta[1] * x
    return y

# readData method reads data from file in columns and loads the columns in x and y data arrays
def readData(file_name, delim=','):
    points = np.genfromtxt(file_name, delimiter=delim)
    y = points[:,0]
    x = points[:,1]
    return x, y

# main method runs the steps of linear regression in sequence 
def main():
    #train model on data
    iterCount = 100
    learn_rate = 0.0001

    # test scores are loaded in y while hours of study are loaded in x
    y, x = readData("input/test_score_vs_hour_studied.csv")
    theta = np.zeros(2)

    print("Initial theta: ", theta)
    
    y_hat, theta, err = checkError(x, y, theta)
    print("Initial Error: {0}  theta: [{1}, {2}]".format(err, theta[0], theta[1]))
    
    y_hat, theta, err = train(iterCount, learn_rate, x, y)
    print("Afer {0} iterations, Error: {1}  theta: [{2}, {3}]".format(iterCount, err, theta[0], theta[1]))
    
    # plot actual values of x and y
    plt.subplot(1,2,1)
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')

    # plot actual values of x against predicted values of y
    plt.subplot(1,2,2)
    plt.scatter(x, y_hat)
    plt.plot(x, y_hat)
    plt.xlabel('x')
    plt.ylabel('y_hat')
    plt.show()

    print(predict(53, theta))


if True:
    main()