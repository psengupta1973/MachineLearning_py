# Given three data sets with square feet area, no. of bedrooms and age in years vs house prices
# We can run a linear regression with multiple variables to predict house pricess for a given sqft area and # of bedrooms

import sys
import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt


class MultivariateLinearRegressor:

    def __init__(self):
        self.theta = None
        self.featMean = None
        self.featStd = None

    # gradientDescent method adjusts theta parameters and returns a minimized theta
    def __gradientDescent(self, X, Y, alpha, iterCount):
        costPath = np.array([])
        m = X.shape[0]
        thetaCols = self.theta.shape[1]
        for c in range(0, iterCount):
            hx = X.dot(self.theta.transpose())
            temp = self.theta
            for i in range(0, thetaCols):
                if i < thetaCols-1:
                    temp[0,i] = self.theta[0,i] - (alpha * (1.0/m) * ((hx - Y) * X[:, i:i+1]).sum(axis=0))
                else:
                    temp[0,i] = self.theta[0,i] - (alpha * (1.0/m) * ((hx - Y) * X[:, -1:]).sum(axis=0))
            self.theta = temp
            costPath = np.append(costPath, self.getCost(X, Y, None))
        return self.theta, costPath

    # normalEquation method adjusts theta parameters and returns a minimized theta
    def __normalEquation(self, X, Y):
        costPath = np.array([])
        costPath = np.append(costPath, self.getCost(X, Y, None))
        self.theta = np.dot(alg.pinv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X),Y))
        self.theta = self.theta.reshape(1, self.theta.shape[0])
        costPath = np.append(costPath, self.getCost(X, Y, None))
        return self.theta, costPath

    # Preprocessing: add a column (with all ones) to X and adjust values in X (feature scaling)
    def preprocess(self, X):
        m, n = X.shape
        X = np.concatenate([np.ones((m, 1)), X], axis=1)                            # add a column as bias in X at index 0 with all ones
        p = X.shape[1]
        if (self.featMean is not None) & (self.featStd is not None):                # During prediction phase
            for i in range(1, n+1):                                                 # for each feature column
                X[:,i]   = ((X[:,i] - self.featMean[i]) / self.featStd[i]).reshape(m)      
        else:                                                                       # First time during training
            self.featMean = np.zeros(p)
            self.featStd  = np.zeros(p)
            for i in range(1, n+1):
                self.featMean[i] = np.mean(X[:,i])
                self.featStd[i]  = np.std(X[:,i])
                X[:,i]   = ((X[:,i] - self.featMean[i]) / self.featStd[i]).reshape(m)
        return X

    # the cost method is the cost function, it calculates the mean squared errors from entire dataset
    def train(self, X, Y, theta=None, alpha=0.3, iterCount=100, minMethod='sgd'):
        if theta is None:
            if self.theta is None:
                self.theta = np.random.rand(1, X.shape[1])
        else:
            self.theta = theta
        if minMethod == 'sgd':
            self.theta, costPath = self.__gradientDescent(X, Y, alpha, iterCount)
        elif minMethod == 'equ':
            self.theta, costPath = self.__normalEquation(X, Y)
        else:
            raise ValueError("Invalid minimization method. Allowed values are 'sgd' or 'equ'.")
        return self.theta, costPath

    # the cost method is the cost function, it calculates the mean squared errors from entire dataset
    def getCost(self, X, Y, theta=None):
        if theta is None:
            if self.theta is None:
                self.theta = np.random.rand(1, X.shape[1])
        else:
            self.theta = theta
        np.seterr(over='raise')
        hx = X.dot(self.theta.transpose())
        totalCost = ( 1.0/(2.0 * X.shape[0])) * (np.square(hx - Y).sum(axis=0))
        return totalCost

    # predict method predicts a y values for a given x value and rectified theta parameters
    def predict(self, X):
        yPred = X.dot(self.theta.transpose())
        yPred = yPred.round().reshape(X.shape[0], 1)
        return yPred

    # validate method measures accuracy of model by predicting with training data 
    def validate(self, X, y):
        yPred = self.predict(X)
        yPred = yPred.round().reshape(X.shape[0], 1)
        print('\nMean Difference (y - yPred): ',(np.mean(y)-np.mean(yPred)))
        return yPred



# readData method loads all columns from left to right (except the last) in X and the last column in Y
def readData(fileName, delim=','):
    data = np.genfromtxt(fileName, delimiter=delim)
    n = data.shape[1]
    X = data[:, 0:n-1]
    Y = data[:,-1:]
    return X, Y

def plot(X, Y, xLabels, yLabel):
    # Plotting example dataset
    plt.figure(figsize=(15,4), dpi=100)
    m, n = X.shape
    if n != len(xLabels):
        return
    for i in range(0, n):
        plt.subplot(1, n, i+1)
        plt.scatter(X[:,i], Y)
        plt.xlabel(xLabels[i])
        plt.ylabel(yLabel)
    plt.show()

# Print house prices with specific number of columns
def printTable(X, Y, xLabels, yLabel):
    m, n = X.shape
    if (m != Y.shape[0]) :
        return
    header = '\n'
    for c in range(0, n):
        header += xLabels[c] + '    '
    header += yLabel
    print(header)
    for r in range(0, m):
        bodyLine = ''
        for c in range(0, n):
            bodyLine += str(X[r, c]) + '        '
        bodyLine += str(Y[r,0])
        print (bodyLine)

# Setting up a series of values for areas and bedrooms to PREDICT corresponding prices
def sampleData4Prediction():
    areas = np.arange(1950.00, 1000.00, -100.00)[0:5].reshape(5,1)
    bedrooms = np.arange(6, 1, -1)[0:5].reshape(5,1)
    years = np.arange(7, 1, -1)[0:5].reshape(5,1)
    X = np.hstack([areas, bedrooms, years])
    return X


########### main method runs the steps of linear regression in sequence ###########
def main():
    method = 'equ'
    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'sgd'):
            method = 'sgd'

    # LOAD house prices in Y while area, rooms and age in X
    X, Y = readData("input/area_rooms_age_prices.csv")
    xLabels = ['Sqft Area','Bedrooms','Age in years']
    yLabel  = 'Price (Y)'
    plot(X, Y, xLabels, yLabel)

    mlr = MultivariateLinearRegressor()
    # TRAIN the model (i.e. theta here)
    print('\nTRAINING:\n')
    iterCount = 100
    Xproc = mlr.preprocess(X)
    theta, costPath = mlr.train(Xproc, Y, None, alpha=0.3, iterCount=iterCount, minMethod=method)      # alpha is learning rate for gradient descent
    costSteps = len(costPath)
    if costSteps > 1:
        print('After ', iterCount, ' iterations, \nCost reduction: ', costPath[0], ' --> ', costPath[costSteps-1],  '\nTheta: ', theta, '\n')
        plt.plot(np.linspace(1, costSteps, costSteps, endpoint=True), costPath)
        plt.title('Iteration vs Cost ')
        plt.xlabel('# of iterations')
        plt.ylabel('theta')
        plt.show()

    # VALIDATE model with training data
    print('\nVAIDATION:\n')
    yPred = mlr.validate(Xproc, Y)
    printTable(X, yPred, xLabels, yLabel)
    plot(X, yPred, xLabels, yLabel)

    # PREDICT using trained model with sample data
    print('\nPREDICTION:\n')
    X = sampleData4Prediction()
    Y = mlr.predict(Xproc)
    printTable(X, Y, xLabels, yLabel)
    plot(X, Y, xLabels, yLabel)


if True:
    main()
