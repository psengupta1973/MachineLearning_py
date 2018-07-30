# Given three data sets with square feet area, no. of bedrooms vs house price
# We can run a linear regression with multiple variables to predict house pricess for a given sqft area and # of bedrooms

import numpy as np
import matplotlib.pyplot as plt


class MultivariateLinearRegressor:

    def __init__(self):
        self.theta = None
        self.alpha = 0.0
        self.iterCount = 100
        self.featMean = None
        self.featStd = None
        self.totalReadCount = 0


    # Preprocessing: add a column (with all ones) to X and adjust values in X (feature scaling)
    def preprocess(self, X):
        m, n = X.shape
        X = np.concatenate([np.ones((m, 1)), X], axis=1)                         # add a column as bias in X at index 0 with all ones
        p = X.shape[1]
        if (self.featMean is not None) & (self.featStd is not None):             # During prediction phase
            self.totalReadCount += m
            for i in range(1, n+1):                                              # for each feature column
                X[:,i]   = ((X[:,i] - self.featMean[i]) / self.featStd[i]).reshape(m)      
        else:                                                                    # First time during training
            self.featMean = np.zeros(p)
            self.featStd  = np.zeros(p)
            self.totalReadCount = m
            for i in range(1, n+1):
                self.featMean[i] = np.mean(X[:,i])
                self.featStd[i]  = np.std(X[:,i])
                X[:,i]   = ((X[:,i] - self.featMean[i]) / self.featStd[i]).reshape(m)
        return X


    # the cost method is the cost function, it calculates the mean squared errors from entire dataset
    def train(self, X, Y, theta=None, alpha=0.3, iterCount=100):
        if theta is None:
            if self.theta is None:
                self.theta = np.random.rand(1, X.shape[1])
        else:
            self.theta = theta
        self.alpha = alpha
        self.interCount = iterCount
        self.theta, costPath = self.gradientDescent(X, Y)
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


    # gradientDescent method adjusts theta parameters and returns a minimized theta
    def gradientDescent(self, X, Y):
        costPath = np.array([])
        m = X.shape[0]
        thetaColCount = self.theta.shape[1]
        for c in range(0, self.iterCount):
            hx = X.dot(self.theta.transpose())
            temp = self.theta
            for i in range(0, thetaColCount):
                if i < thetaColCount-1:
                    temp[0,i] = self.theta[0,i] - (self.alpha * (1.0/m) * ((hx - Y) * X[:, i:i+1]).sum(axis=0))
                else:
                    temp[0,i] = self.theta[0,i] - (self.alpha * (1.0/m) * ((hx - Y) * X[:, -1:]).sum(axis=0))
            self.theta = temp
            costPath = np.append(costPath, self.getCost(X, Y, None))
        return self.theta, costPath


    # predict method predicts a y values for a given x value and rectified theta parameters
    def predict(self, X):
        predicted_Y = X.dot(self.theta.transpose())
        return predicted_Y



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
    header = ''
    for c in range(0, n-1):
        header += xLabels[c] + '    '
    header += yLabel
    print(header)
    for r in range(0, m):
        bodyLine = ''
        for c in range(0, n-1):
            bodyLine += str(X[r, c]) + '        '
        bodyLine += str(Y[r,0])
        print (bodyLine)


########### main method runs the steps of linear regression in sequence ###########
def main():
    iterCount = 100
    alpha = 0.3          # learning rate
    xLabels = ['Sqft Area','Bedrooms','Age in years']
    yLabel  = 'Price (Y)'
    mlr = MultivariateLinearRegressor()

    # LOAD House prices in Y while area, rooms and age in X
    X, Y = readData("input/area_rooms_age_prices.csv")
    plot(X, Y, xLabels, yLabel)
    X = mlr.preprocess(X)

    # Check initial errors before training
    print('\nTRAINING:\n')
    cost = mlr.getCost(X, Y, None)
    print("\nInitial Cost: ", cost)

    # TRAIN the model (i.e. theta here)
    theta, costPath = mlr.train(X, Y, None, alpha, iterCount)
    plt.plot(np.linspace(1, iterCount, iterCount, endpoint=True), costPath)
    plt.title("Iteration vs Cost ")
    plt.xlabel("# of iterations")
    plt.ylabel("theta")
    plt.show()

    cost = mlr.getCost(X, Y, theta)
    print("After ", iterCount, " iterations, \nCost: ", cost, " Theta: ", theta, "\n")

    # Setting up a series of values for areas and bedrooms to PREDICT corresponding prices
    print('\nPREDICTION:\n')
    areas = np.arange(1950.00, 1000.00, -100.00)[0:5].reshape(5,1)
    bedrooms = np.arange(6, 1, -1)[0:5].reshape(5,1)
    years = np.arange(7, 1, -1)[0:5].reshape(5,1)
    Xraw = np.hstack([areas, bedrooms, years])
    Xproc = mlr.preprocess(Xraw)
    Y = mlr.predict(Xproc)
    printTable(Xraw, Y, xLabels, yLabel)
    plot(Xraw, Y, xLabels, yLabel)


if True:
    main()
