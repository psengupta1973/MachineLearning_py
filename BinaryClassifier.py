# Using Logistic Regression for binary classification of houses in categories like Standard (0) & Premium (1) 
# based on input features e.g. sqft area, bedrooms, age in years and price

import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt


class BinaryClassifier:

    def __init__(self, numOfIterations=100, learningRate=0.3, biasNeeded=False, scalingNeeded=False, verbose=False):
        self.epoch = numOfIterations
        self.theta = None
        self.biasNeeded = biasNeeded
        self.alpha = learningRate
        self.scalingNeeded = scalingNeeded
        self.verbose = verbose

    # add a column (with all ones) as the BIAS to X (at 0th index)
    def __addBias(self, X):
        m = X.shape[0]
        return np.concatenate([np.ones((m, 1)), X], axis=1)                            # add a column as bias in X at index 0 with all ones

    # the COST method is the cost function, it calculates the mean squared errors from entire dataset
    def __cost(self, X, y):
        rows = X.shape[0]
        if self.theta is None:
            self.theta = np.random.rand(1, X.shape[1])
        np.seterr(over='raise')
        hx   = np.zeros(rows)
        loss = np.zeros(rows)
        cost = 0.0
        for r in range(0, rows):
            hx[r] = self.__hypothesis(X[r])
            loss[r] = self.__loss(hx[r], y[r, 0])
            cost += loss[r]
        return cost

    # HYPOTHESIS function to calculate hx for a row
    def __hypothesis(self, Xrow):
        cols = len(Xrow)
        z, hx = 0.0, 0.0
        for c in range(0, cols):
            z += (self.theta[0, c] * Xrow[c])
        hx = self.__sigmoid(z)
        return hx

    # LOSS function
    def __loss(self, hx, y):
        np.seterr(divide='ignore')
        np.seterr(invalid='ignore')
        loss = (-y * np.log(hx) - (1-y) * np.log(1-hx)).mean()
        return loss

    # GRADIENTDESCENT (loops) method adjusts theta parameters and returns a minimized theta
    def __gradientDescent(self, X, y):
        costPath = self.__cost(X, y)
        rows, cols = X.shape
        hx, lossFactor = 0.0, 0.0
        for i in range(0, self.epoch):
            for c in range(0, cols):
                for r in range(0, rows):
                    hx = self.__hypothesis(X[r])
                    lossFactor += (hx - y[r,0]) * X[r,c]
                derivative = lossFactor / rows
                self.theta[0, c] -= (self.alpha * derivative)
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

    # Logistic / SIGMOID function
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

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
        costPath = costPath[~np.isnan(costPath)]                                                 # remove nan values if any
        costSteps = len(costPath)
        if (costSteps > 1) & self.verbose == True:
            # VISUALIZE improvement of model after training
            print('Training iterations: ', self.epoch, ' \nCost minization: ', costPath[0],', --> ', np.min(costPath), ' \nTheta: ', self.theta, '\n')
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



# readData method loads all columns from left to right (except the last) in X and the last column in y
def readInput(fileName, delim=','):
    data = np.genfromtxt(fileName, delimiter=delim)
    n = data.shape[1]
    X = data[:, 0:n-1]
    y = data[:,-1:]
    return X, y

def writeOutput(X, y, fileName, delim=','):
    data = np.hstack([X, y])
    np.savetxt(fileName, data, fmt='%.d', delimiter=delim)
    return

def plot(X, y, xLabels, yLabel):
    # Plotting example dataset
    plt.figure(figsize=(15,4), dpi=100)
    rows, cols = X.shape
    if cols != len(xLabels):
        return
    for c in range(0, cols):
        plt.subplot(1, cols, c+1)
        plt.scatter(X[:,c], y)
        plt.plot(X[:,c], y)
        plt.xlabel(xLabels[c])
        plt.ylabel(yLabel)
    plt.show()

# Print house prices with specific number of columns
def printData(X, y, xLabels, yLabel):
    rows, cols = X.shape
    if (rows != y.shape[0]) :
        return
    header = '\n'
    for c in range(0, cols):
        header += xLabels[c] + '    '
    header += yLabel
    print(header)
    for r in range(0, rows):
        bodyLine = ''
        for c in range(0, cols):
            bodyLine += str(X[r, c]) + '        '
        bodyLine += str(y[r,0])
        print (bodyLine)

# Sample data for Prediction
def sampleData4Prediction():
    areas    = np.arange(1950.00, 1000.00, -100.00)[0:5].reshape(5,1)
    bedrooms = np.arange(6, 1, -1)[0:5].reshape(5,1)
    years    = np.arange(7, 1, -1)[0:5].reshape(5,1)
    prices   = np.arange(280000, 220000, -10000)[0:5].reshape(5,1)
    X = np.hstack([areas, bedrooms, years, prices])
    return X



########### main method runs the steps of training & prediction ###########
def main():

    # LOAD house prices in y while area, rooms and age in X
    X, y = readInput("input/area_rooms_age_categories.csv")
    xLabels = ['Area(sqft)','Bedrooms','Age(years)', 'Prices']
    yLabel  = 'Categories (y)'
    #plot(X, y, xLabels, yLabel)
    plt.scatter(X[:,2],X[:,3], label='Training data')
    plt.legend()
    plt.show()

    classifier = BinaryClassifier(numOfIterations=200, learningRate=0.3, scalingNeeded=True, biasNeeded=True, verbose=True)
    # TRAIN the model (i.e. theta here)
    print('\nTRAINING:\n')
    classifier.train(X, y)                                                 # alpha is learning rate for gradient descent
    classifier.saveModel('model/bin_classification.model')

    classifier.loadModel('model/bin_classification.model')
    # VALIDATE model with training data
    print('\nVAIDATION:\n')
    yPred = classifier.validate(X, y)
    printData(X, yPred, xLabels, yLabel)
    #plot(X, yPred, xLabels, yLabel)
    writeOutput(X, yPred, 'output/house_categories_validation.csv')
    
    # Plot after training
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y.ravel() == 0][:, 2], X[y.ravel() == 0][:, 3], color='b', label='Standard')
    plt.scatter(X[y.ravel() == 1][:, 2], X[y.ravel() == 1][:, 3], color='r', label='Premium')
    plt.legend()
    plt.show()

    # PREDICT with trained model using sample data
    print('\nPREDICTION:\n')
    X = sampleData4Prediction()
    yPred = classifier.predict(X)
    printData(X, yPred, xLabels, yLabel)
    #plot(X, yPred, xLabels, yLabel)
    writeOutput(X, yPred, 'output/house_categories_prediction.csv')

if True:
    main()
