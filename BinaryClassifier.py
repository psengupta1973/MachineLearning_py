# Using Logistic Regression for binary classification of houses in categories like Standard (0) & Premium (1) 
# based on input features e.g. sqft area, bedrooms, age in years and price

import numpy as np
import matplotlib.pyplot as plt
from LogisticRegressor import LogisticRegressor 

class BinaryClassifier:

    ########### init method runs the steps of training & prediction ###########
    def __init__(self):

        # LOAD house prices in y while area, rooms and age in X
        X, y = self.readInput("input/area_rooms_age_categories.csv")
        xLabels = ['Area(sqft)','Bedrooms','Age(years)', 'Prices']
        yLabel  = 'Categories (y)'
        self.plot(X, y, xLabels, yLabel, ['Standard', 'Premium'])

        classifier = LogisticRegressor(numOfIterations=100, learningRate=0.3, regularizer=1, scalingNeeded=True, biasNeeded=True, verbose=True)
        # TRAIN the model (i.e. theta here)
        print('\nTRAINING:\n')
        classifier.train(X, y)                                                 # alpha is learning rate for gradient descent
        classifier.saveModel('model/bin_classification.model')

        classifier.loadModel('model/bin_classification.model')
        # VALIDATE model with training data
        print('\nVAIDATION:\n')
        yPred = classifier.validate(X, y)
        self.printData(X, yPred, xLabels, yLabel)
        self.plot(X, yPred, xLabels, yLabel, ['Standard', 'Premium'])
        self.writeOutput(X, yPred, 'output/house_categories_validation.csv')

        # PREDICT with trained model using sample data
        print('\nPREDICTION:\n')
        X = self.sampleData4Prediction()
        yPred = classifier.predict(X)
        self.printData(X, yPred, xLabels, yLabel)
        self.plot(X, yPred, xLabels, yLabel, ['Standard', 'Premium'])
        self.writeOutput(X, yPred, 'output/house_categories_prediction.csv')

    # readData method loads all columns from left to right (except the last) in X and the last column in y
    def readInput(self, fileName, delim=','):
        data = np.genfromtxt(fileName, delimiter=delim)
        n = data.shape[1]
        X = data[:, 0:n-1]
        y = data[:,-1:]
        return X, y

    def writeOutput(self, X, y, fileName, delim=','):
        data = np.hstack([X, y])
        np.savetxt(fileName, data, fmt='%.d', delimiter=delim)
        return

    # Plotting dataset
    def plot(self, X, y, xLabels, yLabel, classLabels):
        plt.figure(figsize=(15,4), dpi=100)
        y = y.ravel()
        rows, cols = X.shape
        if cols != len(xLabels):
            return
        for c in range(0, cols):
            plt.subplot(1, cols, c+1)
            Xy0 = X[y == 0][:, c]
            Xy1 = X[y == 1][:, c]
            plt.scatter(range(1, Xy0.shape[0]+1), Xy0, color='b', label=classLabels[0])
            plt.scatter(range(1, Xy1.shape[0]+1), Xy1, color='r', label=classLabels[1])
            plt.xlabel('House #')
            plt.ylabel(xLabels[c])
        plt.legend()
        plt.show()

    # Print house prices with specific number of columns
    def printData(self, X, y, xLabels, yLabel, delim='\t', fileName=None):
        rows, cols = X.shape
        if (rows != y.shape[0]) :
            return
        headLine = ''
        colheads = len(xLabels)
        for c in range(0, colheads):
            headLine += xLabels[c] + delim
        headLine += yLabel +str('\n')
        bodyLine = ''
        for r in range(0, rows):
            for c in range(0, cols):
                bodyLine += str(X[r, c]) + delim
            bodyLine += str(y[r,0])
            bodyLine += str('\n')
        if fileName is None:
            print(headLine)
            print (bodyLine)
        else:
            with open(fileName, "w") as f:
                f.write(headLine)
                f.write(bodyLine)

    # Sample data for Prediction
    def sampleData4Prediction(self):
        areas    = np.arange(1950.00, 1000.00, -100.00)[0:5].reshape(5,1)
        bedrooms = np.arange(6.0, 1.0, -1.0)[0:5].reshape(5,1)
        years    = np.arange(7.0, 1.0, -1.0)[0:5].reshape(5,1)
        prices   = np.arange(280000.00, 220000.00, -10000.00)[0:5].reshape(5,1)
        X = np.hstack([areas, bedrooms, years, prices])
        return X


if True:
    BinaryClassifier()
