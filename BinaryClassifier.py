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
        #plot(X, y, xLabels, yLabel)
        plt.scatter(X[:,2],X[:,3], label='Training data')
        plt.legend()
        plt.show()

        classifier = LogisticRegressor(numOfIterations=100, learningRate=0.3, scalingNeeded=True, biasNeeded=True, verbose=True)
        # TRAIN the model (i.e. theta here)
        print('\nTRAINING:\n')
        classifier.train(X, y)                                                 # alpha is learning rate for gradient descent
        classifier.saveModel('model/bin_classification.model')

        classifier.loadModel('model/bin_classification.model')
        # VALIDATE model with training data
        print('\nVAIDATION:\n')
        yPred = classifier.validate(X, y)
        self.printData(X, yPred, xLabels, yLabel)
        #plot(X, yPred, xLabels, yLabel)
        self.writeOutput(X, yPred, 'output/house_categories_validation.csv')
        
        # Plot after training
        plt.figure(figsize=(10, 6))
        plt.scatter(X[y.ravel() == 0][:, 2], X[y.ravel() == 0][:, 3], color='b', label='Standard')
        plt.scatter(X[y.ravel() == 1][:, 2], X[y.ravel() == 1][:, 3], color='r', label='Premium')
        plt.legend()
        plt.show()

        # PREDICT with trained model using sample data
        print('\nPREDICTION:\n')
        X = self.sampleData4Prediction()
        yPred = classifier.predict(X)
        self.printData(X, yPred, xLabels, yLabel)
        #plot(X, yPred, xLabels, yLabel)
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

    def plot(self, X, y, xLabels, yLabel):
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
    def printData(self, X, y, xLabels, yLabel):
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
    def sampleData4Prediction(self):
        areas    = np.arange(1950.00, 1000.00, -100.00)[0:5].reshape(5,1)
        bedrooms = np.arange(6.0, 1.0, -1.0)[0:5].reshape(5,1)
        years    = np.arange(7.0, 1.0, -1.0)[0:5].reshape(5,1)
        prices   = np.arange(280000.00, 220000.00, -10000.00)[0:5].reshape(5,1)
        X = np.hstack([areas, bedrooms, years, prices])
        return X


if True:
    BinaryClassifier()
