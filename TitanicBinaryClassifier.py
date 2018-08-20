# Using Logistic Regression for binary classification of Titanic passengers in categories of Survived (1) or not (0) 
# based on input features e.g. Pclass,Sex,Age,SibSp,Parch,Fare,Cabin and Embarked etc.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from LogisticRegressor import LogisticRegressor


class TitanicBinaryClassifier:

    ########### main method runs the steps of training & prediction ###########
    def __init__(self):
        # LOAD data
        xLabels = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
        yLabel  = 'Survived'
        data = pd.read_csv('input/titanic_train.csv')
        X, y = self.preprocessTitanicData(data, xLabels, yLabel)
        self.plot(X, y, xLabels, yLabel, ['Not Survived', 'Survived'])

        classifier = LogisticRegressor(numOfIterations=100, learningRate=0.3, regularizer=100, scalingNeeded=True, biasNeeded=True, verbose=True)
        # TRAIN the model (i.e. theta here)
        print('\nTRAINING:\n')
        classifier.train(X, y)                                                 # alpha is learning rate for gradient descent
        classifier.saveModel('model/titanic_classifier.model')

        classifier.loadModel('model/titanic_classifier.model')
        # VALIDATE model with training data
        print('\nVAIDATION:\n')
        yPred = classifier.validate(X, y)
        self.writeOutput(X, yPred, 'output/titanic_validation.csv')
        # Plot after training
        self.plot(X, yPred, xLabels, yLabel, ['Not Survived', 'Survived'])

        # PREDICT with trained model using sample data
        print('\nPREDICTION:\n')
        data = pd.read_csv("input/titanic_test.csv")
        X, y = self.preprocessTitanicData(data, xLabels, yLabel, training=False)
        yPred = classifier.predict(X)
        self.plot(X, yPred, xLabels, yLabel, ['Not Survived', 'Survived'])
        #printData(X, yPred, xLabels, yLabel)
        indexField = data['PassengerId'].values.reshape(data.shape[0], 1)
        self.writeOutput(indexField, yPred, 'output/titanic_prediction.csv', colHeaders=['PassengerId', 'Survived'])


    def writeOutput(self, X, y, fileName, delim=',', colHeaders=None):
        if colHeaders is None:
            print(' Headless Write in ', fileName)
            data = np.hstack([X, y])
            np.savetxt(fileName, data, fmt='%.d', delimiter=delim)
        else:
            self.printData(X, y, colHeaders[0:len(colHeaders)-1], colHeaders[len(colHeaders)-1], delim=',', fileName=fileName)
        print('Output written to ', fileName)
        return

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

    # Plotting dataset
    def plot(self, X, y, xLabels, yLabel, classLabels):
        plt.figure(figsize=(15,4), dpi=100)
        y = y.ravel()
        rows, cols = X.shape
        if cols != len(xLabels):
            return
        for c in range(0, cols-1):
            plt.subplot(1, cols, c+1)
            plt.scatter(X[y == 0][:, c], X[y == 0][:, c+1], color='b', label=classLabels[0])
            plt.scatter(X[y == 1][:, c], X[y == 1][:, c+1], color='r', label=classLabels[1])
            plt.xlabel(xLabels[c])
            plt.ylabel(xLabels[c+1])
        plt.legend()
        plt.show()

    def preprocessTitanicData(self, data, xLabels, yLabel=None, training=True):
        y = None
        if training:
            y = data[yLabel].values
            y = y.reshape(len(y), 1)
            y = y.astype('int64')
        data = data[xLabels]
        data['Sex'] = data['Sex'].map({'male':1, 'female':0})
        data['Embarked'] = data['Embarked'].map({'C':1, 'Q':2, 'S':3})
        meanAge = np.mean(data['Age'])
        data['Age'] = data['Age'].fillna(meanAge)
        X = data.values.astype('int64')
        return X, y


if True:
    TitanicBinaryClassifier()
