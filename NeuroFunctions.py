# Functions for Neural Network

import numpy as np
import math



class ActivationFunction:

    LINEAR, SIGMOID, RELU, LEAKY_RELU, TANH = 'LINEAR', 'SIGMOID', 'RELU', 'LEAKY_RELU', 'TANH'

    @staticmethod
    def isValid(func):
        if func not in [ActivationFunction.SIGMOID, ActivationFunction.RELU, ActivationFunction.LEAKY_RELU, 
                       ActivationFunction.TANH, ActivationFunction.LINEAR]:
            raise ValueError('ERROR: Invalid activation function')
        return True

    @staticmethod
    def activate(z, actFunc, prime=False):
        #if not ActivationFunction.isValid(actFunc):
        #    exit(1)
        a = z
        if actFunc == ActivationFunction.LINEAR:
            a = ActivationFunction.__linear(z, prime=prime)
        if actFunc == ActivationFunction.SIGMOID:
            a = ActivationFunction.__sigmoid(z, prime=prime)
        if actFunc == ActivationFunction.RELU:
            a = ActivationFunction.__relu(z, prime=prime)
        if actFunc == ActivationFunction.LEAKY_RELU:
            a = ActivationFunction.__leakyRelu(z, prime=prime)
        if actFunc == ActivationFunction.TANH:
            a = ActivationFunction.__tanh(z, prime=prime)
        return a

    @staticmethod
    def __linear(z, prime=False):
        gz = 0.0
        if prime:
            gz = 1
        else:
            gz = z
        return gz

    @staticmethod
    def __sigmoid(z, prime=False):
        gz = 1.0 / (1.0 + np.exp(-z))
        if prime:
            #gz = z - (1-z)
            gz = gz - (1-gz)
        return gz
    
    @staticmethod
    def __tanh(z, prime=False):
        gz = 0.0
        if prime:
            gz = 1.0 - np.tanh(z)**2
        else:
            gz = math.tanh(z)
        return gz
    
    @staticmethod
    def __relu(z, prime=False):
        gz = 0.0
        if prime:
            if z < 0:
                gz = 0
            else:
                gz = 1
        else:
            gz = max(0, z)
        return gz

    @staticmethod
    def __leakyRelu(z, prime=False):
        leak = 0.01
        gz = 0.0
        if prime:
            if z < 0:
                gz = leak
            else:
                gz = 1
        else:
            gz = max(leak*z, z)
        return gz

    @staticmethod
    def __softmax(z):
        gz = 0.0
        expZ = np.exp(z)
        gz = expZ / np.sum(expZ, axis=1, keepdims=True)
        return gz

    @staticmethod
    def __softmaxLoss(y, yHat):
        # Clipping value
        minval = 0.000000000001
        m = y.shape[0]
        loss = -1/m * np.sum(y * np.log(yHat.clip(min=minval)))
        return loss



class LossFunction:

    SSDE, MSE, MSLE, MAE, MAPE  = 'SUM_SQUARED_DIFF_ERR', 'MEAN_SQUARE_ERR', 'MEAN_SQUARE_LOG_ERR', 'MEAN_ABS_ERR', 'MEAN_ABS_PER_ERR'
    SAE, L_ONE, L_TWO, KLD, CROSS_ENTROPY = 'SMOOTH_ABS_ERR', 'L_ONE', 'L_TWO', 'KL_DIVERGENCE', 'CROSS_ENTROPY'
    NLL, POISON, COS_PROX, HINGE, SQUARED_HINGE = 'NEGATIVE_LOG_LIKELIHOOD', 'POISON', 'COSINE_PROXIMITY', 'HINGE', 'SQUARED_HINGE'

    @staticmethod
    def isValid(func):
        if func not in [LossFunction.SSDE, LossFunction.MSE, LossFunction.MSLE, LossFunction.MAE, LossFunction.MAPE, LossFunction.SAE,
                        LossFunction.L_ONE, LossFunction.L_TWO, LossFunction.KLD, LossFunction.CROSS_ENTROPY, LossFunction.NLL, 
                        LossFunction.POISON, LossFunction.COS_PROX, LossFunction.HINGE, LossFunction.SQUARED_HINGE]:
            raise ValueError('ERROR: Invalid loss function')
        return True


    @staticmethod
    def getLoss(y, yHat, lossFunc=MSE):
        if not LossFunction.isValid(lossFunc):
            exit(1)
        loss = 0.0
        if lossFunc == LossFunction.SSDE:
            loss = LossFunction.__sumSquaredDifferenceError(y, yHat)
        if lossFunc == LossFunction.MSE:
            loss = LossFunction.__meanSquareError(y, yHat)
        if lossFunc == LossFunction.MSLE:
            loss = LossFunction.__meanSquareLogError(y, yHat)
        if lossFunc == LossFunction.MAE:
            loss = LossFunction.__meanAbsError(y, yHat)
        if lossFunc == LossFunction.MAPE:
            loss = LossFunction.__meanAbsPercentError(y, yHat)
        if lossFunc == LossFunction.SAE:
            loss = LossFunction.__meanSmoothAbsError(y, yHat)
        if lossFunc == LossFunction.L_ONE:
            loss = LossFunction.__lOne(y, yHat)
        if lossFunc == LossFunction.L_TWO:
            loss = LossFunction.__lTwo(y, yHat)
        if lossFunc == LossFunction.CROSS_ENTROPY:
            loss = LossFunction.__crossEntropy(y, yHat)
        if lossFunc == LossFunction.KLD:
            loss = LossFunction.__klDivergence(y, yHat)
        if lossFunc == LossFunction.NLL:
            loss = LossFunction.__negativeLogLikelihood(y, yHat)
        if lossFunc == LossFunction.POISON:
            loss = LossFunction.__poison(y, yHat)
        if lossFunc == LossFunction.COS_PROX:
            loss = LossFunction.__cosineProximity(y, yHat)
        if lossFunc == LossFunction.HINGE:
            loss = LossFunction.__hinge(y, yHat)
        if lossFunc == LossFunction.SQUARED_HINGE:
            loss = LossFunction.__squaredHinge(y, yHat)
        return loss

    @staticmethod
    def __sumSquaredDifferenceError(y, yHat):
        err = 0.0
        if y.shape == yHat.shape:
            m = y.shape[0]
            for i in range(0, m):
                err += (y[i] - yHat[i])**2
            err = err / 2
        return err

    @staticmethod
    def __meanSquareError(y, yHat):
        err = 0.0
        if y.shape == yHat.shape:
            m = y.shape[0]
            for i in range(0, m):
                err += (y[i] - yHat[i])**2
            err = err / m
        return err

    @staticmethod
    def __meanSquareLogError(y, yHat):
        err = 0.0
        if y.shape == yHat.shape:
            m = y.shape[0]
            for i in range(0, m):
                err += ( np.log(y[i]+1) - (np.log(yHat[i])+1) ) **2
            err = err / m
        return err

    @staticmethod
    def __meanAbsError(y, yHat):                                        # Used for regression
        err = 0.0
        if y.shape == yHat.shape:
            m = y.shape[0]
            for i in range(0, m):
                err += abs((y[i] - yHat[i]))
            err = err / m
        return err
    
    @staticmethod
    def __meanAbsPercentError(y, yHat):
        err = 0.0
        if y.shape == yHat.shape:
            m = y.shape[0]
            for i in range(0, m):
                err += abs( (y[i] - yHat[i]) / y[i] ) * 100             # Not recommended if values contain zeros
            err = err / m
        return err
    
    @staticmethod
    def __meanSmoothAbsError(y, yHat):                                  # Used for regression
        err = 0.0
        return err
    
    @staticmethod
    def __lOne(y, yHat):
        err = 0.0
        if y.shape == yHat.shape:
            m = y.shape[0]
            for i in range(0, m):
                err += abs((y[i] - yHat[i]))
        return err

    @staticmethod
    def __lTwo(y, yHat):
        err = 0.0
        if y.shape == yHat.shape:
            m = y.shape[0]
            for i in range(0, m):
                err += (y[i] - yHat[i])**2
        return err

    @staticmethod
    def __crossEntropy(y, yHat):                                        # Used for classification
        err = 0.0
        if y.shape == yHat.shape:
            m = y.shape[0]
            crossEntropy = 0.0
            for i in range(0, m):
                crossEntropy += ( y[i] * np.log(yHat[i]) ) + ( (1-y[i]) * np.log(1-yHat[i]) )
            err = (-1/m) * crossEntropy
        return err

    @staticmethod
    def __klDivergence(y, yHat):
        err = 0.0
        if y.shape == yHat.shape:
            m = y.shape[0]
            entropy = 0.0
            crossEntropy = 0.0
            for i in range(0, m):
                entropy += y[i] * np.log(y[i])
                crossEntropy += y[i] * np.log(yHat[i])
            err = (entropy - crossEntropy) / m
        return err

    @staticmethod
    def __negativeLogLikelihood(y, yHat):                               # Used for classification
        err = 0.0
        if y.shape == yHat.shape:
            m = y.shape[0]
            for i in range(0, m):
                err += np.log(yHat[i])
            err = (-1/m) * err
        return err

    @staticmethod
    def __poison(y, yHat):
        err = 0.0
        if y.shape == yHat.shape:
            m = y.shape[0]
            for i in range(0, m):
                err += yHat[i] - (y[i] * np.log(yHat[i]))
            err = err / m
        return err

    @staticmethod
    def __cosineProximity(y, yHat):                                     # Embedding loss function: Used to measure similarities in 2 inputs
        err = 0.0
        return err

    @staticmethod
    def __hinge(y, yHat):                                               # Embedding loss function
        err = 0.0
        if y.shape == yHat.shape:
            m = y.shape[0]
            for i in range(0, m):
                err += max(0, 1 - y[i] * yHat[i])
            err = err / m
        return err
    
    @staticmethod
    def __squaredHinge(y, yHat):
        err = 0.0
        if y.shape == yHat.shape:
            m = y.shape[0]
            for i in range(0, m):
                err += (max(0, 1 - y[i] * yHat[i])) **2
            err = err / m
        return err


