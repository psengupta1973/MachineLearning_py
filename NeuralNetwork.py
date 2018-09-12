# Neural Network
# Training: feed(input) -> findError(actualOutput) -> feedBack(error) -> findGradient() -> adjustParams(gradient)
# Predictions: feed(input)

import numpy as np
import matplotlib.pyplot as plt

from NeuroFunctions import ActivationFunction as afun
from NeuroFunctions import LossFunction as lfun
import copy

np.set_printoptions(suppress=True)



class Neuron:

    def __init__(self, id, layerId, layerType, actFunc):
        self.id = id
        self.layerId, self.layerType = layerId, layerType
        self.weights, self.bias = None, None
        self.actFunc = actFunc
        self.inputs, self.z, self.a = [], 0.0, 0.0
        self.dEdA, self.dAdZ = 0.0, 0.0
        self.dZdW, self.dEdW = None, None
        self.dEdB = 0.0

    def combine(self):
        self.z = 0.0
        weightCount = len(self.weights)
        for i in range(0, weightCount):
            self.z += self.weights[i] * self.inputs[i]
        self.z += self.bias
        return self.z

    def feed(self, inputs):
        if self.layerType == NeuralLayer.INPUT:
            self.inputs, self.weights, self.bias, self.z, self.a = [inputs], [], 0.0, 0.0, inputs
        else:
            self.inputs = np.asarray(inputs)
            self.inputs = self.inputs.reshape(self.inputs.shape[0], 1)
            if self.weights is None and self.bias is None:
                self.weights = np.random.normal(size=self.inputs.shape)
                self.bias = np.random.normal()
                #print('INIT WEIGHTS & BIASES for ',self.layerId,self.id)
            try:
                self.a = afun.activate(float(self.combine()), actFunc=self.actFunc)
            except ValueError as ve:
                print('ERROR in Neuron: {0}'.format(ve))
        #print('==>',self.layerId,'-',self.id,'i:',np.asarray(self.inputs).T,' w:', np.asarray(self.weights).T, ' b:', self.bias, ' a:', self.a)
        return self.a

    def feedBack(self, errorSignal, learnRate, nextLayer=None):
        #np.seterr(all='ignore')
        weightCount = len(self.weights)
        if self.layerType == NeuralLayer.OUTPUT and nextLayer is None:                          # propagatedError (@ output layer) =  yHat - y
            self.dEdA = errorSignal                                                             # dA -> d/da of propagatedError
            self.dAdZ = afun.activate(self.z, actFunc=self.actFunc, prime=True)                 # dZ -> d/dz of a
            self.dZdW = copy.deepcopy(self.inputs)                                              # dW -> d/dw of z
            self.dEdW = copy.deepcopy(self.inputs)
            for i in range(0, weightCount):
                self.dEdW[i] = self.dEdA * self.dAdZ * self.dZdW[i]
                self.weights[i] -= (learnRate * self.dEdW[i])
            self.dEdB = self.dEdA * self.dAdZ
            self.bias -= (learnRate * self.dEdB)
        elif self.layerType == NeuralLayer.HIDDEN:
            self.dAdZ = afun.activate(self.z, actFunc=self.actFunc, prime=True)
            self.dEdW = copy.deepcopy(self.weights)
            for i in range(0, weightCount):                                                      # Adjust weights
                dNextSum = 0.0
                for nextNeuron in nextLayer.neurons:
                    dNextSum += nextNeuron.dEdA * nextNeuron.dAdZ * nextNeuron.weights[self.id]
                self.dEdW[i] = self.inputs[i] * self.dAdZ * dNextSum
                self.weights[i] -= (learnRate * self.dEdW[i])
            dNextSum = 0.0
            for nextNeuron in nextLayer.neurons:                                                 # Adjust bias
                dNextSum += nextNeuron.bias * nextNeuron.weights[self.id]
            self.dEdB = self.dAdZ * dNextSum
            self.bias -= (learnRate * self.dEdB)
        #print('<==',self.layerId,'-',self.id,'i:',np.asarray(self.inputs).T,' w:', np.asarray(self.weights).T, ' b:', self.bias)

    def printMe(self):
        print('L',self.layerId,'\b-N',self.id,':',np.asarray(self.inputs).T,' * ', 
                np.asarray(self.weights).T , '->',self.actFunc,'(',self.z,') -->', self.a)





class NeuralLayer:

    INPUT, HIDDEN, OUTPUT = 'INPUT', 'HIDDEN', 'OUTPUT'

    def __init__(self, id, neuronCount, actFunc, layerType=INPUT):
        if layerType not in [NeuralLayer.INPUT, NeuralLayer.HIDDEN, NeuralLayer.OUTPUT]:
            print('ERROR: Invalid layer type in layer (',layerType,') definition')
            exit(1)
        self.id, self.type = id, layerType
        self.neurons = []
        for i in range(0, neuronCount):
            self.neurons.append(Neuron(i, id, layerType, actFunc))

    def feed(self, inputs):
        outputs = [] 
        if self.type == NeuralLayer.INPUT:
            if len(self.neurons) != len(inputs):
                print('ERROR: Number of inputs and neurons in INPUT layer are not same')
                exit(1)
            else:
                for neuron, inp in zip(self.neurons, inputs):
                    outputs.append(neuron.feed(inp))
        else:
            for neuron in self.neurons:
                outputs.append(neuron.feed(inputs))
        outputs = np.asarray(outputs)
        #print('DEBUG: layer',self.id, ' -> ',inputs.T, ' --> ',outputs.T)
        return outputs

    def feedBack(self, errorSignal, learnRate, nextLayer=None):
        for neuron in self.neurons:
            if self.type == NeuralLayer.OUTPUT and nextLayer is None:                       # @ output layer
                neuron.feedBack(errorSignal, learnRate)
            else:
                neuron.feedBack(None, learnRate, nextLayer)    

    def printMe(self):
        for neuron in self.neurons:
            print('L',neuron.layerId,'\b-N',neuron.id,':',np.asarray(neuron.inputs).T,' * ', np.asarray(neuron.weights).T , 
                    '->',neuron.actFunc,'(',neuron.z,') -->', neuron.a)





class NeuralNetwork:

    # weights and biases are 2d arrays with dimensions as [neuron count, input count] in current layer
    def __init__(self, topology, verbose=False):
        if topology is None or len(topology) < 2:
            print('ERROR: Invalid topology')
            exit(1)
        self.verbose = verbose
        self.layers = []
        layerCount = len(topology)
        self.layers.append(NeuralLayer(0, topology[0][0], topology[0][1], layerType=NeuralLayer.INPUT))
        for i in range(1, layerCount-1):
            self.layers.append(NeuralLayer(i, topology[i][0], topology[i][1], layerType=NeuralLayer.HIDDEN))
        self.layers.append(NeuralLayer(len(self.layers), topology[-1][0], topology[-1][1], layerType=NeuralLayer.OUTPUT))

    def feed(self, inputs):
        outputs = None
        for layer in self.layers:
            outputs = layer.feed(inputs)
            inputs = outputs
        return outputs
    
    def lossPerEpoch(self, y, yHat, prime=False):
        if yHat.shape != y.shape:
            print('ERROR: Unequal shapes for yHat and y in loss function')
            exit(1)
        loss = 0.0
        m, n = y.shape
        for x in range(0, m):
            if prime:
                loss += self.lossPerExample(y[x], yHat[x], prime=True)
            else:
                loss += self.lossPerExample(y[x], yHat[x])
            #for f in range(0, n):
            #    loss += ((yHat[x][f] - y[x][f]) **2)/2
        return loss

    def lossPerExample(self, y, yHat, prime=False):
        outputCount = len(yHat)
        loss = 0.0
        if prime:
            for f in range(0, outputCount):
                loss += (yHat[f] - y[f])
        else:
            for f in range(0, outputCount):
                loss += ((yHat[f] - y[f]) **2)/2
        return loss

    def feedBack(self, netError, eta=0.3):
        layerCount = len(self.layers)
        for i in range(layerCount-1, -1, -1):
            if self.layers[i].type == NeuralLayer.OUTPUT:
                self.layers[i].feedBack(netError, eta)
            else:
                self.layers[i].feedBack(None, eta, nextLayer=self.layers[i+1])

    def train(self, X, y, learningRate=0.3, epoch=10):
        X, y = np.asarray(X), np.asarray(y)
        if y.shape[0] != X.shape[0]:
            print('ERROR: number of examples in X (', X.shape[0] ,') and y (',y.shape[0],') are not same')
            exit(1)
        print('\n------------------- TRAINING ---------------------')
        m, n = X.shape
        eLosses = []
        for i in range(0, epoch):
            yHat = []
            for x in range(0, m):
                #print('Feed ------------ > ')
                outputs = self.feed(X[x])
                lossSignal = self.lossPerExample(y[x], outputs, prime=True)
                #print('LOSS @ Epoch',i,'-Example',x,': ', outputs, '-',y[x], '=', lossSignal)
                #print('< --------- FeedBack ')
                self.feedBack(lossSignal, eta=learningRate)
                yHat.append(outputs)
            netLoss = self.lossPerEpoch(y, np.asarray(yHat).reshape(y.shape))
            eLosses.append(netLoss)
            #print('LOSS @ EPOCH',i,': ', netLoss)
            #self.feedBack(netLoss, eta=learningRate)
        print('\nTraining iterations: ', epoch, ' Cost: Init:', eLosses[0],' --> Min:', np.min(eLosses), '-> Final:',eLosses[-1],' \n')
        self.plotLoss(eLosses)

    def predict(self, X):
        yHat = []
        for xRow in X:
            print('\n------------------- PREDICT ---------------------')
            print('#### Predicting example X:', xRow)
            outputs = self.feed(xRow)
            outputs = np.asarray(outputs)
            yHat.append(outputs)
        return yHat

    def printMe(self, back=False):
        if back:
            lCount = len(self.layers)
            for l in range(lCount-1, -1, -1):
                layer = self.layers[l]
                print('\n#### Layer', layer.id,'(type=',layer.type,'; actFunc=', layer.neurons[0].actFunc, '\b) ####')
                nCount = len(layer.neurons)
                for n in range(0, nCount):
                    neuron = layer.neurons[n]
                    print('L',neuron.layerId,'N',neuron.id,': weights: ', np.asarray(neuron.weights).T, 'bias: ', neuron.bias)
        else:
            for layer in self.layers:
                print('\n#### Layer', layer.id,'(type=',layer.type,'; actFunc=',layer.neurons[0].actFunc, '\b) ####')
                for neuron in layer.neurons:
                    print(np.asarray(neuron.inputs).T, '-> N',neuron.id, '-> *', np.asarray(neuron.weights).T , '=',neuron.z,'->', neuron.a)

    def plotLoss(self, losses):
        # VISUALIZE improvement of model after training
        lossCount = len(losses)
        if (lossCount > 0):
            plt.plot(np.linspace(1, lossCount, lossCount, endpoint=True), losses)
            plt.title("Iteration vs Loss ")
            plt.xlabel("# of iterations")
            plt.ylabel("Loss")
            plt.show()





def main():
    topology = [[2, None], [4, afun.SIGMOID], [8, afun.TANH], [16, afun.SIGMOID], [1, afun.SIGMOID]]
    #print('Topology: ',topology)
    X = np.array([[0,0],  [1,0],  [2,0],  [1,1],  [7,2], [1,3], [5,0], [1,5],  [7,1],  [5,5], 
                  [6,5],  [7,7],  [9,7],  [10,8], [1,9], [8,2], [7,3], [9,3],  [8,10], [6,9], 
                  [5,10], [5,12], [4,30], [78,45],[0,45],[34,8],[8,9], [23,12],[33,17],[27,9]])
   # y = np.array([[0,0], [0,0], [0,0], [0,0], [1,0], [0,0], [1,0], [0,1], [1,0], [1,1], [1,1], [1,1], [1,1],  [1,1], [0,1], [1,0]])
    y = np.array([[0],    [0],    [0],    [0],    [0],   [0],   [0],   [0],    [0],    [1], 
                  [1],    [1],    [1],    [1],    [0],   [0],   [0],   [1],    [1],    [1], 
                  [1],    [1],    [0],    [1],    [0],   [1],   [1],   [1],    [1],    [1]])
    print('X: ',X, '\ny: ',y)
    nn = NeuralNetwork(topology, verbose=False)
    nn.train(X, y, learningRate=0.003, epoch=100)
    predX = np.array([[9,7]])
    print(nn.predict(predX))

if True:
    main()

# Adjust bias
# check gPrime(z)
# check error calculations for Ei for more than 1 neuron in output layer
# why output is array of arrays in train()?