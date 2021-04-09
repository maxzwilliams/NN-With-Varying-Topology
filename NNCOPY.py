from Neuron import *
from random import *
from Helper import *
import time
import copy
seed(1)
## hardcoded positions of the input and output layers
inputPos = 0
outputPos = 100
hiddenPos = 50

class NN:

    def __init__(self,dataSet, learningRate, generationRate):
        ## learning properties
        self.learningRate = learningRate
        self.generationRate = generationRate

        ## geometry propeties
        self.inputSize = len(dataSet[0][0])
        self.outputSize = len(dataSet[0][1])
        self.hiddenSize = math.floor((self.inputSize + self.outputSize)/2)
        self.neuronDict = dict()

        ## initialNetworkState
        self.generateInitialNetwork()

    ## generate initial network structure
    def generateInitialNetwork(self):
        self.generateOutputLayer()
        self.generateHiddenLayer()
        self.generateInputLayer()

    def generateInputLayer(self):
        layer = []
        for index1 in range(self.inputSize):
            newNeuron = Neuron(inputPos, uniform(0,1))
            newConnections = []
            for index2 in range(self.hiddenSize):
                newConnections.append([ self.neuronDict[hiddenPos][index2], uniform(0,1)])
            newNeuron.connections = newConnections
            layer.append(newNeuron)
        self.neuronDict[inputPos] = layer

    def generateHiddenLayer(self):
        layer = []
        for index1 in range(self.hiddenSize):
            newNeuron = Neuron(inputPos, uniform(0,1))
            newConnections = []
            for index2 in range(self.outputSize):
                newConnections.append([ self.neuronDict[outputPos][index2], uniform(0,1)])
            newNeuron.connections = newConnections
            layer.append(newNeuron)
        self.neuronDict[hiddenPos] = layer

    def generateOutputLayer(self):
        layer = []
        for index1 in range(self.outputSize):
            newNeuron = Neuron(outputPos, uniform(0,1))
            layer.append(newNeuron)
        self.neuronDict[outputPos] = layer

    ## takes and input and evaluates it using the network returning the result
    def evaluate(self, input):
        ## first we evaluate the input layer

        rtn = []
        keys = getKeys(self.neuronDict)
        for index1 in range(len(self.neuronDict[inputPos])):
            self.neuronDict[inputPos][index1].value = input[index1]

        for key in keys:
            ##print(key)
            for index1 in range(len(self.neuronDict[key])):

                if (key != inputPos):
                    self.neuronDict[key][index1].getValue()

                if (key != outputPos):
                    ## then we need to update the logits of everything infront
                    sum = 0
                    for index2 in range(len(self.neuronDict[key][index1].connections)):
                         self.neuronDict[key][index1].connections[index2][0].logit += self.neuronDict[key][index1].value * self.neuronDict[key][index1].connections[index2][1]
                         ##sum += self.neuronDict[key][index1].value * self.neuronDict[key][index1].connections[index2][1]

                    ##self.neuronDict[key][index1].connections[index2][0].logit = sum
                else:
                    rtn.append(self.neuronDict[key][index1].value)
                ##print("here")
                ##print(self.neuronDict[key][index1].value)
                ##print(key)
                ##print("done")

        return rtn


    def reset(self):
        keys = getKeys(self.neuronDict)
        for key in keys:
            for neuron in self.neuronDict[key]:
                neuron.valueReset()


    def backPropigate(self, output, expectedOutput):
        ## algorithm from: https://en.wikipedia.org/wiki/Backpropagation
        keys = getKeys(self.neuronDict)
        keys.reverse()

        for index in range(len(self.neuronDict[outputPos])):
            self.neuronDict[outputPos][index].delta = (self.neuronDict[outputPos][index].value - expectedOutput[index]) * self.neuronDict[outputPos][index].derivativeActivationFunction(self.neuronDict[outputPos][index].value)
            ##print("output deltas")
            ##print(self.neuronDict[outputPos][index].value)
            ##print((self.neuronDict[outputPos][index].value - output[index]))
            ## update weights


        for key in keys[1:(len(keys)-1)]:
            try:
                for index1 in range(len(self.neuronDict[key])):
                    sum = 0
                    for index2 in range(len(self.neuronDict[key][index1].connections)):
                        sum += self.neuronDict[key][index1].connections[index2][1] * self.neuronDict[key][index1].connections[index2][0].delta
                    self.neuronDict[key][index1].delta = sum * self.neuronDict[key][index1].derivativeActivationFunction(self.neuronDict[key][index1].value)
            except:
                print("Overflow error")
                time.sleep(10000)

        ## update the weights
        for key in keys:
            for index1 in range(len(self.neuronDict[key])):
                dBias = self.neuronDict[key][index1].delta
                self.neuronDict[key][index1].bias += - self.learningRate * dBias
                for index2 in range(len(self.neuronDict[key][index1].connections)):
                    ##print("here is the value")
                    ##print(self.neuronDict[key][index1].value)
                    ##print("")
                    ##print("here is the delta")
                    ##print(self.neuronDict[key][index1].connections[index2][0].delta)
                    ##print("")
                    dWeight = self.neuronDict[key][index1].value * self.neuronDict[key][index1].connections[index2][0].delta
                    dBias = self.neuronDict[key][index1]

                    self.neuronDict[key][index1].connections[index2][1] += -self.learningRate * dWeight


    def basicTrain(self,dataSet):
        ## dataSet contains elements that are lists of inputs and outputs
        counter = 0
        dataSetLength = len(dataSet)
        for element in dataSet:
            input = element[0]
            expectedOutput = element[1]
            eval = self.evaluate(input)
            self.backPropigate(eval, expectedOutput)
            self.reset()

            #print(counter)
            ##print(counter % dataSetLengthPerHundred)
            counter += 1

    def train(self, dataSet, learningRateAdjustNumber, multipliers):
        ## dataSet contains elements that are lists of inputs and outputs
        print("commencing training")
        counter = 0
        dataSetLength = len(dataSet)
        adjustInterval = math.floor(dataSetLength/learningRateAdjustNumber)

        for element in dataSet:
            input = element[0]
            expectedOutput = element[1]
            eval = self.evaluate(input)
            self.backPropigate(eval, expectedOutput)
            self.reset()
            printProgressBar(counter, dataSetLength)

            ##if (counter % adjustInterval == 0 and counter != 0):
                ##self.adjustLearningRate( sample(dataSet, math.floor(adjustInterval/ (100 *len(multipliers)) )) , multipliers)
                ##self.adjustLearningRate( sample(dataSet, 20) , multipliers)
            #print(counter)
            ##print(counter % dataSetLengthPerHundred)
            counter += 1
        print("training complete")

    def adjustLearningRate(self, testSet, multipliers):
        ## testSet is formatted like eveyrother dataSet
        ## multipliers is a list
        scores = []

        counter = 0
        for element in multipliers:
            testNetwork = copy.deepcopy(self)
            testNetwork.learningRate = self.learningRate * element
            testNetwork.basicTrain(testSet)
            scores.append(testPerformance(testSet, testNetwork))
            del testNetwork
            counter += 1
        maxIndex = scores.index(max(scores))
        self.learningRate = self.learningRate * multipliers[maxIndex]
