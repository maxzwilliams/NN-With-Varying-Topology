from Neuron import *
from random import *
from Helper import *
import time

import sys

## for saving large dataStructures
sys.setrecursionlimit(4000)

import copy

import matplotlib.pyplot as plt
##import multiproccessing as mp
import multiprocessing as mp

##seed(2)
## hardcoded positions of the input and output layers
inputPos = 0
outputPos = 100

## only used when hidden layers are used
hiddenPos = 50
hiddenPos2 = 75

class NN:

    def __init__(self, dataSet, learningRate, geometryEditRate, printNetworkInfo = False, marker = None):
        print("initializing a new network")
        self.marker = marker

        ## learning properties
        self.learningRate = learningRate
        self.geometryEditRate = geometryEditRate
        ## changes

        ## accuracy that the network is aiming for
        self.target = 0.99

        ## period of training data points between potential network generations
        self.editPeriod = 10

        ## the probability of adding a neuron when at target
        self.genDecideBias = 0.65

        ## changes how the slope changes how new neurons are generated
        self.slopeSensitivity = 10**4

        ## hyperParameter which dictates which neurons are more connected too
        ## a higher selectionBias inducates that connections between hidden neurons
        ## is more likely than connections to or from input or output neurons
        ## higher selectionBias numbers are suited to more complex systems
        ## and lower selectionBias is for less complex systems that require less
        ## hidden connections
        self.selectionBias = 1000

        ## current score of the network while training
        self.currentScore = -1000
        ## best score predicted by a simulation
        self.currentTrialScore = -1

        ## hyperparameter multipliers to simulate on
        self.multipliers = [[0.01, 1, 100], [math.exp(-1/2), 1, math.exp(1/2)]]
        self.multiplierCopy = [[0.01, 1, 100], [math.exp(-1/2), 1, math.exp(1/2)]]

        ## geometry propeties
        self.inputSize = len(dataSet[0][0])
        self.outputSize = len(dataSet[0][1])

        ## when used these are the hiddenlayer sizes
        self.hiddenSize = 100
        self.hiddenSize2 = 50

        ## number of neurons in the network including input and output neurons
        self.neuronNumber = self.inputSize + self.outputSize

        ## number of random neurons we start with
        self.startingBatchSize = 100

        ## self.neuronInputSize = 100
        ## self.neuronOutputSize = 50

        self.neuronInputSize = 100
        self.neuronOutputSize = 50

        ## dataStructure that stores all neurons in the network. This completely
        ## defines the network
        self.neuronDict = dict()

        ## geometry editing scores
        ## past correct scores (they are either 0 or 1)
        self.pastScores = []
        ## pastscores smoothed by a moving average
        self.smoothedScores = []

        ## keeps track of how many neurons were in the network over time
        self.pastNeuronNumbers = []

        ## keeps track of the number of connections in the network
        self.pastConnections = []
        self.currentConnections = 0

        ##
        self.currentGeometryEdit = 0
        self.pastGeometryEdit= []

        self.currentDecide = 0
        self.pastDecide = []

        ## the number of most recent data points that the moving average takes
        self.movingAverageLookBack = 1000

        ## how far the hyperParameter simulations go for
        self.hyperparameterLookBack = 500
        ## how far back our linear fit looks
        self.linearRegressionLookBack = 500


        ## STATS FOR NEURONS
        ##self.initialNeuronInputs = self.inputSize
        ##self.initialNeuronOutputs = math.floor(self.intputSize + self.outputSize)


        ## initialNetworkState
        self.generateInitialNetwork()

        if (printNetworkInfo == True):
            self.informationPrintOut()

    def informationPrintOut(self):
        print("----------------------------------------------------------------------------------------------------")
        print("general stats:")
        print("self.learningRate", self.learningRate)
        print("self.editperiod", self.editPeriod)
        print("self.target", self.target)
        print("self.movingAverageLookBack", self.movingAverageLookBack)
        print("self.hyperparameterLookBack", self.hyperparameterLookBack)
        print("self.startingBatchSize", self.startingBatchSize)
        print("General Information:")
        print("core learning functions are working")
        print("several hyperparameters are searched")
        print("learning rate boosts of x10 are given to new neurons")
        print("a scoring system for generation is introduced and works mostly")
        print("this program deletes direct connections between neurons when a more complex route between them is formed")
        print("this program has a bias towards connections being made between hidden neurons and outputNeurons")
        print("Currently the best score is ~90 % from overnight training on the whole dataSet")
        print("this program uses a sigmoid activation function for all neurons")
        print("this also has improved speed training by a factor of 10")
        print("----------------------------------------------------------------------------------------------------")

    ## will generate an initial amount of randomly placed neurons when called
    def randomGenerate(self):
        for index in range(self.startingBatchSize):
            ##self.addRandomNeuron(self.inputSize, math.floor((self.outputSize+self.inputSize)/2))
            self.addRandomNeuron(self.neuronInputSize, self.neuronOutputSize)

    ## generate initial network structure
    def generateInitialNetwork(self):
        self.generateOutputLayer()
        #self.hiddenLayer2()
        #self.hiddenLayer()
        self.generateInputLayer()

        self.randomGenerate()

    def generateInputLayer(self):
        layer = []
        for index1 in range(self.inputSize):
            newNeuron = Neuron(inputPos, uniform(-1,1))
            newConnections = []
            ##for index2 in range(self.outputSize):
                ##newConnections.append([ self.neuronDict[outputPos][index2], uniform(-1,1)])
            newNeuron.connections = newConnections
            layer.append(newNeuron)
        self.neuronDict[inputPos] = layer

    ## NOT USED
    def hiddenLayer(self):
        layer = []
        for index1 in range(self.hiddenSize):
            newNeuron = Neuron(inputPos, uniform(-1,1))
            newConnections = []
            for index2 in range(self.hiddenSize2):
                newConnections.append([ self.neuronDict[hiddenPos2][index2], uniform(-1,1)])
            newNeuron.connections = newConnections
            layer.append(newNeuron)
        self.neuronDict[hiddenPos] = layer

    ## NOT USED
    def hiddenLayer2(self):
        layer = []
        for index1 in range(self.hiddenSize2):
            newNeuron = Neuron(inputPos, uniform(-1,1))
            newConnections = []
            for index2 in range(self.outputSize):
                newConnections.append([ self.neuronDict[outputPos][index2], uniform(-1,1)])
            newNeuron.connections = newConnections
            layer.append(newNeuron)
        self.neuronDict[hiddenPos2] = layer

    def generateOutputLayer(self):
        layer = []
        for index1 in range(self.outputSize):
            newNeuron = Neuron(outputPos, uniform(-1,1))
            layer.append(newNeuron)
        self.neuronDict[outputPos] = layer

    ## takes and input and evaluates it using the network returning the result
    def evaluate(self, input):

        rtn = []
        keys = getKeys(self.neuronDict)
        for index1 in range(len(self.neuronDict[inputPos])):
            self.neuronDict[inputPos][index1].value = input[index1]
            ##print("value:",self.neuronDict[inputPos][index1].value)

        for key in keys:
            ##print(key)
            for index1 in range(len(self.neuronDict[key])):

                if (key != inputPos):
                    self.neuronDict[key][index1].getValue()
                    ##print("value2:",self.neuronDict[key][index1].value)

                ## what is this??
                if (key != outputPos):
                    ## then we need to update the logits of everything infront
                    sum = 0
                    for index2 in range(len(self.neuronDict[key][index1].connections)):
                         self.neuronDict[key][index1].connections[index2][0].logit += self.neuronDict[key][index1].value * self.neuronDict[key][index1].connections[index2][1]
                         ##print("logit:",self.neuronDict[key][index1].connections[index2][0].logit)
                         ##sum += self.neuronDict[key][index1].value * self.neuronDict[key][index1].connections[index2][1]

                    ##self.neuronDict[key][index1].connections[index2][0].logit = sum
                else:
                    rtn.append(self.neuronDict[key][index1].value)
                ##print("here")
                ##print(self.neuronDict[key][index1].value)
                ##print(key)
                ##print("done")

        return rtn

    ## resets the states of each neuron in the network
    def reset(self):
        keys = getKeys(self.neuronDict)
        for key in keys:
            for neuron in self.neuronDict[key]:
                neuron.valueReset()

    ## general Backpropagation algorithm
    def backPropigate(self, output, expectedOutput):
        ## algorithm from: https://en.wikipedia.org/wiki/Backpropagation
        keys = getKeys(self.neuronDict)
        keys.reverse()

        for index in range(len(self.neuronDict[outputPos])):
            ##self.neuronDict[outputPos][index].delta = (self.neuronDict[outputPos][index].value - expectedOutput[index]) * self.neuronDict[outputPos][index].derivativeActivationFunction(self.neuronDict[outputPos][index].value)
            self.neuronDict[outputPos][index].delta = (self.neuronDict[outputPos][index].value - expectedOutput[index]) * self.neuronDict[outputPos][index].derivativeActivationFunction(self.neuronDict[outputPos][index].logit) ##[tick1]
            self.neuronDict[outputPos][index].scoreUpdate(1-smoothedUnitLinear(5*abs(self.neuronDict[outputPos][index].delta )))
            ##self.pastDeltas.append( abs(self.neuronDict[outputPos][index].delta) )
            #print(self.neuronDict[outputPos][index].delta)
            ##print("output deltas")
            ##print(self.neuronDict[outputPos][index].value)
            ##print((self.neuronDict[outputPos][index].value - output[index]))
            ## update weights


        ## question:
        ##
        for key in keys[1:(len(keys))]:
            try:
                for index1 in range(len(self.neuronDict[key])):
                    sum = 0
                    for index2 in range(len(self.neuronDict[key][index1].connections)):
                        try:
                            sum += self.neuronDict[key][index1].connections[index2][1] * self.neuronDict[key][index1].connections[index2][0].delta
                        except:
                            print(key, index1, index2)
                            print(sum)
                            print(self.neuronDict[key][index1].connections[index2][1])
                            print(self.neuronDict[key][index1].connections[index2][0].delta)

                            print("overflow")
                            time.sleep(100)
                    self.neuronDict[key][index1].delta = sum * self.neuronDict[key][index1].derivativeActivationFunction(self.neuronDict[key][index1].logit)
                    ##self.neuronDict[key][index1].scores.append(abs( self.neuronDict[key][index1].delta ))
                    self.neuronDict[key][index1].scoreUpdate(1-smoothedUnitLinear(5*abs(self.neuronDict[outputPos][index].delta )))
                    ##if (key == inputPos):
                        ##self.pastDeltas.append( abs(self.neuronDict[outputPos][index].delta) )

            except:
                print("Overflow error")
                time.sleep(10000)

        ## update the weights
        for key in keys:
            for index1 in range(len(self.neuronDict[key])):
                dBias = self.neuronDict[key][index1].delta
                learningRateBuff = 10 - (self.neuronDict[key][index1].age/10)
                if (learningRateBuff <= 1 ):
                    learningRateBuff = 1

                self.neuronDict[key][index1].bias += -self.learningRate * dBias * learningRateBuff
                for index2 in range(len(self.neuronDict[key][index1].connections)):
                    dWeight = self.neuronDict[key][index1].value * self.neuronDict[key][index1].connections[index2][0].delta
                    #dBias = self.neuronDict[key][index1]

                    self.neuronDict[key][index1].connections[index2][1] += -self.learningRate * dWeight * learningRateBuff

    ## an attempt at a fast backPropigate
    ## takes a set of evaluations and the corresponding set of correct outputs
    ## and runs the backpropigation algorithm on them
    ## this may not work at all but lets have a look
    def fastBackPropigate(self, outputs, expectedOuputs):
        batchSize = len(outputs)
        ## we first scale the outputs and expectedOutputs
        scaledOutputs = []
        scaledExpectedOutputs = []

        scaledOutput  = []
        scaledExpectedOutput = []

        for index in range(len(outputs)):
            outputs[index][:] = [x/ batchSize for x in outputs[index]]
            expectedOuputs[index][:] = [x/ batchSize for x in expectedOuputs[index]]
            scaledOutputs.append(outputs[index])
            scaledExpectedOutputs.append(expectedOuputs[index] )

        for index1 in range(batchSize):
            scaledOutputElement = 0
            scaledExpectedOutputElement = 0
            for index2 in range(len(scaledOutputs[index1])):
                scaledOutputElement += scaledOutputs[index1][index2]
            scaledOutput.append(scaledOutputElement)
            for index3 in range(len(scaledExpectedOutputs[index1])):
                scaledExpectedOutputElement += scaledExpectedOutputs[index1][index3]
            scaledExpectedOutput.append(scaledExpectedOutputElement)




        ## now we have our output and expected output which is all we need
        ## for the normal backpropigation algorithm

        self.setBackPropigate(scaledOutput, scaledExpectedOutput, batchSize)

    def setBackPropigate(self, output, expectedOutput, batchSize):
        ## algorithm from: https://en.wikipedia.org/wiki/Backpropagation
        keys = getKeys(self.neuronDict)
        keys.reverse()

        for index in range(len(self.neuronDict[outputPos])):
            ##self.neuronDict[outputPos][index].delta = (self.neuronDict[outputPos][index].value - expectedOutput[index]) * self.neuronDict[outputPos][index].derivativeActivationFunction(self.neuronDict[outputPos][index].value)
            self.neuronDict[outputPos][index].delta = (self.neuronDict[outputPos][index].value - expectedOutput[index]) * self.neuronDict[outputPos][index].derivativeActivationFunction(self.neuronDict[outputPos][index].logit) ##[tick1]
            for repIndex in range(batchSize):
                self.neuronDict[outputPos][index].scoreUpdate(1-smoothedUnitLinear(5*abs(self.neuronDict[outputPos][index].delta )))
            ##self.pastDeltas.append( abs(self.neuronDict[outputPos][index].delta) )
            #print(self.neuronDict[outputPos][index].delta)
            ##print("output deltas")
            ##print(self.neuronDict[outputPos][index].value)
            ##print((self.neuronDict[outputPos][index].value - output[index]))
            ## update weights


        ## question:
        ##
        for key in keys[1:(len(keys))]:
            try:
                for index1 in range(len(self.neuronDict[key])):
                    sum = 0
                    for index2 in range(len(self.neuronDict[key][index1].connections)):
                        try:
                            sum += self.neuronDict[key][index1].connections[index2][1] * self.neuronDict[key][index1].connections[index2][0].delta
                        except:
                            print(key, index1, index2)
                            print(sum)
                            print(self.neuronDict[key][index1].connections[index2][1])
                            print(self.neuronDict[key][index1].connections[index2][0].delta)

                            print("overflow")
                            time.sleep(100)
                    self.neuronDict[key][index1].delta = sum * self.neuronDict[key][index1].derivativeActivationFunction(self.neuronDict[key][index1].logit)
                    ##self.neuronDict[key][index1].scores.append(abs( self.neuronDict[key][index1].delta ))
                    for repIndex in range(batchSize):
                        self.neuronDict[key][index1].scoreUpdate(1-smoothedUnitLinear(5*abs(self.neuronDict[outputPos][index].delta )))
                    ##if (key == inputPos):
                        ##self.pastDeltas.append( abs(self.neuronDict[outputPos][index].delta) )
            except:
                print("Overflow error")
                time.sleep(10000)

        ## update the weights
        for key in keys:
            for index1 in range(len(self.neuronDict[key])):
                dBias = self.neuronDict[key][index1].delta
                learningRateBuff = 10 - (self.neuronDict[key][index1].age/10)
                if (learningRateBuff <= 1 ):
                    learningRateBuff = 1

                self.neuronDict[key][index1].bias += -self.learningRate * dBias * learningRateBuff
                for index2 in range(len(self.neuronDict[key][index1].connections)):
                    dWeight = self.neuronDict[key][index1].value * self.neuronDict[key][index1].connections[index2][0].delta
                    #dBias = self.neuronDict[key][index1]
                    self.neuronDict[key][index1].connections[index2][1] += -self.learningRate * dWeight * learningRateBuff

    ## trains network on specified dataset without adapting learning rate or any other parameters
    def basicTrain(self, dataSet, marker = None):
        ## dataSet contains elements that are lists of inputs and outputs
        counter = 0
        dataSetLength = len(dataSet)
        for element in dataSet:
            input = element[0]
            expectedOutput = element[1]
            eval = self.evaluate(input)
            self.backPropigate(eval, expectedOutput)
            self.reset()

            if (marker != None):
                printProgressBar(counter, dataSetLength)

            counter += 1

    def fastBasicTrain(self, dataSet, marker = None):
        ## dataSet contains elements that are lists of inputs and outputs
        counter = 0
        dataSetLength = len(dataSet)
        evals = []
        expectedOuputs = []

        for element in dataSet:
            input = element[0]
            expectedOutput = element[1]
            eval = self.evaluate(input)

            evals.append(eval)
            expectedOuputs.append(expectedOutput)
            if (counter % 10 == 0 and counter != 0):
                self.fastBackPropigate(evals, expectedOuputs)
                evals = []
                expectedOuputs = []

            ##self.backPropigate(eval, expectedOutput)
            self.reset()

            if (marker != None):
                printProgressBar(counter, dataSetLength)

            counter += 1

    ## simulation version of train

    def simTrain(self, dataSet, return_dict):
        ## dataSet contains elements that are lists of inputs and outputs
        counter = 0
        dataSetLength = len(dataSet)
        for element in dataSet:
            input = element[0]
            expectedOutput = element[1]
            eval = self.evaluate(input)
            self.backPropigate(eval, expectedOutput)
            self.reset()

            if (testCorrect(eval, expectedOutput)):
                self.pastScores.append(1)
            else:
                self.pastScores.append(0)

            printProgressBar(counter, dataSetLength)

            self.smoothedScores.append( movingAverage(self.pastScores, self.movingAverageLookBack) )
            self.pastNeuronNumbers.append( self.neuronNumber - self.inputSize - self.outputSize )
            self.pastConnections.append( self.currentConnections )

            #print(counter)
            ##print(counter % dataSetLengthPerHundred)
            counter += 1
        return_dict[self] = []

    ## trains network on dataSet while adjusting learning rate
    ## learningRateAdjustNumber is the number of times during training the
    ## learning rate is adjusted
    def train(self, dataSet, learningRateAdjustNumber, multipliers):
        ## dataSet contains elements that are lists of inputs and outputs
        print("commencing training")
        counter = 0
        dataSetLength = len(dataSet)
        adjustInterval = math.floor(dataSetLength/learningRateAdjustNumber)
        self.reset()

        for element in dataSet:
            input = element[0]
            expectedOutput = element[1]
            eval = self.evaluate(input)

            if (testCorrect(eval, expectedOutput)):
                self.pastScores.append(1)
            else:
                self.pastScores.append(0)

            self.backPropigate(eval, expectedOutput)
            self.reset()
            printProgressBar(counter, dataSetLength)

            if (counter % adjustInterval == 0 and counter != 0):
            ##if (counter == 0):
                self.adjustLearningRate( sample(dataSet, self.hyperparameterLookBack) , multipliers)
                ##self.adjustLearningRate( sample(dataSet, 20) , multipliers)
            #print(counter)
            ##print(counter % dataSetLengthPerHundred)
            counter += 1
        print("training complete")

    ## simulates learning with different learningRates to produce
    def adjustLearningRate(self, testSet, multipliers):
        print("adjusting learningRate")
        ## testSet is formatted like eveyrother dataSet
        ## multipliers is a list
        scores = []
        counter = 0
        for element in multipliers:
            testNetwork = copy.deepcopy(self)
            testNetwork.learningRate = self.learningRate * element
            testNetwork.reset()
            testNetwork.basicTrain(testSet)
            testNetwork.reset()
            scores.append(testPerformance(testSet, testNetwork))
            del testNetwork
            printProgressBar(counter, len(multipliers))
            counter += 1
        maxIndex = scores.index(max(scores))
        self.learningRate = self.learningRate * multipliers[maxIndex]
        print("new learningRate:", self.learningRate)

    ## search through learningRates and editPeriods to find the optimal one
    ## Does not use multiprocessing
    def hyperParameterSearch(self, testSet, multipliers):
        print("finding hyperparameters")
        ## each element contains a score and the corresponding index in fullMultipliers
        scores = []

        ## generate a list with elements whos first element repressent the learningRate multiplier and the second the editpopulation
        fullMultipliers = []
        for element1 in multipliers[0]:
            for element2 in multipliers[1]:
                fullMultipliers.append([element1, element2])

        ## so element is [learningRate scaler, editpopulation scalar]
        processes = []
        counter = 0
        for element in fullMultipliers:

            counter += 1
            testNetwork = copy.deepcopy(self)
            testNetwork.learningRate = self.learningRate * element[0]
            testNetwork.geometryEditRate = self.geometryEditRate * element[1]
            testNetwork.reset()
            testNetwork.basicGeometryTrain(testSet) ## we can put anything in here
            testNetwork.reset()
            printProgressBar(counter, len(fullMultipliers))

            scores.append( testPerformance(testSet, testNetwork) )

            del testNetwork

        index = scores.index(max(scores))
        element = fullMultipliers[index]
        self.learningRate = self.learningRate * element[0]
        self.geometryEditRate = self.geometryEditRate * element[1]
        ## check here
        print("new learning rate", self.learningRate)
        print("new geometryEditRate", self.geometryEditRate)
        print("selected Score", scores[index])

    ## keep searching hyperparameter scores until one that is better than current is found
    ## this takes the normal defualt multiuplies list with two list elements
    def fastImprovementHyperParameterSearch(self, testSet):
        ##print("in deep param search:", defaultMultipliers)
        foundImprovement = False
        counter = 0

        ndefaultMultipliers = self.multipliers
        while (not foundImprovement):
            print(counter)
            if (counter == 0):
                hyperParamInfo = self.fastHyperParamaterSearch(testSet,  multiplierConvert(ndefaultMultipliers) )
                bestMultiplier = hyperParamInfo[0]
                ##print("bestMultiplier:", bestMultiplier)
                score = hyperParamInfo[1]
            elif (counter != 0):
                newMultipliers = []
                tmp1 = uniform(0,1)
                if (tmp1 > 0.5):
                    newEntry2 = uniform(1,ndefaultMultipliers[0][-1] * ndefaultMultipliers[0][-1] )
                else:
                    newEntry2 = uniform(ndefaultMultipliers[0][0] *ndefaultMultipliers[0][0], 1)

                tmp2 = uniform(0,1)
                if (tmp2 > 0.5):
                    newEntry1 = uniform(1,ndefaultMultipliers[1][-1] * ndefaultMultipliers[1][-1] )
                else:
                    newEntry1 = uniform(ndefaultMultipliers[1][0] *ndefaultMultipliers[1][0] ,1 )

                ## now we create our multipliers
                multiplierlist = [[newEntry2, newEntry1] ]
                for element in ndefaultMultipliers[0]:
                    multiplierlist.append([element, newEntry1])
                for element in ndefaultMultipliers[1]:
                    multiplierlist.append([newEntry2, element])

                hyperParamInfo = self.fastHyperParamaterSearch(testSet, multiplierlist)
                ##bestMultiplier = hyperParamInfo[0]
                score = hyperParamInfo[1]

                ndefaultMultipliers[0].append(newEntry2)
                ndefaultMultipliers[1].append(newEntry1)

                ##self.currentTrialScore = hyperParamInfo[1]
            if (score > self.currentTrialScore):
                bestMultiplier = hyperParamInfo[0]
                self.currentTrialScore = score
                ##print("getting here 1")

                if (score > self.currentScore):
                    ##bestMultiplier = hyperParamInfo[1]
                    self.currentScore = score
                    foundImprovement = True
                    ##print("getting here 2")

            if (counter >= 0):
                foundImprovement = True ## even through we are really just not wanting to search anymore
            counter += 1
        ##print("here is best multiplier", bestMultiplier)
        self.learningRate = self.learningRate * bestMultiplier[0]
        self.geometryEditRate = self.geometryEditRate * bestMultiplier[1]

        if (self.geometryEditRate < 1):
            self.geometryEditRate = 1

        print("self.learningRate:", self.learningRate)
        print("self.geometryEditRate:", self.geometryEditRate)
        print("selected score of:", self.currentTrialScore)
        print("score to beat of:", self.currentScore)
        self.currentTrialScore = -1

    def slowThoroughHyperParameterSearch(self, testSet):
        ##print("in deep param search:", defaultMultipliers)
        foundImprovement = False
        counter = 0

        ndefaultMultipliers = self.multipliers
        while (not foundImprovement):
            print(counter)
            if (counter == 0):
                hyperParamInfo = self.fastHyperParamaterSearch(testSet,  multiplierConvert(ndefaultMultipliers) )
                bestMultiplier = hyperParamInfo[0]
                ##print("bestMultiplier:", bestMultiplier)
                score = hyperParamInfo[1]
            elif (counter != 0):
                newMultipliers = []

                newEntry11 = ndefaultMultipliers[0][0] * ndefaultMultipliers[0][0]
                newEntry12 = ndefaultMultipliers[0][-1] * ndefaultMultipliers[0][-1]

                newEntry21 = ndefaultMultipliers[1][0] * ndefaultMultipliers[1][0]
                newEntry22 = ndefaultMultipliers[1][-1] * ndefaultMultipliers[1][-1]

                multiplierlist = [[newEntry11, newEntry21], [newEntry11, newEntry22], [newEntry12, newEntry21], [newEntry12, newEntry22]]
                ## now we create our multipliers
                for element in ndefaultMultipliers[0]:
                    multiplierlist.append([element, newEntry21])
                    multiplierlist.append([element, newEntry22])
                for element in ndefaultMultipliers[0]:
                    multiplierlist.append([newEntry11, element])
                    multiplierlist.append([newEntry12, element])

                hyperParamInfo = self.fastHyperParamaterSearch(testSet, multiplierlist)
                score = hyperParamInfo[1]

                ndefaultMultipliers[0].append(newEntry11)
                ndefaultMultipliers[0].append(newEntry12)
                ndefaultMultipliers[1].append(newEntry21)
                ndefaultMultipliers[1].append(newEntry22)

            if (score > self.currentTrialScore):
                bestMultiplier = hyperParamInfo[0]
                self.currentTrialScore = score
                if (score > self.currentScore):
                    ##bestMultiplier = hyperParamInfo[1]
                    self.currentScore = score
                    foundImprovement = True
                    ##print("getting here 2")

            if (counter >= 5):
                foundImprovement = True ## even through we are really just not wanting to search anymore

            counter += 1
        ##print("here is best multiplier", bestMultiplier)
        self.learningRate = self.learningRate * bestMultiplier[0]
        self.geometryEditRate = self.geometryEditRate * bestMultiplier[1]

        if (self.geometryEditRate < 1):
            self.geometryEditRate = 1

        print("self.learningRate:", self.learningRate)
        print("self.geometryEditRate:", self.geometryEditRate)
        print("selected score of:", self.currentTrialScore)
        print("score to beat of:", self.currentScore)
        self.currentTrialScore = -1

    ## changing this again. multipliers is now a list of lists that are the combos
    def fastHyperParamaterSearch(self, testSet, multipliers):
        ##print("in fast param search:", multipliers)
        ##print("finding hyperparameters")
        randomSeed = random()
        fullMultipliers = []
        poolInput = []
        marker = 0

        for element in multipliers:
            fullMultipliers.append(element)
            testNetwork = copy.deepcopy(self)
            testNetwork.learningRate = self.learningRate * element[0]
            testNetwork.geometryEditRate = self.geometryEditRate * element[1]

            ##if (testNetwork.geometryEditRate < 1):
                ##testNetwork.geometryEditRate = 1

            poolInput.append( (testNetwork, testSet, element, randomSeed, marker ) )
            marker += 1

        ##print("boopx")
        cores = mp.cpu_count()
        processNumber = len(fullMultipliers)
        if (len(fullMultipliers) > cores):
            processNumber = cores
        p= mp.Pool(processNumber)
        performance = p.map(self.fastHyperParamaterSearchHelper2, poolInput)

        bestScore = - 1000
        bestMultiplier = None

        for element in performance:
            if (element[0] > bestScore ):
                bestScore = element[0]
                bestMultiplier = element[1]
        ##self.currentTrialScore = bestScore

        ##self = bestNetwork

        ##print("boop2")
        seed(randomSeed)
        ##print("here is what fastHyperParamaterSearch is returning")
        ##print([bestMultiplier, bestScore])
        return [bestMultiplier, bestScore]

    def fastHyperParamaterSearchHelper2(self, args):
        return self.fastHyperParamaterSearchHelper(*args)

    def fastHyperParamaterSearchHelper(self, testNetwork, testSet, multiplier, randomSeed, marker):
        ##print("got here")
        ##time.sleep(5)
        seed(randomSeed)
        testNetwork.basicGeometryTrain(testSet, marker)
        print("completed process")
        ## this should be the last entry
        return [testNetwork.smoothedScores[-1:][0], multiplier]

    ## adds a specified neuron with specified connections to the network
    def addNeuron(self, position, backwardsConnections, forwardsConnections, bias):
        if (self.validAddCheck(position, backwardsConnections, forwardsConnections) == False):
            print("neuron invaid")
            return None
        else:
            backwardsNeurons = []
            forwardsNeurons = []

            for element in backwardsConnections:
                backwardsNeurons.append(element[0])

            for element in forwardsConnections:
                forwardsNeurons.append(element[0])

            self.directConnectFilter(backwardsNeurons, forwardsNeurons)

            self.currentConnections += len(backwardsConnections) + len(forwardsConnections)
            self.neuronNumber += 1
            ## if no problems then we add the neuron
            newNeuron = Neuron(position, bias)
            newNeuron.numConnectionsTo += len(backwardsConnections)
            newNeuron.connections = forwardsConnections
            try:
                self.neuronDict[position].append(newNeuron)
            except:
                self.neuronDict[position] = [newNeuron]

            for index in range(len(backwardsConnections)):
                (backwardsConnections[index][0].connections).append([newNeuron, backwardsConnections[index][1]])

            if (len(self.neuronDict[position]) == 1):
                self.respace()

    ## helper function to addNeuron
    ## returns False if proposed neuron placement interconnections within its layer and True otherwise
    def validAddCheck(self, position, backwardsConnections, forwardsConnections):
        ## backwards and forwards connections follow the connection standards as outlined in Standards
        for element in backwardsConnections:
            if (element[0].position == position):
                return False
        for element in forwardsConnections:
            if (element[0].position == position):
                return False
        return True

    ## removes a specified neuron from the network
    def deleteNeuron(self, deleteNeuron):
        ## first we have to find where the neuron is
        position = deleteNeuron.position
        if (position == inputPos or position == outputPos):
            print("Tried to delete input or output layer neuron")
            print("aborting deletion")
            time.sleep(5)
            return None
        ## update the connections to parameter
        for element in deleteNeuron.connections:
            element[0].numConnectionsTo -= 1
            self.currentConnections -= 1

        keys = getKeys(self.neuronDict)

        ## need to delete references to the deleteNeuron
        ##for location in range(position):

        for location in keys:
            if (position > location):
                for neuron in self.neuronDict[location]:
                    copyConnections = neuron.connections.copy()
                    for element in copyConnections:
                        if (element[0] == deleteNeuron):
                            neuron.connections.remove( [element[0], element[1]] )
                            self.currentConnections -= 1

        ## then we delete the neuron from the network
        self.neuronNumber -= 1
        self.neuronDict[position].remove(deleteNeuron)

        if ( len( self.neuronDict[position] ) == 0 ):
            del self.neuronDict[position]
            self.respace()

    ## adds a random neuron to the network
    def addRandomNeuron(self, backwardsConnectionNumber, forwardsConnectionNumber):
        ## first we decide where to place the random neuron
        foundValid = False
        while (not foundValid):
            position = uniform(inputPos, outputPos)
            if (position != inputPos and position != outputPos):
                foundValid = True

        backwardsConnections = []
        forwardsConnections = []

        backwardsNeurons = []
        forwardsNeurons = []

        keys = getKeys(self.neuronDict)
        for key in keys:
            for neuron in self.neuronDict[key]:
                if (key < position):
                    backwardsNeurons.append(neuron)
                elif (key > position):
                    forwardsNeurons.append(neuron)

        if (len(forwardsNeurons) > forwardsConnectionNumber):
            forwardsNeurons = sample(forwardsNeurons, forwardsConnectionNumber)

        if (len(backwardsNeurons) > backwardsConnectionNumber):
            backwardsNeurons = sample(backwardsNeurons, backwardsConnectionNumber)

        for index in range(len(forwardsNeurons)):
            forwardsConnections.append( [ forwardsNeurons[index], uniform(-1,1) ] )

        for index in range(len(backwardsNeurons)):
            backwardsConnections.append( [ backwardsNeurons[index], uniform(-1,1) ] )

        ## adding the neuron to the network
        self.addNeuron(position, backwardsConnections, forwardsConnections, uniform(-1,1) )

    ## trains while editing the hyperparameters and geometry of the network
    def geometryTrain(self, dataSet, learningRateAdjustNumber):
        ## dataSet contains elements that are lists of inputs and outputs
        print("commencing training")
        counter = 0
        dataSetLength = len(dataSet)
        adjustInterval = math.floor(dataSetLength/learningRateAdjustNumber)
        self.reset()

        thingo = self.multipliers.copy()
        print("here is thingo where initialized:", thingo)

        for element in dataSet:

            input = element[0]
            expectedOutput = element[1]
            eval = self.evaluate(input)

            if (testCorrect(eval, expectedOutput)):
                self.pastScores.append(1)
            else:
                self.pastScores.append(0)

            self.smoothedScores.append( movingAverage(self.pastScores, self.movingAverageLookBack) )
            self.pastNeuronNumbers.append(self.neuronNumber - self.inputSize - self.outputSize)

            self.pastConnections.append(self.currentConnections)





            try:
                if (counter % 10 == 0):
                    plt.plot(self.smoothedScores)
                    plt.savefig('scores1.png')
                    plt.clf()
                    plt.plot(self.pastNeuronNumbers)
                    plt.savefig('NeuronNumber1.png')
                    plt.clf()
                    plt.plot(self.pastConnections)
                    plt.savefig('connectionNumber' + '.png')
                    plt.clf()
            except:
                print("",end="")

            self.backPropigate(eval, expectedOutput)
            self.reset()
            printProgressBar(counter, dataSetLength)

            if (counter % adjustInterval == 0 and counter != 0):
            ##if (counter == 0):
                ##print(self.failGen)
                ##self.hyperParameterSearch( sample(dataSet, self.hyperparameterLookBack) , multipliers)
                ##self.fastHyperParamaterSearch( sample(dataSet, self.hyperparameterLookBack) , multipliers )
                ##nmultipliers = multipliers.copy()

                self.multipliers = copy.deepcopy(self.multiplierCopy)
                print("here is multipliers before evaluation:",self.multipliers)
                print("here is the constant before evaluations:", thingo)
                print("here is self.multiplierCopy", self.multiplierCopy)
                ##self.slowThoroughHyperParameterSearch( dataSet[counter: counter + self.hyperparameterLookBack] )
                self.fastImprovementHyperParameterSearch( dataSet[counter:counter+self.hyperparameterLookBack])
                print("here is multipliers after evaluation:",self.multipliers)
                print("here is the contant after evaluation:", thingo)
            ##print(counter % dataSetLengthPerHundred)

            ## this is what I got to work on
            if (counter % (math.ceil(self.editPeriod)) == 0 and counter != 0):
                print("altering geometry")
                self.geometryEdit()
                print("number of network Neurons:", self.neuronNumber)
                print("done altering geometry")
            counter += 1
        print("training complete")

    ## geometryTrain without the addition of adjusting the hyperParameters
    def basicGeometryTrain(self, dataSet, marker = None):
        ## dataSet contains elements that are lists of inputs and outputs
        ##print("commencing training")
        counter = 0
        dataSetLength = len(dataSet)
        self.reset()

        for element in dataSet:
            input = element[0]
            expectedOutput = element[1]
            eval = self.evaluate(input)

            if (testCorrect(eval, expectedOutput)):
                self.pastScores.append(1)
            else:
                self.pastScores.append(0)

            self.smoothedScores.append( movingAverage(self.pastScores, self.movingAverageLookBack) )
            self.pastNeuronNumbers.append(self.neuronNumber - self.inputSize - self.outputSize)
            if (marker == [1,1]):
                printProgressBar(counter , len(dataSet))
                ##plt.plot(self.smoothedScores)
                ##plt.savefig('basicPlot' + str(marker) + '.png')
                ##plt.clf()

            self.backPropigate(eval, expectedOutput)
            self.reset()
                ##self.adjustLearningRate( sample(dataSet, 20) , multipliers)
            #print(counter)
            ##print(counter % dataSetLengthPerHundred)

            if (counter % (math.ceil(self.editPeriod)) == 0 and counter != 0):
                ##print("altering geometry")
                self.geometryEdit()
                ##print("number of network Neurons:", self.neuronNumber)
                ##print("done altering geometry")
            ##printProgressBar(counter, dataSetLength)
            counter += 1

    def simGeometryTrain(self, dataSet, return_dict, marker = None):
        ## dataSet contains elements that are lists of inputs and outputs
        ##print("commencing training
        counter = 0
        dataSetLength = len(dataSet)
        self.reset()

        for element in dataSet:
            input = element[0]
            expectedOutput = element[1]
            eval = self.evaluate(input)

            if (testCorrect(eval, expectedOutput)):
                self.pastScores.append(1)
            else:
                self.pastScores.append(0)

            self.smoothedScores.append( movingAverage(self.pastScores, self.movingAverageLookBack) )
            self.pastNeuronNumbers.append( self.neuronNumber - self.inputSize - self.outputSize )
            self.pastConnections.append( self.currentConnections )
            ## to show progress

            ##if (marker == "standard"):
                ##printProgressBar(counter , len(dataSet))

            printProgressBar(counter , len(dataSet))

            self.backPropigate(eval, expectedOutput)
            self.reset()
                ##self.adjustLearningRate( sample(dataSet, 20) , multipliers)
            #print(counter)
            ##print(counter % dataSetLengthPerHundred)

            if (counter % (math.ceil(self.editPeriod)) == 0 and counter != 0):
                self.geometryEdit()
            counter += 1

        ##self.learningRate = self.learningRate * 0.1
        self.basicTrain( dataSet, marker )
        ##self.learningRate = self.learningRate * 10
        return_dict[self] = []

    ## an attempt at a faster version of simGeometryTrain
    def fastSimGeometryTrain(self, dataSet, return_dict, marker = None):
        ## dataSet contains elements that are lists of inputs and outputs
        ##print("commencing training")
        counter = 0
        dataSetLength = len(dataSet)
        self.reset()
        evals = []
        expectedOutputs = []

        for element in dataSet:
            input = element[0]
            expectedOutput = element[1]
            eval = self.evaluate(input)
            evals.append(eval)
            expectedOutputs.append(expectedOutput)

            if (testCorrect(eval, expectedOutput)):
                self.pastScores.append(1)
            else:
                self.pastScores.append(0)

            self.smoothedScores.append( movingAverage(self.pastScores, self.movingAverageLookBack) )
            self.pastNeuronNumbers.append( self.neuronNumber - self.inputSize - self.outputSize )
            self.pastConnections.append( self.currentConnections )
            ## to show progress
            ##if (marker == [1,1]):
            printProgressBar(counter , len(dataSet))

            if (counter % 10 == 0 and counter != 0):
                self.fastBackPropigate(evals, expectedOutputs)

                evals = []
                expectedOutputs = []
            self.reset()

            if (counter % (math.ceil(self.editPeriod)) == 0 and counter != 0):
                ##print("altering geometry")
                self.geometryEdit()
                ##print("number of network Neurons:", self.neuronNumber)
                ##print("done altering geometry")
            ##printProgressBar(counter, dataSetLength)

            counter += 1

        print("")
        self.fastBasicTrain( dataSet, marker )


        ##self.basicTrain( dataSet, marker)
        return_dict[self] = []

    ## these functions need to be changed for a more probabilistic method for generation
    ## really its only geometry edit that needs to be changed

    ## returns 0 to 1, 0 if geometry should NOT be changed and 1 if it should be changed
    def geometryChange(self, lookBackPeriod):
        if (lookBackPeriod > len(self.smoothedScores)):
            lookBackPeriod = len(self.smoothedScores)
        filteredScores = self.smoothedScores[-lookBackPeriod:]
        getLinearPoperties(self, filteredScores)
        ##slopeScore = unitLinear(1- ((self.slopeSensitivity) * abs(self.smoothedSlope)) )
        slopeScore = 1 - smoothedUnitLinear((self.slopeSensitivity) * abs(self.smoothedSlope))
        ##slopeScore = 1 - smoothedUnitLinear()


        self.currentGeometryEdit = smoothedUnitLinear(slopeScore * self.geometryEditRate * self.r_SQRD)
        self.pastGeometryEdit.append(self.currentGeometryEdit)

        return self.currentGeometryEdit

    ## returns 0-1, 1 if 100 percent of neurons to be editted should be additions
    ## and 0 if 0% of neurons to be editted should be additions to the network.

    ## returns the probability that we will generate a neuron. The not of
    ## of generating a neuron in this case is the elimination of the neurom
    def geometryDecide(self):
        if (self.mean == self.target):
            return self.genDecideBias
        elif (self.mean < self.target):
            return ( ( self.mean * (self.genDecideBias-1)/self.target ) + 1)
            ##return (1 - self.mean/(2*self.target))
        else:
            return ((self.mean - 1)/(self.target - 1))* self.genDecideBias

    ## when called will edit the geometry of the neural network
    def geometryEdit(self):
        randomNum = uniform(0,1)
        geometryChangeScore = self.geometryChange(self.linearRegressionLookBack)
        ##print("geometryChange propability:", geometryChangeScore)
        if (randomNum > geometryChangeScore):
            return None
        else:
            randomNum = uniform(0,1)
            decideScore = self.geometryDecide()
            self.currentDecide = decideScore
            self.pastDecide.append(self.currentDecide)

            ##print("decideScore probaility:", decideScore)
            if (randomNum < decideScore):
                scoreRanking = self.biasScoreRanking()
                ##self.generateBestNeuron(self.inputSize, math.floor( (self.inputSize + self.outputSize)/2 ), scoreRanking) ## Done
                self.generateBestNeuron(self.neuronInputSize, self.neuronOutputSize, scoreRanking) ## Done
            else:
                numConnectionsToDelete = math.floor((self.neuronInputSize + self.neuronOutputSize) * 1.0)
                ##print("degenerate moment")
                self.degenerate( numConnectionsToDelete )

    ## when called removes the worst connections within the network
    ## and neurons if the removal of connections makes the neuron
    ## useless
    def degenerate(self, numConnectionsToDelete):

        keys = getKeys(self.neuronDict)

        connectionList = []
        indexesToDelete = []
        indexCounter = 0
        scoresAndIndexes = []

        for key in keys:
            for neuron in self.neuronDict[key]:
                for element in neuron.connections:
                    ##deltaScore = 1 - unitLinear(5 * abs(element[0].delta))
                    deltaScore = 1 - smoothedUnitLinear(5 * abs(element[0].delta))
                    score = abs(deltaScore * element[1])
                    scoresAndIndexes.append([score, indexCounter])
                    indexCounter += 1

        scoresAndIndexes.sort()
        indexesToDelete = []
        for element in scoresAndIndexes[0:numConnectionsToDelete]:
            indexesToDelete.append(element[1])
        ## now as we go through we only need to check the first element
        ## this drastically improves the speed
        indexesToDelete.sort()

        ## now that we have our indexes to get rid of we can look through
        ## all the connections again to get rid of them
        indexCounter = 0
        for key in keys:
            for neuron in self.neuronDict[key]:

                newNeuronConnections = []
                for element in neuron.connections:
                    if (len(indexesToDelete) != 0):
                        if (indexCounter == indexesToDelete[0]):
                            del indexesToDelete[0]
                            self.currentConnections -= 1
                        else:
                            newNeuronConnections.append(element)

                    else:
                        newNeuronConnections.append(element)
                    indexCounter += 1
                neuron.connections = newNeuronConnections

        ## now we need to look through again and delete all neurons with the
        ## following properties
        ## 1) neuron position != 100 and neuron.connections is empty
        ## 2) neuron positiuon != 0 and no neurons connection to it

        neuronsToRemove = []
        for key in keys:
            for neuron in self.neuronDict[key]:
                if (neuron.position != 100 and neuron.position != 0 and len(neuron.connections) == 0):
                    neuronsToRemove.append(neuron)
                elif (neuron.numConnectionsTo == 0 and neuron.position != 0 and neuron.position != 100):
                    neuronsToRemove.append(neuron)

        for neuron in neuronsToRemove:
            ##print("deleting neuron")
            self.deleteNeuron(neuron)

    ## probabilistically adds the best neuron to the network
    def generateBestNeuron(self, backwardsConnectionSize, forwardsConnectionSize, scoreRanking):
        inputSize = backwardsConnectionSize
        outputSize = math.floor((backwardsConnectionSize + forwardsConnectionSize)/2)
        sampleSize = inputSize + outputSize

        searchingForConnectionPool = True
        counter = 0
        while searchingForConnectionPool:
            ## connectionPool is the set of all the neurons that we want to connect to
            connectionPool = randomWeightedSelection(scoreRanking, sampleSize) ## Done

            ## we first need to find a location for the new neuron to sit.
            ## we get the locations for all of these neurons
            locations = []
            for neuron in connectionPool:
                locations.append(neuron.position)

            ## locations are now sorted from smallest to largest
            locations.sort()
            ##print(locations[0])
            ##print(locations[len(locations)-1])
            if (locations[0] == locations[len(locations)-1]):
                ##print("problem in generateBestNeuron")
                ##time.sleep(100)
                searchingForConnectionPool = True
            else:

                searchingForConnectionPool = False
            counter += 1
            if (counter % 100 == 0):
                print("while loop stuck")
                ##print(searchingForConnectionPool)
                time.sleep(100)
        idealPosition = placeNeuron(locations, inputSize, outputSize)

        ## now we have ideal position which is a valid position for the new neuron

        backwardsConnections = []
        forwardsConnections = []

        backwardsNeurons = []
        forwardsNeurons = []

        for neuron in connectionPool:
            if (neuron.position < idealPosition):
                backwardsConnections.append([neuron, uniform(-1,1)])
                backwardsNeurons.append(neuron)

            elif (neuron.position > idealPosition):
                forwardsConnections.append([neuron, uniform(-1,1)])
                forwardsNeurons.append(neuron)
            else:
                print(idealPosition, neuron.position)
                print("tried to generate an invalid neuron")
                time.sleep(100)

        self.addNeuron(idealPosition, backwardsConnections, forwardsConnections, uniform(-1,1) )

    ## this function takes two sets of neurons. THe first of which all have lower positions compaired to the second.
    ## this function deletes all connections between these two sets
    def directConnectFilter(self, firstNeuronSet, secondNeuronSet):
        return None

        for neuron in firstNeuronSet:
            connectionKeepList = []
            for element in neuron.connections:
                if (not (element[0] in secondNeuronSet)):
                    connectionKeepList.append(element)
                else:
                    self.currentConnections -= 1
            neuron.connections = connectionKeepList

    ## produces a list with elements that are [neuron score, neuron].
    ## it also sorts it from best to worst neurons
    def scoreRanking(self):
        rtn = []
        keys = getKeys(self.neuronDict)
        counter = 0
        for key in keys:
            for neuron in self.neuronDict[key]:
                counter += 1
                rtn.append( [neuron.score, counter, neuron] )
        rtn.sort()
        rtn.reverse()

        return rtn

    ## has a bias towards ranking neurons higher that are not input or output neurons
    def biasScoreRanking(self):
        rtn = []
        keys = getKeys(self.neuronDict)
        counter = 0
        for key in keys:
            for neuron in self.neuronDict[key]:
                counter += 1
                if (neuron.position > inputPos + 0.00001):
                    rtn.append( [neuron.score * self.selectionBias, counter, neuron] )
                else:
                    rtn.append( [neuron.score, counter, neuron] )
        rtn.sort()
        rtn.reverse()

        return rtn

    ## respaces the network to make it both easier to see and prevent bunching
    ## within the network
    def respace(self):
        ## we first need to find out how many layers there are in the network
        layerCount = 0
        keys = getKeys(self.neuronDict)
        layerCount = len(keys)
        space = outputPos - inputPos
        ## now we have to distribute layerCount accross a length space as evenly as possible
        dSpace = space/(layerCount - 1)
        newNeuronDict = dict()
        for index in range(len(keys)):
            newPos = dSpace * index
            if (index == inputPos):
                newPos = inputPos
            if (index == len(keys) - 1):
                newPos = outputPos

            for neuron in self.neuronDict[keys[index]]:
                neuron.position = newPos

            oldEntry = self.neuronDict[keys[index]]
            del self.neuronDict[keys[index]]
            self.neuronDict[newPos] = oldEntry

    ## NEXT BIG PROJECT:
    ## giving neurons plasticity
