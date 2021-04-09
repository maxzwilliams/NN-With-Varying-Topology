from Neuron import *
from random import *
from Helper import *
import time
import statistics

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

class lightNN:

    def __init__(self, bigNet):
        self.marker = marker

        self.span = bigNet.span
        self.repeatBase = bigNet.span


        self.sessionTrainingData = bigNet.span

        ## learning properties
        self.learningRate = bigNet.learningRate
        ##self.hyperStep = 0.0001
        self.hyperStep = bigNet.hyperStep
        self.geometryEditRate = bigNet.geometryEditRate
        self.originalLearningRate = bigNet.originalLearningRate
        ## changes

        ## tested past performance
        self.testPerformance = bigNet.testPerformance

        ## accuracy that the network is aiming for
        self.target = bigNet.target

        ## period of training data points between potential network generations
        ##self.editPeriod = 10

        ## the probability of adding a neuron when at target
        self.genDecideBias = bigNet.genDecideBias

        ## changes how the slope changes how new neurons are generated
        self.slopeSensitivity = bigNet.slopeSensitivity

        ## hyperParameter which dictates which neurons are more connected too
        ## a higher selectionBias inducates that connections between hidden neurons
        ## is more likely than connections to or from input or output neurons
        ## higher selectionBias numbers are suited to more complex systems
        ## and lower selectionBias is for less complex systems that require less
        ## hidden connections


        self.selectionBias = bigNet.selectionBias

        ## current score of the network while training
        self.currentScore = bigNet.currentScore
        ## best score predicted by a simulation
        self.currentTrialScore = bigNet.currentTrialScore


        ## geometry propeties
        self.inputSize = bigNet.inputSize
        self.outputSize = bigNet.outputSize

        ## when used these are the hiddenlayer sizes
        self.hiddenSize = 100
        self.hiddenSize2 = 50

        ## number of neurons in the network including input and output neurons
        self.neuronNumber = self.inputSize + self.outputSize

        ## number of random neurons we start with
        self.startingBatchSize = 100

        ## self.neuronInputSize = 100
        ## self.neuronOutputSize = 50

        self.neuronInputSize = math.floor((2 * self.inputSize + self.outputSize)/3)
        self.neuronOutputSize = math.floor((self.inputSize + 2 * self.outputSize)/3)

        ##self.neuronInputSize = 10
        ##self.neuronOutputSize = 5

        ## dataStructure that stores all neurons in the network. This completely
        ## defines the network
        self.neuronDict = bigNet.neuronDict

        ## geometry editing scores
        ## past correct scores (they are either 0 or 1)
        self.pastScores = bigNet.pastScores
        ## pastscores smoothed by a moving average
        self.smoothedScores = bigNet.smoothedScores

        ## keeps track of how many neurons were in the network over time
        self.pastNeuronNumbers = bigNet.pastNeuronNumbers

        ## keeps track of the number of connections in the network
        self.pastConnections = bigNet.pastConnections
        self.currentConnections = bigNet.currentConnections

        ##
        self.currentGeometryEdit = bigNet.currentGeometryEdit
        self.pastGeometryEdit = bigNet.pastGeometryEdit

        self.currentDecide = bigNet.currentDecide
        self.pastDecide = bigNet.pastDecide

        ## the number of most recent data points that the moving average takes
        self.movingAverageLookBack = bigNet.movingAverageLookBack

        ## how far the hyperParameter simulations go for
        self.hyperparameterLookBack = bigNet.hyperparameterLookBack
        ## how far back our linear fit looks
        self.linearRegressionLookBack = bigNet.linearRegressionLookBack

class NN:

    def __init__(self, dataSet, learningRate, geometryEditRate, printNetworkInfo = False, marker = None):
        print("initializing a new network")
        self.marker = marker

        self.span = 10
        self.repeatBase = 0.1 ## the fraction of past wrong answers we use


        self.sessionTrainingData = []

        ## learning properties
        self.learningRate = learningRate
        ##self.hyperStep = 0.0001
        self.hyperStep = 0
        self.geometryEditRate = geometryEditRate
        self.originalLearningRate = self.learningRate
        ## changes

        ## tested past performance
        self.testPerformance = 0.5

        ## accuracy that the network is aiming for
        self.target = 1.0

        ## period of training data points between potential network generations
        self.editPeriod = 10

        ## the probability of adding a neuron when at target
        self.genDecideBias = 0.90

        ## changes how the slope changes how new neurons are generated
        self.slopeSensitivity = 10**3

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
        self.dataSet = dataSet

        ## when used these are the hiddenlayer sizes
        self.hiddenSize = 100
        self.hiddenSize2 = 50

        ## number of neurons in the network including input and output neurons
        self.neuronNumber = self.inputSize + self.outputSize

        ## number of random neurons we start with
        self.startingBatchSize = 50

        ## self.neuronInputSize = 100
        ## self.neuronOutputSize = 50

        self.neuronInputSize = math.floor((2 * self.inputSize + self.outputSize)/3)
        self.neuronOutputSize = math.floor((self.inputSize + 2 * self.outputSize)/3)

        ##self.neuronInputSize = 10
        ##self.neuronOutputSize = 5

        ## dataStructure that stores all neurons in the network. This completely
        ## defines the network
        self.neuronDict = dict()

        ## marker dict
        ## this dictionary has keys that are neuron markers (unique)
        ## with entries that are neurons so that connections can store only markers
        self.markerNeuronDict = dict()


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
        self.movingAverageLookBack = 100

        ## how far the hyperParameter simulations go for
        self.hyperparameterLookBack = 100
        ## how far back our linear fit looks
        self.linearRegressionLookBack = 100


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
        print("Currently the best score is ~95 % from overnight training on the whole dataSet")
        print("this program uses a sigmoid activation function for all neurons")
        ##print("this also has improved speed training by a factor of 10")
        print("----------------------------------------------------------------------------------------------------")

    ## will generate an initial amount of randomly placed neurons when called
    def randomGenerate(self):
        for index in range(self.startingBatchSize):
            ##self.addRandomNeuron(self.inputSize, math.floor((self.outputSize+self.inputSize)/2))
            self.addRandomNeuron()
            print("generated Neuron")

    def cleanClone(self):
        rtn = [self.neuronDict, self.marker, self.learningRate, self.hyperStep, self.geometryEditRate,
        self.originalLearningRate, self.testPerformance, self.target, self.genDecideBias, self.slopeSensitivity
        , self.markerNeuronDict, self.span]
        return rtn

    def replicate(self, c):
        self.neuronDict = c[0]
        self.marker = c[1]
        self.learningRate = c[2]
        self.hyperStep = c[3]
        self.geometryEditRate = c[4]
        self.originalLearningRate = c[5]
        self.testPerformance = c[6]
        self.target = c[7]
        self.genDecideBias = c[8]
        self.slopeSensitivity = c[9]
        self.markerNeuronDict = c[10]
        self.span = c[11]

    ## generate initial network structure
    def generateInitialNetwork(self):
        self.generateOutputLayer()
        #self.hiddenLayer2()
        #self.hiddenLayer()
        self.generateInputLayer()
        print("print running random generation")
        self.randomGenerate()

    def generateInputLayer(self):
        layer = []
        for index1 in range(self.inputSize):
            ##newNeuron = Neuron(inputPos, uniform(-1,1), (getKeys(self.markerNeuronDict, False )) )
            newNeuron = Neuron(inputPos, uniform(-0.1,0.1), (getKeys(self.markerNeuronDict, False )) )
            newConnections = []
            ##for index2 in range(self.outputSize):
                ##newConnections.append([ self.neuronDict[outputPos][index2], uniform(-1,1)])
            newNeuron.connections = newConnections
            self.markerNeuronDict[newNeuron.marker] = newNeuron
            layer.append(newNeuron)
        self.neuronDict[inputPos] = layer

    def generateOutputLayer(self):
        layer = []
        for index1 in range(self.outputSize):
            newNeuron = Neuron(outputPos, uniform(-0.1,0.1) ,(getKeys(self.markerNeuronDict, False )), True)
            self.markerNeuronDict[newNeuron.marker] = newNeuron
            layer.append(newNeuron)
        self.neuronDict[outputPos] = layer

    ## takes and input and evaluates it using the network returning the result
    ## corrected for marker system
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

                        (self.markerNeuronDict[(self.neuronDict[key][index1].connections[index2][0])]).logit += self.neuronDict[key][index1].value * self.neuronDict[key][index1].connections[index2][1]
                         ##print("logit:",self.neuronDict[key][index1].connections[index2][0].logit)
                         ##sum += self.neuronDict[key][index1].value * self.neuronDict[key][index1].connections[index2][1]

                    ##self.neuronDict[key][index1].connections[index2][0].logit = sum
                else:
                    rtn.append(self.neuronDict[key][index1].value) ## this could be a problem here
        return self.normalise(rtn)
        ##return rtn

    def normalise(self, list):
        sum = 0
        for index in range(len(list)):
            if (list[index] > 0):
                sum += abs(list[index])
        for index in range(len(list)):
            if (list[index] < 0):
                list[index] = 0
            else:
                list[index] = list[index]/sum
        return list

    ## resets the states of each neuron in the network
    def reset(self):
        keys = getKeys(self.neuronDict)
        for key in keys:
            for neuron in self.neuronDict[key]:
                neuron.valueReset()

    ## general Backpropagation algorithm
    ## corrected for marker system
    def backPropigate(self, output, expectedOutput, loopCounter = None):
        ## lets get how correct we were
        resultError = 0
        for index in range(len(output)):
            resultError += abs(output[index] - expectedOutput[index])
        resultError = resultError/len(output) ## 0 if there was no error and higher number if there is

        correctIndex = expectedOutput.index(max(expectedOutput))
        estimatedIndex = output.index(max(output))

        if (correctIndex == estimatedIndex):
            correct = True
        else:
            correct = False

        ## algorithm from: https://en.wikipedia.org/wiki/Backpropagation
        keys = getKeys(self.neuronDict)
        keys.reverse()

        for index in range(len(self.neuronDict[outputPos])):
            ##self.neuronDict[outputPos][index].delta = (self.neuronDict[outputPos][index].value - expectedOutput[index]) * self.neuronDict[outputPos][index].derivativeActivationFunction(self.neuronDict[outputPos][index].value)
            ##self.neuronDict[outputPos][index].delta = (self.neuronDict[outputPos][index].value - expectedOutput[index]) * self.neuronDict[outputPos][index].derivativeActivationFunction( self.neuronDict[outputPos][index].logit ) ##[tick1]
            self.neuronDict[outputPos][index].delta = (output[index] - expectedOutput[index]) * self.neuronDict[outputPos][index].derivativeActivationFunction( self.neuronDict[outputPos][index].logit )

            score = smoothedUnitLinear(resultError) * abs(self.neuronDict[outputPos][index].value) * abs(self.neuronDict[outputPos][index].delta)
            self.neuronDict[outputPos][index].scoreUpdate( score )
            ##self.neuronDict[outputPos][index].scoreUpdate( smoothedUnitLinear(5*abs(self.neuronDict[outputPos][index].delta )) )
            self.neuronDict[outputPos][index].deltaScores.append( self.neuronDict[outputPos][index].delta )

        for key in keys[1:(len(keys))]:
            try:
                for index1 in range(len(self.neuronDict[key])):
                    sum = 0
                    for index2 in range(len(self.neuronDict[key][index1].connections)):
                        try:
                            sum += self.neuronDict[key][index1].connections[index2][1] * (self.markerNeuronDict[(self.neuronDict[key][index1].connections[index2][0])]).delta

                        except:
                            print(key, index1, index2)
                            print(sum)
                            print(self.neuronDict[key][index1].connections[index2][1])
                            print((self.markerNeuronDict[(self.neuronDict[key][index1].connections[index2][0])]).delta)

                            print("overflow")
                            time.sleep(100)
                    try:
                        self.neuronDict[key][index1].delta = sum * self.neuronDict[key][index1].derivativeActivationFunction( self.neuronDict[key][index1].logit )

                    except:
                        print("THE PROBLEM IS HERERERERE")
                        print(sum)
                        print(self.neuronDict[key][index1].logit)
                        print(self.neuronDict[key][index1].outputNeuron)
                        print(self.neuronDict[key][index1].derivativeActivationFunction( self.neuronDict[key][index1].logit ))
                        print("___________________________________")
                        ##self.neuronDict[key][index1].scores.append(abs( self.neuronDict[key][index1].delta ))
                        ##self.neuronDict[key][index1].scoreUpdate(smoothedUnitLinear(5*abs(self.neuronDict[index1][index].delta )))
                    score = smoothedUnitLinear(resultError) * abs(self.neuronDict[key][index1].value) * abs(self.neuronDict[key][index1].delta)
                    self.neuronDict[key][index1].scoreUpdate( score )
                    self.neuronDict[key][index1].deltaScores.append(self.neuronDict[key][index1].delta)


            except:
                print("Overflow error")
                print()
                time.sleep(10000)

        ## update the weights and biases
        for key in keys:
            for index1 in range(len(self.neuronDict[key])):
                dBias = self.neuronDict[key][index1].delta

                ##learningRateBuff = self.neuronDict[key][index1].getBuff(self.hyperStep, self.learningRate, self.originalLearningRate)
                learningRateBuff = self.neuronDict[key][index1].getBuff(self.hyperStep, self.learningRate, self.originalLearningRate) * self.neuronDict[key][index1].stableBuff
                biasReduce = 0 * self.neuronDict[key][index1].bias

                self.neuronDict[key][index1].bias += -self.learningRate * dBias * learningRateBuff
                biasReduce = 0.01 * abs(self.learningRate * dBias * learningRateBuff)
                if (self.learningRate * dBias * learningRateBuff > 1):
                    self.neuronDict[key][index1].bias -= biasReduce
                else:
                    self.neuronDict[key][index1].bias += biasReduce


                ##print(self.neuronDict[key][index1].connectionChanges)
                for index2 in range(len(self.neuronDict[key][index1].connections)):
                    ##learningRateBuffNext = self.neuronDict[key][index1].connections[index2][0].getBuff(self.hyperStep)
                    ##learningRateBuff = max(learningRateBuff, learningRateBuffNext)

                    learningRateBuff = max(learningRateBuff, (self.markerNeuronDict[(self.neuronDict[key][index1].connections[index2][0])]).currentBuff)


                    dWeight = self.neuronDict[key][index1].value * (self.markerNeuronDict[(self.neuronDict[key][index1].connections[index2][0])]).delta
                    #dBias = self.neuronDict[key][index1]

                    self.neuronDict[key][index1].connections[index2][1] += -self.learningRate * dWeight * learningRateBuff
                    ##weightReduce = 0.01*abs(self.learningRate * dWeight * learningRateBuff)
                    weightReduce = 0
                    if (self.neuronDict[key][index1].connections[index2][1] > 1):
                        self.neuronDict[key][index1].connections[index2][1] -= weightReduce
                    else:
                        self.neuronDict[key][index1].connections[index2][1] += weightReduce
                    ##currentWeightChanges.append( [ self.neuronDict[key][index1].connections[index2][0].marker , -self.learningRate * dWeight * learningRateBuff, learningRateBuff ] )

                ##self.neuronDict[key][index1].connectionChanges.append(currentWeightChanges)
                ##if ( len(self.neuronDict[key][index1].connectionChanges) > 2):
                    ##self.neuronDict[key][index1].connectionChanges = self.neuronDict[key][index1].connectionChanges[-2:]
                ##print(len(self.neuronDict[key][index1].connectionChanges))

        scoresAndNeurons = []
        scores = []

        counter = 0
        for key in keys:
            for neuron in self.neuronDict[key]:
                scoresAndNeurons.append([neuron.score * 10**10, counter, neuron])
                scores.append(neuron.score*10**10)
                counter += 1
        scoresAndNeurons.sort()

        try:
            SD = statistics.stdev(scores)
            mean = statistics.mean(scores)
        except:
            SD = 0
            mean = 0

        ##print(SD, mean)
        length = len(scoresAndNeurons)
        largestScore = -1
        for index in range(len(scoresAndNeurons)):
            if (SD != 0):
                zscore = (scoresAndNeurons[index][2].score - mean)/SD
            else:
                zscore = 0
            ##scoresAndNeurons[index][2].currentBuff = ( math.exp( (2/length) * math.log(span)) )**( index - (length/2))
            scoresAndNeurons[index][2].stableBuff = (1 + self.span)**(zscore)

    ## trains network on specified dataset without adapting learning rate or any other parameters
    def basicTrain(self, dataSet, marker = None, trainingTime = None):
        ## dataSet contains elements that are lists of inputs and outputs
        counter = 0
        dataSetLength = len(dataSet)
        startTime = time.time()
        ##retryBound = (1 + self.repeatBase) ** math.ceil(self.testPerformance/(1 - self.testPerformance))
        wrongList = []
        retryBound = 1
        for element in dataSet:
            input = element[0]
            expectedOutput = element[1]
            retryCounter = 0
            eval = self.evaluate(input)
            self.backPropigate(eval, expectedOutput)
            self.reset()
            counter += 1
            currentTime = time.time()
            if (marker != None):
                if (trainingTime == None):
                    printProgressBar(counter, dataSetLength)
                else:
                    printProgressBar(currentTime - startTime, trainingTime)

            ##eval2 = self.evaluate(input)
            ##if (testCorrect(eval2, expectedOutput) == False):
            ##wrongList.append(element)
            ##self.reset()

            if (trainingTime != None):
                if (currentTime - startTime > trainingTime):
                    break

    ## deleted unneeded past information clogging up the program
    def clean(self):
        if (len(self.pastScores) > 1000):
            self.pastScores = self.pastScores[-1000:]

        if (len(self.smoothedScores) > 1000):
            self.smoothedScores = self.smoothedScores[-1000:]

        if (len(self.pastNeuronNumbers) > 1000):
            self.pastNeuronNumbers = self.pastNeuronNumbers[-1000:]

        if (len(self.pastConnections) > 1000):
            self.pastConnections = self.pastConnections[-1000:]

        if (len(self.pastGeometryEdit) > 1000):
            self.pastGeometryEdit = self.pastGeometryEdit[-1000:]

        if (len(self.pastDecide) > 1000):
            self.pastDecide = self.pastDecide[-1000:]

    def addRandomNeuron(self):
        ## select a location
        position = ((inputPos + outputPos)/2) + uniform(-1,1)
        keys = getKeys(self.neuronDict, False)
        forwardsPool = []
        backwardsPool = []
        for key in keys:
            for neuron in self.neuronDict[key]:
                if (neuron.position > position):
                    forwardsPool.append([neuron.marker, uniform(-0.1,0.1)])
                else:
                    backwardsPool.append([neuron.marker, uniform(-0.1,0.1)])
        try:
            forwardsConnections = sample(forwardsPool, 500)
        except:
            forwardsConnections = forwardsPool
        try:
            backwardsConnections = sample(backwardsPool, 1000)
        except:
            backwardsConnections = backwardsPool

        self.respace()
        self.addNeuron(position, backwardsConnections, forwardsConnections, uniform(-0.1,0.1))

    ## adds a specified neuron with specified connections to the network
    ## marker system added
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

            ##self.directConnectFilter(backwardsNeurons, forwardsNeurons)

            self.currentConnections += len(backwardsConnections) + len(forwardsConnections)
            self.neuronNumber += 1
            ## if no problems then we add the neuron
            newNeuron = Neuron(position, bias, (getKeys(self.markerNeuronDict, False )))
            ##newNeuron.currentBuff =  self.originalLearningRate / self.learningRate
            newNeuron.currentBuff = 1

            newNeuron.numConnectionsTo += len(backwardsConnections)
            newNeuron.connections = forwardsConnections ## assumes functions that use addNeuron use marker system
            try:
                self.neuronDict[position].append(newNeuron)
            except:
                self.neuronDict[position] = [newNeuron]

            self.markerNeuronDict[newNeuron.marker] = newNeuron

            for index in range(len(backwardsConnections)):
                (self.markerNeuronDict[(backwardsConnections[index][0])].connections).append([newNeuron.marker, backwardsConnections[index][1]])


            if (len(self.neuronDict[position]) == 1):
                self.respace()

    ## helper function to addNeuron
    ## returns False if proposed neuron placement interconnections within its layer and True otherwise
    ## corrected for marker system
    def validAddCheck(self, position, backwardsConnections, forwardsConnections):
        ## backwards and forwards connections follow the connection standards as outlined in Standards
        for element in backwardsConnections:
            if (self.markerNeuronDict[(element[0])].position == position):
                return False
        for element in forwardsConnections:
            if (self.markerNeuronDict[(element[0])].position == position):
                return False
        return True

    ## removes a specified neuron from the network
    ## setup with marker system
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
            self.markerNeuronDict[element[0]].numConnectionsTo -= 1
            self.currentConnections -= 1

        keys = getKeys(self.neuronDict)

        ## need to delete references to the deleteNeuron
        ##for location in range(position):

        for location in keys:
            if (position > location):
                for neuron in self.neuronDict[location]:
                    copyConnections = neuron.connections.copy()
                    for element in copyConnections:
                        if (self.markerNeuronDict[element[0]] == deleteNeuron):
                            neuron.connections.remove( [element[0] , element[1]] )
                            self.currentConnections -= 1

        ## then we delete the neuron from the network
        self.neuronNumber -= 1
        self.neuronDict[position].remove(deleteNeuron)

        ## remove entry from marker dictionary
        del self.markerNeuronDict[deleteNeuron.marker]


        if ( len( self.neuronDict[position] ) == 0 ):
            del self.neuronDict[position]
            self.respace()

    def simGeometryTrain(self, dataSet, return_list, marker = None, geometryTrainTime = None, normalTrainTime = None):
        ## dataSet contains elements that are lists of inputs and outputs
        ##print("commencing training
        ##self.wrongList = []
        counter = 0
        dataSetLength = len(dataSet)
        self.reset()
        geometryTrainStart = time.time()

        retryBound = 10
        wrongList = []
        for element in dataSet:
            input = element[0]
            expectedOutput = element[1]
            gotCorrect = False
            retryCounter = 0
            retryBound = 1

            ##if (retryBound != 1):
            ##    retryBound = 1000
            ##retryBound = 100
                ##if (retryCounter >= 1):
                    ##print(retryCounter)
            eval = self.evaluate(input)
            self.backPropigate(eval, expectedOutput)
            self.reset()
                ##self.adjustLearningRate( sample(dataSet, 20) , multipliers)
                #print(counter)
                ##print(counter % dataSetLengthPerHundred)
            tmp = uniform(0,1)
            tmp=100
            if (counter != 0 and tmp < smoothedUnitLinear(self.geometryEditRate)):
                self.geometryEdit()
            counter += 1

                ##gotCorrect = True
                ## if we got a sample wrong we try to learn it over and over
            ##eval2 = self.evaluate(input)
            ##if (testCorrect(eval2, expectedOutput) == False):
            ##    wrongList.append(element)
            ##self.reset()
            wrongList.append(element)
            if (testCorrect(eval, expectedOutput)):
                self.pastScores.append(1)
            else:
                self.pastScores.append(0)

            self.currentConnections = networkSize(self)

            self.smoothedScores.append( movingAverage(self.pastScores, self.movingAverageLookBack) )
            self.pastNeuronNumbers.append( self.neuronNumber - self.inputSize - self.outputSize )
            self.pastConnections.append( self.currentConnections )

            ##if (counter % (math.ceil(self.geometryEditRate)) == 0 and counter != 0):
                ##self.geometryEdit()
            ##counter += 1
            geometryCurrentTime = time.time()

            if (geometryTrainTime == None):
                printProgressBar(counter , len(dataSet))
            else:
                printProgressBar(geometryCurrentTime - geometryTrainStart, geometryTrainTime)

            if (geometryTrainTime != None):
                if (geometryCurrentTime - geometryTrainStart > geometryTrainTime):
                    break

        ##self.learningRate = self.learningRate * 0.1
        ##shuffle(dataSet)
        ##self.wrongList = wrongList
        self.basicTrain(dataSet, marker, normalTrainTime)
        ##self.learningRate = self.learningRate * 10
        ##return_dict[self] = []
        ##self.clean()
        return_list.append(self)

    ## these functions need to be changed for a more probabilistic method for generation
    ## really its only geometry edit that needs to be changed

    ## returns 0 to 1, 0 if geometry should NOT be changed and 1 if it should be changed
    def geometryChange(self, lookBackPeriod):
        if (lookBackPeriod > len(self.smoothedScores)):
            lookBackPeriod = len(self.smoothedScores)
        filteredScores = self.smoothedScores[-lookBackPeriod:]
        getLinearPoperties(self, filteredScores)
        ##slopeScore = unitLinear(1- ((self.slopeSensitivity) * abs(self.smoothedSlope)) )
        try:
            self.currentGeometryEdit = 1 - smoothedUnitLinear( (self.slopeSensitivity) * abs(self.smoothedSlope) * self.r_SQRD )

        except:
            print("that wierd error occured")
            return 0
        self.pastGeometryEdit.append(self.currentGeometryEdit)
        return self.currentGeometryEdit

    ## returns 0-1, 1 if 100 percent of neurons to be editted should be additions
    ## and 0 if 0% of neurons to be editted should be additions to the network.

    ## returns the probability that we will generate a neuron. The not of
    ## of generating a neuron in this case is the elimination of the neurom
    def geometryDecide(self):

        return self.genDecideBias
        """
        if (self.mean == self.target):
            return self.genDecideBias
        elif (self.mean < self.target):
            return ( ( self.mean * (self.genDecideBias-1)/self.target ) + 1)
            ##return (1 - self.mean/(2*self.target))
        else:
            return ((self.mean - 1)/(self.target - 1))* self.genDecideBias
        """

    ## when called will edit the geometry of the neural network
    def geometryEdit(self):
        try:
            seed(self.seed)
        except:
            seed(0)
            self.seed = 0
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
            if (randomNum < decideScore):
                scoreRanking = self.biasScoreRanking()

                ##self.generateBestNeuron(self.neuronInputSize, self.neuronOutputSize, scoreRanking)
                ## lets have every new neuron connect in some way to every other current neuron
                ##self.generateBestNeuron(math.floor(self.neuronNumber/2), math.floor(self.neuronNumber/2), scoreRanking)
                self.generateBestNeuron(1000, 500, scoreRanking)

                ##self.generateBestNeuron()
            else:
                ##numConnectionsToDelete = self.neuronInputSize + self.neuronOutputSize
                print("deleting connections")
                numConnectionsToDelete = math.floor( 15 )

                ##math.floor( 0.01 * self.currentConnections)
                ##print("degenerate moment")
                self.degenerate( numConnectionsToDelete )
        self.seed += 1

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
                    ##deltaScore = smoothedUnitLinear(5 * abs(element[0].delta))
                    ##score = abs(deltaScore * element[1])
                    score = self.markerNeuronDict[element[0]].score
                    scoresAndIndexes.append([score, indexCounter])
                    indexCounter += 1

        scoresAndIndexes.sort()
        scoresAndIndexes.reverse()
        indexesToDelete = []
        for element in scoresAndIndexes[0:numConnectionsToDelete]:
            indexesToDelete.append(element[1])
        ## now as we go through we only need to check the first element
        ## this drastically improves the speed
        indexesToDelete.sort()

        ## now that we have our indexes to get rid of we can look through
        ## all the connections again to get rid of them
        ##initial = self.currentConnections
        indexCounter = 0

        initialCount = networkSize(self)
        for key in keys:
            for neuron in self.neuronDict[key]:

                newNeuronConnections = []
                initialConnections = len(neuron.connections)
                ##sum = 0
                for element in neuron.connections:
                    if (len(indexesToDelete) != 0):
                        if (indexCounter in indexesToDelete):
                            ##del indexesToDelete[0]
                            indexesToDelete.remove(indexCounter)
                            ##print("connection number decrease")
                            self.currentConnections -= 1
                            ##print("getting rid of a connection")
                        else:
                            newNeuronConnections.append(element)

                    else:
                        newNeuronConnections.append(element)
                    indexCounter += 1
                neuron.connections = newNeuronConnections
        ##print(self.currentConnections - initial)
        ## now we need to look through again and delete all neurons with the
        ## following properties
        ## 1) neuron position != 100 and neuron.connections is empty
        ## 2) neuron positiuon != 0 and no neurons connection to it
        ##print(networkSize(self) - initialCount)
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
                backwardsConnections.append([neuron.marker, uniform(-0.1,0.1) ])
                ##backwardsNeurons.append(neuron)

            elif (neuron.position > idealPosition):
                forwardsConnections.append([neuron.marker, uniform(-0.1,0.1) ])
                ##forwardsNeurons.append(neuron)
            else:
                print(idealPosition, neuron.position)
                print("tried to generate an invalid neuron")
                time.sleep(100)

        self.addNeuron(idealPosition, backwardsConnections, forwardsConnections, uniform(-0.1,0.1) )

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
