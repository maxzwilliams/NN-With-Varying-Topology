from Neuron import *
from random import *
from Helper import *
import time
import copy

import matplotlib.pyplot as plt
##import multiproccessing as mp
import multiprocessing as mp

seed(1)
## hardcoded positions of the input and output layers
inputPos = 0
outputPos = 100
hiddenPos = 50
hiddenPos2 = 75

class NN:

    def __init__(self, dataSet, learningRate, geometryEditRate):

        ## learning properties
        self.learningRate = learningRate
        self.geometryEditRate = geometryEditRate
        ## changes
        self.editPopulation = 0.01
        ## accuracy that the network is aiming for
        self.target = 0.99

        self.editPeriod = 50
        ##self.generationRate = generationRate
        self.genDecideBias = 0.60

        self.failGen = 0

        ## geometry propeties
        self.inputSize = len(dataSet[0][0])
        self.outputSize = len(dataSet[0][1])
        self.hiddenSize = 100
        self.hiddenSize2 = 50
        self.neuronNumber = self.inputSize + self.outputSize

        ## number of random neurons we start with
        self.startingBatchSize = 0

        ##math.floor((self.inputSize + self.outputSize)/2)
        self.neuronDict = dict()

        ## debugging stuff
        self.pastDeltas = []

        ## geometry editing scores
        ## past correct scores (they are either 0 or 1)
        self.pastScores = []
        self.smoothedScores = []

        self.hyperParameterScores = []

        self.pastNeuronNumbers = []

        self.movingAverageLookBack = 300

        self.hyperparameterLookBack = 500
        self.linearRegressionLookBack = self.hyperparameterLookBack

        ## initialNetworkState
        self.generateInitialNetwork()
        self.randomGenerate()


        self.informationPrintOut()

    def informationPrintOut(self):
        print("----------------------------------------------------------------------------------------------------")
        print("general stats:")
        print("self.learningRate", self.learningRate)
        print("self.editperiod", self.editPeriod)
        print("self.editPopulation", self.editPopulation)
        print("self.target", self.target)
        print("self.movingAverageLookBack", self.movingAverageLookBack)
        print("self.hyperparameterLookBack", self.hyperparameterLookBack)
        print("self.startingBatchSize", self.startingBatchSize)
        print("General Information:")
        print("Core learning of the network is all working")
        print("evaluation and backpropigation seem to be working")
        print("the only hyperparameter searched is learningRate")
        print("No plasticity introduced yet")
        print("no learningRate boosts are given for different neurons")
        print("A basic scoring system is introduced")
        print("Editing of the geometry seems to be working fully, however errors in this process can be really hard to find")
        print("this specific instance wont stop training until it acheives just under the target")
        print("this version also uses an adapted learning rate which gives new neurons a learningrate buff (currently this is x10 learning buff)")
        print("----------------------------------------------------------------------------------------------------")

    ## will generate an initial amount of randomly placed neurons when called
    def randomGenerate(self):
        for index in range(self.startingBatchSize):
            self.addRandomNeuron(self.inputSize, math.floor((self.outputSize+self.inputSize)/2))

    ## generate initial network structure
    def generateInitialNetwork(self):
        self.generateOutputLayer()
        #self.hiddenLayer2()
        #self.hiddenLayer()
        self.generateInputLayer()

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
        ##print("evaluating")
        ## first we evaluate the input layer

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
            self.neuronDict[outputPos][index].scoreUpdate(abs(self.neuronDict[outputPos][index].delta ))
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
                            print(self.neuronDict[key][index1].connections[index2][1])
                            print(self.neuronDict[key][index1].connections[index2][0].delta)

                            print("overflow")
                            time.sleep(100)
                    self.neuronDict[key][index1].delta = sum * self.neuronDict[key][index1].derivativeActivationFunction(self.neuronDict[key][index1].logit)
                    ##self.neuronDict[key][index1].scores.append(abs( self.neuronDict[key][index1].delta ))
                    self.neuronDict[key][index1].scoreUpdate(abs( self.neuronDict[key][index1].delta ) )
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
    ## could add multiproccessing to this one
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

    def fastHyperParamaterSearch(self, testSet, multipliers):
        print("finding hyperparameters")
        ##print("here now")

        ##print("boop1")
        randomSeed = random()
        fullMultipliers = []
        poolInput = []
        marker = 0
        for element1 in multipliers[0]:
            for element2 in multipliers[1]:
                fullMultipliers.append([element1, element2])
                testNetwork = copy.deepcopy(self)
                testNetwork.learningRate = self.learningRate * element1
                testNetwork.geometryEditRate = self.geometryEditRate * element2

                if (testNetwork.geometryEditRate < 1):
                    testNetwork.geometryEditRate = 1

                poolInput.append( (testNetwork, testSet, [element1, element2], randomSeed, marker ) )
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

        ##self = bestNetwork

        ##print("boop2")

        if (bestMultiplier != None):
            self.learningRate = self.learningRate * bestMultiplier[0]
            self.geometryEditRate = self.geometryEditRate * bestMultiplier[1]
            if (self.geometryEditRate < 1):
                self.geometryEditRate = 1

            print("self.learningRate:", self.learningRate)
            print("self.geometryEditRate:", self.geometryEditRate)
            print("predicted score of:", bestScore)
        else:
            print("multipler didnt work")
            time.sleep(10)
        seed(randomSeed)


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
        ##return [testPerformance(testSet, testNetwork), multiplier]
        ##scoreDict[multiplier] = testPerformance(testSet, testNetwork)


    ## adds a specified neuron with specified connections to the network
    def addNeuron(self, position, backwardsConnections, forwardsConnections, bias):
        if (self.validAddCheck(position, backwardsConnections, forwardsConnections) == False):
            return None
        else:
            self.neuronNumber += 1
            ## if no problems then we add the neuron
            newNeuron = Neuron(position, bias)
            newNeuron.connections = forwardsConnections
            try:
                self.neuronDict[position].append(newNeuron)
            except:
                self.neuronDict[position] = [newNeuron]

            for index in range(len(backwardsConnections)):
                (backwardsConnections[index][0].connections).append([newNeuron, backwardsConnections[index][1]])

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

        ## then we delete the neuron from the network
        self.neuronNumber -= 1
        self.neuronDict[position].remove(deleteNeuron)

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
    def geometryTrain(self, dataSet, learningRateAdjustNumber, multipliers):
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

            self.smoothedScores.append( movingAverage(self.pastScores, self.movingAverageLookBack) )
            self.pastNeuronNumbers.append(self.neuronNumber - self.inputSize - self.outputSize)
            try:
                if (counter % 10 == 0):
                    plt.plot(self.smoothedScores)
                    plt.savefig('scoresTest.png')
                    plt.clf()
                    plt.plot(self.pastNeuronNumbers)
                    plt.savefig('NeuronNumberTest.png')
                    plt.clf()
                    plt.plot(self.pastDeltas)
                    plt.savefig('someDeltasTest.png')
                    plt.clf()

            except:
                print("",end="")

            self.backPropigate(eval, expectedOutput)
            self.reset()
            printProgressBar(counter, dataSetLength)

            if (counter % adjustInterval == 0 and counter != 0):
            ##if (counter == 0):
                print(self.failGen)
                ##self.hyperParameterSearch( sample(dataSet, self.hyperparameterLookBack) , multipliers)
                ##self.fastHyperParamaterSearch( sample(dataSet, self.hyperparameterLookBack) , multipliers )
                self.fastHyperParamaterSearch( dataSet[counter:counter+self.hyperparameterLookBack] , multipliers )
            ##print(counter % dataSetLengthPerHundred)

            ## this is what I got to work on
            if (counter % (math.ceil(self.editPeriod)) == 0 and counter != 0):
                print("altering geometry")
                self.geometryEdit()
                print("number of network Neurons:", self.neuronNumber)
                print("done altering geometry")
            counter += 1
        print("training complete")


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
            ##self.pastNeuronNumbers.append(self.neuronNumber - self.inputSize - self.outputSize)
            ##if (marker != None and counter % 10 == 0):
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
        ##print("training complete")


    ## we need some functions that decides 1) if we should edit the geometry and 2) if
    ## such edit should result in the addition or deletion of neuron(s) 3) which neurons we should change




    ## these functions need to be changed for a more probabilistic method for generation
    ## really its only geometry edit that needs to be changed

    ## returns 0 to 1, 0 if geometry should NOT be changed and 1 if it should be changed
    def geometryChange(self, lookBackPeriod):
        if (lookBackPeriod > len(self.smoothedScores)):
            lookBackPeriod = len(self.smoothedScores)
        filteredScores = self.smoothedScores[-lookBackPeriod:]
        getLinearPoperties(self, filteredScores)
        slopeScore = unitLinear(1- ((10**4) * abs(self.smoothedSlope)))


        return unitLinear(slopeScore * self.geometryEditRate * self.r_SQRD)

        ##return (1 - unitLinear(self.geometryEditRate * self.smoothedSlope)) * self.r_SQRD
        ## altering self.geometryEditRate will control the edit rate on its own

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
            ##print("decideScore probaility:", decideScore)
            if (randomNum < decideScore):
                scoreRanking = self.scoreRanking()
                self.generateBestNeuron(self.inputSize, math.floor((self.inputSize + self.outputSize)/2), scoreRanking)
            else:
                self.smartDelete()

    ## simply take the worst performing neuron thats older than some fixed amount
    def smartDelete(self):
        minDeleteAge =  500
        maxDeleteAge = 2000
        deleteAge = math.floor(uniform(minDeleteAge, maxDeleteAge))
        ##deleteAge = 1000
        ## after age 1000 the neuron can be deleted
        worstNeuron = None
        worstScore = -1 ## the worst score is the highest

        keys = getKeys(self.neuronDict)

        for key in keys:
            for neuron in self.neuronDict[key]:
                if (neuron.score > worstScore and neuron.age > deleteAge):
                    if (key != inputPos and key != outputPos):
                        worstNeuron = neuron
                        worstScore = neuron.score

        if (worstNeuron != None):
            self.deleteNeuron(worstNeuron)

    ## helper function
    def test(self, neuronList, testNeuron):
        for neuron in neuronList:
            ## if they are at the same location
            if (neuron.position == testNeuron.position):
                ##print("trace1")
                return False
            ## if they are arlready connected
            ##if (self.neuronContains(neuron.connections, testNeuron) or self.neuronContains(testNeuron.connections, neuron)):
                ##print("trace2")
                ##return False
        return True

    def generateBestNeuron(self, backwardsConnectionSize, forwardsConnectionSize, scoreRanking):
        inputSize = backwardsConnectionSize
        outputSize = math.floor((backwardsConnectionSize + forwardsConnectionSize)/2)
        sampleSize = inputSize + outputSize

        searchingForConnectionPool = True
        counter = 0
        while searchingForConnectionPool:
            ## connectionPool is the set of all the neurons that we want to connect to
            connectionPool = randomWeightedSelection(scoreRanking, sampleSize)

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
        ## okay so now we have a valid pool
        """
        searchingForPosition = True
        ## True if we are going to search to the right if we need
        direction = True
        posL = inputSize - 1
        posR = inputSize
        counter = 1
        deltaPos = (locations[posR] - locations[posL])/2

        while searchingForPosition:
            searchingForPosition = False
            ## test if that location is already taken within the ppol
            idealPosition = (locations[posR] + locations[posL])/2

            for location in locations:
                ##print("in the loop")
                if (idealPosition == location):
                    searchingForPosition = True
                    break
            if (searchingForPosition == True):
                idealPosition += deltaPos
            counter += 1
            if (counter > 100):
                print("problem generating best neuron")
                time.sleep(100)
        """
        idealPosition = placeNeuron(locations, inputSize, outputSize)

        ## now we have ideal position which is a valid position for the new neuron

        backwardsConnections = []
        forwardsConnections = []

        for neuron in connectionPool:
            if (neuron.position < idealPosition):
                backwardsConnections.append([neuron, uniform(-1,1)])

            elif (neuron.position > idealPosition):
                forwardsConnections.append([neuron, uniform(-1,1)])
            else:
                print(idealPosition, neuron.position)
                print("tried to generate an invalid neuron")
                time.sleep(100)

        self.addNeuron(idealPosition, backwardsConnections, forwardsConnections, uniform(-1,1) )


    ## generate a new neuron
    def smartGenerate(self, connectionNumber):

        keys = getKeys(self.neuronDict)
        ## we will have a dictionary with keys that are the neuron score and
        ## values that are the actual Neurons
        NeuronDict = dict()
        ## populate the dictionary with neurons
        for key in keys:
            for element in self.neuronDict[key]:
                try:
                    NeuronDict[element.score].append(element)
                except:
                    NeuronDict[element.score] = [element]
        ## then lets find the top entries that arnt already connected
        keys2 = list( NeuronDict.keys() )

        keys2.sort()

        keys2.reverse()
        ##print("here is keys2")
        ##print(keys2)
        neuronsToConnect = [NeuronDict[ keys2[0] ] [0] ]
        locations = [NeuronDict[ keys2[0] ][0].position]
        counter = 1
        find = False
        for key in keys2:
            if (find == False):
                for element in NeuronDict[key]:
                    if (counter < connectionNumber):
                        #print("triggering")
                        ##if (self.test(neuronsToConnect, element)):
                        ##print("triggering")
                        neuronsToConnect.append(element)
                        locations.append(element.position)
                        counter += 1
                    else:
                        find = True
                        break
        ## now we have to find where to place the neuron.
        placement = placeNeuron(locations, connectionNumber, connectionNumber)
        ## this is just a shitty little patch. Should actually be fixed later
        if (placement == True):
            self.failGen += 1
            return True

        backwardsConnections = []
        forwardsConnections = []
        for element in neuronsToConnect:
            if (element.position > placement):
                ##forwardsConnections.append([element, uniform(-1,1) ])
                forwardsConnections.append([element, 0 ])

            elif (element.position < placement):
                ##backwardsConnections.append([element, uniform(-1,1)])
                backwardsConnections.append([element, 0 ])
            else:
                print("attempting to make invalid connections")
                time.sleep(100)

        ##self.addNeuron(placement, backwardsConnections, forwardsConnections, uniform(-1,1))
        self.addNeuron(placement, backwardsConnections, forwardsConnections, 0)


    ## produces a list with elements that are [neuron score, neuron].
    ## remember that neuron scores are just the deltas. So
    ## low numbers are "better" performing neurons are better.
    ## so this list [in index] is best to worst
    def scoreRanking(self):
        rtn = []
        keys = getKeys(self.neuronDict)
        counter = 0
        for key in keys:
            for neuron in self.neuronDict[key]:
                counter += 1
                rtn.append([neuron.score, counter, neuron])

        rtn.sort()

        return rtn

    ## NEXT BIG PROJECT:
    ## giving neurons plasticity
