import math
from Helper import *
from NN import *
from random import *
import time

class Neuron(object):

    def __init__(self, position, bias, markerList, outputNeuron = False ):
        ## geometry of the Neuron

        self.outputNeuron = outputNeuron
        found=False
        while not found:
            self.marker = randint(- 999999999999, 999999999999)
            if ( (self.marker in markerList) == False):
                found = True

        self.position = position
        self.connections = [] ## contains the current connections


        self.numConnectionsTo = 0
        self.currentBuff = 1
        self.stableBuff = 1
        ## values
        self.logit = 0
        self.value = 0
        self.bias = bias
        self.delta = 0

        ## things for drawing
        self.drawingX = 0
        self.drawingY = 0

        ## scoring for this neuron
        self.scores = []
        self.score = 0
        self.age = 0
        self.deltaScores = []

    def activationFunction(self, input):

        if (self.outputNeuron):
            return input
        else:
            ## sigmoid activation function
            if (input > 100):
                return 1
            if (input < -100):
                return 0
            else:
                ##return 1/(1+math.exp(-input))
                return (math.exp(2 * input) - 1)/(math.exp(2 * input) + 1)


    def derivativeActivationFunction(self, input):
        if (self.outputNeuron):
            return 1
        else:
            ##return self.activationFunction(input) * (1 - self.activationFunction(input))
            try:
                return (math.exp(2*input))/((math.exp(2*input)+1)**2)
            except:
                return 0.001


    ## calculates and updates value of the neuron
    def getValue(self):
        self.value = self.activationFunction(self.logit)

    def valueReset(self):
        self.value = 0
        self.logit = self.bias
        self.delta = 0

    def getWeightSum(self):
        sum = 0
        if (len(self.connections) == 0):
            return 1
        for element in self.connections:
            sum += abs( element[1] )
        return sum

    ## a much more simple getBuff that manipulates the learningRate based on the age of the neuron. A sort of plasticity if you will.
    def getBuff(self, hyperStep, currentLearningRate, originalLearningRate):
        if (self.age < 10):
            self.currentBuff = originalLearningRate/currentLearningRate
        else:
            self.currentBuff = self.currentBuff * (1/(1 + hyperStep))
        return self.currentBuff

    ## calculate and update the scores
    def scoreUpdate(self, newScore):
        ##print("starting score update")
        processScore = (newScore) ## could take into account the number of neurons that are comming into the neuron
        # * self.getWeightSum()
        self.scores.append(processScore)
        ##self.deltaScores.append(newScore)
        if (len(self.scores) > 100):
            self.scores = self.scores[-100:]
            self.deltaScores = []
        self.age += 1
        self.score = self.scoringFunction(100)
        ##print("done scoring")

    ## returns a new score based on passed scores
    def scoringFunction(self, scorelength):
        ## Lets take the last 100 scores and average them
        sum = 0
        for index in range(len(self.scores)):
            sum += self.scores[index]
        return sum/len(self.scores)
        """
        if (scorelength >= len(self.scores)):
            return average(self.scores)
        else:
            return average( sample(self.scores, scorelength) )
        """
