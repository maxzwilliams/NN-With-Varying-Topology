import math
from Helper import *
from NN import *
from random import *

class Neuron:
    def __init__(self, position, bias):
        ## geometry of the Neuron
        self.position = position
        self.connections = []
        self.numConnectionsTo = 0

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

    def activationFunction(self, input):
        ## sigmoid activation function
        if (input > 100):
            return 1
        if (input < -100):
            return 0
        else:
            return 1/(1+math.exp(-input))


    def derivativeActivationFunction(self, input):
        return self.activationFunction(input) * (1 - self.activationFunction(input))

    ## calculates and updates value of the neuron
    def getValue(self):
        self.value = self.activationFunction(self.logit)

    def valueReset(self):
        self.value = 0
        self.logit = self.bias
        self.delta = 0

    def getWeightSum(self):
        sum = 0
        for element in self.connections:
            sum += abs(element[1])
        return sum


    ## calculate and update the scores
    def scoreUpdate(self, newScore):
        processScore = abs(newScore * self.value * self.getWeightSum()) ## could take into account the number of neurons that are comming into the neuron

        self.scores.append(newScore)
        if (len(self.scores) > 1000):
            self.scores = self.scores[-1000:]
        self.age += 1
        self.score = self.scoringFunction(1000)

    ## returns a new score based on passed scores
    def scoringFunction(self, scorelength):
        ## Lets take the last 100 scores and average them

        if (scorelength >= len(self.scores)):
            return average(self.scores)
        else:
            return average( sample(self.scores, scorelength) )
