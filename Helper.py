"""
Helper functions for Atlas
"""

import math
## from random import *
import numpy as np
import scipy.stats
import numpy.random
from random import *

## returns a sorted (assending) list of the keys in the given dictionary
def getKeys(dictionary, sorted = True):

    keys = list(dictionary.keys())
    if sorted:
        keys.sort()
    return keys

## average of a list
def average(list):
    sum = 0
    for element in list:
        sum += element
    return sum/len(list)

def getNeuronLinearProperties(dataSet):
    if (len(dataSet) == 0):
        print("linear properties cannot calculate empty list")
        time.sleep(1000)
    if (len(dataSet) == 1):
        return [0,0]

    indexList = []
    mean = 0
    for index in range(len(dataSet)):
        indexList.append(index)
        mean += dataSet[index]
    mean = mean/len(dataSet)
    try:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(indexList, dataSet)
    except:
        print("stat function crashed")
        print(dataSet)
        print("done")
        time.sleep(1000)
    return [slope, r_value * r_value]

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 3, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def testPerformance(testData, network):
    correctCounter = 0
    totalCounter = 0
    for element in testData:
        eval = network.evaluate(element[0])
        network.reset()
        indexResult = eval.index(max(eval))
        indexCorrect = element[1].index(max(element[1]))
        if (indexResult == indexCorrect):
            correctCounter += 1
        totalCounter += 1
    return (correctCounter/totalCounter) * 100
    ##print("Percent correct on training data:",(correctCounter/totalCounter) * 100)

## takes two lists, returns True if they have highest element with common index
## and False otherwise.
def testCorrect(eval, correctOutput):
    indexResult = eval.index(max(eval))
    indexCorrect = correctOutput.index(max(correctOutput))
    if (indexResult == indexCorrect):
        return True
    else:
        return False


## returns the number of neurons within the network
def networkSize(network):
    counter = 0
    keys = getKeys(network.neuronDict)
    for key in keys:
        for neuron in network.neuronDict[key]:
            for element in neuron.connections:
                counter += 1
    return counter



## sets the linear regession results for the network
def getLinearPoperties(network, dataSet):
    if (len(dataSet) == 0):
        print("linear properties cannot calculate empty list")
        time.sleep(1000)

    indexList = []
    mean = 0
    for index in range(len(dataSet)):
        indexList.append(index)
        mean += dataSet[index]
    mean = mean/len(dataSet)
    try:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(indexList, dataSet)
    except:
        print("stat function crashed")
        print(dataSet)
        print("done")
        time.sleep(1000)

    network.smoothedSlope = slope
    network.r_SQRD = r_value * r_value
    network.mean = mean

## linear gradient=1 function over domain [0,1] and 0 or 1 elsewhere
def unitLinear(input):
    if (input > 0 and input < 1):
        return input
    elif (input <= 0):
        return 0
    elif (input >= 1):
        return 1

def smoothedUnitLinear(input):
    if (input <= 0.9):
        return input
    if (input > 0.9):
        return (-0.01/(input-0.8)) + 1

## returns the average value from the last # period entries of the dataList
def movingAverage(dataList, period):
    rtn = 0
    if (len(dataList) == 0):
        print("Moving average cannot function on empty data")
        time.sleep(1000)
    if (len(dataList) < period):
        period = len(dataList)
    dataList = dataList[-period:]
    for element in dataList:
        rtn += element
    return rtn/len(dataList)

## takes a scoredNeuronlist [atleast for this case] and the number of elemnents to be
## randomy selected from it. It then uses the scores as probabilities of being picked, normalises them and makes
## a selection based on that
def randomWeightedSelection(scoredNeurons, numberOfSelections):

    numpy.random.seed(1)
    ## we first have to normalise everything
    if (numberOfSelections> len(scoredNeurons)):
        numberOfSelections = len(scoredNeurons)

    sum = 0
    for element in scoredNeurons:
        sum += element[0] + 0.00000000001
    for element in scoredNeurons:
        element[0] = (element[0] + 0.00000000001)/sum

    ## so now scoredNeurons has associated with it the probabiltiy of it being picked
    choiceList = []
    probList = []
    for element in scoredNeurons:
        choiceList.append(element[2])
        probList.append(element[0])
    ##print(choiceList)
    ##print("here is a sample of choiceList:", choiceList, 30)
    ##print("here is a sample of probList:", probList, 30)
    ##print(len(choiceList))
    ##print(len(probList))

    randomDraw = numpy.random.choice(choiceList, numberOfSelections, p=probList, replace=False)
    ##print(randomDraw)
    ## this sbhould be a set of neurons that were picked probabilistically
    return randomDraw

def placeNeuron(locations, inputSize, outputSize):
    seed(1)
    locations.sort()
    buckets = []
    ## buckets contains the left and right indexes
    for index in range(len(locations)-1):
        buckets.append([index, index+1])

    ## so now we have a bunch of elements that have L element the left index
    ## and right element the right index of locations within the network

    ## indexes of bucket entries that have non zero delta location between them
    validBucketIndexes = []
    for index in range(len(buckets)):
        if ( abs(locations[buckets[index][0]] -  locations[buckets[index][1]]) > 0.0001 ):
        ##if (locations[buckets[index][0]] != locations[buckets[index][1]] ):
            validBucketIndexes.append(index)
    if (len(validBucketIndexes) == 0):
        print("improper generation")

    ## index of the entry in buckets that contains the indexes of the
    ## two locations that we want to place the new neuron
    startIndex = inputSize - 1
    bestIndex = startIndex
    minDistance = 100000000
    shuffle(validBucketIndexes)
    for element in validBucketIndexes:
        distance = abs(startIndex - element)
        if (distance < minDistance):
            minDistance = distance
            bestIndex = element

    ## now that we have the best element from the bucket we simply set the genLocation
    if (len(buckets) == 0 ):
        print("buckets empty")
        print(locations)
    try:
        leftLocation = locations[buckets[bestIndex][0]]
        rightLocation = locations[buckets[bestIndex][1]]
        genLocation = (leftLocation + rightLocation)/2
    except:
        print(len(locations))
        print(buckets[bestIndex])
        time.sleep(10000)

    for element in locations:
        if (genLocation == element):
            print("we have a problem, here is the locations")
            print(locations)
            print("here is the genLocation")
            print(genLocation)
            time.sleep(100000)
            ##print(genLocation)
            ##print(element)
            ##return True


    return genLocation
