"""
Manages composite network interactions
"""
from Main import *

def compositeTrain(dataSet, testData, target):
    originalDataSet = dataSet
    nets = []
    for index in range(3):
        net = NN(dataSet[0:5], 1, 30)
        net.originalLearningRate = 1
        nets.append(net)
        print("starting training")
        inputNetTrain(dataSet, testData, net, target,index+1)
        print("done Training")
        print("getting wrong data")
        dataSet = getWrongData(dataSet, net)
        testData = sample(dataSet, math.floor(len(dataSet) * 0.05))
        print("done getting wrong data")
    print("we are done")
    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")

    compDataSet = compositeDataSet(originalDataSet, nets)

    compNet = NN(compDataSet[0:5], 1, 30)
    compNet.originalLearningRate = 1
    simultanousTrain(compNet, compDataSet[0:math.floor( len(compDataSet)*0.90 )], compDataSet[-math.floor( len(compDataSet)*0.09):], 10, 60, 60,[[math.exp(-1/2), 1, math.exp(1/2)], [math.exp(-1/2), math.exp(1/2)], [math.exp(-1/2), math.exp(1/2)], [math.exp(-1/2), math.exp(1/2)]], 95, "COMP")
    print("FULL DONE")
    print(len(compDataSet))


def compositeDataSet(dataSet, nets):
    ## we need to generate a dataSet to train the next layer neural networ5
    rtn = []
    for element in dataSet:
        evals = []
        for net in nets:
            evals.append( net.evaluate( element[0] ) )
        evals = concat(evals)

        rtn.append( [evals, element[1]] )

    return rtn

def concat(lists):
    list1 = lists[0]
    for index in range(1, len(lists)):
        list1 = concatHelper(list1, lists[index])

    return list1

def concatHelper(list1, list2):
    for element in list2:
        list1.append(element)
    return list1

def inputNetTrain(trainingData, testData, net, target, marker):
    simultanousTrain(net, trainingData, testData, 10, 60, 60,[[math.exp(-1/2), 1, math.exp(1/2)], [math.exp(-1/2), math.exp(1/2)], [math.exp(-1/2), math.exp(1/2)], [math.exp(-1/2), math.exp(1/2)]], target, marker)

def getWrongData(trainingData, net):
    wrongData = []
    for element in trainingData:
        eval = net.evaluate(element[0])
        net.reset()
        indexResult = eval.index(max(eval))
        indexCorrect = element[1].index(max(element[1]))
        if (indexResult != indexCorrect):
            wrongData.append(element)

    return wrongData
