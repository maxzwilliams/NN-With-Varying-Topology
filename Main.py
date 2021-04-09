from NN import *
from DataReading import *
from Helper import *
from Drawing import *

from CompositeNetwork import *

import pickle

## this evaluates the network on the testData
def thingo(network, testData):
    print("evaluating testData")
    correctCounter = 0
    totalCounter = 0
    totalLength = len(testData)
    for element in testData:
        eval = network.evaluate(element[0])
        network.reset()
        indexResult = eval.index(max(eval))
        indexCorrect = element[1].index(max(element[1]))
        if (indexResult == indexCorrect):
            correctCounter += 1
        totalCounter += 1

        printProgressBar(totalCounter, totalLength)
    print("Correct:", correctCounter, end=" ")
    print("Percent correct on training data:",(correctCounter/totalCounter) * 100)
    return (correctCounter/totalCounter) * 100

def duplicate(items):
    unique = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique

def refineWrongList(wrongList, net):
    newNet = copy.deepcopy(net)
    newWrongList = []
    for element in wrongList:
        newNet.reset()
        if ( testCorrect(newNet.evaluate(element[0]), element[1] ) == False ):
            newWrongList.append(element)
    newNet.reset()
    del newNet
    return newWrongList

def generateSesh(seshData, wrongList, repeatBase):
    tempList = wrongList
    if (len(wrongList) == 0):
        print("wrong list was empty")
        generated = True
    else:
        generated = False

    while not generated:
        if (math.floor(len(tempList) * (repeatBase)) < math.floor(len(seshData) * (1 -repeatBase))):
            tempList = tempList + tempList
        else:
            wrongList = sample(tempList, math.floor(len(tempList) * repeatBase))
            generated = True

    seshData = seshData + wrongList
    shuffle(seshData)
    return seshData

## should be run with no less than 9 logical cores
def simultanousTrain(OGNetwork, trainingData, testingData, numberOfConvergencePoints, geometryTrainTime = None, normalTrainTime = None ,hyperParameterMultipliers = [[math.exp(-1), math.exp(1)], [math.exp(-1/2), math.exp(1/2)], [math.exp(-1/2), math.exp(1/2)], [math.exp(-1/2), math.exp(1/2)], [math.exp(-1/3), math.exp(1/3)]], endTarget = None, CompoundMarker = None):
    numberOfSimulations = len(hyperParameterMultipliers)
    gotTarget = False
    target = OGNetwork.target
    trainingSessionLists = []

    veryBestNetwork = None
    veryBestNetworkScore = -1

    for index in range(numberOfConvergencePoints):
        trainingSessionLists.append(trainingData[index * math.floor( len(trainingData)/numberOfConvergencePoints):(index+1) * math.floor( len(trainingData)/numberOfConvergencePoints) ])

    counter = 0
    wrongList = []
    while (not gotTarget):
        ##wrongList = list(set(wrongList))
        wrongList = duplicate(wrongList)
        print("Wrong list length is", len(wrongList))
        if (len(wrongList) > 1000):
            wrongList = wrongList[-1000:]
        sessionTrainingData = trainingSessionLists[ counter % numberOfConvergencePoints ]
        seshLen = len(sessionTrainingData)

        ##shuffle(trainingSessionLists)
        shuffle(sessionTrainingData)
        networkList = []

        multiplierListList = []

        sessionData = generateSesh(sessionTrainingData, wrongList, OGNetwork.repeatBase)
        """
        for firstElement in hyperParameterMultipliers[0]:
            for secondElement in hyperParameterMultipliers[1]:
                for thirdElement in hyperParameterMultipliers[2]:
                    for fourthElement in hyperParameterMultipliers[3]:
                        for FithElement in hyperParameterMultipliers[4]:
                            for sixthElement in hyperParameterMultipliers[3]:
                                ##for seventhElement in hyperParameterMultipliers[4]:
                                marker = "geometryTraining"
                                learningRate = OGNetwork.learningRate * firstElement
                                geometryEditRate = smoothedUnitLinear(OGNetwork.geometryEditRate * secondElement)
                                selectionBias = OGNetwork.selectionBias * sixthElement
                                slopeSensitivity = OGNetwork.slopeSensitivity * thirdElement
                                genDecideBias = smoothedUnitLinear(OGNetwork.genDecideBias * FithElement)
                                repeatBase = 0.01
                                span = OGNetwork.span * fourthElement

                                if (geometryEditRate < 0.01):
                                    geometryEditRate = 0.01

                                multiplierList = [sessionData, marker, learningRate, geometryEditRate, selectionBias,slopeSensitivity,genDecideBias,span,repeatBase]
                                multiplierListList.append(multiplierList)
        """

        for firstElement in hyperParameterMultipliers[0]:
            for secondElement in hyperParameterMultipliers[1]:
                for thirdElement in hyperParameterMultipliers[1]:
                ##for seventhElement in hyperParameterMultipliers[4]:
                    marker = "geometryTraining"
                    learningRate = OGNetwork.learningRate * firstElement
                    geometryEditRate = 0
                    selectionBias = 0
                    slopeSensitivity = 0
                    genDecideBias = 0
                    repeatBase = smoothedUnitLinear(OGNetwork.repeatBase * thirdElement)
                    span = OGNetwork.span * secondElement


                    multiplierList = [sessionData, marker, learningRate, geometryEditRate, selectionBias,slopeSensitivity,genDecideBias,span,repeatBase]
                    multiplierListList.append(multiplierList)

        """
        for seventhElement in hyperParameterMultipliers[0]:
            sessionData = generateSesh(sessionTrainingData, wrongList, OGNetwork.repeatBase)
            for secondElement in hyperParameterMultipliers[1]:
                for thirdElement in hyperParameterMultipliers[2]:
                    for forthElement in hyperParameterMultipliers[3]:
                        for FithElement in hyperParameterMultipliers[4]:
                            for sixthElement in hyperParameterMultipliers[3]:
                                for firstElement in hyperParameterMultipliers[1]:
                                    ##testNetwork = copy.deepcopy(OGNetwork)
                                    sessionTrainingData = sessionData
                                    marker = "geometryTraining"
                                    learningRate = OGNetwork.learningRate * firstElement
                                    geometryEditRate = smoothedUnitLinear(OGNetwork.geometryEditRate * secondElement)
                                    selectionBias = OGNetwork.selectionBias * thirdElement
                                    slopeSensitivity = OGNetwork.slopeSensitivity * forthElement
                                    genDecideBias = smoothedUnitLinear(OGNetwork.genDecideBias* FithElement)
                                    span = OGNetwork.span * sixthElement
                                    repeatBase = smoothedUnitLinear(OGNetwork.repeatBase * seventhElement)

                                    if (geometryEditRate < 0.001):
                                        geometryEditRate = 0.001

                                    if (slopeSensitivity > 5000):
                                        slopeSensitivity = 5000

                                    if (genDecideBias > 0.50):
                                        genDecideBias = 0.50

                                        ##if (testNetwork.geometryEditRate > 50):
                                    multiplierList = [sessionTrainingData, marker, learningRate, geometryEditRate, selectionBias,slopeSensitivity,genDecideBias,span,repeatBase]
                                    multiplierListList.append(multiplierList)
        """


        if ( len(multiplierListList) > math.floor(mp.cpu_count() - 3)):
            multiplierListList = sample(multiplierListList, math.floor(mp.cpu_count() - 3))

        print(len(multiplierListList))

        for element in multiplierListList:
            testNetwork = copy.deepcopy(OGNetwork)

            testNetwork.sessionTrainingData = element[0]
            testNetwork.marker = element[1]
            testNetwork.learningRate = element[2]
            testNetwork.geometryEditRate = element[3]
            testNetwork.selectionBias = element[4]
            testNetwork.slopeSensitivity = element[5]
            testNetwork.genDecideBias = element[6]
            testNetwork.span = element[7]
            testNetwork.repeatBase = element[8]

            networkList.append(testNetwork)

        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()
        return_list = manager.list()
        ##return_list = []
        print("got here")
        for element in networkList:
            if (element.marker == "geometryTraining" ):
                p = mp.Process(target = element.simGeometryTrain, args = (element.sessionTrainingData, return_list, "standard", geometryTrainTime, normalTrainTime, ) )
            processes.append(p)

        ## then we append the standard baseline
        print("starting threads")
        ##print("starting processes")
        for element in processes:
            element.start()
        ##print("joining processes")
        for element in processes:
            element.join()
        print("eneded threads")

        nets = return_list
        ##nets = getKeys(return_dict, False)
        ##print(return_dict)
        print("here is len nets", len(nets))
        print("got keys")
        ##shuffle(nets)
        print("shuffled")
        netsAndScores = []
        netCounter = 0

        bestNetworkScore = - 1
        for net in nets:
            ##avg = average( net.smoothedScores[-len(sessionTrainingData):] )
            ##if (max(net.smoothedScores[-len(sessionTrainingData):]) > bestNetworkScore):
                ##bestNetworkScore = max(net.smoothedScores[-len(sessionTrainingData):])
                ##bestNetwork = net
            print("sample time")
            try:
                sampleSize = math.ceil(10/(1 - (scoreBest/100)))
            except:
                sampleSize = 10
            if (sampleSize < 10):
                sampleSize = 10
            testScore = thingo(net, sample(testingData, sampleSize))
            netsAndScores.append([testScore, netCounter, net])
            netCounter += 1
        print("done with samples")

        netsAndScores.sort()
        netsAndScores.reverse()

        bestNetwork = netsAndScores[0][2]
        bestNetworkScore = netsAndScores[0][0]
        scoreBest = bestNetworkScore
        wrongList = refineWrongList(bestNetwork.sessionTrainingData, bestNetwork)
        bestNetwork.sessionTrainingData = []
        ## then we get the seshList

        print("wrongList length here", len(wrongList))
        ##wrongList = refineWrongList(wrongList, OGNetwork)


        ##wrongList = wrongList + bestNetwork.wrongList

        bestNetwork.testPerformance = bestNetworkScore/100
        secondNetwork = netsAndScores[1][2]
        thirdNetwork = netsAndScores[2][2]


        if (bestNetwork.marker == "basicTrain"):
            bestNetwork.learningRate = secondNetwork.learningRate
            bestNetwork.geometryEditRate = secondNetwork.geometryEditRate
            bestNetwork.selectionBias = secondNetwork.selectionBias
            bestNetwork.slopeSensitivity = secondNetwork.slopeSensitivity
            bestNetwork.genDecideBias = secondNetwork.genDecideBias
            bestNetwork.span = secondNetwork.span
        else:
            bestNetwork.learningRate = ( (bestNetwork.learningRate * 3) + (secondNetwork.learningRate *2) + thirdNetwork.learningRate)/6
            bestNetwork.geometryEditRate = ( (bestNetwork.geometryEditRate * 3) + (secondNetwork.geometryEditRate *2) + thirdNetwork.geometryEditRate)/6
            bestNetwork.selectionBias = ( (bestNetwork.selectionBias * 3) + (secondNetwork.selectionBias *2) + thirdNetwork.selectionBias)/6
            bestNetwork.slopeSensitivity = ( (bestNetwork.slopeSensitivity * 3) + (secondNetwork.slopeSensitivity *2) + thirdNetwork.slopeSensitivity)/6
            bestNetwork.genDecideBias = ( (bestNetwork.genDecideBias * 3) + (secondNetwork.genDecideBias *2) + thirdNetwork.genDecideBias)/6
            bestNetwork.span = ( (bestNetwork.span * 3) + (secondNetwork.span *2) + thirdNetwork.span)/6
            bestNetwork.repeatBase = ( (bestNetwork.repeatBase * 3) + (secondNetwork.repeatBase *2) + thirdNetwork.repeatBase)/6

        if (endTarget != None):
            if (bestNetworkScore > endTarget):
                print("starting pickle")
                try:
                    pickleOut = open( "CurrentNet", 'wb')
                    pickle.dump( OGNetwork, pickleOut )
                    pickleOut.close()
                except:
                    print("___________________________________________________")
                    print("pickle did not work")
                    print("___________________________________________________")

                print("ended pickle")
                return None

        del netsAndScores

        ##OGNetwork = copy.deepcopy(bestNetwork)
        plt.plot(bestNetwork.smoothedScores)
        plt.savefig('NetworkScore.png')
        plt.clf()
        plt.plot(bestNetwork.pastNeuronNumbers)
        plt.savefig('NetworkNeurons.png')
        plt.clf()
        plt.plot(bestNetwork.pastConnections)
        plt.savefig('NetworkConnections.png')
        plt.clf()
        plt.plot(bestNetwork.pastGeometryEdit)
        plt.savefig('NetworkGeometryEdit.png')
        plt.clf()
        plt.plot(bestNetwork.pastDecide)
        plt.savefig('NetworkDecide.png')
        plt.clf()

        counter += 1

        print("learingRate:", bestNetwork.learningRate)
        print("geometryEditRate:", bestNetwork.geometryEditRate)
        print("bestScore", bestNetworkScore)
        print("current Target:", bestNetwork.target)
        print("selectionBias:", bestNetwork.selectionBias)
        print("genDecideBias", bestNetwork.genDecideBias)
        print("hyperStep", bestNetwork.hyperStep)
        print("span", bestNetwork.span)
        print("repeate base", bestNetwork.repeatBase)
        print("OGNetwork.slopeSensitivity", bestNetwork.slopeSensitivity)
        print("mostRecentScore", average( bestNetwork.pastScores[-100:]) )

        if (bestNetwork.marker == "basicTrain"):
            print("network did basic train")
        else:
            print("network did geometry edit training")
        ##wrongList = refineWrongList(wrongList, OGNetwork)

        print("starting pickle")
        try:
            pickleOut = open( "newPickledNetwork" + str(counter), "wb" )
            pickle.dump( bestNetwork, pickleOut )
            pickleOut.close()
            print("ended pickle")
        except:
            print("Pickled didnt work this time")

        OGNetwork = copy.deepcopy(bestNetwork)
        del bestNetwork


if __name__ == '__main__':

    ##fullDataSet = dataGenerate(60000)
    ##print("time to pickle all the data")
    ##pickleOut = open("fullDataSet", "wb")
    ##pickle.dump(fullDataSet, pickleOut)
    ##pickleOut.close()

    print("getting all the data")
    pickleIn = open("fullDataSet", "rb")
    fullDataSet = pickle.load(pickleIn)
    print("done getting data")

    trainData = fullDataSet[0][0:50000]
    wrongList = []
    testData = fullDataSet[0][50000:60000]
    print("train data has", len(trainData), "entries")
    print("testData has", len(testData), "entries")

    ## lets find how many common elements between the two
    counter = 0
    trainDataSample = sample(trainData, 200)
    for element in trainDataSample:
        if (element in testData):
            counter += 1
    print("test and train data had", counter, "common elements from a sample of 1000")

    ##compositeTrain(trainData, testData, 80)

    ##pickle.close()
    ##dict = oldNetwork.neuronDict

    ##network.genDecideBias = 0.40
    ##network.slopeSensitivity = 1000
    ##network.selectionBias = 10000
    ##network.repeatBase = 0.50

    network = NN(trainData[0:5], 1, 0.05)
    network.originalLearningRate = 1
    network.repeatBase = 0.10
    network.selectionBias = 3000
    ##network.target = 1.0
    network.hyperStep = 0
    ##network.genDecideBias = 0.50
    network.testPerformance = 0.10
    network.span = 15


    ##network = NN(trainData[0:5], 1, 100)
    ##network.neuronDict = pastNetwork.neuronDict

    ##del pastNetwork
    ##for index in range(100):
        ##print(index)
        ##network.degenerate(100)

    network.informationPrintOut()

    simultanousTrain(network, trainData, testData, 50, 20, 5)
    ##network.geometryTrain(trainData,  1)
