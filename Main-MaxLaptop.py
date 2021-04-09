from NN import *
from DataReading import *
from Helper import *
from Drawing import *

import pickle

## this evaluates the network on the testData
def thingo(network):
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

## should be run with no less than 9 logical cores
def simultanousTrain(OGNetwork, trainingData, testingData, numberOfConvergencePoints, hyperParameterMultipliers = [[0.01, 1, 100], [math.exp(-1/2), 1, math.exp(1/2)], [math.exp(-1/2), 1, math.exp(1/2)] ]):
    numberOfSimulations = len(hyperParameterMultipliers)
    gotTarget = False
    target = OGNetwork.target
    trainingSessionLists = []

    for index in range(numberOfConvergencePoints):
        trainingSessionLists.append(trainingData[index * math.floor( len(trainingData)/numberOfConvergencePoints):(index+1) * math.floor( len(trainingData)/numberOfConvergencePoints) ])

    counter = 0
    while (not gotTarget):
        sessionTrainingData = trainingSessionLists[ counter % numberOfConvergencePoints ]
        shuffle(trainingSessionLists)
        networkList = []
        for firstElement in hyperParameterMultipliers[0]:
            for secondElement in hyperParameterMultipliers[1]:
                for thirdElement in hyperParameterMultipliers[2]:
                    testNetwork = copy.deepcopy(OGNetwork)
                    testNetwork.learningRate = OGNetwork.learningRate * firstElement
                    testNetwork.geometryEditRate = OGNetwork.geometryEditRate * secondElement
                    testNetwork.selectionBias = OGNetwork.selectionBias * thirdElement

                    if (testNetwork.geometryEditRate > 50):
                        networkList.append(testNetwork)
        if ( len(networkList) > mp.cpu_count() ):
            networkList = sample(networkList, mp.cpu_count() )

        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()

        for element in networkList:
            print("element.learningRate", element.learningRate, "element.geometryEditRate", element.geometryEditRate, "element.selectionBias",element.selectionBias)
            ##print(len(sessionTrainingData))
            if (element.learningRate == OGNetwork.learningRate * 0.01 and element.geometryEditRate  == OGNetwork.geometryEditRate* math.exp(1/2)):
                p = mp.Process(target = element.simGeometryTrain, args = (sessionTrainingData, return_dict, [1,1], ) )
            elif (element.learningRate == OGNetwork.learningRate and element.geometryEditRate  == OGNetwork.geometryEditRate ):
                p = mp.Process(target = element.simGeometryTrain, args = (sessionTrainingData, return_dict, "standard", ) )
            else:
                p = mp.Process(target = element.simGeometryTrain, args = (sessionTrainingData, return_dict, ) )
            processes.append(p)

        for element in processes:
            element.start()

        for element in processes:
            element.join()

        bestNetworkScore = -1
        bestNetwork = None
        nets = getKeys(return_dict, False)
        for net in nets:
            ##avg = average( net.smoothedScores[-len(sessionTrainingData):] )
            ##if (max(net.smoothedScores[-len(sessionTrainingData):]) > bestNetworkScore):
                ##bestNetworkScore = max(net.smoothedScores[-len(sessionTrainingData):])
                ##bestNetwork = net
            testScore = thingo(net)
            if (testScore > bestNetworkScore):
                bestNetworkScore = testScore
                bestNetwork = net

            ##if ((net.smoothedScores[-1]) > bestNetworkScore):
            ##    bestNetworkScore = net.smoothedScores[-1]
            ##    bestNetwork = net

        OGNetwork = copy.deepcopy(bestNetwork)
        plt.plot(OGNetwork.smoothedScores)
        plt.savefig('NetworkScore.png')
        plt.clf()
        plt.plot(OGNetwork.pastNeuronNumbers)
        plt.savefig('NetworkNeurons.png')
        plt.clf()
        plt.plot(OGNetwork.pastConnections)
        plt.savefig('PastConnections.png')
        plt.clf()
        plt.plot(OGNetwork.pastGeometryEdit)
        plt.savefig('pastGeometryEdit.png')
        plt.clf()
        plt.plot(OGNetwork.pastDecide)
        plt.savefig('pastDecide.png')
        plt.clf()
        
        print("started pickle")
        pickleOut = open("pickledNetwork" + str(counter), "wb")
        pickle.dump(OGNetwork, pickleOut)
        pickleOut.close()
        print("ended pickle")


        counter += 1

        ##OGNetwork.basicTrain( sample( sessionTrainingData, math.floor(0.1*len(sessionTrainingData)) ) )

        ##if ( bestNetworkScore > OGNetwork.target):
            ##OGNetwork.target = bestNetworkScore

        print("learingRate:", OGNetwork.learningRate)
        print("geometryEditRate:", OGNetwork.geometryEditRate)
        print("bestScore", bestNetworkScore)
        print("current Target:", OGNetwork.target)
        print("selectionBias:", OGNetwork.selectionBias)

        drawNetwork(OGNetwork, message = "counter: " + str(counter))



if __name__ == '__main__':

    fullDataSet = dataGenerate(60000)
    trainData = fullDataSet[0][0:50000]
    testData = fullDataSet[0][59000:60000]
    print("train data has", len(trainData), "entries")
    print("testData has", len(testData), "entries")

    ## lets find how many common elements between the two
    counter = 0
    trainDataSample = sample(trainData, 100)
    for element in trainDataSample:
        if (element in testData):
            counter += 1
    print("test and train data had", counter, "common elements from a sample of 1000")


    network = NN(trainData[0:5], 1, 100)
    network.informationPrintOut()
    ##drawNetwork(network)
    network.target = 0.95
    network.genDecideBias = 0.60

    simultanousTrain(network, trainData, testData, 15)
    ##network.geometryTrain(trainData,  1)

    ##for index in range(400):
        ##network.addRandomNeuron(network.inputSize, network.outputSize


## now lets see how well our machine learning model does on test data
