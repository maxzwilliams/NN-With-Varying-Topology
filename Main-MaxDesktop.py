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
def simultanousTrain(OGNetwork, trainingData, testingData, numberOfConvergencePoints, hyperParameterMultipliers = [[math.exp(-1/2), 1, math.exp(1/2)], [math.exp(-1/2), math.exp(1/2)], [math.exp(-1/2), math.exp(1/2)], [math.exp(-1/2), math.exp(1/2)] ]):
    numberOfSimulations = len(hyperParameterMultipliers)
    gotTarget = False
    target = OGNetwork.target
    trainingSessionLists = []

    for index in range(numberOfConvergencePoints):
        trainingSessionLists.append(trainingData[index * math.floor( len(trainingData)/numberOfConvergencePoints):(index+1) * math.floor( len(trainingData)/numberOfConvergencePoints) ])

    counter = 7
    while (not gotTarget):
        sessionTrainingData = trainingSessionLists[ counter % numberOfConvergencePoints ]
        ##shuffle(trainingSessionLists)
        shuffle(sessionTrainingData)
        networkList = []

        for firstElement in hyperParameterMultipliers[0]:
            testNetwork = copy.deepcopy(OGNetwork)
            testNetwork.marker = "basicTrain"
            testNetwork.learningRate = OGNetwork.learningRate * firstElement

                        ##if (testNetwork.geometryEditRate > 50):
            networkList.append(testNetwork)



        ## leaving one core free
        if ( len(networkList) > mp.cpu_count() - 1 ):
            networkList = sample(networkList, mp.cpu_count() - 1 )


        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()

        for element in networkList:
            print("element.learningRate", element.learningRate, "element.geometryEditRate", element.geometryEditRate, "element.selectionBias",element.selectionBias, "testNetwork.slopeSensitivity", element.slopeSensitivity)
            print("above has marker", element.marker)
            ##print(len(sessionTrainingData))
            p = mp.Process(target = element.simTrain, args = (sessionTrainingData, return_dict, ) )
            processes.append(p)

        ## then we append the standard baseline

        ##print("starting processes")
        for element in processes:
            element.start()
        ##print("joining processes")
        for element in processes:
            element.join()


        nets = getKeys(return_dict, False)
        netsAndScores = []
        netCounter = 0
        for net in nets:
            ##avg = average( net.smoothedScores[-len(sessionTrainingData):] )
            ##if (max(net.smoothedScores[-len(sessionTrainingData):]) > bestNetworkScore):
                ##bestNetworkScore = max(net.smoothedScores[-len(sessionTrainingData):])
                ##bestNetwork = net
            testScore = thingo(net)
            netsAndScores.append([testScore, netCounter, net])
            netCounter += 1

        netsAndScores.sort()
        netsAndScores.reverse()

        bestNetwork = netsAndScores[0][2]
        bestNetworkScore = netsAndScores[0][0]

        secondNetwork = netsAndScores[1][2]

        ## lets not carry this around
        del netsAndScores


        OGNetwork = copy.deepcopy(bestNetwork)
        plt.plot(OGNetwork.smoothedScores)
        plt.savefig('NetworkScore.png')
        plt.clf()
        plt.plot(OGNetwork.pastNeuronNumbers)
        plt.savefig('NetworkNeurons.png')
        plt.clf()
        plt.plot(OGNetwork.pastConnections)
        plt.savefig('NetworkConnections.png')
        plt.clf()
        plt.plot(OGNetwork.pastGeometryEdit)
        plt.savefig('NetworkGeometryEdit.png')
        plt.clf()
        plt.plot(OGNetwork.pastDecide)
        plt.savefig('NetworkDecide.png')
        plt.clf()

        ## now we save it
        print("starting pickle")
        pickleOut = open( "pickledNetwork" + str(counter), "wb" )
        pickle.dump( OGNetwork, pickleOut )
        pickleOut.close()
        print("ended pickle")
        counter += 1

        ##OGNetwork.basicTrain( sample( sessionTrainingData, math.floor(0.1*len(sessionTrainingData)) ) )

        ##if ( bestNetworkScore > OGNetwork.target):
            ##OGNetwork.target = bestNetworkScore

        ##OGNetwork.target = (1 - OGNetwork.target) * 0.20 + OGNetwork.target
        OGNetwork.target = ( ((100 - bestNetworkScore) * 0.20) + bestNetworkScore)/100

        print("learingRate:", OGNetwork.learningRate)
        print("geometryEditRate:", OGNetwork.geometryEditRate)
        print("bestScore", bestNetworkScore)
        print("current Target:", OGNetwork.target)
        print("selectionBias:", OGNetwork.selectionBias)
        print("OGNetwork.slopeSensitivity", OGNetwork.slopeSensitivity)

        if (bestNetwork.marker == "basicTrain"):
            print("network did basic train")
        else:
            print("network did geometry edit training")

        ##drawNetwork(OGNetwork, message = "counter: " + str(counter))


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

    ##pickleIn = open("pickledNetwork6","rb")
    ##network = pickle.load(pickleIn)
    network.target = 0.90
    network.genDecideBias = 0.50
    network.informationPrintOut()

    simultanousTrain(network, trainData, testData, 250)
    ##network.geometryTrain(trainData,  1)

    ##for index in range(400):
        ##network.addRandomNeuron(network.inputSize, network.outputSize


## now lets see how well our machine learning model does on test data
