"""
Class that runs a society of NN instances with different hyperparameters.
"""

## should be run with no less than 9 logical cores
def simultanousTrain(OGNetwork, trainingData, testingData, numberOfConvergencePoints, hyperParameterMultipliers = [[0.01, 1, 100], [math.exp(-1/2), 1, math.exp(1/2)]]):
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
                testNetwork = copy.deepcopy(OGNetwork)
                testNetwork.learningRate = OGNetwork.learningRate * firstElement
                testNetwork.geometryEditRate = OGNetwork.geometryEditRate * secondElement
                if (testNetwork.geometryEditRate < 5):
                    testNetwork.geometryEditRate == 5
                networkList.append(testNetwork)
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()

        for element in networkList:
            ##print(len(sessionTrainingData))
            if (element.learningRate == OGNetwork.learningRate * 100 and element.geometryEditRate  == OGNetwork.geometryEditRate* math.exp(1/2)):
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

            if ((net.smoothedScores[-1]) > bestNetworkScore):
                bestNetworkScore = net.smoothedScores[-1]
                bestNetwork = net

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

        if ( average(OGNetwork.smoothedScores[-len(sessionTrainingData):]) > OGNetwork.target):
            drawNetwork(OGNetwork)



        counter += 1

        OGNetwork.basicTrain( sample( sessionTrainingData, math.floor(0.1*len(sessionTrainingData)) ) )
        testScore = thingo(OGNetwork)
        ##if ( bestNetworkScore > OGNetwork.target):
            ##OGNetwork.target = bestNetworkScore

        print("learingRate:", OGNetwork.learningRate)
        print("geometryEditRate:", OGNetwork.geometryEditRate)
        print("bestScore", bestNetworkScore)
        print("current Target:", OGNetwork.target)
