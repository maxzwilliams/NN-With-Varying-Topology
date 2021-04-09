from graphics import *
from NN import *
from Helper import *
from random import *
import math

##from PIL import Image as NewImage

def assignPosition(network, xDim, xOff, yDim, yOff):
    keys = getKeys(network.neuronDict)

    for key in keys:
        numberOfNeuronsInLayer = len(network.neuronDict[key])
        xPoses = generateXPoints(numberOfNeuronsInLayer, xDim)
        counter = 0
        for neuron in network.neuronDict[key]:
            if (neuron.drawingX == 0 and neuron.drawingY == 0):
                neuron.drawingX = xPoses[counter] + xOff
                neuron.drawingY = key * 10 + yOff
            counter += 1


def generateXPoints(numberOfPoints, xDim):
    rtn = []
    ##counter = 0
    if (numberOfPoints == 1):
        return [uniform(xDim/2 - xDim/3, xDim/2 + xDim/3)]
        ##return [xDim/2]

    for counter in range(numberOfPoints):
        rtn.append( uniform(xDim/2 - xDim/3, xDim/2 + xDim/3) )

    return rtn

def drawNetwork(network, xDim= 1800, yDim= 1200, message = ''):
    assignPosition(network, xDim, 0, yDim- 100, 100)
    win = GraphWin('Network', xDim, yDim)
    ##win.yUp()

    keys = getKeys(network.neuronDict)
    ## contains lists that have first element x and second element y position
    ## of the neuron
    positions = []


    for key in keys:
        for neuron in network.neuronDict[key]:

            for connection in neuron.connections:

                if (neuron.position != 100 and neuron.position != 0 and network.markerNeuronDict[connection[0]].position != 0 and network.markerNeuronDict[connection[0]].position != 100):
                    tmp = uniform(0,1)
                    if (tmp > 0.99):
                        line = Line( Point( neuron.drawingX, neuron.drawingY ), Point( network.markerNeuronDict[connection[0]].drawingX, network.markerNeuronDict[connection[0]].drawingY ) )
                        weightStrength = abs(connection[1])
                        if (abs(weightStrength) > 0.80):
                            newColor = color_rgb(255, 0, 0)
                        else:
                            newColor = color_rgb(0, 0, 255)
                        line.setFill(newColor)
                        line.draw(win)
                else:
                    tmp = uniform(0,1)
                    if (tmp > 0.99):
                        line = Line( Point( neuron.drawingX, neuron.drawingY ), Point( network.markerNeuronDict[connection[0]].drawingX, network.markerNeuronDict[connection[0]].drawingY ) )
                        weightStrength = abs(connection[1])
                        if (abs(weightStrength) > 0.80):
                            newColor = color_rgb(255, 0, 0)
                        else:
                            newColor = color_rgb(0, 0, 255)
                            ##newColor = color_rgb(1000*math.ceil(255 * abs(weightStrength)),1000*math.ceil(255 * abs(weightStrength)), 1000* math.ceil(255 * abs(weightStrength)) )
                        line.setFill(newColor)
                        line.draw(win)


            if (neuron.position == 0):
                point = Circle(Point(neuron.drawingX, neuron.drawingY), 2)
                point.setFill('red')
                point.draw(win)
            elif (neuron.position == 100):
                point = Circle(Point(neuron.drawingX, neuron.drawingY), 5)
                point.setFill('blue')
                point.draw(win)
            else:
                point = Circle(Point(neuron.drawingX, neuron.drawingY), 5)
                point.setFill('green')
                point.draw(win)

    message = Text(Point(win.getWidth()/2, 20), message)
    message.draw(win)


    time.sleep(10)
    ##win.close()



    ## now we need to print all the points that we just generate
