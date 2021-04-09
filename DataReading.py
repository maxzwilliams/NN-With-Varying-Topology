import numpy as np
##np.seterr(all='raise')
import matplotlib.pyplot as plt
from math import *
import pickle

def getNum(thingo):
    maxIndex = thingo.index(max(thingo))
    print(maxIndex)
##
def dataGenerate(samples):
    image_size = 28
    no_of_different_labels = 10
    image_pixels = image_size * image_size
    data_path = "Data/"

    ## windows requires:
    num_rows = 0
    max_cols = 0
    for line in open(data_path + "mnist_train.csv"):
        num_rows += 1
        tmp = line.split(",")
    if len(tmp) > max_cols:
        max_cols = len(tmp)

    train_data = np.empty([num_rows, max_cols])
    row = 0
    for line in open(data_path + "mnist_train.csv"):
        train_data[row] = np.fromstring(line, sep=",")
        row += 1
    """
    num_rows = 0
    max_cols = 0

    for line in open(data_path + "mnist_test.csv"):
        num_rows += 1
        tmp = line.split(",")

    if len(tmp) > max_cols:
        max_cols = len(tmp)

    test_data = np.empty([num_rows, max_cols])
    row = 0
    for line in open(data_path + "mnist_test.csv"):
        test_data[row] = np.fromstring(line, sep=",")
        row += 1
    """

    ## linux only needs:
    ##train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter = ",")
    ##test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter = ",")
    ##print(train_data[:10])

    frac = 0.99/255
    train_imgs = np.asfarray(train_data[:, 1:], dtype='float64') * frac + 0.01
    ##test_imgs = np.asfarray(test_data[:, 1:], dtype='float64') * frac + 0.01

    train_labels = np.asfarray(train_data[:, :1], dtype='float64')
    ##test_labels = np.asfarray(train_data[:, :1], dtype='float64')

    ## converting our labels into something usable
    lr = np.arange(10)

    for label in range(10):
        one_hot = (lr ==label).astype(np.int)
        ##print("label: ", label, " in one-hot rep: ", one_hot)


    ## now we want to adjust our labels outputs
    lr = np.arange(no_of_different_labels)

    # transform labels into one hot representation
    train_labels_one_hot = (lr==train_labels).astype(np.float64)
    ##test_labels_one_hot = (lr==test_labels).astype(np.float64)

    # we don't want zeroes and ones in the labels neither:
    train_labels_one_hot[train_labels_one_hot==0] = 0.01
    train_labels_one_hot[train_labels_one_hot==1] = 0.99
    ##test_labels_one_hot[test_labels_one_hot==0] = 0.01
    ##test_labels_one_hot[test_labels_one_hot==1] = 0.99

    ## okay so here is the hard part. We need to turn everything into a big list
    print("forming train data")
    inputData = []
    numTrainImgs = len(train_imgs)
    if (len(train_imgs) > samples):
        numTrainImgs = samples
    print("loading network")

    ##in = open("newPickledNetwork28", "rb")
    ##net = pickle.load(in)
    ##print("done loading")
    ##pickleIn = open("newPickledNetwork28","rb")
    ##net = pickle.load(pickleIn)

    for index in range(numTrainImgs):
        singleInput = []

        for element in train_imgs[index]:
            singleInput.append(element)
        ##print("eval time")
        ##net.reset()
        ##getNum(net.evaluate(singleInput))
        ##net.reset()
        ##print("showing")
        ##img = train_imgs[index].reshape((28,28))
        ##plt.imshow(img, cmap="Greys")
        ##plt.show()
        ##print("done showing")
        ##inputData.append([singleInput, list(train_labels_one_hot[index]) ] )
    print("done forming traing data")

    print("forming test data")
    testData = []
    """
    for index in range(1):
    ##for index in range(1000):
        dataEntry = []
        for element in test_imgs[index]:
            dataEntry.append(element)
        testData.append([dataEntry, list(test_labels_one_hot[index])])
    """
    print("done forming test data ")


    ## this naming doesnt really make sense
    return [inputData, []]
##dataGenerate(100)
