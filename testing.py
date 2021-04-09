import pickle
from Drawing import *
import time

out = open("newPickledNetwork135", "rb")
net = pickle.load(out)


drawNetwork(net)
time.sleep(1000)
##net.evaluate()
"""
print("getting all the data")
pickleIn = open("fullDataSet", "rb")
fullDataSet = pickle.load(pickleIn)
print("done getting data")

trainData = fullDataSet[0][0:50000]

for i in range(10):
    img = trainData[i][0].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()
"""
