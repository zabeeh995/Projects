#########  using tensor flow with cross entropy and converting the labels to vectors ####################

from keras.datasets import cifar10
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
import TrainingMonitor
import os

# load the training and testing data, scale it into the range [0, 1],
# then reshape the design matrix
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# define the 3072-1024-512-10 architecture using Keras
model = Sequential()  # thats why (3072,)
model.add(Dense(128, input_shape=(3072,), activation="relu"))  # dense layers need straight shape [.......]
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

#####################################################################################################
figgPath = "trainingmontor"
jsonnPath = "trainingmonitor"
figPath = os.path.sep.join([figgPath, "{}.png".format(
    os.getpid())])
print(figPath)
jsonPath = os.path.sep.join([jsonnPath, "{}.json".format(
    os.getpid())])
print(jsonPath)
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]
####################################################################################################


# train the network
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, callbacks=callbacks, verbose=1)
