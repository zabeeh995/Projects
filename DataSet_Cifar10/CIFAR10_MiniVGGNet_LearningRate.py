#########Very deep learning convolutional NN for large scale Image Recogniotion ###############################

from keras.datasets import cifar10
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import numpy as np

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

#########################################################################################################
# the shape is already (50000, 32, 32, 3)  so no need to reshape it
##########################################################################################################
# trainX = trainX.reshape((trainX.shape[0], 3072))
# testX = testX.reshape((testX.shape[0], 3072))

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# # initialize the label names for the CIFAR-10 dataset
# label names may beused to classify in the end of cifar for classification function
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

##############################################################################################################
inputShape = (32, 32, 3)
classes = 10

############################# define miniVGGNet architecture ##################################################
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))  ############# depth index  (height , width , depth)
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(classes))
model.add(Activation("softmax"))

############################## Compile the Model ##########################################
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics="Accuracy")

# train the network
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=64)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

model.save("classifier_cifar10_VGGNet.model")

###########learning rate schedulers can be used to reduce alpha with epoch or eveey epoch ###################
########### basically to reduce the step function for greater accuracy but it also overfits the results#######
# def step_decay(epoch):
#     initAlpha = 0.01
#     factor = 0.25
#     dropEvery = 5
#     alpha = initAlpha * (factor ** np.floor((1 + epoch)/dropEvery))
#
#     return alpha
#
# #after claas labels
#
# class LearingRateScheduker(object):
#     pass
#
#
# callbacks = [LearingRateScheduker(step_decay)]
#
# H = model.fit(trainX, trainY, validation_data=(testX,testY) , batch_size=64,
#               epochs=40 , callbacks=callbacks , verbose=1)
