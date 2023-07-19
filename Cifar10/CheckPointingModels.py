#########Very deep learning convolutional NN for large scale Image Recogniotion ###############################

#############check pointing models##############################
from keras.callbacks import ModelCheckpoint
##############################################################

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

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# # initialize the label names for the CIFAR-10 dataset
# label names may be used to classify in the end of cifar for classification function
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

##########################################################################################################
fname = ("weights-{epoch:03d}-{val_loss:.4f}.hdf5")
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
                             save_best_only=True, verbose=1)
callbacks = [checkpoint]
#########################################################################################################

# train the network
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=64, callbacks=callbacks , verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

model.save("classifier_cifar10_VGGNet.model")
