###########using the sparse_cross_entropy with labels treated as integers#############

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, layers

(trainX, trainY), (testX, testy) = datasets.cifar10.load_data()
trainX, testX = trainX / 255, testX / 255

classNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(trainX[i], cmap=plt.cm.binary)
    plt.xlabel(classNames[trainY[i][0]])

# plt.show()

trainX = trainX[:20000]
trainY = trainY[:20000]
testX = testX[:4000]
testy = testy[:4000]

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
#model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(trainX, trainY, epochs=10, validation_data=(testX, testy))

loss,accuracy = model.evaluate(testX,testy)
print(f"loss : {loss}")
print(f"accuracy : {accuracy}")

model.save("classifier_cifar10_.model")
