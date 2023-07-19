import os
import cv2
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.python.keras.preprocessing.image import img_to_array

images = []
labels = []

path = "datasets/SMILEs"
############################################################################################
for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        print(f"Loading files from {folder_path}...")
        for file in os.listdir(folder_path):
            try:
                image = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
                image = img_to_array(image)
                images.append(image)

                #                labels.append("smiling" if int(folder) == '2' else "not_smiling")
                labels.append(int(folder))

            except Exception as e:
                print(f"There is a problem with file: {file}")
                print(str(e))
##############################################################################################
print(labels)
# newlabels = ["not_smiling" if x == "1" else "smiling" for x in labels]

result = []
for i in labels:
    if i == 1:
        result.append('smiling')
    else:
        result.append("not_smiling")

print(result)

images = np.array(images, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
# (x_train, x_test, y_train, y_test) = train_test_split(images,labels, test_size=0.20, stratify=labels, random_state=42)

# Split data into training and testing sets
# labels = tf.keras.utils.to_categorical(labels)

classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, stratify=labels, random_state=42)
#############################################################################################

model = models.Sequential()
model.add(layers.Conv2D(28, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(56, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(112, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(56, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
############################################################################################

# Fit model on training data
# model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

model.fit(x_train, y_train, validation_data=(x_test, y_test),
          batch_size=64,
          epochs=5, verbose=1)

# model.save(args["model"])
model.save('classifier_smile.model')
print(f"Model saved  ")
