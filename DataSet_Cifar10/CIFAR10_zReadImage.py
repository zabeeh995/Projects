###############read image using trained model and classify object using cv2 function

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, layers

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

classNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model = models.load_model("classifier_image.model")

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

path = "anyimage"           #### enter image path to detrect object in picture within 10 classses
img = cv2.imread(path)

plt.imshow(img,cmap=plt.cm.binary)

prediction = model.predict(np.array([img])/255)         #converting image to nummpy so it can be compare with testing images and labels
index = np.argmax(prediction)                       #maximum neuron which wuld indicate the particular class label
print(f" Prediction is {classNames[index]}")


