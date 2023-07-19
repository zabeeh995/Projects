import lenet
from keras.optimizers import SGD

from keras.datasets import mnist

print("[INFO] accessing MNIST...")
#dataset = mnist.load_data()



#if K.image_data_format() == "channels_first":
 #   data = data.reshape(data.shape[0], 1, 28, 28)

# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
#else:
#    data = data.reshape(data.shape[0], 28, 28, 1)

(trainX, testX), (trainY, testY) = mnist.load_data()
print("tarining 111111111")

trainX, testX = trainX / 255, testX / 255
print("tarining 222222222")

# convert the labels from integers to vectors


opt = SGD(learning_rate=0.01)
print("tarining 3333333")
model = lenet.build(width=28, height=28, depth=1, classes=10)
print("tarining  44444")
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(trainX, trainY,  epochs=10)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
