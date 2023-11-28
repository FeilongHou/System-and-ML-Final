import tensorflow as tf
import os, psutil
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

data = keras.datasets.fashion_mnist

# load in data
# this data set have label between 0-9
(train_images, trian_labels), (test_images, test_labels) = data.load_data()
# class name corresponding to the label
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.imshow(train_images[7], cmap = plt.cm.binary)
plt.show()

# now we want to flatten the data where originally were 28x28 matrix, become a 1D array 782 elements
# input layer: 782 inputs
# create model
start = time.time()
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),   # Input layer
    keras.layers.Dense(256, activation="sigmoid"),    # 2nd layer full connected(Dense) with Input layer relu is max(0,x)
    keras.layers.Dense(128, activation="sigmoid"),
    keras.layers.Dense(128, activation="sigmoid"),
    keras.layers.Dense(10, activation="softmax")   # Output layer "softmax" all output value add up to be 1, pick the biggest   
    ])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
model.fit(train_images, trian_labels, epochs = 5) # epochs mean how many times the model will see this information 
                                                  # it randomlly pick images and labels (play around)
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
# evaluate
end = time.time()
total_time = end - start
print(f"Total time is {total_time}")
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested Acc: " + str(test_acc))

# now we want to predict the model
prediction = model.predict(test_images)   # if only predict 1 data point, put in square braket 

for i in range(10,20,1):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])  # ith prediction "argmax" picks the max value and return its index
    plt.show()