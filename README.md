# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit categorization of scanned handwriting images, together with answer verification.
There are a number of handwritten digits in the MNIST dataset. The assignment is to place a handwritten digit picture into one of ten classes that correspond to integer values from 0 to 9, inclusively. The dataset consists of 60,000 handwritten digits that are each 28 by 28 pixels in size. In this case, we construct a convolutional neural network model that can categorise to the relevant numerical value.
## Neural Network Model

![WhatsApp Image 2023-09-15 at 21 59 20](https://github.com/Pavan-Gv/mnist-classification/assets/94827772/ddae93e9-ad0b-4368-8db3-91c8b001930c)


## DESIGN STEPS

### STEP 1: Import the required packages

### STEP 2: Load the dataset

### STEP 3: Scale the dataset

### STEP 4: Use the one-hot encoder

### STEP 5: Create the model

### STEP 6: Compile the model

### STEP 7: Fit the model

### STEP 8: Make prediction with test data and with an external data


## PROGRAM
```
Developed by: Palamakula Deepika
Registration no:212221240035
```

## Importing the required packages
~~~

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
~~~
## Loading the dataset
~~~
(X_train, y_train), (X_test, y_test) = mnist.load_data()
~~~
## Shape of training and testing data
~~~
X_train.shape
X_test.shape
~~~
## Scaling the data
~~~
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
~~~
## Implementing one hot encoder
~~~
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
~~~
## Scaling training and testing data
~~~
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
~~~
## Creating the model
~~~
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
~~~
## Compiling the model
~~~
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
~~~
## Fitting the model
~~~
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
~~~
## Creating Dataframe and getting history
~~~
metrics = pd.DataFrame(model.history.history)
~~~
## Plotting accuracy vs validated accuracy
~~~
metrics[['accuracy','val_accuracy']].plot()
~~~
## Plotting loss vs validated loss
~~~
metrics[['loss','val_loss']].plot()
~~~
## Implementing argmax
~~~
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
~~~
## Confusion matrix
~~~~
print(confusion_matrix(y_test,x_test_predictions))
~~~~
## Classification report
~~~
print(classification_report(y_test,x_test_predictions))
~~~
## Loading an external image
~~~
img = image.load_img('imagethree.jpg')
~~~
## Conversion of the image
~~~
img = image.load_img('imagethree.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
~~~
## Prediction 
~~~
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
~~~

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/Pavan-Gv/mnist-classification/assets/94827772/1254bda3-1c18-4cc9-aada-aa53c7439808)

![image](https://github.com/Pavan-Gv/mnist-classification/assets/94827772/dd286b2f-79c3-4337-94c5-d20ef6740a1f)



### Classification Report

![image](https://github.com/palamakuladeepika/mnist-classification/assets/94154679/f68a6910-2ef5-4d41-b979-f8d5ecda3ce2)


### Confusion Matrix

![image](https://github.com/palamakuladeepika/mnist-classification/assets/94154679/c3a61880-fa81-4764-8a81-da39a725d5ab)


### New Sample Data Prediction

![image](https://github.com/palamakuladeepika/mnist-classification/assets/94154679/e901b6bb-4619-4449-bf71-ce2569dc13ac)


## RESULT
Therefore a model has been successfully created for digit classification using mnist dataset.
