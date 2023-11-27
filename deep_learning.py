# /*-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: deep_learning.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU CAN USE ANY PYTHON LIBRARY TO COMPLETE YOUR CODE.

# importing the libraries
import tensorflow as tf 
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def build_model(n_hidden, n_neurons_hidden, n_neurons_output, learning_rate):
    # Creating the Neural Network using the Sequential API
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))  # input layer

    # iterate over the number of hidden layers to create the hidden layers:
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons_hidden, activation="relu"))  # hidden layer with ReLU activation function

    # output layer
    model.add(keras.layers.Dense(n_neurons_output, activation="softmax"))  # output layer with softmax activation function

    # defining the learning rate
    opt = keras.optimizers.SGD(learning_rate)

    # Compiling the Model specifying the loss function and the optimizer to use.
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

# Using Keras to Load the Dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# creating a validation set and scaling the features
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# For Fashion MNIST, we need the list of class names to know what we are dealing with.
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Iterate over the number of hidden layers, number of neurons in each hidden layer, and the learning rate.
for n_hidden in [2, 5, 10]:  # looking for the best parameters w.r.t the number of hidden layers
    for n_neurons in [10, 50, 100]:  # looking for the best parameters w.r.t the number of neurons
        for l_rate in [0.01, 0.05, 0.1]:  # looking for the best parameters w.r.t the learning rate

            # build the model for each combination by calling the function:
            model = build_model(n_hidden, n_neurons, 10, l_rate)

            # To train the model
            history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))

            # Calculate the accuracy of this neural network and store its value if it is the highest so far.
            accuracy = model.evaluate(X_test, y_test)[1]
            print("Accuracy for current model:", accuracy)

            # Print the highest accuracy so far and the corresponding parameters
            if accuracy > highestAccuracy:
                highestAccuracy = accuracy
                print("Highest accuracy so far:", highestAccuracy)
                print("Parameters: Number of Hidden Layers:", n_hidden, ", Number of Neurons:", n_neurons, ", Learning Rate:", l_rate)
                print()

# After generating all neural networks, print the summary of the best model found
print(model.summary())
img_file = './model_arch.png'
tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)

# Plotting the learning curves of the best model
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()
