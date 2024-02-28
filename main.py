import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf 

def instantiate_weights():
    W1 = np.random.rand(64, 784) - 0.5
    b1 = np.random.rand(64, 1) - 0.5
    W2 = np.random.rand(28, 64) - 0.5
    b2 = np.random.rand(28, 1) - 0.5
    W3 = np.random.rand(10, 28) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

def ReLU(dotp):
    return np.maximum(dotp, 0)

def ReLU_deriv(Z):
    return Z > 0

def softmax(dotp):
    exps = np.exp(dotp - np.max(dotp)) # Stability improvement for softmax
    return exps / np.sum(exps, axis=0, keepdims=True)
    
def forward_prop(W1, b1, W2, b2, W3, b3, input_vector): 
    dotp1 = W1.dot(input_vector) + b1
    activation1 = ReLU(dotp1)
    dotp2 = W2.dot(activation1) + b2
    activation2 = ReLU(dotp2)
    dotp3 = W3.dot(activation2) + b3
    activation3 = softmax(dotp3)
    return dotp1, activation1, dotp2, activation2, dotp3, activation3

def one_hot(labels):
    one_hot_labels = np.zeros((10, labels.size)) 
    one_hot_labels[labels, np.arange(labels.size)] = 1
    return one_hot_labels

def backward_prop(dotp1, activation1, dotp2, activation2, dotp3, activation3, W1, W2, W3, input_vector, labels, batch_size):
    one_hot_labels = one_hot(labels)  
    derivative_dotp3 = activation3 - one_hot_labels
    derivative_W3 = 1 / batch_size * derivative_dotp3.dot(activation2.T)
    derivative_b3 = 1 / batch_size * np.sum(derivative_dotp3, axis=1, keepdims=True)
    
    derivative_dotp2 = W3.T.dot(derivative_dotp3) * ReLU_deriv(dotp2)
    derivative_W2 = 1 / batch_size * derivative_dotp2.dot(activation1.T)
    derivative_b2 = 1 / batch_size * np.sum(derivative_dotp2, axis=1, keepdims=True)
    
    derivative_dotp1 = W2.T.dot(derivative_dotp2) * ReLU_deriv(dotp1)
    derivative_W1 = 1 / batch_size * derivative_dotp1.dot(input_vector.T)
    derivative_b1 = 1 / batch_size * np.sum(derivative_dotp1, axis=1, keepdims=True)
    
    return derivative_W1, derivative_b1, derivative_W2, derivative_b2, derivative_W3, derivative_b3

def update_params(W1, b1, W2, b2, W3, b3, derivative_W1, derivative_b1, derivative_W2, derivative_b2, derivative_W3, derivative_b3, learning_rate):
    W1 -= learning_rate * derivative_W1
    b1 -= learning_rate * derivative_b1    
    W2 -= learning_rate * derivative_W2  
    b2 -= learning_rate * derivative_b2
    W3 -= learning_rate * derivative_W3  
    b3 -= learning_rate * derivative_b3
    return W1, b1, W2, b2, W3, b3

def get_predictions(activation3):
    return np.argmax(activation3, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, learning_rate, epochs, batch_size):
    W1, b1, W2, b2, W3, b3 = instantiate_weights()
    m = X.shape[0]  # number of examples

    for i in range(epochs):
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        Y_shuffled = Y[permutation]

        for j in range(0, m, batch_size):
            end = j + batch_size
            mini_batch_X = X_shuffled[j:end]
            mini_batch_Y = Y_shuffled[j:end]

            dotp1, activation1, dotp2, activation2, dotp3, activation3 = forward_prop(W1, b1, W2, b2, W3, b3, mini_batch_X.T)
            derivative_W1, derivative_b1, derivative_W2, derivative_b2, derivative_W3, derivative_b3 = backward_prop(
                dotp1, activation1, dotp2, activation2, dotp3, activation3, W1, W2, W3, mini_batch_X.T, mini_batch_Y, batch_size
            )
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, derivative_W1, derivative_b1, derivative_W2, derivative_b2, derivative_W3, derivative_b3, learning_rate)

        if i % 10 == 0:
            predictions = get_predictions(activation3)
            accuracy = get_accuracy(predictions, mini_batch_Y)
            print("Epoch: {}, Accuracy: {}".format(i, accuracy))

    return W1, b1, W2, b2, W3, b3

# Load and process the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.array(x_train.reshape(x_train.shape[0], -1))
x_test = np.array(x_test.reshape(x_test.shape[0], -1))
m, n = x_train.shape


# Train
W1, b1, W2, b2 = gradient_descent(x_train, y_train, 0.01, 250, 64)