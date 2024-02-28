# Overview
I am currently learning linear algebra, and I thought the above implementation of a neural network would be a great way to deepen my understanding of machine learning and linear albegra. 

To understand how this code works, all you need is a rudimentary understanding of the dot-product, array manipulation and the core principles of basic neural networks. Within the implementation details below, I have linked a number of youtube videos that - in my opinion - explain the key concepts of neural networks extremely well (e.g chain rule, dot-product, etc).


# Implementation Details
To start, the script in `main.py` is a simple neural network (NN) implemented using NumPy for digit recognition on the MNIST dataset (Note: I did use tensorflow to access the MNIST dataset as I didn't want to store it locally). The NN is structured with three layers: the first with 64 neurons, the second with 28, and the third (output layer) with 10 neurons, corresponding to the 10 possible digits. The `instantiate_weights` function initializes the weights and biases for each layer with random values.

The ReLU function is used for the first two layers to output either 0 or the input value, whichever is higher. The softmax function is applied to the output of the third layer to obtain a probability distribution over the 10 possible digits.

During forward propagation (`forward_prop`), the input vector (a flattened MNIST 28x28 image) is passed through the network, undergoing [linear transformations](https://www.youtube.com/watch?v=LyGKycYT2v0&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=9) (i.e dot product) and non-linear activations.

The training process uses mini-batch gradient descent, as defined in the `gradient_descent` function. In each epoch, the dataset is shuffled and divided into mini-batches. For each batch, forward propagation is performed, followed by backward propagation (`backward_prop`), where the gradient of the loss - with respect to each parameter - is calculated using the [chain rule](https://www.youtube.com/watch?v=wl1myxrtQHQ&t=2s). These gradients are used to update the parameters in the `update_params` function with the aim of minimizing the loss. The `get_accuracy` function calculates the prediction accuracy of the current model against the mini-batch labels.

The `one_hot` function encodes labels (e.g 4 - representing the number in the image) as a one-hot representation (e.g `[0,0,0,1,0,0...]`) for computing the loss from the softmax probabilities. Finally, after training for a specified number of epochs, the model's weights and biases are returned, which can be used for making predictions on new data.
