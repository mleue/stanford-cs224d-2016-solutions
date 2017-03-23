#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z1 = np.dot(data, W1) + b1
    hidden = sigmoid(z1)
    z2 = np.dot(hidden, W2) + b2
    yhat = softmax(z2)
    cost = np.sum(-1 * labels * np.log(yhat))
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    grad_dj_dz2 = yhat - labels #20x10 (batches x classes)
    grad_dz2_dhidden = W2 #5x10 (hidden x outputs)
    grad_dhidden_dz1 = sigmoid_grad(hidden) #20x5 (batches x hidden)
    grad_dz1_dx = W1 #10x5 (inputs x hidden)

    #(5x20)x(20x10) = 5x10
    gradW2 = np.dot(hidden.T, grad_dj_dz2)
    #summing to sum the error over all 20 batches for each of the 10 classes (20x10 -> 1x10)
    gradb2 = np.sum(grad_dj_dz2, axis=0)


    #(20x5)*((20x10)x(10x5)) = 20x5
    grad_dj_dz1 = grad_dhidden_dz1 * np.dot(grad_dj_dz2, grad_dz2_dhidden.T)

    #(10x20)x(20x5) = 10x5
    gradW1 = np.dot(data.T, grad_dj_dz1) 
    #summing to sum the error over all 20 batches for each of the 5 hidden units (20x5 -> 1x5)
    gradb1 = np.sum(grad_dj_dz1, axis=0)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
