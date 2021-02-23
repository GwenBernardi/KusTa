# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:27:36 2021

@author: gwend
"""

import matplotlib.pyplot as plt

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.array_loss = []
        self.epochs = 0

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result
    
    def print_error(self):
        plt.plot([i for i in range(0,self.epochs)], self.array_loss)
        plt.ylabel('Error')
        plt.xlabel('Epochs')
        
        plt.xlim(0, self.epochs)
        plt.ylim(0, max(self.array_loss))
        
        plt.show()
        
    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)
        self.epochs = epochs
        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            self.array_loss.append(err)
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))