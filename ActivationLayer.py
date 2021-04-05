# -*- coding: utf-8 -*-
"""
Classe des couches d'Activations

@author: gwendal
"""

from Layer import Layer

# Hérite de la classe Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # Retourne l'input avec l'application de la fonction d'activation
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Learning Rate pas utilisé étant donné qu'il n'y a pas de paramètre 
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error