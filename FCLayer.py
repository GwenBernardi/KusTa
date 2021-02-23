# -*- coding: utf-8 -*-
"""
Classe des couches FullConnected

@author: gwendal
"""

from Layer import Layer
import numpy as np

# Héritage de la classe Layer
class FullConectedLayer(Layer):
    
    # input = Nombre de neurone d'entrée
    # output = Nombre de neurone de sortie
    def __init__(self, input, output):
        self.weights = np.random.rand(input, output) - 0.5
        self.bias = np.random.rand(1, output) - 0.5

    # Propagation vers les prochaines couches :
    def forward_propagation(self, input_data):
        self.input = input_data
        # Y = XW + B
        self.output = np.dot(self.input, self.weights) + self.bias
        # Fonction d'activation sur une couche d'activation
        return self.output

    # BackPropagation vers les couches précédentes :
    def backward_propagation(self, output_error, learning_rate):
        # Input error pour l'output_error de la couche d'avant
        # X = dE/dY*Wt
        input_error = np.dot(output_error, self.weights.T)
        # Erreur de la couche des poids : W = Xt * dE/dY
        weights_error = np.dot(self.input.T, output_error)
        
        # Erreur du biais = dE/dY

        # Actualisation des paramètres
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        
        return input_error