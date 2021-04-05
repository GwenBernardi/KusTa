# -*- coding: utf-8 -*-
"""
Classe Mère de toutes les layers 

@author: gwendal
"""

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    # Implémentation de la forward propagation
    def forward_propagation(self, input):
        raise NotImplementedError
    
    # Implémentation de la backward_propagation
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError