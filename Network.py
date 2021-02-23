# -*- coding: utf-8 -*-
"""
Classe de la structure du réseau de neurone

@author: gwendal
"""

import matplotlib.pyplot as plt

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.array_loss = []
        self.epochs = 0

    # Ajout d'une couche
    def add(self, layer):
        self.layers.append(layer)

    # Appliquer une fonction de cout au modèle
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # Predire à partir d'une valeur
    def predict(self, input_data):
        # Nb d'exemple
        exemple = len(input_data)
        result = []

        for i in range(exemple):
            # Propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result
    
    # Affichage de la courbe d'erreur
    def print_error(self):
        plt.plot([i for i in range(0,self.epochs)], self.array_loss)
        plt.ylabel('Error')
        plt.xlabel('Epochs')
        
        plt.xlim(0, self.epochs)
        plt.ylim(0, max(self.array_loss))
        
        plt.show()
        
    # Entrainement du réseau 
    def fit(self, x_train, y_train, epochs, learning_rate):
        # nb d'exemple
        exemple = len(x_train)
        self.epochs = epochs
        
        # training loop
        for i in range(epochs):
            err = 0
            for j in range(exemple):
                
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Calcul de la fonction de cout pour l'affichage
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calcul de la moyenne des erreurs
            err /= exemple
            self.array_loss.append(err)
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))