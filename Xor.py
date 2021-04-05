# -*- coding: utf-8 -*-
"""
Application du réseau de neurone sur une porte logique XOR

@author: gwendal
"""

import numpy as np

from Network import Network
from FCLayer import FullConectedLayer
from ActivationLayer import ActivationLayer
from activation import tanh, tanh_prime
from loss import mse, mse_prime

# Donnée d'apprentissage
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# Création du réseau
model = Network()
model.add(FullConectedLayer(2, 3))
model.add(ActivationLayer(tanh, tanh_prime))
model.add(FullConectedLayer(3, 1))
model.add(ActivationLayer(tanh, tanh_prime))

# Entrainement
model.use(mse, mse_prime)
model.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# Test
model = model.predict(x_train)

model.print_error()

print(model)