# -*- coding: utf-8 -*-
"""
Application du réseau de neurone sur MNIST

@author: gwendal
"""


from keras.datasets import mnist
from keras.utils import np_utils

from Network import Network
from FCLayer import FCLayer
from ActivationLayer import ActivationLayer
from activation import tanh, tanh_prime
from loss import mse, mse_prime

# Téléchargement des images
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisation des données
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

y_train = np_utils.to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Réseau de neurone
net = Network()
net.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

# Test sur 3 exemples
out = net.predict(x_test[0:3])

net.print_error()

print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])