# -*- coding: utf-8 -*-
"""
Fonction d'activation

@author: gwendal
"""

import numpy as np

# Fonction d'activation et ces dérivées
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;
