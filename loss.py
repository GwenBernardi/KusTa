# -*- coding: utf-8 -*-
"""
Fonction de coût

@author: gwendal
"""

import numpy as np

# Fonctions de coût et ces dérivées
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;