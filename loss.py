# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:29:00 2021

@author: gwend
"""

import numpy as np

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;