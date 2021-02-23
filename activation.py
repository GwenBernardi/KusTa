# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:29:20 2021

@author: gwend
"""

import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;
