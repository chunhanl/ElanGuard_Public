# -*- coding: utf-8 -*-
import numpy as np

def l2_distance(a, b):
    return np.sqrt(np.sum( np.square( a - b ))+1e-12 )        

def cos_distance(a, b):
    return np.dot(a,b)        