# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:31:53 2019

@author: Ananye
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

import pycuda.autoinit

def s_maxind(A,size,k):
    #function to find index of maximum element in each row 
    t0 = time.time()
    m1 = k + 1
    for i in range(k+2,size):
        if(abs(A[k][i])>abs(A[k][m1])):
            m1 = i
    t1 = time.time()
    return m1
        
def s_update(k,t,e,changed,state):
    #function to update state of changed eigenvalue and its corresponding state
    t0 = time.time()
    y = e[k]
    e[k] = y + t
    if (changed[k]==True and y == e[k]):
        changed[k] = False
        state -= 1
    elif (changed[k]==False and y!=e[k]):
        changed[k] = True
        state += 1
    t1 = time.time()
    return changed, state

def s_rotate(k,l,i,j,A,c,s):
    #function to rotate matrix according to given sine and cos functions
    t0 = time.time()
    kl = c * A[k][l] - s * A[i][j]
    ij = s * A[k][l] + c * A[i][j]
    A[k][l] = kl
    A[i][j] = ij
    t1 = time.time()
    return A

