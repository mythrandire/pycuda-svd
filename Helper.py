# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:31:53 2019

@author: Ananye
"""
def s_maxind(A,size,k):
    #function to find index of maximum element along each column after k
    m1 = k + 1
    for i in range(k+2,size):
        if(abs(A[k][i])>abs(A[k][m1])):
            m1 = i
    return m1
        
def s_update(k,t,e,changed,state):
    #function to update state of changed eigenvalue and its corresponding state
    y = e[k]
    e[k] = y + t
    if (changed[k]==True and y == e[k]):
        changed[k] = False
        state -= 1
    elif (changed[k]==False and y!=e[k]):
        changed[k] = True
        state += 1
    return changed, state

def s_rotate(k,l,i,j,A,c,s):
    #function to rotate matrix according to given sine and cos functions
    kl = c * A[k][l] - s * A[i][j]
    ij = s * A[k][l] + c * A[i][j]
    A[k][l] = kl
    A[i][j] = ij
    return A