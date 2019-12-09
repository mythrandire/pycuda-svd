# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:10:24 2019

@author: Ananye
"""

import numpy as np

def maxind(A,size,k):
    m = k + 1
    for i in range(k+2,size):
        if(np.abs(A[k,i])>np.abs(A[k,m])):
            m = i
    return m
        
def update(k,t,e,changed,state):
    y = e[k]
    e[k] = y + t
    if changed[k] and y == e[k]:
        changed[k] = False
        state -= 1
    elif changed[k]==False and y!=e[k]:
        changed[k] = True
        state += 1
    return changed, state

def rotate(k,l,i,j,A,P,c,s):
    kl = c * A[k,l] - s * A[i,j]
    ij = s * A[k,l] + c * A[i,j]
    A[k,l] = kl
    A[i,j] = ij
    return A
            
def svd_pca_serial(N, P, D, U, sigma, VT, sigmam, sigman, dhat, k, retention):
    MAX_ITER = 1000000
    state = P
    num_iter = 0
    E = np.ones((P,P), dtype = np.float32)
    DT = D.T
    As = np.multiply(D.T,D)
    ind = np.zeros((P),dtype = np.int32)
    e = np.zeros((P), dtype = np.float32)
    changed = np.zeros((P), dtype = np.bool)
    for i in range(P):
        ind[i] = maxind(As,P,i)
        e[i] = As[i,i]
        changed[i] = True
    
    while (state and num_iter<MAX_ITER):
        m=0
        for i in range(1,P-1):
            if(np.abs(As[i,ind[i]])>np.abs(As[m,ind[m]])):
                m = i
        k = m
        l = ind[k]
        p = As[k,l]
        y = 0.5 * (e[l]-e[k])
        d = np.abs(y) + np.sqrt(p*p + y*y)
        r = np.sqrt(p*p + d*d)
        c = d/r
        s = p/r
        t = p*p/d
        if y<0:
            s = -s
            t = -t
        As[k,l] = 0.0
        changed1, state1 = update(k, -t, e, changed, state)
        changed, state = update(k, t, e, changed1, state1)
        
        for i in range(k):
            As = rotate(i,k,i,l,As,P,c,s)
        for i in range(k+1,l):
            As = rotate(k,i,i,l,As,P,c,s)
        for i in range(l+1,P):
            As = rotate(k,i,l,i,As,P,c,s)
        
        #rotate eigenvectors
        for i in range(P):
            eik = c * E[i,k] - s * E[i,l]
            eil = s * E[i,k] + c * E[i,l]
            E[i,k] = eik
            E[i,l] = eil
        
        ind[k] = maxind(As,P,k)
        ind[l] = maxind(As,P,l)
        
        num_iter += 1
        
    sum_eigenvalues = 0.0
    for i in range(P):
        sigma[i] = np.sqrt(e[i])
        sum_eigenvalues += e[i]
        
    
    
    return sigma

if __name__ =='__main__':
    A = np.random.randint(0,9,(3,3))
    A1 = np.dot(A,A.T)
    sigma = np.zeros((3),dtype = np.float32)
    s = svd_pca_serial(A1.shape[0],A1.shape[1],A1,None,sigma,None,None,None,None,None,None)
    print("S",s)
    s1,v1 = np.linalg.eig(A1)
    print("S1",s1)
    
    
            
            
        
        
    