# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:10:24 2019

@author: Ananye
"""
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from Helper import s_maxind, s_update, s_rotate            

def svd_pca_serial(N, P, D):
    MAX_ITER = 1000000
    state = P
    num_iter = 0
    
    #initializing eigenvector matrix to diag{1xP}
    E = np.diag(np.ones((P), dtype = np.float32))
    U = np.empty((P,P), dtype = np.float32)
    
    #calculating covariance matrix
    DT = D.T
    As = np.dot(DT,D)
    
    #initializing some useful variables
    ind = np.empty((P),dtype = np.int32)
    e = np.empty((P), dtype = np.float32)
    changed = np.zeros((P), dtype = np.bool)
    
    #setting ind to index of maximum value in each column and setting 
    #eigenvalues to diagonal elements of covariance matrix
    for i in range(P):  
        ind[i] = s_maxind(As,P,i)
        e[i] = As[i][i]
        changed[i] = True
    t0 = time.time()
    #start iteration of jaboi method
    
    while (state>0 and num_iter<MAX_ITER):
        m=0
        #find index of maximum element in each column
        for i in range(1,P-1):
            if(abs(As[i][ind[i]])>abs(As[m][ind[m]])):
                m = i
        
        #calculate sine, cosine values for rotation and tolerance for stopsign
        k = m
        l = ind[k]
        p = As[k][l]
        y = 0.5 * (e[l]-e[k])
        d = abs(y) + np.sqrt(p*p + y*y)
        r = np.sqrt(p*p + d*d)
        c = d/r
        s = p/r
        t = p*p/d
        if y<0:
            s = -s
            t = -t
        As[k][l] = 0.0
        
        #update state of eigenvalues if their values have been changed
        changed1, state1 = s_update(k, -t, e, changed, state)
        changed, state= s_update(l, t, e, changed1, state1)
        #rotate covariance matrix on offdiagonal elements to reduce it to 
        #eigenvalue matrix
        for i in range(0,k):
            As = s_rotate(i,k,i,l,As,c,s)
        for j in range(k+1,l):
            As = s_rotate(k,j,j,l,As,c,s)
        for z in range(l+1,P):
            As = s_rotate(k,z,l,z,As,c,s)
        
        #rotate eigenvectors
        for i in range(0,P):
            ik = c * E[i][k] - s * E[i][l]
            il = s * E[i][k] + c * E[i][l]
            E[i][k] = ik
            E[i][l] = il
        
        ind[k] = s_maxind(As,P,k)
        ind[l] = s_maxind(As,P,l)
        
        num_iter += 1
        
    sum_eigenvalues = 0.0
    sigma = np.empty((P),dtype = np.float32)
    
    #sort eigenvalues in descending order along with corresponding indices
    e = np.sort(e)
    newind = np.argsort(e)
    e = np.flip(e)
    newind = np.flip(newind)
    
    #calculate singular values of D
    for i in range(P):
        sigma[i] = np.sqrt(e[i])
        sum_eigenvalues += e[i]
    
    #calculate eigenvector U of D
    for i in range(P):
        for j in range(P):
            U[i][j] = E[i][newind[j]]
    
    #calculating eigenvector VT of D        
    inv_sigma = np.zeros((N,P),dtype = np.float32)
    for i in range(P):
        inv_sigma[i][i] = 1.0/sigma[i]
    UT = U.T
    prod = np.dot(inv_sigma,UT)
    VT = np.dot(prod, DT)
    t1 = time.time()
    #return calculated eigenvalue and eigenvector matrices
    return sigma, U, VT, t1-t0

if __name__ =='__main__':
    random.seed(1)
    t = []
    for i in range(1,15):
        A = np.random.randint(0,9,(2*i,2*i)).astype(np.float32)
    #initialize A
    #A = np.array([[4,0],[3,-5]])
    #A = A.astype(np.float32)
    
    #calculate covaiance matrix of A for numpy verification
        A1 = np.dot(A.T,A)
    
    #serial jacobi method for SVD
        s, u, vt, tt = svd_pca_serial(A.shape[0],A.shape[1],A)
        t.append(tt)
    #numpy verification
        s1,v1 = np.linalg.eig(A1)
    
    #print results
        print("Serial Eigenvalues: \n", s)
        print("Numpy Eigenvalues: \n",np.sqrt(s1))
        print("Serial Eigenvectors: \n", u)
        print("Numpy Eigenvectors: \n", v1)
    #print timing results
    print("Time taken serially: \n")
    plt.plot(range(1,15),t, label = 'Serial Training Time')
    plt.xlabel("Iteration of input array (x2)")
    plt.ylabel("Running time(s)")
    plt.legend()
    plt.savefig('final.png')  
    
    
            
            
        
        
    