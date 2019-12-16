# svd_4750/JPCA
Project repo for E4750 Final Project: SVD

## Introduction
Principal Component Analysis is a useful dimensionality reduction tool used to essentially summarize high dimensional data. A key underlying step in PCA is Singular Value Decomposition. SVD can be computed using a variety of algorithms. Of the many available algorithms that can be used to iteratively converge to the singular decomposition of a matrix, we demonstrate that the Given’s rotation calculation in the Jacobi method is parallelizable and can thus allow for faster computation of the eigenvalues and eigenvectors. 

## Getting Started
Serial code: serial.py 

Serial code requires Helper.py function. Please store both files in same directory while running.

Parallel code was tested on Nvidia GeForce RTX2070 and Nvidia GeForce Tesseract.

## Conclusion
One of the major issues of efficient implementation of CUDA code is efficient device memory management. The total amount of memory on the GPU device is limited and hence it is not uncommon to run into memory errors for larger data matrices.

Although our code works for data matrices upto size 17, we can further strive to improve the memory efficiency by introducing memory pooling or deallocating memory more regularly while interfacing with pycuda.

A parallel implementation of Jacobi SVD algorithm was thus implemented by us. 

### Contributors
Ananye Pandey (ap3885@columbia.edu) UNI: ap3885
Dwiref Oza (dso2119@columbia.edu UNI: dso2119
