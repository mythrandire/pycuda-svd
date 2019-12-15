import numpy as np
import random
import matplotlib.pyplot as plt

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import compiler, gpuarray, tools

import time

"""
###############################################################################
                    define kernel codes and how to call them
###############################################################################
"""



class cuda_Transpose:
    """
    Class of functions pertaining to computing a matrix transpose
    using for loops and parallelized pyCuda code.
    """
    def __init__(self):

        # Kernal code:
        self.transpose_kernel_code = """
        __global__ void parTranspose(float *idata, float *odata, int cols, int rows) {
            int ix = blockIdx.x * blockDim.x + threadIdx.x;
            int iy = blockIdx.y * blockDim.y + threadIdx.y;
            if ((ix < cols) && (iy < rows)) {
                odata[iy*cols + ix] = idata[ix*rows + iy];
            }
        }
        """

    def transpose_parallel(self, a_cpu):
        self.x = a_cpu
        x_gpu = gpuarray.to_gpu(self.x)
        self.y_gpu = gpuarray.empty((self.x.shape[1], self.x.shape[0]), np.float32)

        M = self.x.shape[0]
        N = self.x.shape[1]

        mod = compiler.SourceModule(self.transpose_kernel_code)
        timing = []
        cTranspose = mod.get_function("parTranspose")
        cTranspose(
            x_gpu,
            self.y_gpu,
            np.int32(self.x.shape[0]),
            np.int32(self.x.shape[1]),
            block = (32, 32, 1),
            grid = (int(np.ceil(np.float32(M)/np.float32(32))), int(np.ceil(np.float32(N)/np.float32(32))), 1)
        )

        return self.y_gpu.get()


class gpuMul:
    def __init__(self):

        self.mul_kernel_code = """
            #define BLOCK_SIZE 32
            __global__ void kernel_MatMul(double *A, int rA, int cA, double *B, int rB, int cB, double *C) {
                int bIDx = blockIdx.x, bIDy = blockIdx.y, tIDx = threadIdx.x, tIDy = threadIdx.y;
                int row_ = bIDy * BLOCK_SIZE + tIDy;
                int col_ = bIDx * BLOCK_SIZE + tIDx;
                __shared__ double A_sub[BLOCK_SIZE][BLOCK_SIZE];
                __shared__ double B_sub[BLOCK_SIZE][BLOCK_SIZE];
                double C_sub = 0.0;
                for (int m = 0; m < (BLOCK_SIZE + cA - 1) / BLOCK_SIZE; m++) {
                    if (m * BLOCK_SIZE + tIDx < cA && row_ < rA) {
                        A_sub[tIDy][tIDx] = A[row_ * cA + m * BLOCK_SIZE + tIDx];
                    }
                    else {
                        A_sub[tIDy][tIDx] = 0.0;
                    }
                    if (m * BLOCK_SIZE + tIDy < rB && col_ < cB) {
                        B_sub[tIDy][tIDx] = B[(m * BLOCK_SIZE + tIDy) * cB + col_];
                    }
                    else {
                        B_sub[tIDy][tIDx] = 0.0;
                    }
                    __syncthreads();
            #pragma unroll
                    for (int k = 0; k < BLOCK_SIZE; k++) {
                        C_sub += A_sub[tIDy][k] * B_sub[k][tIDx];
                    }
                    __syncthreads();
                }
                if (row_ < rA && col_ < cB) {
                    C[cB * BLOCK_SIZE * bIDy + BLOCK_SIZE * bIDx + cB * tIDy + tIDx] = C_sub;
                }
            }
        """

    def MatMul(self, A, rA, cA, B, rB, cB):

            self.C_gpu = gpuarray.empty((A.shape[0], B.shape[1]), dtype = np.float32)
            self.A_gpu = gpuarray.to_gpu(A)
            self.B_gpu = gpuarray.to_gpu(B)

            mod = compiler.SourceModule(self.mul_kernel_code)
            dev_mul = mod.get_function("kernel_MatMul")

            grid_x = int(np.ceil(cB*1.0/32))
            grid_y = int(np.ceil(rA*1.0/32))

            dev_mul(
                self.A_gpu, rA, cA,
                self.B_gpu, rB, cA,
                self.C_gpu,
                block = (32, 32, 1),
                grid = (grid_x, grid_y, 1)
            )

            return self.C_gpu.get()

# computeParams.compute_params
class computeParams:
    def __init__(self):

        self.compute_params_kernel_code = """
            __global__ void kernel_compute_params(double *device_A, int P, int iter, double *device_sine, double *device_cosine, int *device_IterBlockToElem) {
                /*1 Block, P/2 threads: threadID t handles params for its alloted pair (for a particular device_iter)*/
                int localID = threadIdx.x;
                int k, l;
                double elem, y, d, r, c, s; //,t
                k = device_IterBlockToElem[iter*P+localID*2]; //row
                l = device_IterBlockToElem[iter*P+localID*2+1]; //col
                elem = device_A[k * P + l];
                y = (device_A[l * P + l] - device_A[k * P + k]) * 0.5;
                d = fabs(y) + sqrt(elem * elem + y * y);
                r = sqrt(elem * elem + d * d);
                if (r < EPSILON) {
                    c = 1.0;
                    s = 0.0;
                }
                else {
                    c = d / r;
                    s = y / fabs(y) * elem / r; //t=y/fabs(y)*p*p/d;
                }
                device_cosine[k * P + l] = c;
                device_sine[k * P + l] = s;
            }
        """

    def compute_params(self, A, P, iter, iterblock):
        self.A_gpu = gpuarray.to_gpu(A)
        self.dev_sin = gpuarray.empty((P, P))
        self.dev_cos = gpuarray.empty((P, P))
        self.iterBlock_device = gpuarray.to_gpu(iterblock)

        # self.iterBlock_device = gpuarray.empty((P-1)*P / 2 * 2), astype.int)
        mod = compiler.SourceModule(self.compute_params_kernel_code)
        compute_params_code = mod.get_function(kernel_compute_params)

        compute_params_code(
            self.A_gpu, P, iter,
            self.dev_sin,
            self.dev_cos,
            self.iterBlock_device,
            block = (32, 32, 1)
        )
        print('here!')

        return self.dev_sin.get(), self.dev_cos.get()




class dimUpdate:

    def __init__(self):

        self.row_update_kernel_code = """
            __global__ void kernel_row_update(int iter, double *device_A, double *device_X, int P, double *device_sine, double *device_cosine, int *device_IterBlockToElem) {
                int localID = threadIdx.x;
                int blockID = blockIdx.x;
                /*Based on blockID [total blocks=P/2], compute the corresponding two rows: p,q for device_iter*/
                __shared__ int row_pair[2];
                __shared__ double params[2]; //[sin_, cos_]
                if (localID == 0)            //to minimize global memory access latency at the cost of divergence
                {
                    row_pair[0] = device_IterBlockToElem[iter*P+blockID * 2];
                    row_pair[1] = device_IterBlockToElem[iter*P+blockID * 2 + 1];
                    params[0] = device_sine[row_pair[0] * P + row_pair[1]];
                    params[1] = device_cosine[row_pair[0] * P + row_pair[1]];
                }
                __syncthreads(); //all "P" threads in the block are synchronized and have access to row_pair(k,l) and params
                //CHECKPOINT: Can you reduce shared-memory bank conflicts here?
                int k = row_pair[0], l = row_pair[1];
                double sin_ = params[0], cos_ = params[1], elem_k=device_A[k*P+localID], elem_l=device_A[l * P + localID];
                /*Concurrent modifications to all row pairs(k,l) [different blocks]*/
                /*Concurrent modifications to different-column elements of a row pair: ["P" threads of the block]*/
                /*X is col-major, i.e. write in X-transpose*/
                device_X[localID * P + k] = elem_k * cos_ - elem_l * sin_;
                device_X[localID * P + l] = elem_k * sin_ + elem_l * cos_;
            }
        """

        self.col_update_kernel_code = """
            __global__ void kernel_col_update(int iter, double *device_A, double *device_X, int P, double *device_eigenvectors, double *device_sine, double *device_cosine, int *device_IterBlockToElem) {
                int localID = threadIdx.x;
                int blockID = blockIdx.x;
                /*Based on blockID [total blocks=P/2], compute the corresponding two cols: p,q for device_iter*/
                __shared__ int col_pair[2];
                __shared__ double params[2]; //[sin_, cos_]
                if (localID == 0)            //to minimize global memory access latency at the cost of divergence
                {
                    col_pair[0] = device_IterBlockToElem[iter*P+blockID * 2];
                    col_pair[1] = device_IterBlockToElem[iter*P+blockID * 2 + 1];
                    params[0] = device_sine[col_pair[0] * P + col_pair[1]];
                    params[1] = device_cosine[col_pair[0] * P + col_pair[1]];
                }
                __syncthreads(); //all "P" threads in the block are synchronized and have access to row_pair(k,l) and params
                //CHECKPOINT: Can you reduce shared-memory bank conflicts here? Is this better than computing pair(p,q) all over again
                int k = col_pair[0], l = col_pair[1];
                double sin_ = params[0], cos_ = params[1];
                /*Concurrent modifications to all row pairs(k,l) [different blocks]*/
                /*Concurrent modifications to different-column elements of a row pair: ["P" threads of the block]*/
                double new_eigen_k, new_eigen_l;
                /* col-wise access (inefficient):*/
                //device_A[localID * P + k] = device_X[k * P + localID] * cos_ - device_X[l * P + localID] * sin_;
                //device_A[localID * P + l] = device_X[k * P + localID] * sin_ + device_X[l * P + localID] * cos_;
                //new_eigen_k = device_eigenvectors[localID * P + k]*cos_ - device_eigenvectors[localID*P+l]*sin_;
                //new_eigen_l = device_eigenvectors[localID * P+k]*sin_ + device_eigenvectors[localID*P+l]*cos_;
                //device_eigenvectors[localID * P + k] = new_eigen_k;
                //device_eigenvectors[localID * P+l] = new_eigen_l;
                /*row-wise access (efficient):*/
                int kp = k*P + localID, lp = l *P+localID;
                device_A[kp] = device_X[kp] * cos_ - device_X[lp] * sin_;
                device_A[lp] = device_X[kp] * sin_ + device_X[lp] * cos_;
                new_eigen_k = device_eigenvectors[kp]*cos_ - device_eigenvectors[lp]*sin_;
                new_eigen_l = device_eigenvectors[kp]*sin_ + device_eigenvectors[lp]*cos_;
                device_eigenvectors[kp] = new_eigen_k;
                device_eigenvectors[lp] = new_eigen_l;
            }
        """

    def row_update(self, iter, A, P, sin, cos, iterBlock):
        self.A_device = gpuarray.to_gpu(A)
        self.dev_sin = gpuarray.to_gpu(sin)
        self.dev_cos = gpuarray.to_gpu(cos)
        self.iterBlock_device = gpuarray.to_gpu(iterBlock)
        self.X_device = gpuarray.empty((P, P))

        mod1 = compiler.SourceModule(row_update_kernel_code)
        row_update_code = mod1.get_function(kernel_row_update)
        if (P % 2 == 0):
            grid_size = P / 2
        else:
            grid_size = P / 2 + 1

        row_update_code(
            iter, self.A_device,
            self.X_device, P,
            self.dev_sin, self.dev_cos,
            self.iterBlock_device,
            block = (P, P, 1),
            grid = (grid_size, grid_size)
        )

        return self.X_device.get()

    def col_update(self, iter, A, X, P, sin, cos, iterBlock):
        self.A_device = gpuarray.to_gpu(A)
        self.dev_sin = gpuarray.to_gpu(sin)
        self.dev_cos = gpuarray.to_gpu(cos)
        self.iterBlock_device = gpuarray.to_gpu(iterBlock)
        self.X_device = gpuarray.to_gpu(X)
        self.device_eigenvectors = gpuarray.empty((P, P))

        if (P % 2 == 0):
            grid_size = P / 2
        else:
            grid_size = P / 2 + 1

        mod2 = compiler.SourceModule(col_update_kernel_code)
        col_update_code = mod2.get_function(kernel_col_update)

        col_update_code(
            iter, self.A_device,
            self.X_device, P,
            self.device_eigenvectors,
            self.dev_sin, self.device_cos,
            self.iterBlock_device,
            block = (P, P, 1),
            grid = (grid_size, grid_size)
        )

        return self.device_eigenvectors.get()

"""
###############################################################################
                                 On to PCA and SVD
###############################################################################
"""


def cudaSVD(N, P, D):

    # Perform SVD for D_T
    # Get eigen values and eigen vectors for D_T*D

    chess_params_kernel_code = """
        __device__ void chess_tourney_params(int P, int *row_pair, int iter) {
            //NOTE: here, row_pair is thread-local
            int localID = threadIdx.x;
            int index1, index2;
            index1 = (localID + iter) % (P - 1);
            if (localID != 0) {
                index2 = (P - localID + iter - 1) % (P - 1);
            }
            else {
                index2 = P - 1;
            }
            row_pair[0] = min(index1, index2);
            row_pair[1] = max(index1, index2);
        }
    __global__ void kernel_compute_all_chess_params(int P, int *device_IterBlockToElem) {
        int blockID = blockIdx.x;
        //each ONE of the P-1 blocks is responsible for computing chess-tourney parameters for ONE of the P-1 iterations
        int index = blockID*P + threadIdx.x*2;
        int *row_pair = (int *) malloc(sizeof(int)*2);
        chess_tourney_params(P, row_pair, blockID);
        device_IterBlockToElem[index] = row_pair[0]; //|=(P-1)X(P/2*2)
        device_IterBlockToElem[index+1] = row_pair[1];
        free(row_pair);
    }
    """
    ###########################################################################
    # STREAM PARALLELIZATION
    t = cuda_Transpose()
    g = gpuMul()

    iterBlock_device = gpuarray.empty(((P-1), np.int(np.ceil(P/2)), 2), np.int32)
    mod = compiler.SourceModule(chess_params_kernel_code)
    dev_chess = mod.get_function("kernel_compute_all_chess_params")

    dev_chess(np.int32(P), iterBlock_device, block = (np.int(P-1), np.int(np.ceil(P/2)), 1),
              grid = (np.int(P-1), np.int(P-1),1))
    iterBlock = iterBlock_device.get()

    # cudaAsynccopy something
    D_T = t.transpose_parallel(D)
    ###########################################################################

    A = g.MatMul(D_T, np.int32(P), np.int32(N), D, np.int32(N), np.int32(P))
    eigenvectors = np.ones((P, P), np.float32)
    iter = 0
    counter = 0

    MAX_SWEEPS = 30
    EPSILON = 1e-4
    THRESHOLD = 1e-4
    MAX_BLOCK_SIZE = 1024
    MAX_SWEEPS = 30
    MAX_ITER = 10000000
    MULTIPLY_BLOCK_SIZE = 64

    iter = 0
    while(iter < P - 1):
        # Compute rotation parameters: sine and cosine
        # for all (p, q), q>p
        print('here!')
        sin, cos = computeParams.compute_params(A, P, iter, iterBlock)

        # row update
        X = dimUpdate.row_update(iter, A, P, sin, cos, iterBlock)

        # col update
        eigenvectors = dimUpdate.col_update(iter, A, X, P, sin, cos, iterBlock)
        iter += 1

    eigenvectors_T = t.transpose_parallel(eigenvectors)

    eigenvalues = np.ones(P)
    e_indices = np.ones(P)

    for i in range(P):
        eigenvalues[i] = A[i * P + i]
        e_indices[i] = i

     # sort eigenvalues in descending order along with corresponding indices
    eigenvalues = np.sort(eigenvalues)
    new_indices = np.argsort(e_indices)
    eigenvalues = np.flip(eigenvalues)
    new_indices = np.flip(new_indices)

    # compute sigma
    SIGMA = np.ones(P, np.float32)
    sum_variance = 0.0
    sum_variance = np.sum(eigenvalues)
    SIGMA = np.sqrt(eigenvalues)

    # compute U
    for i in range(P):
        for j in range(P):
            U[i][j] = E[i][new_indices[j]]

    # calculate V_T

    inv_SIGMA = np.ones((N, P), np.float32)
    for i in range(P):
        inv_SIGMA[i][i] = 1.0 / SIGMA[i]

    U_T = t.transpose_parallel(U)
    prod = g.MatMul(inv_SIGMA, N, P, U_T, P, P)
    V_T = g.MatMul(prod, N, P, D_T, P, N)

    return SIGMA, U, V_T


if __name__ =='__main__':
    random.seed(1)
    A = np.random.randint(0,9,(3,3)).astype(np.float32)
    #initialize A
    #A = np.array([[4,0],[3,-5]])
    #A = A.astype(np.float32)

    #calculate covaiance matrix of A for numpy verification
    A1 = np.dot(A.T,A)

    #serial jacobi method for SVD
    s, u, vt = cudaSVD(A.shape[0],A.shape[1],A)
    print(A.shape[1])

    #numpy verification
    s1,v1 = np.linalg.eig(A1)

    #print results
    print("Serial Eigenvalues: \n", s)
    print("Numpy Eigenvalues: \n",np.sqrt(s1))
    print("Serial Eigenvectors: \n", u)
    print("Numpy Eigenvectors: \n", v1)
