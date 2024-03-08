# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:30:12 2023

@author: yihanZzz
"""

import time
from numba import cuda, float64, int32
#from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import math
import cupy as cp

@cuda.jit(device = True)
def get_T1(i, j, dW1, dW2, Xbd, Vbd, uniform_St_1, St_1, St_T, uniform_Vt_1, Vt_1, Vt_T):
    fxl1 = 1 / (1 + math.exp(-mu * Xbd[i]))
    fxu1 = 1 / (1 + math.exp(-mu * Xbd[i+1]))
    fxl2 = 1 / (1 + math.exp(-mu * Vbd[j]))
    fxu2 = 1 / (1 + math.exp(-mu * Vbd[j+1]))
    for i in range(noi):
        St_1[i,0] = -(1 / mu) * math.log(1/(fxl1 + uniform_St_1[i,0] * (fxu1 - fxl1)) - 1)
        Vt_1[i,0] = -(1 / mu) * math.log(1/(fxl2 + uniform_Vt_1[i,0] * (fxu2 - fxl2)) - 1)
        Vt_T[i,0] = abs(Vt_1[i,0] + kappa*(theta - Vt_1[i,0])*dt + sigma*math.sqrt(Vt_1[i,0]*dt)*dW1[i,0])
        St_T[i,0] = St_1[i,0] + (r - 0.5 * Vt_1[i,0]) * dt + math.sqrt(Vt_1[i,0]) \
        * (rho * dW1[i,0] + math.sqrt(1 - rho**2) * dW2[i,0]) * math.sqrt(dt)
    

@cuda.jit(device = True)
def compute_Xt(St_T, Vt_T, Xt): #form polynomials
    for i in range(noi):
        Xt[i,0] = 1
        Xt[i,1] = 1 - St_T[i,0]
        Xt[i,2] = 1 - Vt_T[i,0]

@cuda.jit(device = True)
def compute_Yt(K, St_T, Xt, betaYt, Yt): #compute least square fitted value
    for i in range(noi):
        X=0
        for j in range(NX):
            X += Xt[i,j]*betaYt[i,j]
        Yt[i,0] =  X
        Yt[i,0] = max(max(K - math.exp(St_T[i,0]), 0),Yt[i,0])
        Yt[i,0] = Yt[i,0] - r * Yt[i,0]*dt

@cuda.jit(device = True)
def searchsorted(Xbd, St_T, indbdX, Vbd, Vt_T, indbdV):
    for i in range(noi):
        j = 0
        k = 0
        if (St_T[i,0] <= Xbd[0]):
            indbdX[i,0] = 0
        elif (St_T[i,0] >= Xbd[-1]):
            indbdX[i,0] = noh
        else:
            while (St_T[i,0] > Xbd[j]):
                j += 1
            indbdX[i,0] = j-1
        if (Vt_T[i,0] <= Vbd[0]):
            indbdV[i,0] = 0
        elif (Vt_T[i,0] >= Vbd[-1]):
            indbdV[i,0] = noh
        else:
            while (Vt_T[i,0] > Vbd[k]):
                k += 1
            indbdV[i,0] = k-1

@cuda.jit(device=True)
def CholeskyInc(A, L):
    n = A.shape[0]
    for i in range(n):
        for j in range(i):
            for k in range(j):
                A[i, j] -= A[i, k] * A[j, k]
            A[i, j] /= A[j, j]
        for k in range(i):
            A[i, i] -= A[i, k] * A[i, k]
        A[i, i] = math.sqrt(A[i, i])
    for i in range(n):
        for j in range(i,n):
            L[j,i] = A[j,i]

@cuda.jit(device=True)
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp

#user defined identity matrix maker for local memory
@cuda.jit(device=True)
def identity(mat0):
    for r in range(mat0.shape[0]):
        for c in range(mat0.shape[1]):
            if r == c:
                mat0[r, c] = 1
            else:
                mat0[r, c] = 0
    return mat0

# user defined matrix copy
@cuda.jit(device=True)
def copy_matrix(mat0, mat1):
    for r in range(mat0.shape[0]):
        for c in range(mat0.shape[1]):
            mat1[r,c] = mat0[r,c]

# user defined matrix inversion
@cuda.jit(device=True)
def invert_matrix(AM, IM):
    for fd in range(AM.shape[0]):
        fdScaler = 1.0 / AM[fd,fd]

        for j in range(AM.shape[0]):
            AM[fd,j] *= fdScaler
            IM[fd,j] *= fdScaler

        for i in range(AM.shape[0]):
            if fd != i:
                crScaler = AM[i,fd]
                for j in range(AM.shape[0]):
                    AM[i,j] = AM[i,j] - crScaler * AM[fd,j]
                    IM[i,j] = IM[i,j] - crScaler * IM[fd,j]

@cuda.jit(device=True)
def linalg_lstsq(Xt_1,Yt,betaY_temp): #NX is Xt_1.shape[1]
    Xt_1_sq = cuda.local.array((NX,NX), float64)
    matmul(Xt_1.transpose(), Xt_1, Xt_1_sq)
    L = cuda.local.array((NX,NX), float64)
    CholeskyInc(Xt_1_sq, L)
    '''betaYt = np.linalg.lstsq(Xt_1, Yt)[0] #regression
    return betaYt:
    Xt_1xbetaYt = Yt, Xt_1.TxXt_1 = LxL.T,
    so betaYt = (L.T)^(-1)x(L)^(-1)xXt_1.TxYt'''
    temp_identity = identity(cuda.local.array((NX,NX), float64))
    temp_matrix = cuda.local.array((NX,NX), float64)
    L_1 = cuda.local.array((NX,NX), float64)
    # get inverse of L
    copy_matrix(L, temp_matrix)
    invert_matrix(temp_matrix, temp_identity)
    copy_matrix(temp_identity, L_1)
    L_sq = cuda.local.array((NX,NX), float64)
    Xt_1_Yt = cuda.local.array((NX,1), float64)
    matmul(L_1.transpose(), L_1, L_sq)
    matmul(Xt_1.transpose(), Yt, Xt_1_Yt)
    # compute betaY_temp
    matmul(L_sq, Xt_1_Yt, betaY_temp)

@cuda.jit(device=True)
def compute_Yt_final(t, K, betaY, Xbd, Vbd, St_T, Vt_T, Xt_1, Yt):
    if t == Nt - 2:
        for jj in range(noi):
            Yt[jj,0] = max(K - math.exp(St_T[jj,0]), 0)
            Yt[jj,0] = Yt[jj,0] -r * Yt[jj,0]*dt
    else:
        betaYt = cuda.local.array(shape=(noi,NX), dtype=float64)
        indbdX = cuda.local.array(shape=(noi,1), dtype=int32)
        indbdV = cuda.local.array(shape=(noi,1), dtype=int32)
        #index of box for each path
        searchsorted(Xbd, St_T, indbdX, Vbd, Vt_T, indbdV)
        compute_Xt(St_T, Vt_T, Xt_1)
        for jj in range(noi):
            for jjj in range(NX):
                betaYt[jj,jjj] = betaY[jjj,indbdX[jj,0],indbdV[jj,0],Nt-t-3]
        compute_Yt(K, St_T, Xt_1, betaYt, Yt)

@cuda.jit(device=True)
def compute_betaYt(i, j, t, K, betaY, dW1, dW2, Xbd, Vbd, uniform_St_1, uniform_Vt_1):
    St_1 = cuda.local.array(shape=(noi,1), dtype=float64)
    St_T = cuda.local.array(shape=(noi,1), dtype=float64)
    Vt_1 = cuda.local.array(shape=(noi,1), dtype=float64)
    Vt_T = cuda.local.array(shape=(noi,1), dtype=float64)
    Xt_1 = cuda.local.array(shape=(noi,NX), dtype=float64)
    Yt = cuda.local.array(shape=(noi,1), dtype=float64)
    get_T1(i, j, dW1, dW2, Xbd, Vbd, uniform_St_1, St_1, St_T, uniform_Vt_1, Vt_1, Vt_T)
    compute_Yt_final(t, K, betaY, Xbd, Vbd, St_T, Vt_T, Xt_1, Yt)
    compute_Xt(St_1, Vt_1, Xt_1)
    betaY_temp = cuda.local.array((NX,1), float64)
    linalg_lstsq(Xt_1,Yt,betaY_temp)
    for k in range(NX):
        betaY[k,i,j,Nt-t-2] = betaY_temp[k,0]

@cuda.jit
def compute_betaYt_(t, K, betaY, dW1, dW2, Xbd, Vbd, uniform_St_1, uniform_Vt_1):
    uniform_St_1_ = cuda.shared.array((noi,1), float64)
    uniform_Vt_1_ = cuda.shared.array((noi,1), float64)
    copy_matrix(uniform_St_1, uniform_St_1_)
    copy_matrix(uniform_Vt_1, uniform_Vt_1_)
    Xbd_ = cuda.shared.array((noh_1,), float64)
    Vbd_ = cuda.shared.array((noh_1,), float64)
    dW1_ = cuda.const.array_like(dW1)
    dW2_ = cuda.const.array_like(dW2)
    for ii in range(noh_1):
        Xbd_[ii] = Xbd[ii]
        Vbd_[ii] = Vbd[ii]
    i, j = cuda.grid(2)
    if i < noh and j < noh:
        compute_betaYt(i, j, t, K, betaY, dW1_, dW2_, Xbd_, Vbd_, uniform_St_1_, uniform_Vt_1_)
    #cuda.syncthreads()

def compute_betaY(K, Xbd, Vbd, betaY, threads_per_block, blocks):
    for t in range(Nt-2,-1,-1): #t is index of time
        uniform_St_1 = cp.random.uniform(0, 1, (noi, 1))
        uniform_Vt_1 = cp.random.uniform(0, 1, (noi, 1))
        dW1 = cp.random.randn(noi,1)
        dW2 = cp.random.randn(noi,1)
        compute_betaYt_[blocks, threads_per_block](t, K, betaY, dW1, dW2, Xbd, Vbd, uniform_St_1, uniform_Vt_1)
    return betaY

#@cuda.jit
def P(K, Xbd, Vbd, betaY, threads_per_block, blocks):
    betaY = compute_betaY(K, Xbd, Vbd, betaY, threads_per_block, blocks)
    Vt_1 = V0
    St_1 = cp.log(S0)
    dZ1 = cp.random.randn(noi,1)
    dZ2 = cp.random.randn(noi,1)
    Vt_T = cp.abs(Vt_1 + kappa*(theta - Vt_1) * dt + sigma*cp.sqrt(Vt_1*dt) * dZ1)
    St_T = St_1 + (r - 0.5 * Vt_1) * dt + cp.sqrt(Vt_1) * (rho * dZ1 + cp.sqrt(1 - rho**2) * dZ2) * cp.sqrt(dt)
    T2 = cp.where((St_T >= Xmin) & (St_T <= Xmax) & (Vt_T >= Vmin) & (Vt_T <= Vmax))[0]
    St_T = St_T[T2,:]
    lenT2 = len(T2)
    Yt = cp.zeros((lenT2,1))
    indbdX = cp.searchsorted(Xbd, St_T)-1
    indbdV = cp.searchsorted(Vbd, Vt_T)-1
    XS1 = 1 - St_T
    XV1 = 1 - Vt_T
    X0 = cp.ones((lenT2,1))
    Xt = cp.hstack((X0,XS1,XV1))
    betaYt = cp.zeros((lenT2,NX))
    for i in range(lenT2):
        betaYt[i,:] = betaY[:, indbdX.flatten()[i], indbdV.flatten()[i], Nt-2]
    Yt = cp.empty((noi,1),dtype=cp.float64)
    for i in range(lenT2):
        for j in range(NX):
            Yt[i,0] += Xt[i,j]*betaYt[i,j]
    Yt = cp.maximum(cp.maximum(K - cp.exp(St_T),0),Yt)
    cum_gen = - r * Yt
    Yt = Yt + cum_gen * dt
    RBSDEvalue = cp.maximum(cp.mean(Yt), cp.maximum(K - S0,0))
    return RBSDEvalue

def compute_(K, Xbd, Vbd, betaY, threads_per_block, blocks, iter):
    Pvalue = cp.zeros((iter,1))
    for i in range(iter):
        #print('iteration =', i)
        Pvalue[i,:] = P(K, Xbd, Vbd, betaY, threads_per_block, blocks)
    return Pvalue

global S0, K, V0, r, kappa, theta, sigma, rho, T, Nt, noh, noi, mu, Xmin, Xmax, Vmin, Vmax, NX, dt, BD, Xbd, Vbd

S0 = 100
Nt = 25 # number of time steps
mu = 1 # parameter for logistic sample generating
V0 = 0.04 # initial volatility
T = 0.25 # time horizon
r = 0.04 # risk-free rate
kappa = 1.58
theta = 0.03
sigma = 0.2
rho = -0.26
noh = 50 # number of hypercubes
noi = 3000 # number of samples
dt = T/Nt
Xmin = 0 #space domain for log stock price
Xmax = 6
Vmin = 0 #space domain for volatility
Vmax = 1
BD = cp.array([[Xmin, Xmax],[Vmin, Vmax]])
Xbd = cp.linspace(BD[0,0],BD[0,1],num=noh+1)
Vbd = cp.linspace(BD[1,0],BD[1,1],num=noh+1)
NX = 3 #number of polynomial functions
noh_1 = noh+1
betaY = cp.zeros((NX,noh,noh,Nt-1),dtype=cp.float64)

time_a = time.time()
threads_per_block = (8,8)
blocks_x = math.ceil(noh / threads_per_block [0])
blocks_y = math.ceil(noh / threads_per_block [1])
blocks = (blocks_x, blocks_y)
iter = 50
K_range = [90,100,110]
meanValue = cp.empty(len(K_range))
stdValue = cp.empty(len(K_range))
for k in range(len(K_range)):
    Pvalue = compute_(K_range[k], Xbd, Vbd, betaY, threads_per_block, blocks, iter)
    meanValue[k] = cp.mean(Pvalue)
    stdValue[k] = cp.std(Pvalue, ddof=1)
    print('meanValue=',meanValue[k])
    print('stdValue=',stdValue[k])
    print('time=',time.time()-time_a)
