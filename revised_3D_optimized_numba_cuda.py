#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter sweep script for 3D CUDA RBSDE solver (S, V, R)
REVISED: Memory-optimized version storing only last time-step (time-homogeneous)
"""

import time
import json
from datetime import datetime
from numba import cuda, float64, int32
import math
import cupy as cp
import numpy as np

# ============================================================================
# CUDA DEVICE FUNCTIONS - OPTIMIZED FOR SINGLE PATH PROCESSING
# ============================================================================

@cuda.jit(device=True)
def get_T1_single(i, j, k, idx, dW1, dW2, dW3, Xbd, Vbd, Rbd, 
                   uniform_St_1, uniform_Vt_1, uniform_Rt_1,
                   St_1_out, St_T_out, Vt_1_out, Vt_T_out, Rt_1_out, Rt_T_out,
                   mu, kappa, theta, sigma, Rkappa, Rtheta, Rsigma, rho, dt):
    """Process single path instead of array"""
    fxl1 = 1 / (1 + math.exp(-mu * Xbd[i]))
    fxu1 = 1 / (1 + math.exp(-mu * Xbd[i+1]))
    fxl2 = 1 / (1 + math.exp(-mu * Vbd[j]))
    fxu2 = 1 / (1 + math.exp(-mu * Vbd[j+1]))
    fxl3 = 1 / (1 + math.exp(-mu * Rbd[k]))
    fxu3 = 1 / (1 + math.exp(-mu * Rbd[k+1]))
    
    St_1_out[0] = -(1 / mu) * math.log(1/(fxl1 + uniform_St_1[idx,0] * (fxu1 - fxl1)) - 1)
    Vt_1_out[0] = -(1 / mu) * math.log(1/(fxl2 + uniform_Vt_1[idx,0] * (fxu2 - fxl2)) - 1)
    Rt_1_out[0] = -(1 / mu) * math.log(1/(fxl3 + uniform_Rt_1[idx,0] * (fxu3 - fxl3)) - 1)
    
    Vt_T_out[0] = abs(Vt_1_out[0] + kappa*(theta - Vt_1_out[0])*dt + sigma*math.sqrt(Vt_1_out[0]*dt)*dW1[idx,0])
    Rt_T_out[0] = abs(Rt_1_out[0] + Rkappa*(Rtheta - Rt_1_out[0])*dt + Rsigma*math.sqrt(Rt_1_out[0]*dt)*dW3[idx,0])
    St_T_out[0] = St_1_out[0] + (Rt_1_out[0] - 0.5 * Vt_1_out[0]) * dt + math.sqrt(Vt_1_out[0]) \
        * (rho * dW1[idx,0] + math.sqrt(1 - rho**2) * dW2[idx,0]) * math.sqrt(dt)

@cuda.jit(device=True)
def compute_Xt_single(St, Vt, Rt, Xt):
    """Compute polynomials for single path"""
    Xt[0] = 1.0
    Xt[1] = 1.0 - St
    Xt[2] = 1.0 - Vt
    Xt[3] = 1.0 - Rt

@cuda.jit(device=True)
def compute_YtZt_single(K, St_T, Rt_1, Xt, betaYt, betaZ1t, betaZ2t, betaZ3t,
                         dW1_val, dW2_val, dW3_val,
                         Yt_out, Z1t_out, Z2t_out, Z3t_out, dt, chi):
    """Compute least square fitted value for single path"""
    NX = 4
    X = 0.0
    Z1 = 0.0
    Z2 = 0.0
    Z3 = 0.0
    for j in range(NX):
        X += Xt[j] * betaYt[j]
        Z1 += Xt[j] * betaZ1t[j] / dt
        Z2 += Xt[j] * betaZ2t[j] / dt
        Z3 += Xt[j] * betaZ3t[j] / dt
    
    Yt_out[0] = X
    Z1t_out[0] = Z1
    Z2t_out[0] = Z2
    Z3t_out[0] = Z3
    Yt_out[0] = max(max(K - math.exp(St_T), 0.0), Yt_out[0])
    Yt_out[0] = Yt_out[0] - (Rt_1 * Yt_out[0] + math.sqrt(((0.1395**2)*Z1t_out[0]**2 + (0.1900**2)*Z2t_out[0]**2 + (0.1992**2)*Z3t_out[0]**2)*chi))*dt
    Z1t_out[0] = Yt_out[0] * dW1_val * math.sqrt(dt)
    Z2t_out[0] = Yt_out[0] * dW2_val * math.sqrt(dt)
    Z3t_out[0] = Yt_out[0] * dW3_val * math.sqrt(dt)

@cuda.jit(device=True)
def searchsorted_single(Xbd, St_T, Vbd, Vt_T, Rbd, Rt_T, noh_val):
    """Binary search for single value"""
    indbdX = 0
    indbdV = 0
    indbdR = 0
    
    if St_T <= Xbd[0]:
        indbdX = 0
    elif St_T >= Xbd[noh_val]:
        indbdX = noh_val - 1
    else:
        j = 0
        while St_T > Xbd[j]:
            j += 1
        indbdX = j - 1
    
    if Vt_T <= Vbd[0]:
        indbdV = 0
    elif Vt_T >= Vbd[noh_val]:
        indbdV = noh_val - 1
    else:
        k = 0
        while Vt_T > Vbd[k]:
            k += 1
        indbdV = k - 1
    
    if Rt_T <= Rbd[0]:
        indbdR = 0
    elif Rt_T >= Rbd[noh_val]:
        indbdR = noh_val - 1
    else:
        m = 0
        while Rt_T > Rbd[m]:
            m += 1
        indbdR = m - 1
    
    return indbdX, indbdV, indbdR

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
        A[i, i] = math.sqrt(max(A[i, i], 1e-10))
    for i in range(n):
        for j in range(i, n):
            L[j, i] = A[j, i]

@cuda.jit(device=True)
def matmul(A, B, C):
    """Matrix multiplication C = A * B"""
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp

@cuda.jit(device=True)
def identity(mat0):
    for r in range(mat0.shape[0]):
        for c in range(mat0.shape[1]):
            mat0[r, c] = 1.0 if r == c else 0.0
    return mat0

@cuda.jit(device=True)
def copy_matrix(mat0, mat1):
    for r in range(mat0.shape[0]):
        for c in range(mat0.shape[1]):
            mat1[r, c] = mat0[r, c]

@cuda.jit(device=True)
def invert_matrix(AM, IM):
    for fd in range(AM.shape[0]):
        fdScaler = 1.0 / AM[fd, fd]
        for j in range(AM.shape[0]):
            AM[fd, j] *= fdScaler
            IM[fd, j] *= fdScaler
        for i in range(AM.shape[0]):
            if fd != i:
                crScaler = AM[i, fd]
                for j in range(AM.shape[0]):
                    AM[i, j] = AM[i, j] - crScaler * AM[fd, j]
                    IM[i, j] = IM[i, j] - crScaler * IM[fd, j]

@cuda.jit(device=True)
def compute_betaYt(i, j, k, t, K, betaY_curr, betaZ1_curr, betaZ2_curr, betaZ3_curr,
                   betaY_next, betaZ1_next, betaZ2_next, betaZ3_next,
                   dW1, dW2, dW3, Xbd, Vbd, Rbd,
                   uniform_St_1, uniform_Vt_1, uniform_Rt_1, noi_val, noh, Nt,
                   mu, kappa, theta, sigma, Rkappa, Rtheta, Rsigma, rho, chi, dt):
    """Compute regression coefficients using streaming approach - stores only current step"""
    NX = 4
    XtX = cuda.local.array((4, 4), float64)
    XtY = cuda.local.array((4, 1), float64)
    XtZ1 = cuda.local.array((4, 1), float64)
    XtZ2 = cuda.local.array((4, 1), float64)
    XtZ3 = cuda.local.array((4, 1), float64)
    
    for ii in range(NX):
        XtY[ii, 0] = 0.0
        XtZ1[ii, 0] = 0.0
        XtZ2[ii, 0] = 0.0
        XtZ3[ii, 0] = 0.0
        for jj in range(NX):
            XtX[ii, jj] = 0.0
    
    for idx in range(noi_val):
        St_1 = cuda.local.array(1, float64)
        Vt_1 = cuda.local.array(1, float64)
        Rt_1 = cuda.local.array(1, float64)
        St_T = cuda.local.array(1, float64)
        Vt_T = cuda.local.array(1, float64)
        Rt_T = cuda.local.array(1, float64)
        
        get_T1_single(i, j, k, idx, dW1, dW2, dW3, Xbd, Vbd, Rbd,
                     uniform_St_1, uniform_Vt_1, uniform_Rt_1,
                     St_1, St_T, Vt_1, Vt_T, Rt_1, Rt_T,
                     mu, kappa, theta, sigma, Rkappa, Rtheta, Rsigma, rho, dt)
        
        Xt_1 = cuda.local.array(4, float64)
        compute_Xt_single(St_1[0], Vt_1[0], Rt_1[0], Xt_1)
        
        if t == Nt - 2:
            # Terminal condition
            Yt_val = max(K - math.exp(St_T[0]), 0.0)
            Yt_val = Yt_val - Rt_1[0] * Yt_val * dt
            Z1t_val = 0.0
            Z2t_val = 0.0
            Z3t_val = 0.0
        else:
            # Use next time step coefficients (already computed)
            indbdX, indbdV, indbdR = searchsorted_single(Xbd, St_T[0], Vbd, Vt_T[0], Rbd, Rt_T[0], noh)
            
            betaYt = cuda.local.array(4, float64)
            betaZ1t = cuda.local.array(4, float64)
            betaZ2t = cuda.local.array(4, float64)
            betaZ3t = cuda.local.array(4, float64)
            
            for jjj in range(NX):
                betaYt[jjj] = betaY_next[jjj, indbdX, indbdV, indbdR]
                betaZ1t[jjj] = betaZ1_next[jjj, indbdX, indbdV, indbdR]
                betaZ2t[jjj] = betaZ2_next[jjj, indbdX, indbdV, indbdR]
                betaZ3t[jjj] = betaZ3_next[jjj, indbdX, indbdV, indbdR]
            
            Xt_T = cuda.local.array(4, float64)
            compute_Xt_single(St_T[0], Vt_T[0], Rt_T[0], Xt_T)
            
            Yt = cuda.local.array(1, float64)
            Z1t = cuda.local.array(1, float64)
            Z2t = cuda.local.array(1, float64)
            Z3t = cuda.local.array(1, float64)
            
            compute_YtZt_single(K, St_T[0], Rt_1[0], Xt_T, betaYt, betaZ1t, betaZ2t, betaZ3t,
                               dW1[idx,0], dW2[idx,0], dW3[idx,0],
                               Yt, Z1t, Z2t, Z3t, dt, chi)
            
            Yt_val = Yt[0]
            Z1t_val = Z1t[0]
            Z2t_val = Z2t[0]
            Z3t_val = Z3t[0]
        
        for ii in range(NX):
            XtY[ii, 0] += Xt_1[ii] * Yt_val
            XtZ1[ii, 0] += Xt_1[ii] * Z1t_val
            XtZ2[ii, 0] += Xt_1[ii] * Z2t_val
            XtZ3[ii, 0] += Xt_1[ii] * Z3t_val
            for jj in range(NX):
                XtX[ii, jj] += Xt_1[ii] * Xt_1[jj]
    
    L = cuda.local.array((4, 4), float64)
    CholeskyInc(XtX, L)
    
    temp_identity = identity(cuda.local.array((4, 4), float64))
    temp_matrix = cuda.local.array((4, 4), float64)
    L_1 = cuda.local.array((4, 4), float64)
    
    copy_matrix(L, temp_matrix)
    invert_matrix(temp_matrix, temp_identity)
    copy_matrix(temp_identity, L_1)
    
    L_sq = cuda.local.array((4, 4), float64)
    betaY_temp = cuda.local.array((4, 1), float64)
    betaZ1_temp = cuda.local.array((4, 1), float64)
    betaZ2_temp = cuda.local.array((4, 1), float64)
    betaZ3_temp = cuda.local.array((4, 1), float64)
    
    matmul(L_1.transpose(), L_1, L_sq)
    matmul(L_sq, XtY, betaY_temp)
    matmul(L_sq, XtZ1, betaZ1_temp)
    matmul(L_sq, XtZ2, betaZ2_temp)
    matmul(L_sq, XtZ3, betaZ3_temp)
    
    # Store in current time step arrays
    for kk in range(NX):
        betaY_curr[kk, i, j, k] = betaY_temp[kk, 0]
        betaZ1_curr[kk, i, j, k] = betaZ1_temp[kk, 0]
        betaZ2_curr[kk, i, j, k] = betaZ2_temp[kk, 0]
        betaZ3_curr[kk, i, j, k] = betaZ3_temp[kk, 0]

@cuda.jit
def compute_betaYt_(t, K, betaY_curr, betaZ1_curr, betaZ2_curr, betaZ3_curr,
                    betaY_next, betaZ1_next, betaZ2_next, betaZ3_next,
                    dW1, dW2, dW3, 
                    Xbd, Vbd, Rbd, uniform_St_1, uniform_Vt_1, uniform_Rt_1,
                    noi_val, noh, Nt, mu, kappa, theta, sigma, Rkappa, Rtheta, Rsigma, rho, chi, dt):
    """Kernel to compute beta coefficients"""
    i, j, k = cuda.grid(3)
    if i < noh and j < noh and k < noh:
        compute_betaYt(i, j, k, t, K, betaY_curr, betaZ1_curr, betaZ2_curr, betaZ3_curr,
                      betaY_next, betaZ1_next, betaZ2_next, betaZ3_next,
                      dW1, dW2, dW3, Xbd, Vbd, Rbd,
                      uniform_St_1, uniform_Vt_1, uniform_Rt_1, noi_val, noh, Nt,
                      mu, kappa, theta, sigma, Rkappa, Rtheta, Rsigma, rho, chi, dt)

# ============================================================================
# HOST FUNCTIONS
# ============================================================================

def compute_betaY(K, Xbd, Vbd, Rbd, betaY, betaZ1, betaZ2, betaZ3, threads_per_block, blocks,
                  noi_val, noh, Nt, mu, kappa, theta, sigma, Rkappa, Rtheta, Rsigma, rho, chi, dt):
    """Main function to compute beta coefficients backward in time - stores only last step"""
    NX = 4
    
    # Create temporary arrays for ping-pong buffering
    betaY_curr = cp.zeros((NX, noh, noh, noh), dtype=cp.float64)
    betaZ1_curr = cp.zeros((NX, noh, noh, noh), dtype=cp.float64)
    betaZ2_curr = cp.zeros((NX, noh, noh, noh), dtype=cp.float64)
    betaZ3_curr = cp.zeros((NX, noh, noh, noh), dtype=cp.float64)
    
    betaY_next = cp.zeros((NX, noh, noh, noh), dtype=cp.float64)
    betaZ1_next = cp.zeros((NX, noh, noh, noh), dtype=cp.float64)
    betaZ2_next = cp.zeros((NX, noh, noh, noh), dtype=cp.float64)
    betaZ3_next = cp.zeros((NX, noh, noh, noh), dtype=cp.float64)
    
    for t in range(Nt-2, -1, -1):
        uniform_St_1 = cp.random.uniform(0, 1, (noi_val, 1))
        uniform_Vt_1 = cp.random.uniform(0, 1, (noi_val, 1))
        uniform_Rt_1 = cp.random.uniform(0, 1, (noi_val, 1))
        dW1 = cp.random.randn(noi_val, 1)
        dW2 = cp.random.randn(noi_val, 1)
        dW3 = cp.random.randn(noi_val, 1)
        
        compute_betaYt_[blocks, threads_per_block](
            t, K, betaY_curr, betaZ1_curr, betaZ2_curr, betaZ3_curr,
            betaY_next, betaZ1_next, betaZ2_next, betaZ3_next,
            dW1, dW2, dW3, Xbd, Vbd, Rbd,
            uniform_St_1, uniform_Vt_1, uniform_Rt_1,
            noi_val, noh, Nt, mu, kappa, theta, sigma,
            Rkappa, Rtheta, Rsigma, rho, chi, dt)
        
        # Swap: current becomes next for the previous time step
        betaY_curr, betaY_next = betaY_next, betaY_curr
        betaZ1_curr, betaZ1_next = betaZ1_next, betaZ1_curr
        betaZ2_curr, betaZ2_next = betaZ2_next, betaZ2_curr
        betaZ3_curr, betaZ3_next = betaZ3_next, betaZ3_curr
    
    # After the loop, betaY_next contains the last computed step (t=0)
    # Copy to output arrays
    betaY[:, :, :, :] = betaY_next
    betaZ1[:, :, :, :] = betaZ1_next
    betaZ2[:, :, :, :] = betaZ2_next
    betaZ3[:, :, :, :] = betaZ3_next
    
    return betaY, betaZ1, betaZ2, betaZ3

def P(K, Xbd, Vbd, Rbd, betaY, betaZ1, betaZ2, betaZ3, threads_per_block, blocks, noi_val,
      S0, V0, R0, kappa, theta, sigma, Rkappa, Rtheta, Rsigma, rho, chi, dt, Nt,
      Xmin, Xmax, Vmin, Vmax, Rmin, Rmax, noh, mu):
    """Compute option price using time-homogeneous coefficients"""
    NX = 4
    betaY, betaZ1, betaZ2, betaZ3 = compute_betaY(K, Xbd, Vbd, Rbd, betaY, betaZ1, betaZ2, betaZ3,
                                                   threads_per_block, blocks, noi_val, noh, Nt,
                                                   mu, kappa, theta, sigma, Rkappa, Rtheta, Rsigma, rho, chi, dt)
    Vt_1 = V0
    Rt_1 = R0
    St_1 = cp.log(S0)
    dZ1 = cp.random.randn(noi_val, 1)
    dZ2 = cp.random.randn(noi_val, 1)
    dZ3 = cp.random.randn(noi_val, 1)
    Vt_T = cp.abs(Vt_1 + kappa*(theta - Vt_1) * dt + sigma*cp.sqrt(Vt_1*dt) * dZ1)
    Rt_T = cp.abs(Rt_1 + Rkappa*(Rtheta - Rt_1) * dt + Rsigma*cp.sqrt(Rt_1*dt) * dZ3)
    St_T = St_1 + (Rt_1 - 0.5 * Vt_1) * dt + cp.sqrt(Vt_1) * (rho * dZ1 + cp.sqrt(1 - rho**2) * dZ2) * cp.sqrt(dt)
    T2 = cp.where((St_T >= Xmin) & (St_T <= Xmax) & (Vt_T >= Vmin) & (Vt_T <= Vmax) & (Rt_T >= Rmin) & (Rt_T <= Rmax))[0]
    St_T = St_T[T2, :]
    Vt_T = Vt_T[T2, :]
    Rt_T = Rt_T[T2, :]
    lenT2 = len(T2)
    
    if lenT2 == 0:
        return cp.maximum(K - S0, 0)
    
    Yt = cp.zeros((lenT2, 1))
    indbdX = cp.searchsorted(Xbd, St_T) - 1
    indbdV = cp.searchsorted(Vbd, Vt_T) - 1
    indbdR = cp.searchsorted(Rbd, Rt_T) - 1
    XS1 = 1 - St_T
    XV1 = 1 - Vt_T
    XR1 = 1 - Rt_T
    X0 = cp.ones((lenT2, 1))
    Xt = cp.hstack((X0, XS1, XV1, XR1))
    betaYt = cp.zeros((lenT2, NX))
    betaZ1t = cp.zeros((lenT2, NX))
    betaZ2t = cp.zeros((lenT2, NX))
    betaZ3t = cp.zeros((lenT2, NX))
    
    # Use the single time-step coefficients (time-homogeneous)
    for i in range(lenT2):
        betaYt[i, :] = betaY[:, indbdX.flatten()[i], indbdV.flatten()[i], indbdR.flatten()[i]]
        betaZ1t[i, :] = betaZ1[:, indbdX.flatten()[i], indbdV.flatten()[i], indbdR.flatten()[i]]
        betaZ2t[i, :] = betaZ2[:, indbdX.flatten()[i], indbdV.flatten()[i], indbdR.flatten()[i]]
        betaZ3t[i, :] = betaZ3[:, indbdX.flatten()[i], indbdV.flatten()[i], indbdR.flatten()[i]]
    
    Yt = cp.sum(Xt * betaYt, axis=1, keepdims=True)
    Z1t = cp.sum(Xt * betaZ1t, axis=1, keepdims=True) / dt
    Z2t = cp.sum(Xt * betaZ2t, axis=1, keepdims=True) / dt
    Z3t = cp.sum(Xt * betaZ3t, axis=1, keepdims=True) / dt
    
    Yt = cp.maximum(cp.maximum(K - cp.exp(St_T), 0), Yt)
    cum_gen = - R0 * Yt - cp.sqrt(((0.1395**2)*Z1t**2 + (0.1900**2)*Z2t**2 + (0.1992**2)*Z3t**2)*chi)
    Yt = Yt + cum_gen * dt
    RBSDEvalue = cp.maximum(cp.mean(Yt), cp.maximum(K - S0, 0))
    return RBSDEvalue

def compute_(K, Xbd, Vbd, Rbd, betaY, betaZ1, betaZ2, betaZ3, threads_per_block, blocks, iter, noi_val,
             S0, V0, R0, kappa, theta, sigma, Rkappa, Rtheta, Rsigma, rho, chi, dt, Nt,
             Xmin, Xmax, Vmin, Vmax, Rmin, Rmax, noh, mu):
    """Run multiple iterations"""
    Pvalue = cp.zeros((iter, 1))
    for i in range(iter):
        Pvalue[i, :] = P(K, Xbd, Vbd, Rbd, betaY, betaZ1, betaZ2, betaZ3, threads_per_block, blocks, noi_val,
                        S0, V0, R0, kappa, theta, sigma, Rkappa, Rtheta, Rsigma, rho, chi, dt, Nt,
                        Xmin, Xmax, Vmin, Vmax, Rmin, Rmax, noh, mu)
    return Pvalue

# ============================================================================
# PARAMETER SWEEP
# ============================================================================

def run_single_config(chi, noi, noh, T_val, Nt, K_range, iterations, log_data):
    """Run a single parameter configuration"""
    # Fixed parameters
    S0 = 100
    mu = 1
    V0 = 0.04
    R0 = 0.04
    kappa = 1.58
    theta = 0.03
    sigma = 0.2
    Rkappa = 0.26
    Rtheta = 0.04
    Rsigma = 0.08
    rho = -0.26
    
    dt = T_val / Nt
    Xmin = 0
    Xmax = 5.5
    Vmin = 0
    Vmax = 1
    Rmin = 0
    Rmax = 1
    BD = cp.array([[Xmin, Xmax], [Vmin, Vmax], [Rmin, Rmax]])
    Xbd = cp.linspace(BD[0, 0], BD[0, 1], num=noh+1)
    Vbd = cp.linspace(BD[1, 0], BD[1, 1], num=noh+1)
    Rbd = cp.linspace(BD[2, 0], BD[2, 1], num=noh+1)
    NX = 4
    
    # REVISED: Only 4D arrays now (no time dimension)
    betaY = cp.zeros((NX, noh, noh, noh), dtype=cp.float64)
    betaZ1 = cp.zeros((NX, noh, noh, noh), dtype=cp.float64)
    betaZ2 = cp.zeros((NX, noh, noh, noh), dtype=cp.float64)
    betaZ3 = cp.zeros((NX, noh, noh, noh), dtype=cp.float64)
    
    threads_per_block = (4, 4, 4)
    blocks_x = math.ceil(noh / threads_per_block[0])
    blocks_y = math.ceil(noh / threads_per_block[1])
    blocks_z = math.ceil(noh / threads_per_block[2])
    blocks = (blocks_x, blocks_y, blocks_z)
    
    config_results = {
        'chi': chi,
        'M': noi,
        'J': noh,
        'T': T_val,
        'Nt': Nt,
        'iterations': iterations,
        'K_results': {}
    }
    
    config_start_time = time.time()
    
    for K in K_range:
        k_start_time = time.time()
        print(f"    Computing K={K}...")
        
        Pvalue = compute_(K, Xbd, Vbd, Rbd, betaY, betaZ1, betaZ2, betaZ3,
                         threads_per_block, blocks, iterations, noi,
                         S0, V0, R0, kappa, theta, sigma, Rkappa, Rtheta, Rsigma, rho, chi, dt, Nt,
                         Xmin, Xmax, Vmin, Vmax, Rmin, Rmax, noh, mu)
        
        mean_val = float(cp.mean(Pvalue))
        std_val = float(cp.std(Pvalue, ddof=1))
        k_time = time.time() - k_start_time
        
        config_results['K_results'][K] = {
            'mean': mean_val,
            'std': std_val,
            'time_seconds': k_time
        }
        
        print(f"      K={K}: mean={mean_val:.4f}, std={std_val:.4f}, time={k_time:.2f}s")
    
    config_time = time.time() - config_start_time
    config_results['total_time_seconds'] = config_time
    
    log_data['configurations'].append(config_results)
    
    return config_results

def main():
    """Main function to run parameter sweep"""
    # Parameter combinations
    chi_list = [0.5844]
    M_list = [2000, 5000, 10000]  # noi
    J_list = [80, 160, 320]  # noh
    TN_list = [(0.5, 25), (0.3, 15)]  # (T, Nt)
    K_range = [90, 100, 110]
    iterations = 50
    
    # Initialize log data
    log_data = {
        'start_time': datetime.now().isoformat(),
        'model': '3D RBSDE (S, V, R) - Stochastic Interest Rate - TIME-HOMOGENEOUS',
        'parameters': {
            'chi_list': chi_list,
            'M_list': M_list,
            'J_list': J_list,
            'TN_list': TN_list,
            'K_range': K_range,
            'iterations': iterations
        },
        'total_combinations': len(chi_list) * len(M_list) * len(J_list) * len(TN_list),
        'configurations': []
    }
    
    print("="*80)
    print("3D CUDA RBSDE SOLVER - PARAMETER SWEEP (MEMORY OPTIMIZED)")
    print("Model: Heston with Stochastic Interest Rate (S, V, R)")
    print("Strategy: Time-homogeneous (stores only last time-step)")
    print("="*80)
    print(f"Total configurations to run: {log_data['total_combinations']}")
    print(f"chi values: {chi_list}")
    print(f"M (noi) values: {M_list}")
    print(f"J (noh) values: {J_list}")
    print(f"(T, Nt) values: {TN_list}")
    print(f"K values: {K_range}")
    print(f"Iterations per K: {iterations}")
    print("="*80)
    
    overall_start_time = time.time()
    config_count = 0
    
    # Run all combinations
    for chi in chi_list:
        for noi in M_list:
            for noh in J_list:
                for T_val, Nt in TN_list:
                    config_count += 1
                    print(f"\n[{config_count}/{log_data['total_combinations']}] Configuration:")
                    print(f"  chi={chi}, M={noi}, J={noh}, T={T_val}, Nt={Nt}")
                    
                    try:
                        run_single_config(chi, noi, noh, T_val, Nt, K_range, iterations, log_data)
                        print(f"  ✓ Configuration completed successfully")
                    except Exception as e:
                        print(f"  ✗ Error in configuration: {str(e)}")
                        log_data['configurations'].append({
                            'chi': chi,
                            'M': noi,
                            'J': noh,
                            'T': T_val,
                            'Nt': Nt,
                            'error': str(e)
                        })
    
    # Finalize log
    overall_time = time.time() - overall_start_time
    log_data['end_time'] = datetime.now().isoformat()
    log_data['total_time_seconds'] = overall_time
    log_data['total_time_hours'] = overall_time / 3600
    
    print("\n" + "="*80)
    print(f"PARAMETER SWEEP COMPLETED")
    print(f"Total time: {overall_time/3600:.2f} hours ({overall_time:.2f} seconds)")
    print("="*80)
    
    # Save log as JSON
    log_filename = f"rbsde_3d_sweep_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nLog saved to: {log_filename}")
    
    return log_data, log_filename

if __name__ == "__main__":
    log_data, log_filename = main()