import time
from numba import cuda, float64, int32
#from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import math
import cupy as cp

@cuda.jit(device = True)
def get_T1(i, j, k, dW1, dW2, dW3, Xbd, Vbd, Rbd, uniform_St_1, St_1, St_T, uniform_Vt_1, Vt_1, Vt_T, uniform_Rt_1, Rt_1, Rt_T):
    fxl1 = 1 / (1 + math.exp(-mu * Xbd[i]))
    fxu1 = 1 / (1 + math.exp(-mu * Xbd[i+1]))
    fxl2 = 1 / (1 + math.exp(-mu * Vbd[j]))
    fxu2 = 1 / (1 + math.exp(-mu * Vbd[j+1]))
    fxl3 = 1 / (1 + math.exp(-mu * Rbd[k]))
    fxu3 = 1 / (1 + math.exp(-mu * Rbd[k+1]))
    for i in range(noi):
        St_1[i,0] = -(1 / mu) * math.log(1/(fxl1 + uniform_St_1[i,0] * (fxu1 - fxl1)) - 1)
        Vt_1[i,0] = -(1 / mu) * math.log(1/(fxl2 + uniform_Vt_1[i,0] * (fxu2 - fxl2)) - 1)
        Rt_1[i,0] = -(1 / mu) * math.log(1/(fxl3 + uniform_Rt_1[i,0] * (fxu3 - fxl3)) - 1)
        Vt_T[i,0] = abs(Vt_1[i,0] + kappa*(theta - Vt_1[i,0])*dt + sigma*math.sqrt(Vt_1[i,0]*dt)*dW1[i,0])
        Rt_T[i,0] = abs(Rt_1[i,0] + Rkappa*(Rtheta - Rt_1[i,0])*dt + Rsigma*math.sqrt(Rt_1[i,0]*dt)*dW3[i,0])
        St_T[i,0] = St_1[i,0] + (Rt_1[i,0] - 0.5 * Vt_1[i,0]) * dt + math.sqrt(Vt_1[i,0]) \
        * (rho * dW1[i,0] + math.sqrt(1 - rho**2) * dW2[i,0]) * math.sqrt(dt)

@cuda.jit(device = True)
def compute_Xt(St_T, Vt_T, Rt_T, Xt): #form polynomials
    for i in range(noi):
        Xt[i,0] = 1
        Xt[i,1] = 1 - St_T[i,0]
        Xt[i,2] = 1 - Vt_T[i,0]
        Xt[i,3] = 1 - Rt_T[i,0]

@cuda.jit(device = True)
def compute_Yt(Xt, betaYt, Yt): #compute least square fitted value
    for i in range(noi):
        X=0
        for j in range(NX):
            X += Xt[i,j]*betaYt[i,j]
        Yt[i,0] = Yt[i,0] + X

@cuda.jit(device = True)
def compute_Zt(Xt, betaZt, Zt): #compute least square fitted value
    for i in range(noi):
        X=0
        for j in range(NX):
            X += Xt[i,j]*betaZt[i,j]/dt
        Zt[i,0] = Zt[i,0] + X

@cuda.jit(device = True)
def searchsorted(Xbd, St_T, indbdX):
    for i in range(noi):
        j = 0
        if (St_T[i,0] <= Xbd[0]):
            indbdX[i,0] = 0
        elif (St_T[i,0] >= Xbd[-1]):
            indbdX[i,0] = noh
        else:
            while (St_T[i,0] > Xbd[j]):
                j += 1
            indbdX[i,0] = j-1

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
def compute_Yt_final(t, K, betaY, betaZ1, betaZ2, betaZ3, dW1, dW2, dW3, Xbd, Vbd, Rbd, St_T, Vt_T, Rt_1, Rt_T, Xt_1, Yt, Z1t, Z2t, Z3t):
    if t == Nt - 2:
        for jj in range(noi):
            Yt[jj,0] = max(K - math.exp(St_T[jj,0]), 0)
            Yt[jj,0] = Yt[jj,0] - Rt_1[jj,0] * Yt[jj,0]*dt
    else:
        betaYt = cuda.local.array(shape=(noi,NX), dtype=float64)
        betaZ1t = cuda.local.array(shape=(noi,NX), dtype=float64)
        betaZ2t = cuda.local.array(shape=(noi,NX), dtype=float64)
        betaZ3t = cuda.local.array(shape=(noi,NX), dtype=float64)
        indbdX = cuda.local.array(shape=(noi,1), dtype=int32)
        indbdV = cuda.local.array(shape=(noi,1), dtype=int32)
        indbdR = cuda.local.array(shape=(noi,1), dtype=int32)
        searchsorted(Xbd, St_T, indbdX) #index of box for each path
        searchsorted(Vbd, Vt_T, indbdV)
        searchsorted(Rbd, Rt_T, indbdR)
        compute_Xt(St_T, Vt_T, Rt_T, Xt_1)
        for jj in range(noi):
            for jjj in range(NX):
                betaYt[jj,jjj] = betaY[jjj,indbdX[jj,0],indbdV[jj,0],indbdR[jj,0]]
                betaZ1t[jj,jjj] = betaZ1[jjj,indbdX[jj,0],indbdV[jj,0],indbdR[jj,0]]
                betaZ2t[jj,jjj] = betaZ2[jjj,indbdX[jj,0],indbdV[jj,0],indbdR[jj,0]]
                betaZ3t[jj,jjj] = betaZ3[jjj,indbdX[jj,0],indbdV[jj,0],indbdR[jj,0]]
        compute_Yt(Xt_1, betaYt, Yt)
        compute_Zt(Xt_1, betaZ1t, Z1t)
        compute_Zt(Xt_1, betaZ2t, Z2t)
        compute_Zt(Xt_1, betaZ3t, Z3t)
        for jj in range(noi):
            Yt[jj,0] = max(max(K - math.exp(St_T[jj,0]), 0),Yt[jj,0])
            Yt[jj,0] = Yt[jj,0] - (Rt_1[jj,0] * Yt[jj,0] + math.sqrt((0.1395*Z1t[jj,0]**2 + 0.1900*Z2t[jj,0]**2 + 0.1992*Z3t[jj,0]**2)*0.5844))*dt
            Z1t[jj,0] = Yt[jj,0]*dW1[jj,0]*math.sqrt(dt)
            Z2t[jj,0] = Yt[jj,0]*dW2[jj,0]*math.sqrt(dt)
            Z3t[jj,0] = Yt[jj,0]*dW3[jj,0]*math.sqrt(dt)


@cuda.jit(device=True)
def compute_betaYt(i, j, k, t, K, betaY, betaZ1, betaZ2, betaZ3, dW1, dW2, dW3, Xbd, Vbd, Rbd, uniform_St_1, uniform_Vt_1, uniform_Rt_1):
    St_1 = cuda.local.array(shape=(noi,1), dtype=float64)
    St_T = cuda.local.array(shape=(noi,1), dtype=float64)
    Vt_1 = cuda.local.array(shape=(noi,1), dtype=float64)
    Vt_T = cuda.local.array(shape=(noi,1), dtype=float64)
    Rt_1 = cuda.local.array(shape=(noi,1), dtype=float64)
    Rt_T = cuda.local.array(shape=(noi,1), dtype=float64)
    Xt_1 = cuda.local.array(shape=(noi,NX), dtype=float64)
    Yt = cuda.local.array(shape=(noi,1), dtype=float64)
    Z1t = cuda.local.array(shape=(noi,1), dtype=float64)
    Z2t = cuda.local.array(shape=(noi,1), dtype=float64)
    Z3t = cuda.local.array(shape=(noi,1), dtype=float64)
    get_T1(i, j, k, dW1, dW2, dW3, Xbd, Vbd, Rbd, uniform_St_1, St_1, St_T, uniform_Vt_1, Vt_1, Vt_T, uniform_Rt_1, Rt_1, Rt_T)
    compute_Yt_final(t, K, betaY, betaZ1, betaZ2, betaZ3, dW1, dW2, dW3, Xbd, Vbd, Rbd, St_T, Vt_T, Rt_1, Rt_T, Xt_1, Yt, Z1t, Z2t, Z3t)
    compute_Xt(St_1, Vt_1, Rt_1, Xt_1)
    betaY_temp = cuda.local.array((NX,1), float64)
    betaZ1_temp = cuda.local.array((NX,1), float64)
    betaZ2_temp = cuda.local.array((NX,1), float64)
    betaZ3_temp = cuda.local.array((NX,1), float64)
    linalg_lstsq(Xt_1,Yt,betaY_temp)
    linalg_lstsq(Xt_1,Z1t,betaZ1_temp)
    linalg_lstsq(Xt_1,Z2t,betaZ2_temp)
    linalg_lstsq(Xt_1,Z3t,betaZ3_temp)
    for kk in range(NX):
        betaY[kk,i,j,k] = betaY_temp[kk,0]
        betaZ1[kk,i,j,k] = betaZ1_temp[kk,0]
        betaZ2[kk,i,j,k] = betaZ2_temp[kk,0]
        betaZ3[kk,i,j,k] = betaZ3_temp[kk,0]

@cuda.jit
def compute_betaYt_(t, K, betaY, betaZ1, betaZ2, betaZ3, dW1, dW2, dW3, Xbd, Vbd, Rbd, uniform_St_1, uniform_Vt_1, uniform_Rt_1):
    Xbd_ = cuda.shared.array((noh_1,), float64)
    Vbd_ = cuda.shared.array((noh_1,), float64)
    Rbd_ = cuda.shared.array((noh_1,), float64)
    dW1_ = cuda.const.array_like(dW1)
    dW2_ = cuda.const.array_like(dW2)
    dW3_ = cuda.const.array_like(dW3)
    for ii in range(noh_1):
        Xbd_[ii] = Xbd[ii]
        Vbd_[ii] = Vbd[ii]
        Rbd_[ii] = Rbd[ii]
    i, j, k = cuda.grid(3)
    if i < noh and j < noh and k < noh:
        compute_betaYt(i, j, k, t, K, betaY, betaZ1, betaZ2, betaZ3, dW1_, dW2_, dW3_, Xbd_, Vbd_, Rbd_, uniform_St_1, uniform_Vt_1, uniform_Rt_1)
    #cuda.syncthreads()

def compute_betaY(K, Xbd, Vbd, Rbd, betaY, betaZ1, betaZ2, betaZ3, threads_per_block, blocks):
    for t in range(Nt-2,-1,-1): #t is index of time
        uniform_St_1 = cp.random.uniform(0, 1, (noi, 1))
        uniform_Vt_1 = cp.random.uniform(0, 1, (noi, 1))
        uniform_Rt_1 = cp.random.uniform(0, 1, (noi, 1))
        dW1 = cp.random.randn(noi,1)
        dW2 = cp.random.randn(noi,1)
        dW3 = cp.random.randn(noi,1)
        #print('time =',t)
        compute_betaYt_[blocks, threads_per_block](t, K, betaY, betaZ1, betaZ2, betaZ3, dW1, dW2, dW3, Xbd, Vbd, Rbd, uniform_St_1, uniform_Vt_1, uniform_Rt_1)
    return betaY, betaZ1, betaZ2, betaZ3

#@cuda.jit
def P(K, Xbd, Vbd, Rbd, betaY, betaZ1, betaZ2, betaZ3, threads_per_block, blocks):
    betaY, betaZ1, betaZ2, betaZ3 = compute_betaY(K, Xbd, Vbd, Rbd, betaY, betaZ1, betaZ2, betaZ3, threads_per_block, blocks)
    Vt_1 = V0
    Rt_1 = R0
    St_1 = cp.log(S0)
    dZ1 = cp.random.randn(noi,1)
    dZ2 = cp.random.randn(noi,1)
    dZ3 = cp.random.randn(noi,1)
    Vt_T = cp.abs(Vt_1 + kappa*(theta - Vt_1) * dt + sigma*cp.sqrt(Vt_1*dt) * dZ1)
    Rt_T = cp.abs(Rt_1 + Rkappa*(Rtheta - Rt_1) * dt + Rsigma*cp.sqrt(Rt_1*dt) * dZ3)
    St_T = St_1 + (Rt_1 - 0.5 * Vt_1) * dt + cp.sqrt(Vt_1) * (rho * dZ1 + cp.sqrt(1 - rho**2) * dZ2) * cp.sqrt(dt)
    T2 = cp.where((St_T >= Xmin) & (St_T <= Xmax) & (Vt_T >= Vmin) & (Vt_T <= Vmax) & (Rt_T >= Rmin) & (Rt_T <= Rmax))[0]
    St_T = St_T[T2,:]
    lenT2 = len(T2)
    Yt = cp.zeros((lenT2,1))
    indbdX = cp.searchsorted(Xbd, St_T)-1
    indbdV = cp.searchsorted(Vbd, Vt_T)-1
    indbdR = cp.searchsorted(Rbd, Rt_T)-1
    XS1 = 1 - St_T
    XV1 = 1 - Vt_T
    XR1 = 1 - Rt_T
    X0 = cp.ones((lenT2,1))
    Xt = cp.hstack((X0,XS1,XV1,XR1))
    betaYt = cp.zeros((lenT2,NX))
    betaZ1t = cp.zeros((lenT2,NX))
    betaZ2t = cp.zeros((lenT2,NX))
    betaZ3t = cp.zeros((lenT2,NX))
    for i in range(lenT2):
        betaYt[i,:] = betaY[:, indbdX.flatten()[i], indbdV.flatten()[i], indbdR.flatten()[i]]
        betaZ1t[i,:] = betaZ1[:, indbdX.flatten()[i], indbdV.flatten()[i], indbdR.flatten()[i]]
        betaZ2t[i,:] = betaZ2[:, indbdX.flatten()[i], indbdV.flatten()[i], indbdR.flatten()[i]]
        betaZ3t[i,:] = betaZ3[:, indbdX.flatten()[i], indbdV.flatten()[i], indbdR.flatten()[i]]
    Yt = cp.empty((noi,1),dtype=cp.float64)
    Z1t = cp.empty((noi,1),dtype=cp.float64)
    Z2t = cp.empty((noi,1),dtype=cp.float64)
    Z3t = cp.empty((noi,1),dtype=cp.float64)
    for i in range(lenT2):
        for j in range(NX):
            Yt[i,0] += Xt[i,j]*betaYt[i,j]
            Z1t[i,0] += Xt[i,j]*betaZ1t[i,j]/dt
            Z2t[i,0] += Xt[i,j]*betaZ2t[i,j]/dt
            Z3t[i,0] += Xt[i,j]*betaZ3t[i,j]/dt
    Yt = cp.maximum(cp.maximum(K - cp.exp(St_T),0),Yt)
    cum_gen = - R0 * Yt - cp.sqrt((0.1395*Z1t**2 + 0.1900*Z2t**2 + 0.1992*Z3t**2)*0.5844)
    Yt = Yt + cum_gen * dt
    RBSDEvalue = cp.mean(Yt)
    return RBSDEvalue

def compute_(K, Xbd, Vbd, Rbd, betaY, betaZ1, betaZ2, betaZ3, threads_per_block, blocks, iter):
    Pvalue = cp.zeros((iter,1))
    for i in range(iter):
        #print('iteration =', i)
        Pvalue[i,:] = P(K, Xbd, Vbd, Rbd, betaY, betaZ1, betaZ2, betaZ3, threads_per_block, blocks)
    return Pvalue

global S0,  K,  V0, R0, kappa, theta, sigma, Rkappa, Rtheta, Rsigma, rho, T, Nt, noh, noi, mu, Xmin, Xmax, Vmin, Vmax, Rmin, Rmax, NX, dt, BD, Xbd, Vbd, Rbd

S0 = 100
Nt = 5 # number of time steps
mu = 1 # parameter for logistic sample generating
V0 = 0.04 # initial volatility
T = 0.1 # time horizon
R0 = 0.04 # risk-free rate
kappa = 1.58
theta = 0.03
sigma = 0.2
Rkappa = 0.26
Rtheta = 0.04
Rsigma = 0.08
rho = -0.26
noh = 80 # number of hypercubes
noi = 1000 # number of samples
dt = T/Nt
Xmin = 0;
Xmax = 5.5;
Vmin = 0;
Vmax = 1;
Rmin = 0;
Rmax = 1;
BD = cp.array([[Xmin, Xmax],[Vmin, Vmax],[Rmin, Rmax]])
Xbd = cp.linspace(BD[0,0],BD[0,1],num=noh+1)
Vbd = cp.linspace(BD[1,0],BD[1,1],num=noh+1)
Rbd = cp.linspace(BD[2,0],BD[2,1],num=noh+1)
NX=4 #number of polynomial functions
noh_1 = noh+1
betaY = cp.zeros((NX,noh,noh,noh),dtype=cp.float64)
betaZ1 = cp.zeros([NX, noh, noh, noh],dtype=cp.float64)
betaZ2 = cp.zeros([NX, noh, noh, noh],dtype=cp.float64)
betaZ3 = cp.zeros([NX, noh, noh, noh],dtype=cp.float64)

time_a = time.time()
threads_per_block = (4,4,4)
blocks_x = math.ceil(noh / threads_per_block [0])
blocks_y = math.ceil(noh / threads_per_block [1])
blocks_z = math.ceil(noh / threads_per_block [2])
blocks = (blocks_x, blocks_y, blocks_z)
iter = 50
K_range = [90,100,110]
meanValue = cp.empty(len(K_range))
stdValue = cp.empty(len(K_range))
for k in range(len(K_range)):
    Pvalue = compute_(K_range[k], Xbd, Vbd, Rbd, betaY, betaZ1, betaZ2, betaZ3, threads_per_block, blocks, iter)
    meanValue[k] = cp.mean(Pvalue)
    stdValue[k] = cp.std(Pvalue, ddof=1)
    print('meanValue=',meanValue[k])
    print('stdValue=',stdValue[k])
    print('time=',time.time()-time_a)
