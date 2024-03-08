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
        Vt_T[i,0] = Vt_1[i,0] + kappa*(alpha - Vt_1[i,0])*dt + sigma2*dW1[i,0]*math.sqrt(dt)
        St_T[i,0] = St_1[i,0] + (r - Vt_1[i,0] - (sigma1**2)/2)*dt + sigma1*(rho*dW1[i,0]*math.sqrt(dt) + math.sqrt(1-rho**2)*dW2[i,0]*math.sqrt(dt))

@cuda.jit(device = True)
def compute_Xt(St_T, Vt_T, Xt): #form polynomials
    for i in range(noi):
        Xt[i,0] = 1
        Xt[i,1] = 1 - St_T[i,0]
        Xt[i,2] = 1 - Vt_T[i,0]
        '''Xt[i,2] = St_T[i,0]**2/2 - 2*St_T[i,0] + 1
        Xt[i,3] = 1 - Vt_T[i,0]
        Xt[i,4] = Vt_T[i,0]**2/2 - 2*Vt_T[i,0] + 1
        Xt[i,5] = 1 - Rt_T[i,0]
        Xt[i,6] = Rt_T[i,0]**2/2 - 2*Rt_T[i,0] + 1
        Xt[i,7] = Xt[i,1]*Xt[i,3]
        Xt[i,8] = Xt[i,1]*Xt[i,5]
        Xt[i,9] = Xt[i,3]*Xt[i,5]'''

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
def compute_Yt_final(t, S0, betaY, betaZ1, betaZ2, dW1, dW2, Xbd, Vbd, St_T, Vt_T, Xt_1, Yt, Z1t, Z2t):
    if t == Nt - 2:
        for jj in range(noi):
            y1 = R*math.exp(-m*Nt*dt)*w0*(a-b*math.exp(-c*Nt*dt))**3
            y2 = R*math.exp(-m*(t+1)*dt)*w0*((a-b*math.exp(-c*(t+1)*dt))**3)
            y3 = f*3*w0*b*c*math.exp(-3*c*(t+1)*dt)*(b - a*math.exp(c*(t+1)*dt))**2
            Yt[jj,0] = max((math.exp(St_T[jj,0]) - H)*y1, 0)
            Yt[jj,0] = Yt[jj,0] + (-r*Yt[jj,0] - C*y2*y3)*dt
    else:
        betaYt = cuda.local.array(shape=(noi,NX), dtype=float64)
        betaZ1t = cuda.local.array(shape=(noi,NX), dtype=float64)
        betaZ2t = cuda.local.array(shape=(noi,NX), dtype=float64)
        indbdX = cuda.local.array(shape=(noi,1), dtype=int32)
        indbdV = cuda.local.array(shape=(noi,1), dtype=int32)
        searchsorted(Xbd, St_T, indbdX) #index of box for each path
        searchsorted(Vbd, Vt_T, indbdV)
        #Xt = cuda.local.array(shape=(noi,NX), dtype=float64)
        compute_Xt(St_T, Vt_T, Xt_1)
        for jj in range(noi):
            for jjj in range(NX):
                betaYt[jj,jjj] = betaY[jjj,indbdX[jj,0],indbdV[jj,0]]
                betaZ1t[jj,jjj] = betaZ1[jjj,indbdX[jj,0],indbdV[jj,0]]
                betaZ2t[jj,jjj] = betaZ2[jjj,indbdX[jj,0],indbdV[jj,0]]
        compute_Yt(Xt_1, betaYt, Yt)
        compute_Zt(Xt_1, betaZ1t, Z1t)
        compute_Zt(Xt_1, betaZ2t, Z2t)
        for jj in range(noi):
            y2 = R*math.exp(-m*(t+1)*dt)*w0*((a-b*math.exp(-c*(t+1)*dt))**3)
            y3 = f*3*w0*b*c*math.exp(-3*c*(t+1)*dt)*(b - a*math.exp(c*(t+1)*dt))**2
            y4 = R*math.exp(-m*(t+2)*dt)*w0*((a-b*math.exp(-c*(t+2)*dt))**3)
            Yt[jj,0] = max(max((math.exp(St_T[jj,0]) - H)*y4, 0),Yt[jj,0])
            Yt[jj,0] = Yt[jj,0] + (- r*Yt[jj,0] - C*y2*y3 - math.sqrt((0.1992*Z1t[jj,0]**2 + 0.1900*Z2t[jj,0]**2)*chi))*dt
            Z1t[jj,0] = Yt[jj,0]*dW1[jj,0]*math.sqrt(dt)
            Z2t[jj,0] = Yt[jj,0]*dW2[jj,0]*math.sqrt(dt)


@cuda.jit(device=True)
def compute_betaYt(i, j, t, S0, betaY, betaZ1, betaZ2, dW1, dW2, Xbd, Vbd, uniform_St_1, uniform_Vt_1):
    St_1 = cuda.local.array(shape=(noi,1), dtype=float64)
    St_T = cuda.local.array(shape=(noi,1), dtype=float64)
    Vt_1 = cuda.local.array(shape=(noi,1), dtype=float64)
    Vt_T = cuda.local.array(shape=(noi,1), dtype=float64)
    #cum_gen = cuda.local.array(shape=(noi,1), dtype=float64)
    Xt_1 = cuda.local.array(shape=(noi,NX), dtype=float64)
    Yt = cuda.local.array(shape=(noi,1), dtype=float64)
    Z1t = cuda.local.array(shape=(noi,1), dtype=float64)
    Z2t = cuda.local.array(shape=(noi,1), dtype=float64)
    get_T1(i, j, dW1, dW2, Xbd, Vbd, uniform_St_1, St_1, St_T, uniform_Vt_1, Vt_1, Vt_T)
    compute_Yt_final(t, S0, betaY, betaZ1, betaZ2, dW1, dW2, Xbd, Vbd, St_T, Vt_T, Xt_1, Yt, Z1t, Z2t)
    compute_Xt(St_1, Vt_1, Xt_1)
    betaY_temp = cuda.local.array((NX,1), float64)
    betaZ1_temp = cuda.local.array((NX,1), float64)
    betaZ2_temp = cuda.local.array((NX,1), float64)
    linalg_lstsq(Xt_1,Yt,betaY_temp)
    linalg_lstsq(Xt_1,Z1t,betaZ1_temp)
    linalg_lstsq(Xt_1,Z2t,betaZ2_temp)
    for kk in range(NX):
        betaY[kk,i,j] = betaY_temp[kk,0]
        betaZ1[kk,i,j] = betaZ1_temp[kk,0]
        betaZ2[kk,i,j] = betaZ2_temp[kk,0]

@cuda.jit
def compute_betaYt_(t, S0, betaY, betaZ1, betaZ2, dW1, dW2, Xbd, Vbd, uniform_St_1, uniform_Vt_1):
    Xbd_ = cuda.shared.array((noh_1,), float64)
    Vbd_ = cuda.shared.array((noh_1,), float64)
    dW1_ = cuda.const.array_like(dW1)
    dW2_ = cuda.const.array_like(dW2)
    #BetaY_ = cuda.const.array_like(betaY)
    for ii in range(noh_1):
        Xbd_[ii] = Xbd[ii]
        Vbd_[ii] = Vbd[ii]
    i, j = cuda.grid(2)
    if i < noh and j < noh:
        compute_betaYt(i, j, t, S0, betaY, betaZ1, betaZ2, dW1_, dW2_, Xbd_, Vbd_, uniform_St_1, uniform_Vt_1)
    #cuda.syncthreads()

def compute_betaY(S0, Xbd, Vbd, betaY, betaZ1, betaZ2, threads_per_block, blocks):
    for t in range(Nt-2,-1,-1): #t is index of time
        uniform_St_1 = cp.random.uniform(0, 1, (noi, 1))
        uniform_Vt_1 = cp.random.uniform(0, 1, (noi, 1))
        dW1 = cp.random.randn(noi,1)
        dW2 = cp.random.randn(noi,1)
        #print('time =',t)
        compute_betaYt_[blocks, threads_per_block](t, S0, betaY, betaZ1, betaZ2, dW1, dW2, Xbd, Vbd, uniform_St_1, uniform_Vt_1)
        #betaY[:,:,:,Nt-t-2] = betaY_
        #cuda.synchronize()
    return betaY, betaZ1, betaZ2

#@cuda.jit
def P(S0, Xbd, Vbd, betaY, betaZ1, betaZ2, threads_per_block, blocks):
    betaY, betaZ1, betaZ2 = compute_betaY(S0, Xbd, Vbd, betaY, betaZ1, betaZ2, threads_per_block, blocks)
    Vt_1 = V0
    St_1 = cp.log(S0)
    dZ1 = cp.random.randn(noi,1)
    dZ2 = cp.random.randn(noi,1)
    Vt_T = Vt_1 + kappa*(alpha - Vt_1) * dt + sigma2*cp.sqrt(dt) * dZ1
    St_T = St_1 + (r - Vt_1 - (sigma1**2)/2) * dt + sigma1 * (rho * dZ1 + cp.sqrt(1 - rho**2) * dZ2) * cp.sqrt(dt)
    T2 = cp.where((St_T >= Xmin) & (St_T <= Xmax) & (Vt_T >= Vmin) & (Vt_T <= Vmax))[0]
    St_T = St_T[T2,:]
    lenT2 = len(T2)
    Yt = cp.zeros((lenT2,1))
    indbdX = cp.searchsorted(Xbd, St_T)-1
    indbdV = cp.searchsorted(Vbd, Vt_T)-1
    XS1 = 1 - St_T
    #XS2 = St_T**2/2 - 2*St_T + 1
    XV1 = 1 - Vt_T
    #XV2 = Vt_T**2/2 - 2*Vt_T + 1
    X0 = cp.ones((lenT2,1))
    Xt = cp.hstack((X0,XS1,XV1))#Xt = cp.hstack((X0,XS1,XS2,XV1,XV2,XR1,XR2,XS1*XV1,XS1*XR1,XV1*XR1))
    betaYt = cp.zeros((lenT2,NX))
    betaZ1t = cp.zeros((lenT2,NX))
    betaZ2t = cp.zeros((lenT2,NX))
    for i in range(lenT2):
        betaYt[i,:] = betaY[:, indbdX.flatten()[i], indbdV.flatten()[i]]
        betaZ1t[i,:] = betaZ1[:, indbdX.flatten()[i], indbdV.flatten()[i]]
        betaZ2t[i,:] = betaZ2[:, indbdX.flatten()[i], indbdV.flatten()[i]]
    Yt = cp.empty((noi,1),dtype=cp.float64)
    Z1t = cp.empty((noi,1),dtype=cp.float64)
    Z2t = cp.empty((noi,1),dtype=cp.float64)
    for i in range(lenT2):
        for j in range(NX):
            Yt[i,0] += Xt[i,j]*betaYt[i,j]
            Z1t[i,0] += Xt[i,j]*betaZ1t[i,j]/dt
            Z2t[i,0] += Xt[i,j]*betaZ2t[i,j]/dt
    t = 0
    y2 = R*cp.exp(-m*(t+1)*dt)*w0*((a-b*cp.exp(-c*(t+1)*dt))**3)
    y3 = f*3*w0*b*c*cp.exp(-3*c*(t+1)*dt)*(b - a*cp.exp(c*(t+1)*dt))**2
    y4 = R*cp.exp(-m*(t+2)*dt)*w0*((a-b*cp.exp(-c*(t+2)*dt))**3)
    Yt = cp.maximum(cp.maximum((cp.exp(St_T) - H)*y4, 0),Yt)
    Yt = Yt + (- r*Yt - C*y2*y3 - cp.sqrt((0.1992*Z1t**2 + 0.1900*Z2t**2)*chi))*dt
    return cp.mean(Yt)

def compute_(S0, Xbd, Vbd, betaY, betaZ1, betaZ2, threads_per_block, blocks, iter):
    Pvalue = cp.zeros((iter,1))
    for i in range(iter):
        #print('iteration =', i)
        Pvalue[i,:] = P(S0, Xbd, Vbd, betaY, betaZ1, betaZ2, threads_per_block, blocks)
    return Pvalue

global S0,  K,  V0, r, kappa, alpha, sigma1, sigma2, rho, chi, R, m, w0, a, b, c, f, C, H, T, Nt, noh, noi, mu, Xmin, Xmax, Vmin, Vmax, NX, dt, BD, Xbd, Vbd

Nt = 100 # number of time steps
mu = 1 # parameter for logistic sample generating
V0 = -0.15 # initial volatility
T = 2 # time horizon
r = 0.0303 # risk-free rate
kappa = 1.092
# alpha = alpha - lambda/kappa
alpha = -0.0017
sigma1 = 0.158
sigma2 = 0.221
rho = 0.803
chi = 0# 0.2107
# parameters for the growth function
R = 10000
m = 0.1
w0 = 6
a = 1.113
b = 1.097
c = 1.43
f = 1.1
C = 7
H = 3
noh = 100 # number of hypercubes
noi = 1000 # number of samples
dt = T/Nt
Xmin = 0
Xmax = 5.5
Vmin = -1.5
Vmax = 1.5
BD = cp.array([[Xmin, Xmax],[Vmin, Vmax]])
Xbd = cp.linspace(BD[0,0],BD[0,1],num=noh+1)
Vbd = cp.linspace(BD[1,0],BD[1,1],num=noh+1)
NX=3#NX = 10 #number of polynomial functions
noh_1 = noh+1
betaY = cp.zeros((NX,noh,noh),dtype=cp.float64)
betaZ1 = cp.zeros([NX, noh, noh],dtype=cp.float64)
betaZ2 = cp.zeros([NX, noh, noh],dtype=cp.float64)

time_a = time.time()
threads_per_block = (4,4)
blocks_x = math.ceil(noh / threads_per_block [0])
blocks_y = math.ceil(noh / threads_per_block [1])
blocks = (blocks_x, blocks_y)
iter = 50
S = [35,45,55]
meanValue = cp.empty(len(S))
stdValue = cp.empty(len(S))
for k in range(len(S)):
    Pvalue = compute_(S[k], Xbd, Vbd, betaY, betaZ1, betaZ2, threads_per_block, blocks, iter)
    meanValue[k] = cp.mean(Pvalue)
    stdValue[k] = cp.std(Pvalue, ddof=1)
    print('meanValue=',meanValue[k])
    print('stdValue=',stdValue[k])
    print('time=',time.time()-time_a)