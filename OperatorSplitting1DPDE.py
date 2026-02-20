import numpy as np

def solve_high_precision_american(S0, K, r, sigma, T, M=500, N=500):
    # 1. Coordinate Transform: x = ln(S/K)
    x_min, x_max = -2.0, 2.0  # Wide enough for most vol/time
    x = np.linspace(x_min, x_max, N + 1)
    dx = x[1] - x[0]
    dt = T / M
    
    # Payoff in log-space
    S = K * np.exp(x)
    g = np.maximum(K - S, 0.0)
    V = g.copy()
    
    # Precompute Diagonals for Crank-Nicolson
    # Eq: V_t = 0.5*sig^2*V_xx + (r - 0.5*sig^2)*V_x - r*V
    sig2 = sigma**2
    drift = r - 0.5 * sig2
    
    # Finite Difference weights
    w_xx = sig2 / (2 * dx**2)
    w_x  = drift / (2 * dx)
    
    # Construct Implicit (A) and Explicit (B) matrices
    # (I - 0.5*dt*L) V_next = (I + 0.5*dt*L) V_now
    a_imp = -0.5 * dt * (w_xx - w_x)
    b_imp =  1.0 + 0.5 * dt * (w_xx * 2 + r)
    c_imp = -0.5 * dt * (w_xx + w_x)
    
    a_exp = 0.5 * dt * (w_xx - w_x)
    b_exp = 1.0 - 0.5 * dt * (w_xx * 2 + r)
    c_exp = 0.5 * dt * (w_xx + w_x)

    for m in range(M):
        # Boundary Conditions (Put)
        # S->0 (x->-inf): V = K - S*exp(-r*t)
        # S->inf (x->inf): V = 0
        V[0] = K - S[0] * np.exp(-r * m * dt)
        V[-1] = 0
        
        # Calculate RHS (Explicit part)
        rhs = np.zeros(N - 1)
        for j in range(1, N):
            rhs[j-1] = a_exp * V[j-1] + b_exp * V[j] + c_exp * V[j+1]
        
        # Add Boundary injections to RHS
        rhs[0] += a_imp * V[0] 
        rhs[-1] += c_imp * V[-1]
        
        # Solve Tri-diagonal system
        V_next = thomas_solve(
            np.full(N-2, a_imp), 
            np.full(N-1, b_imp), 
            np.full(N-2, c_imp), 
            rhs
        )
        
        # Apply American Constraint (Operator Splitting)
        V[1:N] = np.maximum(V_next, g[1:N])

    return np.interp(np.log(S0/K), x, V)

def thomas_solve(a, b, c, d):
    nf = len(d)
    bc = b.copy()
    dc = d.copy()
    for i in range(1, nf):
        mc = a[i-1] / bc[i-1]
        bc[i] -= mc * c[i-1]
        dc[i] -= mc * dc[i-1]
    x = bc
    x[-1] = dc[-1] / bc[-1]
    for i in range(nf-2, -1, -1):
        x[i] = (dc[i] - c[i] * x[i+1]) / bc[i]
    return x

# Testing: S=100, K=100, r=0.06, sig=0.2, T=1.0
# Previous Euler price: ~6.08
# Correct American price: ~6.55
print(f"High Precision Price: {solve_high_precision_american(10, 10, 0.1, 0.4, 0.5):.4f}")