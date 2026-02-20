import numpy as np

# ---------------------------------------------------------------------------
# Thomas algorithm — O(N) tri-diagonal solver
# ---------------------------------------------------------------------------
def thomas_solve(lo, diag, hi, rhs):
    n = len(diag)
    b, d, c, a = diag.copy(), rhs.copy(), hi.copy(), lo.copy()
    for k in range(1, n):
        m = a[k-1] / b[k-1]
        b[k] -= m * c[k-1]
        d[k] -= m * d[k-1]
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for k in range(n - 2, -1, -1):
        x[k] = (d[k] - c[k] * x[k + 1]) / b[k]
    return x

# ---------------------------------------------------------------------------
# Bilinear Interpolation
# ---------------------------------------------------------------------------
def bilinear_interp(x_nodes, y_nodes, Z, xq, yq):
    nx_n, ny_n = Z.shape
    ix = np.searchsorted(x_nodes, xq) - 1
    iy = np.searchsorted(y_nodes, yq) - 1
    ix, iy = max(0, min(ix, nx_n-2)), max(0, min(iy, ny_n-2))
    x1, x2, y1, y2 = x_nodes[ix], x_nodes[ix+1], y_nodes[iy], y_nodes[iy+1]
    wx, wy = (xq - x1)/(x2 - x1), (yq - y1)/(y2 - y1)
    return (1-wx)*(1-wy)*Z[ix, iy] + wx*(1-wy)*Z[ix+1, iy] + \
           (1-wx)*wy*Z[ix, iy+1] + wx*wy*Z[ix+1, iy+1]

# ---------------------------------------------------------------------------
# PDE Solver: Heston American Put (Operator Splitting ADI)
# ---------------------------------------------------------------------------
def heston_american_OS(spot, strike, rate, maturity, kappa, theta, sigma_v, rho, var_init,
                      nx=120, nv=60, nt=100, xlo=-3.0, xhi=3.0, vhi=2.5):
    K, r, T, S0, v0 = float(strike), float(rate), float(maturity), float(spot), float(var_init)
    dx, dv, dt = (xhi - xlo)/(nx+1), vhi/(nv+1), T/nt
    
    x_full, var_full = np.linspace(xlo, xhi, nx+2), np.linspace(0, vhi, nv+2)
    x_int, var_int = x_full[1:-1], var_full[1:-1]
    S_full = K * np.exp(x_full)
    
    # Payoff Grid
    g_full = np.maximum(K - S_full[:, np.newaxis], 0.0) * np.ones((nx+2, nv+2))
    V, lam = g_full.copy(), np.zeros((nx, nv))
    
    # Precompute Variance-direction (Lv) coefficients (constant over x)
    d2_v = 0.5 * sigma_v**2 * var_int / dv**2
    d1_v = kappa * (theta - var_int) / (2 * dv)
    lo_v, md_v, hi_v = -0.5*dt*(d2_v - d1_v), 1.0 + dt*d2_v, -0.5*dt*(d2_v + d1_v)

    for step in range(nt):
        tau = (step + 1) * dt
        # American Put Boundary Conditions
        V[0, :] = K - S_full[0]  # S -> 0 limit
        V[-1, :] = 0.0           # S -> inf limit
        V[:, 0] = np.maximum(K - S_full, 0.0) # v -> 0 limit (Payoff)
        V[:, -1] = K * np.exp(-r * tau)       # v -> inf limit (European Bond)

        # 1. Explicit Step (RHS calculation)
        Lx_n, Lv_n = np.zeros((nx, nv)), np.zeros((nx, nv))
        for jj in range(nv):
            v = var_int[jj]
            d2, d1 = v/dx**2, (r - 0.5*v)/(2*dx)
            Lx_n[:, jj] = (0.5*d2-d1)*V[:-2, jj+1] + (-d2-r)*V[1:-1, jj+1] + (0.5*d2+d1)*V[2:, jj+1]
            d2v, d1v = 0.5*sigma_v**2*v/dv**2, kappa*(theta-v)/(2*dv)
            Lv_n[:, jj] = (d2v-d1v)*V[1:-1, jj] - 2*d2v*V[1:-1, jj+1] + (d2v+d1v)*V[1:-1, jj+2]
            
        coeff_xv = (rho * sigma_v * var_int[np.newaxis, :]) / (4 * dx * dv)
        Lxv_n = coeff_xv * (V[2:, 2:] - V[2:, :-2] - V[:-2, 2:] + V[:-2, :-2])
        
        Y0 = V[1:-1, 1:-1] + dt*(Lx_n + Lv_n + Lxv_n) + dt*lam
        
        # 2. Implicit X-solve
        RHS1 = Y0 - 0.5*dt*Lx_n
        Y1 = np.zeros((nx, nv))
        for jj in range(nv):
            v = var_int[jj]
            d2, d1 = v/dx**2, (r - 0.5*v)/(2*dx)
            lox, mdx, hix = -0.5*dt*(0.5*d2-d1), 1.0 + 0.5*dt*(d2+r), -0.5*dt*(0.5*d2+d1)
            col = RHS1[:, jj].copy()
            col[0] -= lox * V[0, jj+1]
            col[-1] -= hix * V[-1, jj+1]
            Y1[:, jj] = thomas_solve(np.full(nx-1, lox), np.full(nx, mdx), np.full(nx-1, hix), col)
            
        # 3. Implicit V-solve
        RHS2 = Y1 - 0.5*dt*Lv_n
        Y2 = np.zeros((nx, nv))
        for ii in range(nx):
            row = RHS2[ii, :].copy()
            row[0] -= lo_v[0] * V[ii+1, 0]
            row[-1] -= hi_v[-1] * V[ii+1, -1]
            Y2[ii, :] = thomas_solve(lo_v[1:], md_v, hi_v[:-1], row)
            
        # 4. Operator Splitting Update (Constraint Projection)
        V_next = np.maximum(Y2, g_full[1:-1, 1:-1])
        lam = (V_next - Y2) / dt
        V[1:-1, 1:-1] = V_next
        
    return bilinear_interp(x_full, var_full, V, np.log(S0/K), v0)

if __name__ == "__main__":
    # Test Parameters (Case A)
    params = dict(spot=10, strike=10.0, rate=0.10, maturity=0.25, 
                  kappa=5.0, theta=0.16, sigma_v=0.9, rho=0.1, var_init=0.0625)
    
    # params = dict(spot=100.0, strike=110.0, rate=0.04, maturity=0.5, 
            #   kappa=1.58, theta=0.03, sigma_v=0.2, rho=-0.26, var_init=0.04)
    price = heston_american_OS(**params, nx=512, nv=256, nt=514)
    print(f"Corrected PDE Price: {price:.6f}")
