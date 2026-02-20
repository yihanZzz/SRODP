# ---------------------------------------------------------------------------
# Reference: Longstaff-Schwartz MC for independent validation
# ---------------------------------------------------------------------------
import numpy as np
from numpy.polynomial.laguerre import lagval

def weighted_laguerre(x, degree):
    # Create an identity matrix to pick specific polynomial degrees
    coeffs = np.eye(degree + 1)
    return [np.exp(-x / 2) * lagval(x, c) for c in coeffs]

def mc_american_put(spot, strike, rate, maturity,
                    kappa, theta, sigma_v, rho, var_init,
                    n_paths=80000, n_steps=200, seed=42):
    """
    Longstaff-Schwartz Monte Carlo for American put under Heston.
    Uses full-truncation Euler for the variance SDE.
    Returns (price, standard_error).
    """
    rng     = np.random.default_rng(seed)
    S0      = float(spot)
    K       = float(strike)
    r       = float(rate)
    T       = float(maturity)
    var0    = float(var_init)

    dt      = T / n_steps
    sqdt    = np.sqrt(dt)
    disc    = np.exp(-r * dt)

    Z1 = rng.standard_normal((n_steps, n_paths))
    Z2 = rho*Z1 + np.sqrt(1.0 - rho**2)*rng.standard_normal((n_steps, n_paths))

    S_paths   = np.zeros((n_steps + 1, n_paths))
    var_paths = np.zeros((n_steps + 1, n_paths))
    S_paths[0]   = S0
    var_paths[0] = var0

    for t in range(n_steps):
        vp = np.maximum(var_paths[t], 0.0)
        sv = np.sqrt(vp)
        S_paths[t + 1]   = S_paths[t] * np.exp((r - 0.5*vp)*dt + sv*sqdt*Z1[t])
        var_paths[t + 1] = vp + kappa*(theta - vp)*dt + sigma_v*sv*sqdt*Z2[t]

    cashflow = np.maximum(K - S_paths[-1], 0.0)
    ex_step  = np.full(n_paths, n_steps)

    for t in range(n_steps - 1, 0, -1):
        payoff = np.maximum(K - S_paths[t], 0.0)
        itm    = payoff > 0.0
        if itm.sum() < 10:
            continue
        disc_cf = cashflow[itm] * disc**(ex_step[itm] - t)
        Sr = S_paths[t][itm];   vr = var_paths[t][itm]

        ####################################################
        ######## Laguerre basis functions for regression (normalized) ########
        S_norm = Sr / strike
        v_norm = vr / var0
        L_S = weighted_laguerre(S_norm, 3) # Returns [L0, L1, L2, L3]
        L_v = weighted_laguerre(v_norm, 2) # Returns [L0, L1, L2]
        # We include the marginal polynomials and the cross-terms
        A = np.column_stack([
            L_S[0], L_S[1], L_S[2], L_S[3],  # Spot terms
            L_v[1], L_v[2],                  # Var terms (L0 is redundant constant)
            L_S[1] * L_v[1],                 # Cross term S*v
            L_S[2] * L_v[1],                 # Cross term S^2*v
            L_S[1] * L_v[2]                  # Cross term S*v^2
        ])
        ######################################################
        
        #######################################################
        ###### Alternative: simple polynomial basis (unscaled) ######
        # A  = np.column_stack([
        #     np.ones(itm.sum()), Sr, Sr**2 , vr, vr**2, Sr*vr, Sr**2*vr, Sr*vr**2, Sr**3, vr**3
        # ])
        #######################################################
        coef, *_ = np.linalg.lstsq(A, disc_cf, rcond=None)
        cont     = A @ coef
        ex       = itm.copy()
        ex[itm]  = payoff[itm] > cont
        cashflow[ex] = payoff[ex]
        ex_step[ex]  = t

    cf0   = cashflow * disc**ex_step
    price = float(np.mean(cf0))
    se    = float(np.std(cf0) / np.sqrt(n_paths))
    return price, se

if __name__ == "__main__":

    print("=" * 65)
    print("  American Put — Heston Model")
    print("=" * 65)

# params_A = dict(
#         spot     = 100.0,
#         strike   = 90.0,
#         rate     = 0.04,
#         maturity = 0.5,
#         kappa    = 1.58,
#         theta    = 0.03,
#         sigma_v  = 0.2,
#         rho      = -0.26,
#         var_init = 0.04,
#     )

# mc_A, se_A = mc_american_put(**params_A, n_paths=1000000, n_steps=50)
# print(f"  MC reference            : {mc_A:.6f} ± {1.96*se_A:.4f} (95% CI)")
# print(f"  MC SE            : {se_A:.6f}")
strikes = [90.0, 100.0, 110.0]
n_steps_list = [5, 25, 50, 100, 200]

results = {}

for strike in strikes:
    for n_steps in n_steps_list:
        params = dict(
            spot     = 100.0,
            strike   = strike,
            rate     = 0.04,
            maturity = 0.5,
            kappa    = 1.58,
            theta    = 0.03,
            sigma_v  = 0.2,
            rho      = -0.26,
            var_init = 0.04,
        )

        mc, se = mc_american_put(**params, n_paths=1_000_000, n_steps=n_steps)
        results[(strike, n_steps)] = (mc, se)
        print(f"strike={strike:.1f}, n_steps={n_steps:>3d} -> price={mc:.6f}, se={se:.6f}")