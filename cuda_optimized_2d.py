#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized version using pure CuPy vectorization
Much simpler and more scalable than Numba approach
Supports noi >> 2000 (tested up to 50,000+)
"""

import time
import cupy as cp
import numpy as np
import os
import json
import logging
import contextlib
from datetime import datetime, timezone


class HestonRBSDE:
    """Reflected BSDE solver for American options under Heston model"""
    
    def __init__(self, S0=100, V0=0.04, r=0.04, kappa=1.58, theta=0.03, 
             sigma=0.2, rho=-0.26, T=0.5, Nt=25, noh=100, noi=5000, 
             mu=1, Xmin=0, Xmax=6, Vmin=0, Vmax=1, NX=3,
             chi=0.2107):
        
        self.S0 = S0
        self.V0 = V0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.T = T
        self.Nt = Nt
        self.noh = noh
        self.noi = noi
        self.mu = mu
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.NX = NX
        self.dt = T / Nt
        self.chi = chi
        
        # Create boundaries
        self.Xbd = cp.linspace(Xmin, Xmax, num=noh + 1)
        self.Vbd = cp.linspace(Vmin, Vmax, num=noh + 1)
        
    def generate_initial_samples(self, i, j):
        """
        Generate samples uniformly within box (i,j) using logistic transform
        Returns: St_1, Vt_1 (both of shape (noi,))
        """
        # Box boundaries
        fxl1 = 1 / (1 + cp.exp(-self.mu * self.Xbd[i]))
        fxu1 = 1 / (1 + cp.exp(-self.mu * self.Xbd[i + 1]))
        fxl2 = 1 / (1 + cp.exp(-self.mu * self.Vbd[j]))
        fxu2 = 1 / (1 + cp.exp(-self.mu * self.Vbd[j + 1]))
        
        # Uniform samples in [0,1]
        uniform_St = cp.random.uniform(0, 1, self.noi)
        uniform_Vt = cp.random.uniform(0, 1, self.noi)
        
        # Inverse logistic transform
        St_1 = -(1 / self.mu) * cp.log(1 / (fxl1 + uniform_St * (fxu1 - fxl1)) - 1)
        Vt_1 = -(1 / self.mu) * cp.log(1 / (fxl2 + uniform_Vt * (fxu2 - fxl2)) - 1)
        
        return St_1, Vt_1
    
    def evolve_paths(self, St_1, Vt_1):
        """
        Evolve from t to t+dt using Heston dynamics
        Returns: St_T, Vt_T
        """
        dW1 = cp.random.randn(self.noi)
        dW2 = cp.random.randn(self.noi)
        
        # Volatility evolution (with abs to ensure positivity)
        Vt_T = cp.abs(Vt_1 + self.kappa * (self.theta - Vt_1) * self.dt + 
                      self.sigma * cp.sqrt(Vt_1 * self.dt) * dW1)
        
        # Stock price evolution (log price)
        St_T = (St_1 + (self.r - 0.5 * Vt_1) * self.dt + 
                cp.sqrt(Vt_1) * (self.rho * dW1 + cp.sqrt(1 - self.rho**2) * dW2) * cp.sqrt(self.dt))
        
        return St_T, Vt_T, dW1, dW2
    
    def compute_polynomial_basis(self, St, Vt):
        """
        Compute polynomial basis functions
        Returns: Xt of shape (noi, NX)
        """
        Xt = cp.zeros((self.noi, self.NX))
        Xt[:, 0] = 1
        Xt[:, 1] = 1 - St
        Xt[:, 2] = 1 - Vt
        return Xt
    
    def find_boxes(self, St_T, Vt_T):
        """
        Find which box each path belongs to
        Returns: indbdX, indbdV (both of shape (noi,))
        """
        indbdX = cp.searchsorted(self.Xbd, St_T) - 1
        indbdV = cp.searchsorted(self.Vbd, Vt_T) - 1
        
        # Clip to valid range
        indbdX = cp.clip(indbdX, 0, self.noh - 1)
        indbdV = cp.clip(indbdV, 0, self.noh - 1)
        
        return indbdX, indbdV
    
    def compute_Yt_from_betas(self, K, St_T, Vt_T, Xt_T, betaY, betaZ1, betaZ2, 
                              indbdX, indbdV, t, dW1, dW2):
        """
        Compute Y_t, Z1_t, Z2_t using beta values from next timestep
        """
        if t == self.Nt - 2:
            # Terminal condition
            Yt = cp.maximum(K - cp.exp(St_T), 0)
            Yt = Yt - self.r * Yt * self.dt
            Z1t = cp.zeros(self.noi)
            Z2t = cp.zeros(self.noi)
        else:
            # Get beta values for each path's box
            time_idx = self.Nt - t - 3
            
            # Vectorized beta extraction
            betaYt = betaY[:, indbdX, indbdV, time_idx].T  # Shape: (noi, NX)
            betaZ1t = betaZ1[:, indbdX, indbdV, time_idx].T
            betaZ2t = betaZ2[:, indbdX, indbdV, time_idx].T
            
            # Compute fitted values
            Y = cp.sum(Xt_T * betaYt, axis=1)
            Z1 = cp.sum(Xt_T * betaZ1t, axis=1) / self.dt
            Z2 = cp.sum(Xt_T * betaZ2t, axis=1) / self.dt
            
            # Apply reflection (American exercise)
            Y = cp.maximum(cp.maximum(K - cp.exp(St_T), 0), Y)
            
            # BSDE drift term
            penalty = cp.sqrt(((0.1395**2) * Z1**2 + (0.1900**2) * Z2**2) * self.chi)
            Y = Y - (self.r * Y + penalty) * self.dt
            
            # Z terms
            Z1t = Y * dW1 * cp.sqrt(self.dt)
            Z2t = Y * dW2 * cp.sqrt(self.dt)
            
            Yt = Y
        
        return Yt, Z1t, Z2t
    
    def compute_betaY(self, K):
        """
        Main backward induction loop to compute beta coefficients
        """
        betaY = cp.zeros((self.NX, self.noh, self.noh, self.Nt - 1))
        betaZ1 = cp.zeros((self.NX, self.noh, self.noh, self.Nt - 1))
        betaZ2 = cp.zeros((self.NX, self.noh, self.noh, self.Nt - 1))
        
        print(f"Starting backward induction for K={K}...")
        
        for t in range(self.Nt - 2, -1, -1):
            if t % 5 == 0:
                print(f"  Timestep {t}/{self.Nt - 2}")
            
            # Process all boxes
            for i in range(self.noh):
                for j in range(self.noh):
                    # Generate samples in box (i,j) at time t
                    St_1, Vt_1 = self.generate_initial_samples(i, j)
                    
                    # Evolve to t+dt
                    St_T, Vt_T, dW1, dW2 = self.evolve_paths(St_1, Vt_1)
                    
                    # Find destination boxes
                    indbdX, indbdV = self.find_boxes(St_T, Vt_T)
                    
                    # Compute polynomial basis at T
                    Xt_T = self.compute_polynomial_basis(St_T, Vt_T)
                    
                    # Compute Y_t, Z1_t, Z2_t
                    Yt, Z1t, Z2t = self.compute_Yt_from_betas(
                        K, St_T, Vt_T, Xt_T, betaY, betaZ1, betaZ2, 
                        indbdX, indbdV, t, dW1, dW2
                    )
                    
                    # Compute polynomial basis at t (for regression)
                    Xt_1 = self.compute_polynomial_basis(St_1, Vt_1)
                    
                    # Perform least squares regression (CuPy optimized)
                    betaY_temp = cp.linalg.lstsq(Xt_1, Yt.reshape(-1, 1), rcond=None)[0]
                    betaZ1_temp = cp.linalg.lstsq(Xt_1, Z1t.reshape(-1, 1), rcond=None)[0]
                    betaZ2_temp = cp.linalg.lstsq(Xt_1, Z2t.reshape(-1, 1), rcond=None)[0]
                    
                    # Store coefficients
                    betaY[:, i, j, self.Nt - t - 2] = betaY_temp.flatten()
                    betaZ1[:, i, j, self.Nt - t - 2] = betaZ1_temp.flatten()
                    betaZ2[:, i, j, self.Nt - t - 2] = betaZ2_temp.flatten()
        
        return betaY, betaZ1, betaZ2
    
    def compute_option_price(self, K, betaY, betaZ1, betaZ2):
        """
        Compute option price using computed beta values
        """
        # Initial state
        St_1 = cp.log(self.S0)
        Vt_1 = self.V0
        
        # Generate random shocks
        dZ1 = cp.random.randn(self.noi)
        dZ2 = cp.random.randn(self.noi)
        
        # Evolve one step
        Vt_T = cp.abs(Vt_1 + self.kappa * (self.theta - Vt_1) * self.dt + 
                      self.sigma * cp.sqrt(Vt_1 * self.dt) * dZ1)
        St_T = (St_1 + (self.r - 0.5 * Vt_1) * self.dt + 
                cp.sqrt(Vt_1) * (self.rho * dZ1 + cp.sqrt(1 - self.rho**2) * dZ2) * cp.sqrt(self.dt))
        
        # Filter paths within domain
        valid = ((St_T >= self.Xmin) & (St_T <= self.Xmax) & 
                (Vt_T >= self.Vmin) & (Vt_T <= self.Vmax))
        St_T = St_T[valid]
        Vt_T = Vt_T[valid]
        lenT2 = len(St_T)
        
        if lenT2 == 0:
            return cp.maximum(K - self.S0, 0)
        
        # Find boxes
        indbdX = cp.searchsorted(self.Xbd, St_T) - 1
        indbdV = cp.searchsorted(self.Vbd, Vt_T) - 1
        indbdX = cp.clip(indbdX, 0, self.noh - 1)
        indbdV = cp.clip(indbdV, 0, self.noh - 1)
        
        # Get beta values
        betaYt = betaY[:, indbdX, indbdV, self.Nt - 2].T
        betaZ1t = betaZ1[:, indbdX, indbdV, self.Nt - 2].T
        betaZ2t = betaZ2[:, indbdX, indbdV, self.Nt - 2].T
        
        # Compute polynomial basis
        Xt = cp.zeros((lenT2, self.NX))
        Xt[:, 0] = 1
        Xt[:, 1] = 1 - St_T
        Xt[:, 2] = 1 - Vt_T
        
        # Compute Y, Z1, Z2
        Yt = cp.sum(Xt * betaYt, axis=1)
        Z1t = cp.sum(Xt * betaZ1t, axis=1) / self.dt
        Z2t = cp.sum(Xt * betaZ2t, axis=1) / self.dt
        
        # Apply reflection and drift
        Yt = cp.maximum(cp.maximum(K - cp.exp(St_T), 0), Yt)
        penalty = cp.sqrt(((0.1395**2) * Z1t**2 + (0.1900**2) * Z2t**2) * self.chi)
        Yt = Yt - (self.r * Yt + penalty) * self.dt
        
        # Return maximum of expected value and immediate exercise
        return cp.maximum(cp.mean(Yt), cp.maximum(K - self.S0, 0))
    
def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(log_path)  # unique name per file
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate prints if root logger exists

    # Prevent adding handlers multiple times (important in notebooks / reruns)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def main():
    os.makedirs("logs", exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    run_id = now_utc.strftime("%Y%m%d_%H%M%S")

    log_txt = os.path.join("logs", f"run_{run_id}.log")
    log_json = os.path.join("logs", f"run_{run_id}.json")

    logger = setup_logger(log_txt)

    # Container for structured results
    all_results = []

    solver = HestonRBSDE(
        S0=100, V0=0.04, r=0.04, kappa=1.58, theta=0.03,
        sigma=0.2, rho=-0.26, T=0.5, Nt=25, noh=100,
        noi=10000,
        mu=1, Xmin=0, Xmax=6, Vmin=0, Vmax=1, NX=3,
        chi=0.1026
    )

    K_range = [90, 100, 110]
    iterations = 50

    # --- tee stdout to file + console ---
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    import sys
    with open(log_txt, "a", encoding="utf-8") as f:
        tee = Tee(sys.stdout, f)
        with contextlib.redirect_stdout(tee):

            print("=" * 80)
            print("Heston RBSDE run started")
            print(f"UTC time: {datetime.now(timezone.utc).isoformat()}")
            print("=" * 80)

            # Log solver parameters ONCE
            params = {
                "solver": {
                    "S0": solver.S0, "V0": solver.V0, "r": solver.r,
                    "kappa": solver.kappa, "theta": solver.theta,
                    "sigma": solver.sigma, "rho": solver.rho,
                    "T": solver.T, "Nt": solver.Nt, "dt": solver.dt,
                    "noh": solver.noh, "noi": solver.noi, "mu": solver.mu,
                    "Xmin": solver.Xmin, "Xmax": solver.Xmax,
                    "Vmin": solver.Vmin, "Vmax": solver.Vmax, "NX": solver.NX,
                    "chi": solver.chi
                },
                "iterations": iterations,
                "K_range": K_range
            }
            logger.info("Run parameters: %s", json.dumps(params))

            for K in K_range:
                print(f"\n{'-'*60}")
                print(f"Strike Price K = {K}")
                print(f"{'-'*60}")

                t0 = time.time()

                betaY, betaZ1, betaZ2 = solver.compute_betaY(K)

                prices = cp.zeros(iterations)
                for i in range(iterations):
                    if i % 10 == 0:
                        print(f"  Pricing iteration {i}/{iterations}")
                    prices[i] = solver.compute_option_price(K, betaY, betaZ1, betaZ2)

                mean_price = float(cp.mean(prices))
                std_price = float(cp.std(prices, ddof=1))
                elapsed = time.time() - t0

                print("\nResults:")
                print(f"  Mean Value: {mean_price:.6f}")
                print(f"  Std Value:  {std_price:.6f}")
                print(f"  Time:       {elapsed:.2f} seconds")

                # append structured result
                all_results.append({
                    "K": K,
                    "mean_price": mean_price,
                    "std_price": std_price,
                    "elapsed_sec": elapsed
                })

                logger.info(
                    "K=%s | mean=%.6f | std=%.6f | time=%.2f",
                    K, mean_price, std_price, elapsed
                )

            print("\n" + "=" * 80)
            print("Run completed")
            print("=" * 80)

    # Save all results ONCE
    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "results": all_results
    }

    with open(log_json, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)


if __name__ == '__main__':
    main()

