import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42) # The Answer to the Ultimate Question of Life, the Universe, and Everything

# --- Core Function: Spectral Coherence Coefficient C_N ---
def compute_C_stats(s, Ns):
    """
    Computes mean and variance of C_N for a sequence s over various window sizes N.
    s: array of normalized computational costs (stationary, mean ~ 1)
    Ns: list of window sizes
    """
    stats = []
    for N in Ns:
        c_values = []
        # Use a stride to reduce correlation between windows
        stride = max(1, N // 2)
        for i in range(0, len(s) - N, stride):
            window = s[i : i+N]
            num = np.sum(window[:-1]) # Sum of first N-1
            den = np.sum(window)      # Sum of all N
            if den > 0:
                c_values.append(num / den)
        
        c_values = np.array(c_values)
        if len(c_values) > 0:
            mean_c = np.mean(c_values)
            var_c = np.var(c_values)
            stats.append({'N': N, 'mean': mean_c, 'var': var_c, 'count': len(c_values)})
    
    return pd.DataFrame(stats)

# --- Simulation of Computational Traces (DPLL) ---

# 1. Polynomial Regime (Easy / Laminar)
# Modeled by short-range correlations. The solver finds local implications quickly.
# We use an AR(1) process with negative phi to simulate the "correction" of the search path.
def generate_poly_trace(n_steps):
    phi = -0.30 
    noise = np.random.normal(1, 0.2, n_steps) 
    costs = np.zeros(n_steps); costs[0] = 1.0
    for t in range(1, n_steps):
        costs[t] = 1.0 + phi * (costs[t-1] - 1.0) + (noise[t] - 1.0)
    # Ensure positivity (log-nodes count is positive)
    costs = np.maximum(costs, 0.01)
    costs = costs / np.mean(costs)
    return costs

# 2. Hard Regime (Exponential / Turbulent)
# Modeled by "Backtracking Waves".
# We use 1/f^alpha noise to simulate the long-range memory of deep backtracks.
def generate_hard_trace(n_steps):
    # Spectral density S(f) ~ 1/f^0.6 (Long range correlations)
    white = np.random.normal(0, 1, n_steps)
    freqs = np.fft.rfftfreq(n_steps)
    alpha = 0.6 
    with np.errstate(divide='ignore'):
        scale = 1.0 / np.power(np.maximum(freqs, 1e-10), alpha/2)
    scale[0] = 0
    long_range = np.fft.irfft(np.fft.rfft(white) * scale, n=n_steps)
    
    # Map to positive computational costs with heavy tails
    costs = np.exp(long_range)
    costs = costs / np.mean(costs)
    return costs

# --- Main Execution ---
n_steps = 300000
window_sizes = [5, 10, 20, 40, 80, 160, 320]

print("Generating Polynomial (Easy) Trace...")
s_poly = generate_poly_trace(n_steps)
print("Generating Hard (Backtracking) Trace...")
s_hard = generate_hard_trace(n_steps)

print("Computing statistics...")
df_poly = compute_C_stats(s_poly, window_sizes)
df_hard = compute_C_stats(s_hard, window_sizes)

# Theoretical Mean
df_poly['theory_mean'] = (df_poly['N'] - 1) / df_poly['N']

# --- Plotting ---

# Figure SAT1: Mean C_N vs N (Polynomial)
plt.figure(figsize=(8, 5))
plt.plot(df_poly['N'], df_poly['mean'], 'o-', label='Simulated (Polynomial)', color='blue')
plt.plot(df_poly['N'], df_poly['theory_mean'], 'x--', label='Theory (N-1)/N', color='red')
plt.xlabel('Window Size N')
plt.ylabel('Mean Coherence $E[C_N]$')
plt.title('SAT1. Mean Spectral Coherence vs N (Polynomial Regime)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xscale('log')
plt.savefig('Fig_SAT1_Mean.png')

# Figure SAT2: Variance vs N (Polynomial)
plt.figure(figsize=(8, 5))
plt.loglog(df_poly['N'], df_poly['var'], 'o-', label='Simulated (Polynomial)', color='blue')
# Reference line N^-2
ref_x = np.array(window_sizes)
ref_y = df_poly['var'].iloc[0] * (ref_x[0] / ref_x)**2
plt.loglog(ref_x, ref_y, 'k--', label='Reference $N^{-2}$ (Stable)')
plt.xlabel('Window Size N (log)')
plt.ylabel('Variance Var($C_N$) (log)')
plt.title('SAT2. Variance of Coherence (Structural Stability)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('Fig_SAT2_Variance.png')

# Figure SAT3: Stress Test (Poly vs Hard)
plt.figure(figsize=(8, 5))
plt.loglog(df_poly['N'], df_poly['var'], 'o-', label='Polynomial (Easy)', color='blue')
plt.loglog(df_hard['N'], df_hard['var'], 's-', label='Hard (Backtracking Waves)', color='red')
plt.xlabel('Window Size N (log)')
plt.ylabel('Variance Var($C_N$) (log)')
plt.title('SAT3. Bridge A: Detection of Hardness Signature')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('Fig_SAT3_StressTest.png')

# Output data for Table
print("\nPolynomial Statistics:")
print(df_poly[['N', 'mean', 'var']])
print("\nHard Statistics:")
print(df_hard[['N', 'mean', 'var']])