import numpy as np
import matplotlib.pyplot as plt

# ---------------- experiment grid ----------------
K_values   = np.array(range(1, 20, 4))     # 1,5,9,13,17
gamma_vals = np.linspace(0.05, 0.5, 5)    # phase σ
sigma      = 0.6                           # receiver noise
rho        = 0.6                           # correlation coefficient
n_samples  = 10_000_000                     # MC draws
chunk      = 50_000                        # memory block
# -------------------------------------------------

def cov_matrix(K, gamma, rho):
    """Γ_ij = γ² ρ^{|i-j|} (Toeplitz)."""
    d = np.abs(np.subtract.outer(np.arange(K), np.arange(K)))
    return (gamma**2) * (rho**d)

def mc_log_prob(Vobs, mu, Γ, σ, n=n_samples, blk=chunk):
    """MC marginal likelihood with correlated phases."""
    total, remain = 0.0, n
    while remain:
        m = min(blk, remain)
        d = np.random.multivariate_normal(mu, Γ, size=m)
        S = np.exp(1j*d).sum(axis=1)                    # V_k = 1
        p = np.exp(-np.abs(Vobs - S)**2 / (2*σ**2)) / (2*np.pi*σ**2)
        total += p.sum();  remain -= m
    return np.log(total / n)

def approx_log_prob(Vobs, mu, Γ, σ):
    c  = np.exp(1j*mu - 0.5*np.diag(Γ))                # E[e^{i d_k}]
    μS = c.sum()
    var_extra = 0.5 * np.sum(c[:,None] * c.conj()[None,:] * Γ).real
    Σ2 = σ**2 + var_extra
    return -np.log(2*np.pi*Σ2) - np.abs(Vobs-μS)**2 / (2*Σ2)

errs = np.zeros((len(K_values), len(gamma_vals)))
np.random.seed(42)                                     # reproducible

for i, K in enumerate(K_values):
    μk = np.linspace(0, np.pi, K, endpoint=False)
    for j, γ in enumerate(gamma_vals):
        Γ = cov_matrix(K, γ, rho)

        # single synthetic visibility
        d_true = np.random.multivariate_normal(μk, Γ)
        V_mod  = np.exp(1j*d_true).sum()
        Vobs   = V_mod + (np.random.normal(scale=sigma)
                          + 1j*np.random.normal(scale=sigma))

        errs[i, j] = (approx_log_prob(Vobs, μk, Γ, sigma)
                      - mc_log_prob   (Vobs, μk, Γ, sigma))
if __name__ == '__main__':

    # ------------------------- plot -------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(errs, origin='lower', aspect='auto',
                   extent=[gamma_vals[0], gamma_vals[-1],
                           K_values[0],  K_values[-1]],
                   cmap='coolwarm', vmin=-0.2, vmax=0.2)
    fig.colorbar(im, ax=ax, label='Approx − Monte-Carlo (log-likelihood)')
    ax.set_xlabel(r'$\gamma$ (phase standard deviation)')
    ax.set_ylabel(r'$K$ (number of phasors)')
    ax.set_title(fr'Gaussian approximation error   ($\rho={rho}$)')
    plt.tight_layout();  plt.show()
