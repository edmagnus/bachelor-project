import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geninvgauss
from scipy.special import kv
from scipy.stats import gaussian_kde

# Sample from a d-dimensional Generalized Hyperbolic distribution
def sample_ghd(
    lam: float,
    alpha: float,
    beta: np.ndarray,
    delta: float,
    mu: np.ndarray,
    Lambda: np.ndarray,
    n: int = 1,
    rng: np.random.Generator = None
) -> np.ndarray:
    
    # Set up random number generator (for reproducibility)
    if rng is None:
        rng = np.random.default_rng()

    # Ensure inputs are numpy arrays of the correct dtype
    beta = np.asarray(beta, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)

    # Find dimension of input (could also infer from beta or Lambda)
    d = mu.shape[0]

    # Validate parameters and compute the GIG parameter γ
    gamma2 = alpha**2 - beta @ Lambda @ beta # γ² = α² - βᵀΛβ

    # Ensures that the GIG distribution is well-defined (γ² > 0)
    if gamma2 < 0:
        raise ValueError(
            f"Invalid parameters: need α² > βᵀΛβ, but input was "
            f"α²={alpha**2:.6g}, βᵀΛβ={beta @ Lambda @ beta:.6g}."
        )
    
    # Save gamma for sampling Y ~ GIG(λ, δ, γ) below
    gamma = np.sqrt(gamma2)

    # Not implemented
    if gamma == 0.0 or delta == 0.0:
        raise NotImplementedError(
            "The boundary cases γ = 0 (symmetric Student-t limit) and δ = 0 "
            "(gamma/normal-gamma limit) require separate treatment not "
            "implemented here. Perturb parameters slightly to stay interior."
        )
    
    # Sample Y ~ GIG(λ, δ, γ) via scipy's geninvgauss
    # scipy parameterisation: p = λ,  b = δγ,  scale = δ/γ
    # Output is shape (n,)
    Y = geninvgauss.rvs(    
        p=lam,
        b=delta * gamma,
        scale=delta / gamma,
        size=n,
        random_state=rng
    )

    # Reshape to (n, 1) for use in building X
    Y = Y[:, None] 

    # Square root of Lambda
    eigvals, Q = np.linalg.eigh(Lambda)
    if np.any(eigvals <= 0):
        raise ValueError("Lambda must be positive definite.")
    Lambda_sqrt = Q @ np.diag(np.sqrt(eigvals)) @ Q.T  

    # Sample n copies of η ~ N(0, I_d)
    eta = rng.standard_normal(size=(n, d))

    # Build components of X = μ + Y·Λβ + √Y·Λ^(1/2)·η
    Lambda_beta = Lambda @ beta # (d, 1)
    middle_term = Y * Lambda_beta[None, :] # (n, d)
    last_term = np.sqrt(Y) * (eta @ Lambda_sqrt.T) # (n, d)

    # Build X = μ + Y·Λβ + √Y·Λ^(1/2)·η
    # (Note: mu[None, :] is only shape(1, d), but simply broadcast to (n, d) for addition)
    X = mu[None, :] + middle_term + last_term # (n, d)

    # Return the samples as a numpy array of shape (n, d)
    return X

# Univariate GH density function for plotting marginals
def gh1_pdf(x, lam, alpha, beta, delta, mu):
    
    x = np.asarray(x, dtype=float)
    gamma2 = alpha**2 - beta**2
    if gamma2 <= 0:
        raise ValueError("Need alpha^2 > beta^2 for the univariate GH density.")

    gamma = np.sqrt(gamma2)
    v = np.sqrt(delta**2 + (x - mu)**2) 

    prefactor = (gamma / delta) ** lam
    prefactor /= (np.sqrt(2.0 * np.pi) * kv(lam, delta * gamma))

    return prefactor * np.exp(beta * (x - mu)) * (v / alpha) ** (lam - 0.5) * kv(lam - 0.5, alpha * v)

# Use Lemma A.1 to compute marginal parameters for component j
def ghd_marginal_params(j, lam, alpha, beta, delta, mu, Lambda):

    # Ensure inputs are numpy arrays of the correct dtype
    beta = np.asarray(beta, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)

    d = len(mu)
    lambda_jj = Lambda[j, j]

    if lambda_jj <= 0:
        raise ValueError("Lambda must be positive definite.")

    # d = 1 is a special case of the same transformation
    if d == 1:
        return {
            "lam": lam,
            "alpha": alpha / np.sqrt(lambda_jj),
            "beta": beta[0],
            "delta": delta * np.sqrt(lambda_jj),
            "mu": mu[0],
        }

    beta_rest = np.delete(beta, j)
    Lambda12 = np.delete(Lambda[j, :], j)               
    Lambda22 = np.delete(np.delete(Lambda, j, axis=0), j, axis=1)

    diff = Lambda22 - np.outer(Lambda12, Lambda12) / lambda_jj

    alpha_j_sq = (alpha**2 - beta_rest @ diff @ beta_rest) / lambda_jj

    alpha_j = np.sqrt(alpha_j_sq)
    beta_j = beta[j] + (Lambda12 @ beta_rest) / lambda_jj
    delta_j = delta * np.sqrt(lambda_jj)
    mu_j = mu[j]

    return {
        "lam": lam,
        "alpha": alpha_j,
        "beta": beta_j,
        "delta": delta_j,
        "mu": mu_j,
    }

# Plot the marginal densities for each component and compare to empirical KDE from samples
def plot_ghd_marginals(
    X,
    lam,
    alpha,
    beta,
    delta,
    mu,
    Lambda,
    bw_method="scott",
    gridsize=600,
    q_low=0.001,
    q_high=0.999,
    pad_frac=0.15,
    title=None,
):
    """
    Overlay empirical KDE and theoretical marginal GH density for each component.

    Parameters
    ----------
    X        : (n, d) array
               Simulated samples.
    lam, alpha, beta, delta, mu, Lambda
               Same parameters used in sample_ghd.
    bw_method : str or float
               Passed to scipy.stats.gaussian_kde.
    gridsize : int
    q_low, q_high : float
               Quantiles used to define plotting range.
    pad_frac : float
               Extra padding added to each component's x-range.
    title    : str or None
    """
    
    # Ensure inputs are numpy arrays of the correct dtype
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)

    # Save dimensions
    n, d = X.shape
    
    # Create figure and color cycle for plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for j in range(d):
        xj = X[:, j]
        params_j = ghd_marginal_params(j, lam, alpha, beta, delta, mu, Lambda)

        # empirical KDE
        kde = gaussian_kde(xj, bw_method=bw_method)

        # component-specific plotting grid
        lo, hi = np.quantile(xj, [q_low, q_high])
        width = hi - lo
        if width <= 0:
            width = max(np.std(xj), 1.0)

        lo -= pad_frac * width
        hi += pad_frac * width
        grid = np.linspace(lo, hi, gridsize)

        empirical = kde(grid)
        theoretical = gh1_pdf(grid, **params_j)

        color = colors[j % len(colors)]
        ax.plot(grid, theoretical, color=color, lw=2.0, label=f"Component {j+1} theory")
        ax.plot(grid, empirical, color=color, lw=2.0, ls="--", label=f"Component {j+1} empirical")

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(title or f"{d}-dim GH: empirical vs theoretical marginals")
    ax.legend(ncol=2, frameon=True)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

# Plot the marginal densities for each component and compare to empirical KDE from samples
def plot_ghd_marginals(
    X,
    lam,
    alpha,
    beta,
    delta,
    mu,
    Lambda,
    bw_method="scott",
    gridsize=600,
    q_low=0.001,
    q_high=0.999,
    pad_frac=0.15,
    title=None,
):
    """
    Overlay empirical KDE and theoretical marginal GH density for each component.

    Parameters
    ----------
    X        : (n, d) array
               Simulated samples.
    lam, alpha, beta, delta, mu, Lambda
               Same parameters used in sample_ghd.
    bw_method : str or float
               Passed to scipy.stats.gaussian_kde.
    gridsize : int
    q_low, q_high : float
               Quantiles used to define plotting range.
    pad_frac : float
               Extra padding added to each component's x-range.
    title    : str or None
    """
    
    # Ensure inputs are numpy arrays of the correct dtype
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)

    # Save dimensions
    n, d = X.shape
    
    # Create figure and color cycle for plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for j in range(d):
        xj = X[:, j]
        params_j = ghd_marginal_params(j, lam, alpha, beta, delta, mu, Lambda)

        # empirical KDE
        kde = gaussian_kde(xj, bw_method=bw_method)

        # component-specific plotting grid
        lo, hi = np.quantile(xj, [q_low, q_high])
        width = hi - lo
        if width <= 0:
            width = max(np.std(xj), 1.0)

        lo -= pad_frac * width
        hi += pad_frac * width
        grid = np.linspace(lo, hi, gridsize)

        empirical = kde(grid)
        theoretical = gh1_pdf(grid, **params_j)

        color = colors[j % len(colors)]
        ax.plot(grid, theoretical, color=color, lw=2.0, label=f"Component {j+1} theory")
        ax.plot(grid, empirical, color=color, lw=2.0, ls="--", label=f"Component {j+1} empirical")

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(title or f"{d}-dim GH: empirical vs theoretical marginals")
    ax.legend(ncol=2, frameon=True)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

# Compare sample expectation and variance to theoretical values using (A.7) and (A.8)
# Plot empirical vs theoretical marginal densities for each component
if __name__ == "__main__":
    
    # modified Bessel K (needed for theoretical mean/cov calculations)
    from scipy.special import kv

    # Choose seed for reproducibility
    rng = np.random.default_rng(seed=2026) 

    # Parameters
    dim     = 2
    lam     = -0.5
    alpha   = 0.01
    delta   = 1.0
    mu      = np.array([-1.0, 1.0])
    beta    = np.zeros(dim)
    beta[0] = 0.4 * alpha
    Lambda  = np.array([[1.0, 0.25], [0.25, 1.0]])

    # Gamma and zeta are calculated
    gamma = np.sqrt(alpha**2 - beta @ Lambda @ beta)
    zeta = delta * gamma

    print(f"gamma = {gamma:.6f},  delta*gamma = {zeta:.6f}\n")

    # Theoretical mean given by (A.7)
    R_lam = kv(lam + 1, zeta) / kv(lam, zeta)
    theoretical_mean = mu + (delta / gamma) * R_lam * (Lambda @ beta)

    # Draw samples
    N = 100_000
    X = sample_ghd(lam, alpha, beta, delta, mu, Lambda, n=N, rng=rng)

    sample_mean = X.mean(axis=0)
    sample_cov  = np.cov(X.T)

    print("=== Mean ===")
    print(f"  Theoretical : {theoretical_mean}")
    print(f"  Sample      : {sample_mean}")

    # Theoretical covariance Var[X] = (A.8)
    S_lam = (kv(lam + 2, zeta) * kv(lam, zeta) - kv(lam + 1, zeta)**2) \
            / kv(lam, zeta)**2
    theoretical_cov = ((delta / gamma) * R_lam) * Lambda \
                + ((delta**2 / gamma**2) * S_lam) * np.outer(Lambda @ beta, Lambda @ beta)

    print("\n=== Covariance ===")
    print(f"  Theoretical :\n{theoretical_cov}")
    print(f"  Sample      :\n{sample_cov}")

    # Compare quantiles
    component_1_sim = X[:, 0]
    component_2_sim = X[:, 1]

    for q in [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
        print(f"q = {q}")
        print("comp. 1 true: ", np.quantile(component_1_sim, q))
        print("comp. 2 true: ", np.quantile(component_2_sim, q))
        print()

    # Plot marginals
    plot_ghd_marginals(
        X,
        lam=lam,
        alpha=alpha,
        beta=beta,
        delta=delta,
        mu=mu,
        Lambda=Lambda,
        bw_method="scott",
        title="Empirical vs theoretical marginal GH densities",
    )