import numpy as np
from scipy.stats import bernoulli, uniform, beta


def simulate_data_without_confounding(n_samples, d, e, mu0, mu1):
    # Generate the feature vectors
    sigma = np.eye(d)  # Identity matrix as a placeholder for Σ
    X = np.random.multivariate_normal(np.zeros(d), sigma, n_samples)

    # Generate the potential outcomes
    Y0 = np.array([mu0(x) + np.random.normal() for x in X])
    Y1 = np.array([mu1(x) + np.random.normal() for x in X])

    # Generate the treatment assignments
    W = bernoulli.rvs(e, size=n_samples)

    # Generate the observed outcomes
    Y = W * Y1 + (1 - W) * Y0

    # Calculate ATE
    ate = np.mean(Y1 - Y0)
    TE = Y1 - Y0

    # Combine everything into a dataset
    data = np.column_stack([X, W, Y])
    return d, ate, X, W, Y, TE


def simulate_data_with_confounding(n_samples, d, mu0, mu1):
    X = uniform.rvs(size=(n_samples, d))

    # Generate the potential outcomes
    Y0 = np.array([mu0(x) + np.random.normal() for x in X])
    Y1 = np.array([mu1(x) + np.random.normal() for x in X])

    # Define the propensity score
    e = lambda x: 0.25 * (1 + beta.pdf(x[0], 2, 4))

    # Generate the treatment assignments
    W = bernoulli.rvs(np.array([e(x) for x in X]))

    # Generate the observed outcomes
    Y = W * Y1 + (1 - W) * Y0

    # Calculate ATE
    ate = np.mean(Y1 - Y0)
    TE = Y1 - Y0

    # Combine everything into a dataset
    data = np.column_stack([X, W, Y])
    return d, ate, X, W, Y, TE


def simulate_data_with_confounding_binary(n_samples, d, mu0, mu1):
    X = uniform.rvs(size=(n_samples, d))

    # Generate the potential outcomes and cap them to 0 and 1
    Y0 = np.clip(np.round([mu0(x) + np.random.normal() for x in X]), 0, 1).astype(int)
    Y1 = np.clip(np.round([mu1(x) + np.random.normal() for x in X]), 0, 1).astype(int)

    # Define the propensity score
    e = lambda x: 0.25 * (1 + beta.pdf(x[0], 2, 4))

    # Generate the treatment assignments
    W = bernoulli.rvs(np.array([e(x) for x in X]))

    # Generate the observed outcomes
    Y = W * Y1 + (1 - W) * Y0

    # Calculate ATE
    ate = np.mean(Y1 - Y0)
    TE = Y1 - Y0

    # Combine everything into a dataset
    data = np.column_stack([X, W, Y])
    return d, ate, X, W, Y, TE