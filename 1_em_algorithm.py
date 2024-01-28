#  An example from: https://www.geeksforgeeks.org/ml-expectation-maximization-algorithm/

import numpy as np
import seaborn as sb  # to find more about this package
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

# Generate two datasets with from two Gaussian distributions
# It is not use set.seed() so everytime the results would be different because of the
# randomness
mu1, sigma1 = 2, 1
mu2, sigma2 = -1, 0.8
X1 = np.random.normal(mu1, sigma1, size=200)
X2 = np.random.normal(mu2, sigma2, size=600)

# concatenating a sequence of arrays along an existing axis
# https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
X = np.concatenate([X1, X2])

# Plot the density estimation using seaborn (a Python data visualization library based
# on matplotlib)
sb.kdeplot(X)  # Plot univariate/bivariate distributions using kernel density estimation
plt.xlabel('X')
plt.ylabel('Density')
plt.title('Density Estimation of X')
plt.show()

# Initialize parameters: Use the expectations and standard deviations of the two samples
# as initial parameter estimates.
mu1_hat, sigma1_hat = np.mean(X1), np.std(X1)
mu2_hat, sigma2_hat = np.mean(X2), np.std(X2)
pi1_hat, pi2_hat = len(X1) / len(X), len(X2) / len(X)

# pi1_hat and pi2_hat are the initial estimates for the mixing coefficients or prior
# probabilities associated with the two components (or clusters) in a mixture model. In
# the context of the EM algorithm applied to mixture models, these coefficients represent
# the proportion of the overall dataset that is attributed to each component.


# Performing the EM algorithm for 20 epochs, i.e. iterating the Expectation (E) and
# Maximization (M) steps of the algorithm for a total of 20 cycles or iterations. Each
# cycle involves updating the estimates of parameters based on the observed data and the
# current estimates of latent variables.

num_epochs = 20
log_likelihoods = []

for epoch in range(num_epochs):
    # E step: compute responsibilities
    gamma1 = pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat)
    print(gamma1)
    gamma2 = pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)
    print(gamma2)
    total = gamma1 + gamma2
    gamma1 /= total
    gamma2 /= total

    # M-step: update parameteters
    mu1_hat = np.sum(gamma1 * X) / np.sum(gamma1)
    mu2_hat = np.sum(gamma2 * X) / np.sum(gamma2)
    sigma1_hat = np.sqrt(np.sum(gamma1 * (X - mu1_hat) ** 2) / np.sum(gamma1))
    sigma2_hat = np.sqrt(np.sum(gamma2 * (X - mu2_hat) ** 2) / np.sum(gamma2))
    pi1_hat = np.mean(gamma1)
    pi2_hat = np.mean(gamma2)

    # Compute log-likelihood
    log_likelihood = np.sum(np.log(pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat)
                                   + pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)))
    log_likelihoods.append(log_likelihood)


# Plot log-likelihood values over epochs
plt.plot(range(1, num_epochs+1), log_likelihoods)
plt.xlabel('Epoch')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs. Epoch')
plt.show()

# Plot the final estimated density
X_sorted = np.sort(X)
density_estimation = pi1_hat * norm.pdf(X_sorted,
                                        mu1_hat,
                                        sigma1_hat) + pi2_hat * norm.pdf(X_sorted,
                                                                         mu2_hat,
                                                                         sigma2_hat)

plt.plot(X_sorted, gaussian_kde(X_sorted)(X_sorted), color='green', linewidth=2)
plt.plot(X_sorted, density_estimation, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('Density')
plt.title('Density Estimation of X')
plt.legend(['Kernel Density Estimation', 'Mixture Density'])
plt.show()