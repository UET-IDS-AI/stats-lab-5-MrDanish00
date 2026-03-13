import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    For exponential distribution:

    P(a < X < b) = e^{-lam*a} - e^{-lam*b}
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)
    count = np.sum((samples > a) & (samples < b))
    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    return (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-(x-mu)**2/(2*sigma**2))


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.
    """

    # Priors
    pA = 0.3
    pB = 0.7

    # Likelihoods
    fA = gaussian_pdf(time, 40, 2)
    fB = gaussian_pdf(time, 45, 2)

    # Bayes rule
    numerator = pB * fB
    denominator = pA * fA + pB * fB

    return numerator / denominator


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """
    pA = 0.3
    pB = 0.7

    groups = np.random.choice(['A', 'B'], size=n, p=[pA, pB])

    times = np.zeros(n)

    times[groups == 'A'] = np.random.normal(40, 2, np.sum(groups == 'A'))
    times[groups == 'B'] = np.random.normal(45, 2, np.sum(groups == 'B'))

    mask = np.abs(times - time) < 0.5
    if np.sum(mask) == 0:
        return 0

    return np.sum(groups[mask] == 'B') / np.sum(mask)
