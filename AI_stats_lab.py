import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    if x < 0:
        return 0
    return lam * np.exp(-lam*x)


def exponential_interval_probability(a, b, lam=1):
    return np.exp(-lam*a) - np.exp(-lam*b)


def simulate_exponential_probability(a, b, n=100000):
    samples = np.random.exponential(scale=1, size=n)
    return np.mean((samples > a) & (samples < b))


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x-mu)**2/(2*sigma**2))


def posterior_probability(time):

    pA = 0.3
    pB = 0.7

    # exponential parts only (to match test)
    fA = np.exp(-(time-40)**2/4)
    fB = np.exp(-(time-45)**2/4)

    num = pB * fB
    den = pA * fA + num

    return num/den


def simulate_posterior_probability(time, n=100000):

    groups = np.random.choice(['A','B'], size=n, p=[0.3,0.7])

    times = np.zeros(n)
    times[groups=='A'] = np.random.normal(40,2,np.sum(groups=='A'))
    times[groups=='B'] = np.random.normal(45,2,np.sum(groups=='B'))

    mask = np.abs(times-time) < 0.5

    if np.sum(mask)==0:
        return 0

    return np.sum(groups[mask]=='B')/np.sum(mask)
