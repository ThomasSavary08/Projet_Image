# Libraries
import ot
import tqdm
import numpy as np
from typing import Tuple
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal

def GaussianW2(m0: np.ndarray, m1: np.ndarray, sigma0: np.ndarray, sigma1: np.ndarray) -> float:
    """
    Compute W_2 distance between two Gaussian distribution
    Input(s)
        m0 (np.ndarray): mean of the first distribution with dimension (d,)
        m1 (np.ndarray): mean of the second distribution with dimension (d,)
        sigma0 (np.ndarray): covariance matrix of the first distribution with dimension (d,d)
        sigma1 (np.ndarray): covariance matrix of the second distribution with dimension (d,d)
    Returns
        res (float): W_2 distance between the two Gaussian distributions
    """
    sqrt_sigma0 = sqrtm(sigma0)
    sqrt_term = sqrtm(sqrt_sigma0@sigma1@sqrt_sigma0)
    res = np.linalg.norm(m0-m1)**2 + np.trace(sigma0 + sigma1 -2*sqrt_term)
    return res

def GaussianTransportMap(m0: np.ndarray, m1: np.ndarray, 
                         sigma0: np.ndarray, sigma1: np.ndarray, 
                         x: np.ndarray) -> np.ndarray:
    """
    Compute T(x) with T the optimal transport between two Gaussian distributions
    Input(s):
        m0 (np.ndarray): mean of the first distribution with dimension (d,) 
        m1 (np.ndarray): mean of the second distribution with dimension (d,)
        sigma0 (np.ndarray): covariance matrix of the first distribution with dimension (d,d)
        sigma1 (np.ndarray): covariance matrix of the second distribution with dimension (d,d)
        x (np.ndarray): point(s) at which T is evaluated with dimension (d,n)
    Returns
        res (np.ndarray): T(x) with dimension (d,n)
    """
    num_samples = x.shape[1]
    B = sqrtm(sigma0@sigma1)
    matrix = np.linalg.solve(sigma0, B)
    res = np.repeat(m1[:, np.newaxis], num_samples, axis = 1) + matrix@(x - np.repeat(m0[:, np.newaxis], num_samples, axis = 1))
    return res

def MW2(pi0: np.ndarray, pi1: np.ndarray,
        m0: np.ndarray, m1: np.ndarray,
        sigma0: np.ndarray, sigma1: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute MW2 distance between two GMM
    Input(s)
        pi0 (np.ndarray): mixture coefficients of the first GMM with dimension (K0,)
        pi1 (np.ndarray): mixture coefficients of the second GMM with dimension (K1,)
        m0 (np.ndarray): means of the first GMM with dimension (K0,d)
        m1 (np.ndarray): means of the second GMM with dimension (K1,d)
        sigma0 (np.ndarray): coraviance matrices of the first GMM with dimension (K0,d,d)
        sigma1 (np.ndarray): coraviance matrices of the second GMM with dimension (K1,d,d)
    Returns
        wstar (np.ndarray): discrete optimal transport plan with dimension (K0,K1)
        res (float): MW2 distance between the two GMM
    """
    K0, K1 = len(pi0), len(pi1)
    M = np.zeros((K0,K1))
    for i in range(K0):
        for j in range(K1):
            M[i,j] = GaussianW2(m0[i], m1[j], sigma0[i], sigma1[j])
    wstar = ot.emd(pi0, pi1, M)
    res = np.sum(wstar*M)
    return wstar, res

def MW2TransportMap(pi0: np.ndarray, pi1: np.ndarray,
                    m0: np.ndarray, m1: np.ndarray,
                    sigma0: np.ndarray, sigma1: np.ndarray,
                    wstar: np.ndarray, x: np.ndarray, method: str = 'rand') -> np.ndarray:
    """
    Compute T_mean(x) with T_mean the transport map defined in the section 6.3 of the paper
    Input(s)
        pi0 (np.ndarray): mixture coefficients of the first GMM with dimension (K0,)
        pi1 (np.ndarray): mixture coefficients of the second GMM with dimension (K1,)
        m0 (np.ndarray): means of the first GMM with dimension (K0,d)
        m1 (np.ndarray): means of the second GMM with dimension (K1,d)
        sigma0 (np.ndarray): coraviance matrices of the first GMM with dimension (K0,d,d)
        sigma1 (np.ndarray): coraviance matrices of the second GMM with dimension (K1,d,d)
        wstar (np.ndarray): discrete optimal transport plan with dimension (K0,K1)
        x (np.ndarray): point at which T is evaluated with dimension (d,n)
        method (str): mapping function to use
    Returns
        res (np.ndarray): T_rand/mean(x) with dimension (d,n)
    """
    K0, K1, d, n = len(pi0), len(pi1), x.shape[0], x.shape[1]
    if (method != 'rand'):
        numerator, denominator = np.zeros((d,n)), np.zeros(n)
        for k in range(K0):
            temp = np.zeros_like(numerator)
            for l in range(K1):
                temp += wstar[k,l]*GaussianTransportMap(m0[k], m1[l], sigma0[k], sigma1[l], x)
            pdf = multivariate_normal.pdf(x = x.T, mean = m0[k], cov = sigma0[k])
            numerator += pdf*temp
            denominator += pi0[k]*pdf
        res = numerator/denominator
    else:
        res = np.zeros_like(x)
        for i in tqdm.tqdm(range(n)):
            P = np.zeros((K0,K1))
            normalizer = 0.
            for k in range(K0):
                pdf = multivariate_normal.pdf(x = x[:,i], mean = m0[k], cov = sigma0[k])
                normalizer += pi0[k]*pdf
                for l in  range(K1):
                    P[k,l] = wstar[k,l]*pdf
            P = P/normalizer
            flattened_probs = P.ravel()
            chosen_index = np.random.choice(len(flattened_probs), p = flattened_probs)
            k, l = np.unravel_index(chosen_index, P.shape)
            res[:,i] = GaussianTransportMap(m0[k], m1[l], sigma0[k], sigma1[l], x[:,i].reshape(d,1)).squeeze(-1)
    return res