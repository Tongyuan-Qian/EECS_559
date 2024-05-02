import numpy as np
from scipy import stats
from numpy.linalg import norm
from numpy.fft import fft, ifft
from numpy import conj, real


def soft(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)


def ber(size, p=0.5):
    return stats.bernoulli.rvs(size=int(size), p=float(p))


def nor(size, mean=0, std_dev=1):
    return stats.norm.rvs(size=int(size), loc=float(mean), scale=float(std_dev))


def c_conv(a, b, n):
    p_a = np.pad(a, (0, n - len(a)), "constant")
    p_b = np.pad(b, (0, n - len(b)), "constant")
    return real(ifft(fft(p_a, n) * fft(p_b, n)))


def e2r_grad(a, e_grad):
    return e_grad - np.inner(a, e_grad) * a


def exp_f(a, u):
    u_norm = norm(u)
    return np.cos(u_norm) * a + np.sin(u_norm) * u / u_norm
