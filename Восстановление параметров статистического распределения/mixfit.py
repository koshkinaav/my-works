import matplotlib.pyplot as plt
import numpy as np
import random
import math
import json
from scipy.optimize import minimize

tau = 0.25
mu1 = 0.5
mu2 = 1.5
sigma1 = 0.2
sigma2 = 0.7
n = 10000
x_1 = np.random.normal(mu1, sigma1, int(tau * n))
x_2 = np.random.normal(mu2, sigma2, int((1 - tau) * n))
x = np.r_[x_1, x_2]


def max_likelihood(th, x):
    x = np.array(x)
    tau = th[0]
    mu1 = th[1]
    mu2 = th[2]
    sigma1 = th[3]
    sigma2 = th[4]

    f = lambda x: np.log(
        tau / np.sqrt(2 * np.pi * sigma1 ** 2) * np.exp(-0.5 * ((x - mu1) ** 2) / sigma1 ** 2) + (1 - tau) / + np.sqrt(
            2 * np.pi * sigma2 ** 2) * np.exp(-0.5 * ((x - mu2) ** 2) / sigma2 ** 2))
    return -np.sum(f(x))


th_init = (0.1, 0.5, 2, 0.1, 1)
res = minimize(max_likelihood, th_init, args=x, tol=1e-3).x
print(res)


def e(x, tau, mu1, mu2, sigma1, sigma2):
    tau0 = tau
    tau1 = 1 - tau

    T1 = tau0 / np.sqrt(2 * np.pi * sigma1) * np.exp(-0.5 * ((x - mu1) ** 2) / sigma1)
    T2 = tau1 / np.sqrt(2 * np.pi * sigma2) * np.exp(-0.5 * ((x - mu2) ** 2) / sigma2)
    T = T1 + T2

    T1 = np.divide(T1, T, out=np.full_like(T, 0.5), where=T != 0)
    T2 = np.divide(T2, T, out=np.full_like(T, 0.5), where=T != 0)
    return np.vstack((T1, T2))


def m(x, *old):
    T1, T2 = e(x, *old)
    tau = np.sum(T1) / np.sum(T1 + T2)
    mu1 = np.sum(x * T1) / np.sum(T1)
    mu2 = np.sum(x * T2) / np.sum(T2)
    sigma1 = np.sum((x - mu1) ** 2 * T1) / np.sum(T1)
    sigma2 = np.sum((x - mu2) ** 2 * T2) / np.sum(T2)
    return tau, mu1, mu2, sigma1, sigma2


def em_double_gauss(x, tau, mu1, mu2, sigma1, sigma2):
    th = (tau, mu1, mu2, sigma1, sigma2)
    for i in range(100):
        th = m(x, *th)

    return (th[0], th[1], th[2], th[3] ** 0.5, th[4] ** 0.5)


def t(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2):
    x = np.array(x)
    tau0 = 1.0 - tau1 - tau2
    T0 = tau0 / (2 * np.pi * sigma02) * np.exp(-0.5 * np.sum(x[2:] ** 2, axis=0) / sigma02)
    l = 1.0 / (2 * np.pi * sigmav2) * np.exp(-0.5 * np.sum((x[2:] - muv.reshape(-1, 1)) ** 2, axis=0) / sigmav2)
    T1 = tau1 / (2 * np.pi * sigmax2) * np.exp(-0.5 * np.sum((x[:2] - mu1.reshape(-1, 1)) ** 2, axis=0) / sigmax2) * l
    T2 = tau2 / (2 * np.pi * sigmax2) * np.exp(-0.5 * np.sum((x[:2] - mu2.reshape(-1, 1)) ** 2, axis=0) / sigmax2) * l
    T = T0 + T1 + T2

    T0 = np.divide(T0, T, out=np.full_like(T, 1.0 / 3), where=T != 0)
    T1 = np.divide(T1, T, out=np.full_like(T, 1.0 / 3), where=T != 0)
    T2 = np.divide(T2, T, out=np.full_like(T, 1.0 / 3), where=T != 0)
    tau1 = np.sum(T1) / np.sum(T0 + T1 + T2)
    tau2 = np.sum(T2) / np.sum(T0 + T1 + T2)
    muv = np.sum((T1 + T2) * x[2:], axis=1) / np.sum(T1 + T2)
    mu1 = np.sum(T1 * x[:2], axis=1) / np.sum(T1)
    mu2 = np.sum(T2 * x[:2], axis=1) / np.sum(T2)
    sigma02 = np.sum(T0 * np.sum(x[2:] ** 2, axis=0)) / np.sum(T0) / 2
    sigmax2 = np.sum(T1 * np.sum((x[:2] - mu1.reshape(-1, 1)) ** 2, axis=0) +
                     T2 * np.sum((x[:2] - mu2.reshape(-1, 1)) ** 2, axis=0)) / np.sum(T1 + T2) / 2
    sigmav2 = np.sum((T1 + T2) * np.sum((x[2:] - muv.reshape(-1, 1)) ** 2, axis=0)) / np.sum(T1 + T2) / 2
    return (tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2)


def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2):
    th = (tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2)
    for i in range(100):
        th = t(x, *th)

    return (th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7])


print(em_double_gauss(x, tau, mu1, mu2, sigma1, sigma2))
