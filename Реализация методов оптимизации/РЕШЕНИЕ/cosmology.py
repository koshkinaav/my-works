import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from collections import namedtuple
from scipy.optimize import least_squares
import pylab as p
from scipy import integrate
from scipy.integrate import quad

Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
df = pd.read_csv('/Users/akonkina/Downloads/jla_mub 2.txt', sep=' ')
df = df.rename({'#': 'z', 'z': 'mu', 'mu': 'a'}, axis=1)
df = df.drop('a', axis=1)

z = np.array(df['z'])
mu = np.array(df['mu'])
z = np.array(df['z'])
mu = np.array(df['mu'])


def integral(z, Omega):
    intg = np.empty_like(z)
    for i, Z_i in np.ndenumerate(z):
        intg[i] = quad(lambda x: 1 / np.sqrt((1 - Omega) * (1 + x) ** 3 + Omega), 0, Z_i)[0]
    return intg


def f(z, H_0, Omega):
    ras = 3e10 / (1e-1 * H_0) * (1 + z) * integral(z, Omega)
    return 5 * np.log10(ras) - 5


def j(z, H_0, Omega):
    jac = np.empty((z.size, 2), dtype=float)
    jac[:, 0] = -5.0 / H_0 / np.log(10)
    for i, Z_i in np.ndenumerate(z):
        jac[i, 1] = 5.0 / np.log(10) / integral(Z_i, Omega) * \
                    quad(lambda x: -0.5 * ((1 - Omega) * (1 + x) ** 3 + Omega) ** (-1.5) * (1 - (1 + x) ** 3), 0, Z_i)[
                        0]
    return jac


def gauss_newton(y, f, j, x0, k=0.1, tol=1e-4, max_iter=1000):
    x = np.asarray(x0, dtype=float)
    i = 0
    cost = []
    while True:
        i += 1
        res = y - f(*x)
        cost.append(0.5 * np.dot(res, res))

        jac = j(*x)
        g = np.dot(jac.T, res)
        # g_norm = np.linalg.norm(g)
        delta_x = np.linalg.solve(np.dot(jac.T, jac), g)
        # delta_x += delta_x *k
        x = x + k * delta_x
        if i > max_iter:
            break
        if np.linalg.norm(delta_x) <= tol * np.linalg.norm(x):
            break
    cost = np.array(cost)
    return Result(nfev=1, cost=cost, gradnorm=np.linalg.norm(g), x=x)


def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-5, max_iter=1000):
    x = np.asarray(x0, dtype=float)
    i = 0
    cost = []
    delta_x = 0

    while True:
        i += 1
        res = y - f(*x)
        cost.append(0.5 * np.dot(res, res))
        jac = j(*x)
        g = np.dot(jac.T, res)
        I = np.eye(len(np.dot(jac.T, jac)))

        delta_x_1 = np.linalg.solve((np.dot(jac.T, jac) + lmbd0 * I), g)

        delta_x_2 = np.linalg.solve(np.dot(jac.T, jac) + lmbd0 / nu * I, g)

        F_1 = np.dot(y - f(*(x + delta_x_1)), y - f(*(x + delta_x_1)))

        F_2 = np.dot(y - f(*((x + delta_x_2))), y - f(*(x + delta_x_2)))

        P = np.dot(y - f(*x), y - f(*x))

        if F_2 <= P:
            lmbd0 = lmbd0 / nu
            x = x + delta_x_2
            delta_x = delta_x_2
        if (F_2 > P) & (F_1 <= P):
            lmbd0 = lmbd0
            x = x + delta_x_1
            delta_x = delta_x_1

        if (F_2 > P) & (F_1 > P):
            lambd = lmbd0 * nu
            delta = np.linalg.solve(np.dot(jac.T, jac) + lambd * I, g)
            while (np.dot(y - f(*(x + delta)), y - f(*(x + delta))) > P):
                lambd = lambd * nu
                delta = np.linalg.solve(np.dot(jac.T, jac) + lambd * I, g)
            x = x + delta
            delta_x = delta
            lmbd0 = lambd
        if i > 1000:
            break
        if np.linalg.norm(delta_x) <= tol * np.linalg.norm(x):
            break
    cost = np.array(cost)
    return Result(nfev=1, cost=cost, gradnorm=np.linalg.norm(g), x=x)


r_1 = gauss_newton(mu,
                   lambda *args: f(z, *args),
                   lambda *args: j(z, *args),
                   (50, 0.5))
r_2 = lm(mu,
         lambda *args: f(z, *args),
         lambda *args: j(z, *args),
         (50, 0.5))

plt.figure(figsize=(14, 7))
plt.plot(z, mu, '*', label='data')
plt.plot(z, f(z, *r_2.x), label='fit_1')
plt.plot(z, f(z, *r_1.x), label='fit_2')
plt.xlabel('Z')
plt.ylabel('Mu')
plt.legend()
plt.savefig('mu-z.png')
plt.clf()
plt.plot(r_1.cost, label='Gauss newton')
plt.plot(r_2.cost, label='Levenberg')
plt.xlabel('Итерационный шаг')
plt.ylabel('Функция потерь')
plt.legend()
plt.savefig('cost.png')
d = {
    "Gauss-Newton": {"H0": r_1.x[0], "Omega": r_1.x[1], "nfev": r_1.nfev},
    "Levenberg-Marquardt": {"H0": r_2.x[0], "Omega": r_2.x[1], "nfev": r_2.nfev}
}
with open('parameters.json', 'w') as f:
    json.dump(d, f)
