import numpy as np
def lstsq_svd(a, b, rcond=None):
    a = np.atleast_2d(a)
    b = np.atleast_1d(b)
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    if rcond is None:
        where = (s != 0.0)
    else:
        where = s > s[0] * rcond
    x = vh.T @ np.divide(u.T[:s.shape[0], :] @ b, s, out=np.zeros(a.shape[1]), where=where)
    r = a @ x - b
    cost = np.inner(r, r)
    sigma0 = cost / (b.shape[0] - x.shape[0])
    var = vh.T @ np.diag(s ** (-2)) @ vh * sigma0

    return x, cost, var



def lstsq_ne(a, b):
    x = (np.linalg.inv(a.T @ a)) @ a.T @ b
    cost = (np.linalg.norm((b - (a @ x)))) ** 2
    var = (b - a @ x) @ (b - a @ x).T
    return x, cost, var

def lstsq_ne_x(a, b):
    x = (np.linalg.inv(a.T @ a)) @ a.T @ b
    return x

def lstsq(a, b, method):
    if (method == 'ne'):
        return lstsq_ne(a, b)
    if method == 'svd':
        return lstsq_svd(a, b, rcond=None)
