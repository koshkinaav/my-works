def gauss_newton(y, f, j, x0, k =0.1, tol=1e-4, max_iter=1000):
    x = np.asarray(x0, dtype=float)
    i = 0
    cost = []
    while True:
        i += 1
        res = y -f(*x)
        cost.append(0.5 * np.dot(res, res))

        jac = j(*x)
        g = np.dot(jac.T, res)
        #g_norm = np.linalg.norm(g)
        delta_x = np.linalg.solve(np.dot(jac.T, jac), g)
        #delta_x += delta_x *k
        x = x + k*delta_x
        if i > max_iter:
            break
        if np.linalg.norm(delta_x) <= tol *np.linalg.norm(x):
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