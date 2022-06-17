import numpy as np
import pandas as pd


class BaseDescent:

    def __init__(self, X, y):
        self.X = X
        self.l, self.d = X.shape[0], X.shape[1]
        self.y = y.reshape((self.l, 1))
        self.w = np.zeros((self.d, 1))

        self.lambda_ = 1e-11
        self.s0 = 1
        self.p = 0.01
        self.iteration = 1

        self.h = 0
        self.alpha = 1e-9

    def Learning_Rate(self):
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p

    def partial_k(self, k):
        partial = float(2 / self.l * ((-self.y + self.X @ self.w).T @ self.X[:, k]))

        return partial

    def solution(self, X_test):
        weights = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        y_pred = np.dot(X_test, weights)
        return y_pred.reshape((len(y_pred), 1))

    def predict(self, x):
        y_pred = x @ self.fit()

        return y_pred

    def loss(self, y_test, y_pred):
        return 1 / len(y_test) * np.sum((y_test - y_pred) ** 2)


class VanillaGradientDescent(BaseDescent):

    def calc_gradient(self):

        gradient = 2 / self.l * ((-self.y + self.X @ self.w).T @ self.X)

        return np.array(gradient, dtype=float).reshape(self.d, 1)

    def update_weights(self):
        gradient = self.calc_gradient()
        return self.Learning_Rate() * gradient

    def fit(self, max_iter=10000, tolerance=0.000001):
        w = self.w - self.update_weights()

        while self.iteration < max_iter and np.linalg.norm(w - self.w) > tolerance and None not in w:
            self.w = w
            w = self.w - self.update_weights()
        return self.w


class StochasticDescent(BaseDescent):

    def calc_gradient(self):

        A = int(np.random.randint(1, self.d, size=1))
        B = np.random.randint(1, self.d, size=A)
        gradient = []

        for k in B:
            gradient.append(self.partial_k(k))
        stoch_grad = np.sum(gradient)

        return A, stoch_grad

    def update_weights(self):
        A, gradient = self.calc_gradient()
        return self.Learning_Rate() / A * gradient

    def fit(self, max_iter=10, tolerance=0.01):

        w = self.w - self.update_weights()

        while self.iteration < max_iter and np.linalg.norm(w - self.w) > tolerance and None not in w:
            self.w = w
            w = self.w - self.update_weights()
        return self.w


class MomentumDescent(BaseDescent):

    def update_weights(self):
        eta = self.Learning_Rate()
        gradient = self.partial_k(self.iteration)
        self.h = self.alpha * self.h + eta * gradient

        return self.h

    def fit(self, tolerance=0.01):
        max_iter = self.d - 1
        w = self.w - self.update_weights()

        while self.iteration < max_iter and np.linalg.norm(w - self.w) > tolerance and None not in w:
            self.w = w
            w = self.w - self.update_weights()

        return self.w


class Adam(BaseDescent):

    def update_weights(self):
        eps = 1e-8
        m = np.zeros(self.d)
        v = np.zeros(self.d)

        beta_1 = 0.9
        beta_2 = 0.999

        eta = self.Learning_Rate()

        m = beta_1 * m + (1 - beta_1) * self.partial_k(self.iteration)
        v = beta_2 * v + (1 - beta_2) * (self.partial_k(self.iteration) ** 2)
        M = m / (1 - beta_1 ** self.iteration)
        V = v / (1 - beta_2 ** self.iteration)

        delta = eta * M / (np.sqrt(V) + eps)

        return delta

    def fit(self, tolerance=0.01):
        max_iter = self.d
        w = self.w - self.update_weights()

        while self.iteration < max_iter and np.linalg.norm(w - self.w) > tolerance and None not in w:
            self.w = w
            w = self.w - self.update_weights()

        return self.w


class BaseDescentReg(VanillaGradientDescent):

    def partial_k_Reg(self, k):

        mu = 0.2

        return self.partial_k(k) + mu * np.linalg.norm(self.w) * self.w[k][0]

    def calc_gradient(self):
        gradient = []
        for k in range(self.d):
            gradient.append(self.partial_k_Reg(k))

        return np.array(gradient, dtype=float).reshape(self.d, 1)

    def update_weights(self):
        gradient = self.calc_gradient()
        return self.Learning_Rate() * gradient

    def fit(self, max_iter=10000, tolerance=0.0001):

        w = self.w - self.update_weights()

        while self.iteration < max_iter and np.linalg.norm(w - self.w) > tolerance:
            self.w = w
            w = self.w - self.update_weights()
        return self.w
