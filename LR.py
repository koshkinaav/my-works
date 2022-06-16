import numpy as np
import pandas as pd


class BaseDescent:

    def __init__(self, X, y):
        self.X = X
        self.l, self.d = X.shape[0], X.shape[1]
        self.y = y.reshape((self.l, 1))
        self.w = np.random.rand(self.d, 1)

        self.lambda_ = 1e-6
        self.s0 = 1
        self.p = 0.5
        self.iteration = 1

        self.h = 0
        self.alpha = 1e-9

    def Learning_Rate(self):
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p

    def partial_k(self, k):
        partial = 0
        for i in range(self.l):
            partial += (np.dot(self.X[i], self.w) - self.y[i]) * self.X[i][k]

        return float(2 / self.l * partial)

    def solution(self, X_test):
        weights = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        y_pred = np.dot(X_test, weights)
        # return y_pred.reshape((len(y_pred), 1))
        return weights

    def predict(self, x):
        y_pred = x @ self.fit()

        return y_pred


class VanillaGradientDescent(BaseDescent):

    def calc_gradient(self):
        gradient = []
        for k in range(self.d):
            gradient.append(self.partial_k(k))

        return np.array(gradient, dtype=float).reshape(self.d, 1)

    def update_weights(self):
        gradient = self.calc_gradient()
        return self.Learning_Rate() * gradient

    def fit(self, max_iter=100, eps=0.001):

        w = self.w - self.update_weights()

        while self.iteration < max_iter and np.linalg.norm(w - self.w) > eps:
            self.w = w
            w = self.w - self.update_weights()
        return self.w


class StochasticDescent(BaseDescent):

    def calc_gradient(self):

        B = int(np.random.randint(1, self.d, size=1))
        B = np.random.randint(1, self.d, size=B)
        gradient = []

        for k in B:
            gradient.append(self.partial_k(k))
        stoch_grad = np.sum(gradient)

        return stoch_grad

    def update_weights(self):
        gradient = self.calc_gradient()
        return self.Learning_Rate() * gradient

    def fit(self, max_iter=100, eps=0.001):

        w = self.w - self.update_weights()

        while self.iteration < max_iter and np.linalg.norm(w - self.w) > eps:
            self.w = w
            w = self.w - self.update_weights()
        return self.w


class MomentumDescent(BaseDescent):

    def update_weights(self):
        eta = self.Learning_Rate()
        gradient = self.partial_k(self.iteration)
        self.h = self.alpha * self.h + eta * gradient

        return self.h

    def fit(self, eps=0.01):
        max_iter = self.d
        w = self.w - self.update_weights()

        while self.iteration < max_iter and np.linalg.norm(w - self.w) > eps:
            self.w = w
            w = self.w - self.update_weights()

        return self.w
    
