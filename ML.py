import numpy as np
import pandas as pd


class Linear_Regression():

    def __init__(self, X, y):

        self.X = X
        self.y = y
        self.l = X.shape[0]
        self.d = X.shape[1]
        self.w0 = np.zeros(self.d)

    def solution(self):

        return np.linalg.inv(self.X.T @ self.X) @ self.X.T * self.y

    def partial_k(self, w, k):

        partial = 0
        for i in range(self.l):
            partial += (np.dot(self.X[i], w.reshape((self.d, 1))) - self.y[i]) * self.X[i][k]

        return float(2 / self.l * partial)

    def grad_L(self, w):

        gradient = []
        for k in range(self.d):
            gradient.append(self.partial_k(w, k))

        return np.array(gradient)

    def gradient_descent(self, lambd, max_iter, eps):

        iter = 0

        w = self.w0

        W = w - lambd * self.grad_L(w)

        while iter < max_iter or np.linalg.norm(W - w) > eps:
            w = W
            W = w - lambd * self.grad_L(w)
            iter += 1
            print(W)

        return W
