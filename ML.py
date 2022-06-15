import numpy as np
import pandas as pd


class Linear_Regression():

    def __init__(self, X, y):

        self.X = X
        self.y = y
        self.l = X.shape[0]
        self.d = X.shape[1]
        self.w0 = np.zeros(self.d)

    def solution(self, X_test):

        weights = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        y_pred = np.dot(X_test, weights)
        return y_pred.reshape((len(y_pred), 1))

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

    def gradient_descent(self, lambd = 1e-6, max_iter=1000, eps=0.1):

        iter = 0

        w = self.w0

        W = w - lambd * self.grad_L(w)

        while iter < max_iter or np.linalg.norm(W - w) > eps:
            w = W
            W = w - lambd * self.grad_L(w)
            iter += 1


        return W

    def predict(self, X_test):

        weights = self.gradient_descent()
        y_pred = np.dot(X_test, weights)
        return y_pred.reshape((len(y_pred), 1))

    def score(self, y1, y2):

        score = (1/len(y1)) * np.sum((y1 - y2)**2)

        return score

